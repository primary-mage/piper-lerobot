#!/usr/bin/env python3
"""松灵 Piper 真机环境：piper_sdk 读/写 CAN + D435i 相机 + MuJoCo 数字孪生做 IK。

与仿真版 PiperEnv 接口对齐（get_qpos / send_action / get_grasp_pos / solve_ik / render_image），
这样采集脚本只需切换底层环境即可，手柄 + IK + 采集逻辑完全复用。

注意：真机无"方块高度"这种地面真值，所以没有 reward / cube_height。
"""
import time

import cv2
import mujoco
import numpy as np

from piper_sim import PiperEnv  # 数字孪生（只为 IK + 夹爪中点，不参与真机物理）

try:
    from piper_sdk import C_PiperInterface, C_PiperInterface_V2
except ImportError:  # pragma: no cover
    C_PiperInterface = C_PiperInterface_V2 = None

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None

# piper_sdk 关节角单位：0.001 度（见 GetArmJointMsgs 文档）
DEG_TO_RAD = np.pi / 180.0
JOINT_INT_TO_RAD = 0.001 * DEG_TO_RAD   # 读：int → rad
JOINT_RAD_TO_INT = 1.0 / JOINT_INT_TO_RAD  # 发：rad → int

ARM_JOINT_ATTRS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_MAX_INT = 100000  # 夹爪满开编码器值（物理约 100mm）；0=闭合


class RealPiperEnv:
    """真机 Piper 环境。接口与 PiperEnv 对齐，供采集/部署脚本调用。"""

    def __init__(self, can_interface="can0", image_size=96, enable_arm=True, use_depth=False, move_spd_rate=40):
        self.can_interface = can_interface
        self.image_size = image_size
        self.use_depth = use_depth
        self.move_spd_rate = move_spd_rate  # 运动速度百分比 0~100，越低越平滑（默认 40）

        # 1. 真机连接（piper_sdk V2 协议）
        if C_PiperInterface_V2 is None:
            raise RuntimeError("未安装 piper_sdk，请先 pip install piper_sdk")
        self.piper = C_PiperInterface_V2(self.can_interface)
        self.piper.ConnectPort()
        if enable_arm:
            t0 = time.time()
            while not self.piper.EnablePiper():
                if time.time() - t0 > 5:
                    raise RuntimeError("使能手臂超时")
                time.sleep(0.01)
        # 预热：等 CAN 反馈更新，避免首帧读到全 0
        time.sleep(0.5)
        for _ in range(5):
            self.piper.GetArmJointMsgs()
            time.sleep(0.05)

        # 2. MuJoCo 数字孪生（只为 IK + 夹爪中点）
        self.twin = PiperEnv(image_size=image_size)

        # 3. D435i 相机（RGB 流；可选深度）
        self.pipeline = None
        if rs is not None:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            if use_depth:
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(cfg)
            for _ in range(30):  # 预热，等自动曝光稳定
                self.pipeline.wait_for_frames()
        else:
            print("[警告] 未安装 pyrealsense2，相机不可用。安装：pip install pyrealsense2")

    # ---- 真机读关节角 ----
    def get_qpos(self):
        """读真机关节角：6 手臂(rad) + 1 夹爪(归一化 0~1)。"""
        msgs = self.piper.GetArmJointMsgs()
        js = msgs.joint_state  # 关节角在 joint_state 里
        arm_rad = np.array(
            [getattr(js, n) * JOINT_INT_TO_RAD for n in ARM_JOINT_ATTRS], dtype=np.float32
        )
        gripper = self._read_gripper()
        return np.concatenate([arm_rad, [gripper]]).astype(np.float32)

    def _read_gripper(self):
        """读夹爪，映射到仿真约定：0=张开，0.035=闭合。"""
        g = self.piper.GetArmGripperMsgs().gripper_state
        raw = g.grippers_angle  # 编码器值：0=闭合，GRIPPER_MAX_INT=满开
        return 0.035 * (1.0 - raw / GRIPPER_MAX_INT)  # 满开→0，闭合→0.035

    # ---- 真机发关节指令 ----
    def send_action(self, action):
        """发 7 维目标：6 手臂关节(rad) + 夹爪(0~1)。"""
        action = np.asarray(action, dtype=np.float64)
        arm_int = [int(round(a * JOINT_RAD_TO_INT)) for a in action[:6]]
        self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd_rate, 0x00)  # CAN 控制 + MOVE J，速度可调
        self.piper.JointCtrl(*arm_int)
        gripper_int = int(round((1.0 - action[6] / 0.035) * GRIPPER_MAX_INT))  # 0=开(满开) 0.035=闭(0)
        self.piper.GripperCtrl(gripper_int, 1000, 0x01, 0)

    # ---- 数字孪生：把真机关节角同步进 MuJoCo 模型，再算夹爪中点/IK ----
    def _sync_twin(self):
        q = self.get_qpos()
        for i, qid in enumerate(self.twin.arm_qpos_ids):
            self.twin.data.qpos[qid] = q[i]
        self.twin.data.qpos[self.twin.gripper_qpos_id] = 0.0
        mujoco.mj_forward(self.twin.model, self.twin.data)
        return q

    def get_grasp_pos(self):
        """夹爪两指中点（世界坐标），用数字孪生算。"""
        self._sync_twin()
        return self.twin.get_grasp_pos()

    def solve_ik(self, target_pos):
        """用数字孪生做 IK，返回 6 个手臂关节角(rad)。"""
        self._sync_twin()
        return self.twin.solve_ik(target_pos)

    # ---- 相机 ----
    def render_image(self, size=None):
        """从 D435i 读 RGB 图（BGR→RGB，resize 到 size）。"""
        if self.pipeline is None:
            raise RuntimeError("D435i 相机未初始化")
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        img = np.asanyarray(color.get_data())  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size = size or self.image_size
        if img.shape[0] != size or img.shape[1] != size:
            img = cv2.resize(img, (size, size))
        return img

    def close(self):
        try:
            self.piper.GripperCtrl(0, 1000, 0x00, 0)  # 失能夹爪
            # 进入拖动示教悬停（重力补偿+阻尼），不断电，避免退出时手臂砸落
            self.piper.MotionCtrl_1(0x00, 0x00, 0x01)
            time.sleep(0.2)
            self.piper.DisconnectPort()
            print("[提示] 手臂已进入拖动示教悬停（重力补偿），可手推调整或断电")
        except Exception:
            pass
        if self.pipeline is not None:
            self.pipeline.stop()


if __name__ == "__main__":
    env = RealPiperEnv(can_interface="can0")
    print("真机关节角:", np.round(env.get_qpos(), 4))
    env.close()
    print("RealPiperEnv 自检通过 ✓")
