#!/usr/bin/env python3
"""松灵 Piper 六轴 + 夹爪的 MuJoCo 仿真环境（基于 DeepMind mujoco_menagerie 的 agilex_piper 模型）。

动作空间：7 维 = 6 个手臂关节(joint1~joint6) + 1 个夹爪(joint7，joint8 镜像联动)。
观测：state = 当前关节角(7维)，image = 顶部相机渲染图。
"""
import os

import mujoco
import numpy as np

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "piper")

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "joint7"  # joint8 通过 equality 镜像
ACTUATOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]

# 初始关节角（6 手臂 + 夹爪张开）
# 2026-11-12 为「眼在手上相机」重新标定：工具轴朝下俯视桌面，方块落在画面中心附近。
# （原值 [0, 0.8, -1.2, 0, 0, 0] 是给固定俯视相机用的，腕部相机会看不到桌面。）
HOME_QPOS = np.array([0.1, 0.7, -1.0, 0.0, 1.2, 0.0, 0.0], dtype=np.float32)

# ===== 相机配置 =====
# 腕部相机外参（相机光学系 → link6），来自真机手眼标定（2026-11-04）。
# 与视觉端 text_grasp.py 的 T_L6_CAM 同源；format：[R|t] 4x4，行主序。
WRIST_CAM_EXTRINSICS = np.array(
    [
        [+0.010210, +0.928004, +0.372430, -0.0730],
        [-0.999915, +0.006436, +0.011376, +0.0027],
        [+0.008160, -0.372514, +0.927991, +0.0557],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# ROS 光学系(x 右, y 下, z 光轴) → MuJoCo 相机系(x 右, y 上, 光轴 -Z)
# 两者差一个绕 X 轴 180° 的旋转。
_OPTICAL_TO_MJ = np.diag([1.0, -1.0, -1.0])

# D435i 彩色：16:9 时垂直 FOV≈42.5°；采集用 640x480(4:3) 时约 55°（水平约 69°）。
# 取 55° 既贴近真机，又能让接近/抓取阶段的方块留在画面内（42.5° 时抓取位会贴边）。
WRIST_FOVY = 55.0
TOP_FOVY = 50.0


def _matrix_to_quat_wxyz(R):
    """3x3 旋转矩阵 → MuJoCo 四元数 (w, x, y, z)。"""
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(R, dtype=np.float64).reshape(9))
    return quat.tolist()

# 关节限位（弧度），对应 piper.xml 的 range
JOINT_LIMITS = {
    "joint1": (-2.618, 2.618),
    "joint2": (0.0, 3.14),
    "joint3": (-2.697, 0.0),
    "joint4": (-1.832, 1.832),
    "joint5": (-1.22, 1.22),
    "joint6": (-3.14, 3.14),
    "gripper": (0.0, 0.035),  # 0=张开, 0.035=闭合
}

# 手臂关节限位（弧度），顺序 = joint1..joint6，供 IK 用
ARM_LIMITS = [
    (-2.618, 2.618), (0.0, 3.14), (-2.697, 0.0),
    (-1.832, 1.832), (-1.22, 1.22), (-3.14, 3.14),
]


class PiperEnv:
    """Piper 抓取仿真环境。"""

    def __init__(self, image_size=96, render_width=480, render_height=480, camera="wrist"):
        self.image_size = image_size
        self.render_width = render_width
        self.render_height = render_height
        self.camera = camera

        spec = mujoco.MjSpec.from_file(os.path.join(ASSET_DIR, "piper.xml"))
        # 地面 + 桌子 + 方块 + 相机
        spec.worldbody.add_geom(
            name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05], rgba=[0.2, 0.3, 0.4, 1],
        )
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.35, 0.30, 0.04], pos=[0.30, 0, 0.04], rgba=[0.75, 0.75, 0.75, 1],
        )
        cube = spec.worldbody.add_body(name="cube", pos=[0.30, 0.05, 0.105])
        cube.add_freejoint(name="cube_joint")
        cube.add_geom(
            name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.025, 0.025, 0.025], mass=0.05, rgba=[0.9, 0.1, 0.1, 1],
        )
        # 相机：wrist = 眼在手上（挂在 link6），top = 固定俯视（eye-to-hand）
        if camera == "wrist":
            R_l6_mj = WRIST_CAM_EXTRINSICS[:3, :3] @ _OPTICAL_TO_MJ
            t_l6 = WRIST_CAM_EXTRINSICS[:3, 3]
            quat = _matrix_to_quat_wxyz(R_l6_mj)
            spec.body("link6").add_camera(
                name="wrist", pos=t_l6.tolist(), quat=quat, fovy=WRIST_FOVY
            )
        else:
            spec.worldbody.add_camera(
                name="top", pos=[0.30, 0, 1.0], quat=[1, 0, 0, 0], fovy=TOP_FOVY,
            )
        # 目标位置标记（绿色小球，mocap，显示 IK 目标点）
        marker = spec.worldbody.add_body(name="target_marker", pos=[0.30, 0.05, 0.20])
        marker.mocap = True
        marker.add_geom(
            name="target_marker_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.012], rgba=[0.0, 1.0, 0.0, 0.7], contype=0, conaffinity=0,
        )

        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, render_height, render_width)

        self.arm_qpos_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINTS]
        self.gripper_qpos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ACTUATOR_NAMES]
        self.cube_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        # 夹爪两指 body id：IK 参考点用「两指中点」——开合对称，中点稳定不随夹爪动
        self.grasp_body_id7 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.grasp_body_id8 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.target_mocap_id = 0  # 唯一的 mocap 体
        self.marker_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_marker_geom")

    @property
    def action_dim(self):
        return 7

    def reset(self, seed=0):
        """重置：手臂回 home，方块随机落在桌面上。返回当前关节角(7维)。"""
        mujoco.mj_resetData(self.model, self.data)
        for i, qid in enumerate(self.arm_qpos_ids):
            self.data.qpos[qid] = HOME_QPOS[i]
        self.data.qpos[self.gripper_qpos_id] = 0.0  # 夹爪张开

        rng = np.random.default_rng(seed)
        cube_x = 0.28 + rng.uniform(-0.03, 0.03)
        cube_y = rng.uniform(-0.05, 0.10)
        jid = self.cube_joint_id
        self.data.qpos[jid:jid + 3] = [cube_x, cube_y, 0.105]
        self.data.qpos[jid + 3:jid + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)
        return self.get_qpos()

    def get_qpos(self):
        """当前关节角：6 手臂 + 1 夹爪，float32 (7,)。"""
        q = [self.data.qpos[i] for i in self.arm_qpos_ids] + [self.data.qpos[self.gripper_qpos_id]]
        return np.asarray(q, dtype=np.float32)

    def step(self, action, substeps=1):
        """施加 7 维目标关节角，推进 substeps 帧物理。返回 (新关节角, 奖励)。"""
        action = np.asarray(action, dtype=np.float64)
        for i, aid in enumerate(self.act_ids):
            self.data.ctrl[aid] = float(action[i])
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        qpos = self.get_qpos()
        cube_z = float(self.data.qpos[self.cube_joint_id + 2])
        # 奖励：方块被抬得越高越好（>0.16 视为抓到并抬起）
        reward = float(np.clip((cube_z - 0.11) / 0.06, 0.0, 1.0))
        return qpos, reward

    def cube_position(self):
        return self.data.qpos[self.cube_joint_id:self.cube_joint_id + 3].copy()

    def cube_height(self):
        return float(self.data.qpos[self.cube_joint_id + 2])

    def render_image(self, size=None):
        """顶部相机渲染图（RGB uint8）。size 指定输出边长，默认 image_size（数据集用）。

        渲染时临时隐藏绿色目标标记球（alpha=0），避免它污染观测图。
        """
        rgba = self.model.geom_rgba[self.marker_geom_id].copy()
        self.model.geom_rgba[self.marker_geom_id, 3] = 0.0  # 隐藏
        self.renderer.update_scene(self.data, camera=self.camera)
        img = self.renderer.render()  # HWC RGB uint8，原生 render_width×render_height
        self.model.geom_rgba[self.marker_geom_id] = rgba  # 恢复
        size = size or self.image_size
        if img.shape[0] != size:
            import cv2
            img = cv2.resize(img, (size, size))
        return img

    def get_grasp_pos(self):
        """当前夹爪参考点（两指中点）的世界坐标 (3,)。中点不随夹爪开合变化。"""
        return (self.data.xpos[self.grasp_body_id7] + self.data.xpos[self.grasp_body_id8]) / 2.0

    def set_target_marker(self, pos):
        """移动绿色目标标记球到 pos。"""
        self.data.mocap_pos[self.target_mocap_id] = pos

    def solve_ik(self, target_pos, max_iter=60, tol=1e-3, damping=1e-6, null_gain=0.2):
        """阻尼最小二乘 IK + 零空间镇定：求使夹爪中点到达 target_pos 的 6 个手臂关节角。

        零空间镇定：6 关节解 3 维位置是冗余的，DLS 会在冗余自由度间漂移导致关节抖动。
        加一项"拉向当前关节姿态"的零空间任务，让 IK 优先保持关节不动（消抖）。

        不改动真实物理状态（迭代用临时 qpos，结束后恢复）。
        """
        orig = np.array([self.data.qpos[i] for i in self.arm_qpos_ids])
        qpos = orig.copy()
        q_pref = orig.copy()  # 偏好配置 = 当前关节角（零空间拉向这里）
        jacp7 = np.zeros((3, self.model.nv))
        jacr7 = np.zeros((3, self.model.nv))
        jacp8 = np.zeros((3, self.model.nv))
        jacr8 = np.zeros((3, self.model.nv))
        for _ in range(max_iter):
            for i, qid in enumerate(self.arm_qpos_ids):
                self.data.qpos[qid] = qpos[i]
            mujoco.mj_forward(self.model, self.data)
            # 两指中点（开合对称，中点稳定）
            cur = (self.data.xpos[self.grasp_body_id7] + self.data.xpos[self.grasp_body_id8]) / 2.0
            err = np.asarray(target_pos, dtype=np.float64) - cur
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacBody(self.model, self.data, jacp7, jacr7, self.grasp_body_id7)
            mujoco.mj_jacBody(self.model, self.data, jacp8, jacr8, self.grasp_body_id8)
            J = (jacp7[:, :6] + jacp8[:, :6]) / 2.0  # 中点位置雅可比（6 个手臂关节）
            # 主任务：DLS 到达目标
            A = J @ J.T + damping * np.eye(3)
            dq_task = J.T @ np.linalg.solve(A, err)
            # 零空间任务：拉向偏好配置（保持关节不动，消除冗余漂移）
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))  # (6,3) 伪逆
            N = np.eye(6) - J_pinv @ J  # (6,6) 零空间投影
            dq_null = null_gain * (N @ (q_pref - qpos))
            qpos += dq_task + dq_null
            for i in range(6):
                lo, hi = ARM_LIMITS[i]
                qpos[i] = float(np.clip(qpos[i], lo, hi))
        # 恢复真实物理状态
        for i, qid in enumerate(self.arm_qpos_ids):
            self.data.qpos[qid] = orig[i]
        mujoco.mj_forward(self.model, self.data)
        return qpos.astype(np.float32)

    def close(self):
        self.renderer.close()


if __name__ == "__main__":
    env = PiperEnv()
    q = env.reset(seed=0)
    print("初始关节角:", np.round(q, 3))
    for _ in range(10):
        q, r = env.step(q)
    print("步进 10 步后关节角:", np.round(q, 3), "方块高度:", round(env.cube_height(), 3))
    img = env.render_image()
    print("渲染图 shape:", img.shape, img.dtype)
    env.close()
    print("PiperEnv 自检通过 ✓")
