#!/usr/bin/env python3
"""主从机械臂遥控数据采集（Leader-Follower Teleop）

架构：软件桥接
    主臂(leader)  ──零力拖动,只读──>  关节角  ──CAN──>  从臂(follower) 跟随
                                                        └─> 腕部相机 + 实际状态 → 数据集

与 real_collect.py（手柄 + IK）的区别：
    - 不需要 IK，关节空间直通 → 消除零空间抖动（我们诊断出的 J4/J6 抖动主因）
    - 人手直接拖主臂，轨迹天然平滑
    - action 记录「主臂角度」（人示范的目标），state 记录「从臂实际角度」

数据语义：
    observation.image  从臂腕部相机 RGB 96x96
    observation.state  从臂实际关节角(rad) + 夹爪    ← 机器人当前状态
    action             主臂关节角(rad) + 夹爪目标    ← 人给的目标动作

硬件：
    - 主臂：零力拖动模式（不需要使能）
    - 从臂：使能 + 关节控制模式
    - 两臂各占一个 CAN 通道（如 can0=从臂, can1=主臂）
    - 相机装在从臂腕部

用法：
    python leader_follower_collect.py \
        --leader_can can1 --follower_can can0 \
        --repo_id local/piper_lf --task "grasp the fruit with leader-follower teleop" \
        --num_episodes 30

    # 没有主臂时，用从臂自身模拟主臂以测试数据管线
    python leader_follower_collect.py --dry_run --repo_id local/piper_lf_test
"""
import argparse
import os
import shutil
import threading
import time

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sim_collect_piper import GRIPPER_RAMP_STEPS, build_features

# ===== 单位换算（与 real_piper.py 一致）=====
DEG_TO_RAD = np.pi / 180.0
JOINT_INT_TO_RAD = 0.001 * DEG_TO_RAD     # 读：0.001度 → rad
JOINT_RAD_TO_INT = 1.0 / JOINT_INT_TO_RAD  # 发：rad → 0.001度
ARM_JOINT_ATTRS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# ===== 夹爪 =====
GRIPPER_MAX_INT = 100000  # 满开编码器值（0=闭合）
GRIPPER_CODE_ENABLE = 0x01
GRIPPER_EFFORT = 1000

# ===== 关节 =====
JOINT_LIMIT_RAD = [
    (-2.618, 2.618),  # J1
    (0.0, 3.14),      # J2
    (-2.697, 0.0),    # J3
    (-1.832, 1.832),  # J4
    (-1.22, 1.22),    # J5
    (-3.14, 3.14),    # J6
]

# ===== UI =====
WIN = "Piper Leader-Follower Teleop"
IMG_W, IMG_H, BAR = 480, 360, 90
BUTTONS = {
    "start":   (5, IMG_H + 10, 90, 70),
    "end":     (100, IMG_H + 10, 90, 70),
    "discard": (195, IMG_H + 10, 90, 70),
    "quit":    (290, IMG_H + 10, 90, 70),
}


def rad_to_int(q_rad):
    """rad → 0.001 度整数（SDK 单位）。"""
    return [int(round(float(v) * JOINT_RAD_TO_INT)) for v in q_rad]


def int_to_rad(q_int):
    """0.001 度整数 → rad。"""
    return np.asarray(q_int, dtype=np.float32) * JOINT_INT_TO_RAD


class ArmPair:
    """主臂（只读）+ 从臂（读写）封装。"""

    def __init__(self, leader_can, follower_can, follower_spd=40, dry_run=False, arm_type="V2"):
        from piper_sdk import C_PiperInterface_V2

        self.dry_run = dry_run
        self.follower_spd = follower_spd

        # --- 主臂：只连不使能，进零力拖动 ---
        self.leader = None
        if not dry_run:
            print(f"[1/3] 连接主臂 {leader_can} …", flush=True)
            self.leader = C_PiperInterface_V2(leader_can)
            self.leader.ConnectPort()
            time.sleep(0.3)
            # 零力拖动（重力补偿 + 阻尼）：主臂可被手推动
            self.leader.MotionCtrl_1(0x00, 0x00, 0x01)
            time.sleep(0.3)
            print("      主臂已进入零力拖动模式", flush=True)
        else:
            print("[1/3] dry_run：跳过主臂，用从臂自身模拟", flush=True)

        # --- 从臂：使能 + 关节控制 ---
        print(f"[2/3] 连接从臂 {follower_can} …", flush=True)
        self.follower = C_PiperInterface_V2(follower_can)
        self.follower.ConnectPort()
        t0 = time.time()
        while not self.follower.EnablePiper():
            if time.time() - t0 > 5:
                raise RuntimeError("从臂使能超时（检查电源/CAN/急停）")
            time.sleep(0.01)
        print("      从臂已使能", flush=True)

        # 关节控制模式
        self.follower.MotionCtrl_2(0x01, 0x01, follower_spd, 0x00)
        # 预热：等 CAN 反馈更新，避免首帧全 0
        time.sleep(0.5)
        for _ in range(5):
            self.follower.GetArmJointMsgs()
            self.follower.GetArmGripperMsgs()
            time.sleep(0.05)
        print("      从臂进入关节控制模式", flush=True)

    # ---- 主臂读取 ----
    def read_leader(self):
        """返回 (6 关节 rad, 夹爪 0~0.035)。"""
        if self.dry_run:
            return self.read_follower_state()
        return self._read_arm(self.leader)

    @staticmethod
    def _read_arm(arm):
        """读某个臂的 (关节 rad, 夹爪)。与 real_piper.py 一致。"""
        msgs = arm.GetArmJointMsgs()
        js = msgs.joint_state
        q = [getattr(js, n) for n in ARM_JOINT_ATTRS]
        try:
            g = arm.GetArmGripperMsgs().gripper_state
            raw = float(g.grippers_angle)
            grip = float(np.clip(0.035 * (1.0 - raw / GRIPPER_MAX_INT), 0.0, 0.035))
        except Exception:
            grip = 0.0
        return int_to_rad(q), grip

    # ---- 从臂 ----
    def read_follower_state(self):
        return self._read_arm(self.follower)

    def send_follower(self, q_rad, grip):
        """向从臂下发目标关节角 + 夹爪。"""
        q_rad = np.clip(np.asarray(q_rad, dtype=np.float32),
                        [lo for lo, _ in JOINT_LIMIT_RAD],
                        [hi for _, hi in JOINT_LIMIT_RAD])
        q_int = rad_to_int(q_rad)
        self.follower.JointCtrl(*q_int)
        g_int = int(round((1.0 - float(grip) / 0.035) * GRIPPER_MAX_INT))
        self.follower.GripperCtrl(max(0, min(GRIPPER_MAX_INT, g_int)), GRIPPER_EFFORT,
                                  GRIPPER_CODE_ENABLE, 0)

    def close(self):
        try:
            if self.leader is not None:
                self.leader.MotionCtrl_1(0x00, 0x00, 0x01)  # 保持悬停，不掉臂
                self.leader.DisconnectPort()
        except Exception:
            pass
        try:
            self.follower.GripperCtrl(0, 1000, 0x00, 0)
            self.follower.MotionCtrl_1(0x00, 0x00, 0x01)     # 悬停而非断电
            time.sleep(0.2)
            self.follower.DisconnectPort()
        except Exception:
            pass
        print("[提示] 两臂已进入拖动示教悬停（重力补偿），可手推调整或断电", flush=True)


class Camera:
    """D435i RGB（腕部安装）。"""

    def __init__(self, width=640, height=480, warmup=30):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        self.pipeline.start(cfg)
        for _ in range(warmup):          # 等自动曝光稳定
            self.pipeline.wait_for_frames()

    def read(self):
        frames = self.pipeline.wait_for_frames()
        img = np.asanyarray(frames.get_color_frame().get_data())
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def read_preview(self, width=IMG_W):
        """保持 4:3 的预览图（宽=width）。"""
        img = self.read()
        scale = width / img.shape[1]
        return cv2.resize(img, (width, max(1, int(round(img.shape[0] * scale)))))

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def draw_ui(img_rgb, mode, saved, total):
    canvas = np.zeros((IMG_H + BAR, IMG_W, 3), dtype=np.uint8)
    canvas[:IMG_H] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, f"MODE: {mode}   SAVED: {saved}/{total}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for name, (x, y, w, h) in BUTTONS.items():
        active = (name == "start" and mode == "idle") or (name == "end" and mode == "recording")
        color = (0, 140, 0) if active else (70, 70, 70)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        label = {"start": "START", "end": "END", "discard": "DISCARD", "quit": "QUIT"}[name]
        cv2.putText(canvas, label, (x + 8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leader_can", default="can1", help="主臂 CAN 通道")
    ap.add_argument("--follower_can", default="can0", help="从臂 CAN 通道")
    ap.add_argument("--repo_id", default="local/piper_lf")
    ap.add_argument("--task", default="grasp the object with leader-follower teleop")
    ap.add_argument("--num_episodes", type=int, default=30, help="显示用目标条数（手动 QUIT 退出）")
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ctrl_fps", type=int, default=50)
    ap.add_argument("--follower_spd", type=int, default=40, help="从臂速度百分比")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="无主臂：用从臂自身模拟（测试数据管线）")
    ap.add_argument("--debug_log", default=None)
    args = ap.parse_args()

    # ---- 数据集目录 ----
    ds_dir = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    if os.path.exists(ds_dir):
        if args.overwrite:
            bak = f"{ds_dir}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(ds_dir, bak)
            print(f"[提示] 旧数据集已备份 → {bak}", flush=True)
        else:
            has_data = os.path.isdir(os.path.join(ds_dir, "data")) or os.path.isdir(
                os.path.join(ds_dir, "videos"))
            if not has_data:
                shutil.rmtree(ds_dir)
                print(f"[提示] 空壳数据集已自动清理：{ds_dir}", flush=True)
            else:
                print(f"[错误] 数据集已存在且含数据：{ds_dir}")
                print("       如需重采请加 --overwrite（会先备份）")
                raise SystemExit(1)

    print(f"[0/3] 主从遥操作采集 | 主臂={args.leader_can} 从臂={args.follower_can}"
          f"{' | DRY-RUN' if args.dry_run else ''}", flush=True)
    arms = ArmPair(args.leader_can, args.follower_can,
                   follower_spd=args.follower_spd, dry_run=args.dry_run)
    cam = Camera()
    print("[3/3] 相机就绪", flush=True)

    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps,
                               features=build_features(), use_videos=True)

    ui = {"start": False, "end": False, "discard": False, "quit": False}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, (bx, by, bw, bh) in BUTTONS.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                ui[name] = True
                print(f"  [UI] {name.upper()}", flush=True)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, IMG_W, IMG_H + BAR)
    cv2.setMouseCallback(WIN, on_mouse)

    print("\n[操作] 拖动主臂即可控制从臂；点 UI 按钮 START/END/DISCARD/QUIT", flush=True)
    print("       夹爪：按 g 键（或手柄 A）切换开合", flush=True)

    saved = 0
    mode = "idle"
    ep_frames = []
    step = 0
    record_every = max(1, args.ctrl_fps // args.fps)
    ctrl_step = 0
    img_preview = None
    grip_goal = 0.0
    grip_cur = 0.0
    ramp = 0.035 / GRIPPER_RAMP_STEPS
    debug_rows = []
    last_t = time.time()

    try:
        while True:
            t_now = time.time()
            # ---- 读状态 ----
            leader_q, leader_grip = arms.read_leader()
            follower_q, follower_grip = arms.read_follower_state()

            # ---- 夹爪斜坡 + 下发 ----
            if grip_cur < grip_goal:
                grip_cur = min(grip_cur + ramp, grip_goal)
            elif grip_cur > grip_goal:
                grip_cur = max(grip_cur - ramp, grip_goal)
            arms.send_follower(leader_q, grip_cur)

            # ---- 预览 ----
            ctrl_step += 1
            if ctrl_step % 3 == 0:
                img_preview = cam.read_preview(IMG_W)
                cv2.imshow(WIN, draw_ui(img_preview, mode, saved, args.num_episodes))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("g"):
                grip_goal = 0.035 if grip_goal < 0.02 else 0.0
                print(f"  夹爪目标 → {'闭合' if grip_goal > 0.02 else '张开'}", flush=True)
            elif key in (ord("q"), 27):
                ui["quit"] = True

            # ---- 录制 ----
            if mode == "recording" and step % record_every == 0:
                if img_preview is None:
                    img_preview = cam.read_preview(IMG_W)
                img96 = cv2.resize(img_preview, (96, 96))
                ep_frames.append({
                    "observation.image": img96,
                    "observation.state": np.concatenate([follower_q, [follower_grip]]).astype(np.float32),
                    "action": np.concatenate([leader_q, [grip_cur]]).astype(np.float32),
                    "next.reward": np.array([0.0], dtype=np.float32),
                    "next.done": np.array([False]),
                    "next.success": np.array([False]),
                    "task": args.task,
                })
                if args.debug_log:
                    debug_rows.append([len(ep_frames)] + list(follower_q) + list(leader_q) + [grip_cur])
                step += 1
                if step >= args.max_steps:
                    ui["end"] = True
                    print(f"  达到 max_steps={args.max_steps}，自动结束本条", flush=True)

            # ---- UI 动作 ----
            if ui["start"] and mode == "idle":
                mode = "recording"
                ep_frames = []
                step = 0
                print(f"→ 开始录制第 {saved + 1} 条（拖动主臂示范）", flush=True)
            if (ui["end"] or ui["discard"]) and mode == "recording":
                if ui["end"] and len(ep_frames) > 5:
                    for f in ep_frames:
                        ds.add_frame(f)
                    ds.save_episode()
                    saved += 1
                    print(f"✓ 已保存 {saved}/{args.num_episodes} | 帧数 {len(ep_frames)}", flush=True)
                else:
                    print(f"✗ 已丢弃本条（{len(ep_frames)} 帧）", flush=True)
                mode = "idle"
                ep_frames = []
                step = 0
            if ui["quit"]:
                break
            for k in ui:
                ui[k] = False

            # ---- 控制频率 ----
            dt = 1.0 / args.ctrl_fps - (time.time() - t_now)
            if dt > 0:
                time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[中断] Ctrl+C", flush=True)
    finally:
        if ep_frames:
            print(f"[提示] 未保存的 {len(ep_frames)} 帧已丢弃", flush=True)
        if args.debug_log and debug_rows:
            import csv
            with open(args.debug_log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["n"] + [f"fq{i}" for i in range(6)] + [f"lq{i}" for i in range(6)] + ["grip"])
                w.writerows(debug_rows)
            print(f"[debug] 已写 {args.debug_log}", flush=True)
        ds.finalize()
        cv2.destroyAllWindows()
        cam.close()
        arms.close()
        print(f"采集完成 → {args.repo_id}（共 {saved} 条）", flush=True)


if __name__ == "__main__":
    main()
