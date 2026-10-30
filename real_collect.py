#!/usr/bin/env python3
"""真机 Piper 手柄/键盘遥操作采集 LeRobot 数据集（笛卡尔 IK 控制）。

与仿真版 sim_collect_piper.py 逻辑完全一致，只是底层环境换成 RealPiperEnv（piper_sdk + D435i）。
复用 sim_collect_piper 里的 build_features / stick_to_delta / 夹爪斜坡等常量。

用法：
  conda activate lerobot
  python real_collect.py --num_episodes 30 --controller gamepad

手柄：左摇杆=前后左右 右摇杆上下=升降 A=夹爪 B=丢弃 Start=下一条 Select=退出
键盘：w/s 前后 a/d 左右 r/f 升降 g 夹爪 x 丢弃 n 下一条 q 退出
"""
import argparse
import os
import shutil
import threading
import time

import cv2
import numpy as np
from pynput import keyboard as kb

from real_piper import RealPiperEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sim_collect_piper import build_features, stick_to_delta, GRIPPER_RAMP_STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="local/piper_real")
    ap.add_argument("--can_interface", default="can0", help="CAN 接口名（先 bash 01_setup_can.sh 激活）")
    ap.add_argument("--num_episodes", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--pos_delta", type=float, default=0.01, help="键盘每步末端移动量(米)")
    ap.add_argument("--controller", choices=["keyboard", "gamepad"], default="keyboard")
    ap.add_argument("--overwrite", action="store_true", help="覆盖旧数据（会先备份）")
    args = ap.parse_args()

    if args.overwrite:
        ds_dir = args.repo_id and os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
        if ds_dir and os.path.exists(ds_dir):
            backup = f"{ds_dir}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(ds_dir, backup)
            print(f"[提示] 旧数据集已备份（未删除）→ {backup}")

    env = RealPiperEnv(can_interface=args.can_interface, image_size=96)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps,
        features=build_features(), use_videos=True,
    )

    # 键盘状态
    state = {"pressed": set(), "toggle_count": 0, "last_g": 0.0, "new_episode": False, "quit": False, "discard_episode": False}

    def on_press(key):
        state["pressed"].add(key)
        try:
            ch = getattr(key, "char", None)
            if ch == "g":
                now = time.time()
                if now - state["last_g"] > 0.3:
                    state["toggle_count"] += 1
                    state["last_g"] = now
            elif ch in ("n", "\r"):
                state["new_episode"] = True
            elif ch == "x":
                state["discard_episode"] = True
            elif ch == "q":
                state["quit"] = True
        except Exception:
            pass
        if key == kb.Key.esc:
            state["quit"] = True

    def on_release(key):
        state["pressed"].discard(key)

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    if args.controller == "gamepad":
        state["stick"] = {"ABS_X": 0, "ABS_Y": 0, "ABS_RY": 0}
        state["stop"] = False

        def gamepad_thread():
            from inputs import get_gamepad
            while not state["stop"]:
                try:
                    for event in get_gamepad():
                        code, val = event.code, event.state
                        if code in ("ABS_X", "ABS_Y", "ABS_RX", "ABS_RY"):
                            state["stick"][code] = val
                        elif val == 1:
                            if code == "BTN_SOUTH":
                                state["toggle_count"] += 1
                            elif code == "BTN_START":
                                state["new_episode"] = True
                            elif code == "BTN_EAST":
                                state["discard_episode"] = True
                            elif code == "BTN_SELECT":
                                state["quit"] = True
                except Exception:
                    time.sleep(0.02)

        threading.Thread(target=gamepad_thread, daemon=True).start()
        print("[提示] Xbox 手柄：左摇杆=前后左右 右摇杆上下=升降 A=夹爪 B=丢弃 Start=下一条 Select=退出")

    saved = 0
    ep = 0
    try:
        while saved < args.num_episodes and not state["quit"]:
            qpos = env.get_qpos()
            target_pos = env.get_grasp_pos().copy()
            gripper = 0.0
            gripper_goal = 0.0
            ep_frames = []
            state["new_episode"] = False
            state["discard_episode"] = False
            ep_done = False
            discard = False
            step = 0

            while not ep_done and step < args.max_steps and not state["quit"]:
                # 控制器 → 目标末端位置
                if args.controller == "gamepad":
                    st = state["stick"]
                    target_pos[0] += stick_to_delta(st.get("ABS_X", 0))
                    target_pos[1] += -stick_to_delta(st.get("ABS_Y", 0))
                    target_pos[2] += -stick_to_delta(st.get("ABS_RY", 0))
                else:
                    if kb.KeyCode.from_char("w") in state["pressed"] or kb.Key.up in state["pressed"]:
                        target_pos[0] += args.pos_delta
                    if kb.KeyCode.from_char("s") in state["pressed"] or kb.Key.down in state["pressed"]:
                        target_pos[0] -= args.pos_delta
                    if kb.KeyCode.from_char("a") in state["pressed"] or kb.Key.left in state["pressed"]:
                        target_pos[1] -= args.pos_delta
                    if kb.KeyCode.from_char("d") in state["pressed"] or kb.Key.right in state["pressed"]:
                        target_pos[1] += args.pos_delta
                    if kb.KeyCode.from_char("r") in state["pressed"]:
                        target_pos[2] += args.pos_delta
                    if kb.KeyCode.from_char("f") in state["pressed"]:
                        target_pos[2] -= args.pos_delta

                # 夹爪切换 + 斜坡
                while state["toggle_count"] > 0:
                    state["toggle_count"] -= 1
                    gripper_goal = 0.035 if gripper_goal < 0.02 else 0.0  # 0=开, 0.035=闭（与仿真一致）
                    print(f"  夹爪目标 → {'闭合' if gripper_goal > 0.02 else '张开'}")
                ramp = 0.035 / GRIPPER_RAMP_STEPS
                if gripper < gripper_goal:
                    gripper = min(gripper + ramp, gripper_goal)
                elif gripper > gripper_goal:
                    gripper = max(gripper - ramp, gripper_goal)

                # IK（数字孪生）
                arm_q = env.solve_ik(target_pos)
                action = np.concatenate([arm_q, [gripper]]).astype(np.float32)

                if step % 50 == 0:
                    print(f"  [已存 {saved}/{args.num_episodes}] step={step} 末端={np.round(target_pos, 3)} 夹爪={'合' if gripper > 0.02 else '开'}")

                img = env.render_image(96)  # 数据集用 96×96
                ep_frames.append({
                    "observation.image": img,
                    "observation.state": qpos,
                    "action": action,
                    "next.reward": np.array([0.0], dtype=np.float32),  # 真机无地面真值
                    "next.done": np.array([False]),
                    "next.success": np.array([False]),
                    "task": "grasp the object and lift it",
                })

                env.send_action(action)  # 发真机
                time.sleep(1.0 / args.fps)
                qpos = env.get_qpos()  # 读回新关节角
                step += 1

                if state["discard_episode"]:
                    discard = True
                    ep_done = True
                if state["new_episode"]:
                    ep_done = True

            if discard:
                print(f"✗ 第 {ep+1} 次尝试已丢弃（失败数据不保存）")
            else:
                for frame in ep_frames:
                    ds.add_frame(frame)
                ds.save_episode()
                saved += 1
                print(f"✓ 已保存 {saved}/{args.num_episodes} | 帧数 {len(ep_frames)}")
            ep += 1
    finally:
        state["stop"] = True
        ds.finalize()
        try:
            listener.stop()
        except Exception:
            pass
        env.close()
    print(f"采集完成 → {args.repo_id}")


if __name__ == "__main__":
    main()
