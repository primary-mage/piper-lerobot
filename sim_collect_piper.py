#!/usr/bin/env python3
"""键盘遥操作 Piper 仿真采集 LeRobot 数据集（sim2sim 用）。

笛卡尔控制：键盘控制末端位置 x/y/z，IK 自动解算关节角（免去逐关节调的痛苦）。
显示：MuJoCo 原生 viewer（可旋转视角）+ 顶部相机预览窗口。

用法：
  conda activate lerobot
  python sim_collect_piper.py --num_episodes 20 --overwrite

按键：
  w/s 末端前后(x)   a/d 末端左右(y)   r/f 末端升降(z)
  g 夹爪开合切换   n/Enter 下一条   x 丢弃本条   q/Esc 退出
"""
import argparse
import os
import shutil
import threading
import time

import cv2
import numpy as np
from mujoco import viewer as mujoco_viewer
from pynput import keyboard as kb

from piper_sim import PiperEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

POS_DELTA = 0.01  # 键盘每步末端移动 0.01 m
GRIPPER_RAMP_STEPS = 15  # 夹爪开合斜坡步数（10fps 下 1.5 秒），避免惯性冲击带歪手臂


def stick_to_delta(v, deadzone=3000, scale=0.015):
    """把摇杆值 [-32768,32767] 映射到位移增量（含死区，摇杆推满 = scale 米/帧）。"""
    if abs(v) < deadzone:
        return 0.0
    return (v / 32768.0) * scale


def build_features():
    return {
        "observation.image": {"dtype": "video", "shape": (96, 96, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (7,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="local/piper_keyboard")
    ap.add_argument("--root", default="", help="数据集目录；留空用 LeRobot 默认缓存")
    ap.add_argument("--num_episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--pos_delta", type=float, default=POS_DELTA, help="键盘每步末端移动量(米)")
    ap.add_argument("--controller", choices=["keyboard", "gamepad"], default="keyboard", help="控制方式")
    ap.add_argument("--no_display", action="store_true", help="无窗口（headless）模式，用于调试")
    ap.add_argument("--overwrite", action="store_true", help="删除同名旧数据集后重新采集")
    args = ap.parse_args()

    if args.overwrite:
        ds_dir = args.root or os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
        if os.path.exists(ds_dir):
            backup = f"{ds_dir}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(ds_dir, backup)
            print(f"[提示] 旧数据集已备份（未删除）→ {backup}")
        else:
            print(f"[提示] 无旧数据集，直接新建")

    env = PiperEnv(image_size=96)
    # 每帧推进的物理步数：让 1 帧 ≈ 1/fps 秒模拟时间（10fps 下 = 50 步 × 0.002s）
    substeps = max(1, round(1.0 / args.fps / env.model.opt.timestep))
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps,
        features=build_features(), root=(args.root or None), use_videos=True,
    )

    # 原生 viewer
    viewer = None
    if not args.no_display:
        viewer = mujoco_viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        print("[提示] MuJoCo 原生 viewer 已打开：鼠标左键旋转视角、右键平移、滚轮缩放。")
        print("       w/s 前后  a/d 左右  r/f 升降  g 夹爪开合  n 下一条  q 退出")

    # 键盘状态（pynput，线程间共享）
    state = {"pressed": set(), "toggle_count": 0, "last_g": 0.0, "new_episode": False, "quit": False, "discard_episode": False}

    def on_press(key):
        state["pressed"].add(key)
        try:
            ch = getattr(key, "char", None)
            if ch == "g":
                now = time.time()
                if now - state["last_g"] > 0.3:  # 300ms 防抖，避免系统按键重复
                    state["toggle_count"] += 1
                    state["last_g"] = now
            elif ch in ("n", "\r"):
                state["new_episode"] = True
            elif ch == "x":
                state["discard_episode"] = True
            elif ch == "q":
                state["quit"] = True
        except Exception:  # noqa: BLE001
            pass
        if key == kb.Key.esc:
            state["quit"] = True

    def on_release(key):
        state["pressed"].discard(key)

    listener = kb.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 手柄线程（可选）
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
                        elif val == 1:  # 按键按下（边沿）
                            if code == "BTN_SOUTH":
                                state["toggle_count"] += 1
                            elif code == "BTN_START":
                                state["new_episode"] = True
                            elif code == "BTN_EAST":  # B 键：丢弃当前条
                                state["discard_episode"] = True
                            elif code == "BTN_SELECT":
                                state["quit"] = True
                except Exception:  # noqa: BLE001
                    time.sleep(0.02)

        threading.Thread(target=gamepad_thread, daemon=True).start()
        print("[提示] 使用 Xbox 手柄：左摇杆=前后左右 右摇杆上下=升降 A=夹爪 B=丢弃本条 Start=下一条 Select=退出")

    ep = 0      # 尝试计数（seed 用）
    saved = 0   # 已保存（好数据）计数
    try:
        while saved < args.num_episodes and not state["quit"] and (viewer is None or viewer.is_running()):
            qpos = env.reset(seed=ep)
            target_pos = env.get_grasp_pos().copy()  # 初始目标 = 当前末端位置
            gripper = 0.0        # 当前夹爪目标（斜坡平滑）
            gripper_goal = 0.0   # 夹爪目标（切换）
            ep_frames = []       # 本集帧缓存（结束才决定是否写入）
            state["new_episode"] = False
            state["discard_episode"] = False
            ep_done = False
            discard = False
            step = 0
            last_reward = 0.0
            while not ep_done and step < args.max_steps and not state["quit"] and (viewer is None or viewer.is_running()):
                # 控制器 → 目标末端位置
                if args.controller == "gamepad":
                    st = state["stick"]
                    target_pos[0] += stick_to_delta(st.get("ABS_X", 0))
                    target_pos[1] += -stick_to_delta(st.get("ABS_Y", 0))  # 摇杆上推为负，反转
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
                # 夹爪切换（计数器 → 切换目标）
                while state["toggle_count"] > 0:
                    state["toggle_count"] -= 1
                    gripper_goal = 0.035 if gripper_goal < 0.02 else 0.0
                    print(f"  夹爪目标 → {'张开' if gripper_goal > 0.02 else '闭合'}")
                # 斜坡平滑过渡（避免惯性冲击带歪手臂）
                ramp = 0.035 / GRIPPER_RAMP_STEPS
                if gripper < gripper_goal:
                    gripper = min(gripper + ramp, gripper_goal)
                elif gripper > gripper_goal:
                    gripper = max(gripper - ramp, gripper_goal)

                # IK 解算手臂关节 + 移动绿色目标标记
                arm_q = env.solve_ik(target_pos)
                env.set_target_marker(target_pos)
                action = np.concatenate([arm_q, [gripper]]).astype(np.float32)

                if step % 50 == 0:
                    print(f"  [已存 {saved}/{args.num_episodes}] step={step} 末端={np.round(target_pos, 3)} "
                          f"夹爪={'开' if gripper > 0.02 else '合'} 目标高={env.cube_height():.3f}")

                img_native = env.render_image(480)   # 原生高分辨率顶部相机
                img = cv2.resize(img_native, (96, 96))  # 数据集 96×96
                qpos_next, reward = env.step(action, substeps)
                last_reward = reward
                ep_frames.append({
                    "observation.image": img,
                    "observation.state": qpos,
                    "action": action,
                    "next.reward": np.array([reward], dtype=np.float32),
                    "next.done": np.array([False]),
                    "next.success": np.array([False]),
                    "task": "grasp the red cube and lift it",
                })
                qpos = qpos_next
                step += 1

                if viewer is not None:
                    viewer.sync()
                    cv2.imshow("Top Camera (observation)", cv2.cvtColor(img_native, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                time.sleep(1.0 / args.fps)

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
                print(f"✓ 已保存 {saved}/{args.num_episodes} | 方块高度 {env.cube_height():.3f} | 奖励 {last_reward:.3f} | 帧数 {len(ep_frames)}")
            ep += 1
    finally:
        state["stop"] = True
        ds.finalize()
        try:
            listener.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            if viewer is not None:
                viewer.close()
        except Exception:  # noqa: BLE001
            pass
        env.close()
    print(f"采集完成 → {args.repo_id}")


if __name__ == "__main__":
    main()
