#!/usr/bin/env python3
"""真机 Piper 手柄/键盘遥操作采集 LeRobot 数据集（笛卡尔 IK 控制）。

流程：
  每轮开始 → 自动复位到 home 姿态 → 摆好道具（可自由遥控，不记录）
  → 按 Y（再次复位，确保起点一致）→ 开始录制 → 抓取
  → Start=结束并保存 / B=丢弃本条 / Select=退出（示教悬停）

手柄：左摇杆=前后左右 右摇杆上下=升降 A=夹爪 B=丢弃 Y=开始录制 Start=结束并保存 Select=退出
键盘：w/s 前后 a/d 左右 r/f 升降 g 夹爪 y 开始录制 n 结束并保存 x 丢弃 q 退出
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

# 真机 home 姿态（6 关节角，弧度）。先用仿真的 home，实测后可微调
HOME_QPOS_REAL = np.array([0.0, 0.8, -1.2, 0.0, 0.0, 0.0], dtype=np.float32)


def smooth_move_to_home(env, ctrl_fps, duration=2.5):
    """平滑移动到 home 姿态（关节空间线性插值）。"""
    q_cur = env.get_qpos()
    q_start = q_cur[:6].copy()
    grip = float(q_cur[6])
    n = max(1, int(duration * ctrl_fps))
    for k in range(n + 1):
        t = k / n
        q = (1.0 - t) * q_start + t * HOME_QPOS_REAL
        env.send_action(np.concatenate([q.astype(np.float32), np.array([grip], dtype=np.float32)]))
        time.sleep(1.0 / ctrl_fps)


def do_teleop(env, args, state, tel):
    """一步遥操：更新目标位置、解 IK、发指令。tel = {"target","gripper","gripper_goal"}。"""
    if args.controller == "gamepad":
        st = state["stick"]
        sc = args.stick_scale
        tel["target"][0] += stick_to_delta(st.get("ABS_X", 0), scale=sc)
        tel["target"][1] += -stick_to_delta(st.get("ABS_Y", 0), scale=sc)
        tel["target"][2] += -stick_to_delta(st.get("ABS_RY", 0), scale=sc)
    else:
        pd = args.pos_delta
        if kb.KeyCode.from_char("w") in state["pressed"] or kb.Key.up in state["pressed"]:
            tel["target"][0] += pd
        if kb.KeyCode.from_char("s") in state["pressed"] or kb.Key.down in state["pressed"]:
            tel["target"][0] -= pd
        if kb.KeyCode.from_char("a") in state["pressed"] or kb.Key.left in state["pressed"]:
            tel["target"][1] -= pd
        if kb.KeyCode.from_char("d") in state["pressed"] or kb.Key.right in state["pressed"]:
            tel["target"][1] += pd
        if kb.KeyCode.from_char("r") in state["pressed"]:
            tel["target"][2] += pd
        if kb.KeyCode.from_char("f") in state["pressed"]:
            tel["target"][2] -= pd
    # 夹爪切换 + 斜坡
    while state["toggle_count"] > 0:
        state["toggle_count"] -= 1
        tel["gripper_goal"] = 0.035 if tel["gripper_goal"] < 0.02 else 0.0
        print(f"  夹爪目标 → {'闭合' if tel['gripper_goal'] > 0.02 else '张开'}")
    ramp = 0.035 / GRIPPER_RAMP_STEPS
    if tel["gripper"] < tel["gripper_goal"]:
        tel["gripper"] = min(tel["gripper"] + ramp, tel["gripper_goal"])
    elif tel["gripper"] > tel["gripper_goal"]:
        tel["gripper"] = max(tel["gripper"] - ramp, tel["gripper_goal"])
    # IK + 发指令
    arm_q = env.solve_ik(tel["target"])
    action = np.concatenate([arm_q, [tel["gripper"]]]).astype(np.float32)
    env.send_action(action)
    time.sleep(1.0 / args.ctrl_fps)
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="local/piper_real")
    ap.add_argument("--can_interface", default="can0", help="CAN 接口名（先 bash 01_setup_can.sh 激活）")
    ap.add_argument("--num_episodes", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10, help="数据记录帧率（数据集 fps）")
    ap.add_argument("--ctrl_fps", type=int, default=50, help="控制环频率，高于 fps 时目标连续更新、消除顿挫")
    ap.add_argument("--pos_delta", type=float, default=0.01, help="键盘每步末端移动量(米)")
    ap.add_argument("--controller", choices=["keyboard", "gamepad"], default="keyboard")
    ap.add_argument("--overwrite", action="store_true", help="覆盖旧数据（会先备份）")
    ap.add_argument("--move_spd_rate", type=int, default=40, help="运动速度百分比 0~100，越低越平滑（默认 40）")
    ap.add_argument("--stick_scale", type=float, default=0.002, help="摇杆满偏时每控制帧位移(米)，越小越慢（50Hz 下 0.002≈0.1m/s）")
    ap.add_argument("--debug_log", default=None, help="调试日志 CSV 路径（记录目标/IK 关节角）")
    args = ap.parse_args()

    if args.overwrite:
        ds_dir = args.repo_id and os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
        if ds_dir and os.path.exists(ds_dir):
            backup = f"{ds_dir}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(ds_dir, backup)
            print(f"[提示] 旧数据集已备份（未删除）→ {backup}")

    env = RealPiperEnv(can_interface=args.can_interface, image_size=96, move_spd_rate=args.move_spd_rate)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps,
        features=build_features(), use_videos=True,
    )

    state = {
        "pressed": set(), "toggle_count": 0, "last_g": 0.0,
        "new_episode": False, "quit": False, "discard_episode": False, "start_record": False,
    }

    def on_press(key):
        state["pressed"].add(key)
        try:
            ch = getattr(key, "char", None)
            if ch == "g":
                now = time.time()
                if now - state["last_g"] > 0.3:
                    state["toggle_count"] += 1
                    state["last_g"] = now
            elif ch == "y":
                state["start_record"] = True
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
                            if code == "BTN_SOUTH":       # A 夹爪
                                state["toggle_count"] += 1
                            elif code == "BTN_WEST":      # Y 开始录制
                                state["start_record"] = True
                            elif code == "BTN_START":     # Start 结束并保存
                                state["new_episode"] = True
                            elif code == "BTN_EAST":      # B 丢弃
                                state["discard_episode"] = True
                            elif code == "BTN_SELECT":    # Select 退出
                                state["quit"] = True
                except Exception:
                    time.sleep(0.02)

        threading.Thread(target=gamepad_thread, daemon=True).start()
        print("[提示] 手柄：左摇杆=前后左右 右摇杆上下=升降 A=夹爪 B=丢弃 Y=开始录制 Start=结束并保存 Select=退出")

    saved = 0
    debug_f = None
    debug_w = None
    if args.debug_log:
        import csv
        debug_f = open(args.debug_log, "w", newline="")
        debug_w = csv.writer(debug_f)
        debug_w.writerow(["ctrl_step", "tx", "ty", "tz"] + [f"raw_j{i}" for i in range(1, 7)] + ["gripper"])

    record_every = max(1, args.ctrl_fps // args.fps)
    window = "D435i (camera)"
    try:
        while saved < args.num_episodes and not state["quit"]:
            print(f"\n=== 第 {saved+1}/{args.num_episodes} 条 ===")

            # 1. 复位到 home
            print("→ 机械臂复位到 home 姿态...")
            smooth_move_to_home(env, args.ctrl_fps)

            # 2. 待录制（摆道具，可自由遥控，不记录）
            print("→ 摆好道具后按【Y】开始录制（可自由遥控，不记录）")
            state["start_record"] = False
            tel = {"target": env.get_grasp_pos().copy(), "gripper": 0.0, "gripper_goal": 0.0}
            img_cache = None
            preview_step = 0
            while not state["start_record"] and not state["quit"]:
                preview_step += 1
                do_teleop(env, args, state, tel)
                if preview_step % 3 == 0 or img_cache is None:
                    img_cache = env.render_image(480)
                    cv2.imshow(window, cv2.cvtColor(img_cache, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            if state["quit"]:
                break
            state["start_record"] = False

            # 3. 再次复位，确保起点一致
            print("→ 确认起点，复位到 home 后开始录制...")
            smooth_move_to_home(env, args.ctrl_fps)

            # 4. 录制
            tel = {"target": env.get_grasp_pos().copy(), "gripper": 0.0, "gripper_goal": 0.0}
            ep_frames = []
            state["new_episode"] = False
            state["discard_episode"] = False
            ep_done = False
            discard = False
            step = 0
            ctrl_step = 0
            img_cache = None
            qpos = env.get_qpos()
            while not ep_done and step < args.max_steps and not state["quit"]:
                ctrl_step += 1
                action = do_teleop(env, args, state, tel)
                qpos = env.get_qpos()
                # 相机预览（每 3 个控制帧刷新）+ 记录（复用缓存帧缩到 96）
                if ctrl_step % 3 == 0 or img_cache is None:
                    img_cache = env.render_image(480)
                    cv2.imshow(window, cv2.cvtColor(img_cache, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                if ctrl_step % record_every == 0:
                    if step % 50 == 0:
                        print(f"  [已存 {saved}/{args.num_episodes}] step={step} 末端={np.round(tel['target'], 3)} 夹爪={'合' if tel['gripper'] > 0.02 else '开'}")
                    img = cv2.resize(img_cache, (96, 96))
                    ep_frames.append({
                        "observation.image": img,
                        "observation.state": qpos,
                        "action": action,
                        "next.reward": np.array([0.0], dtype=np.float32),
                        "next.done": np.array([False]),
                        "next.success": np.array([False]),
                        "task": "grasp the object and lift it",
                    })
                    if debug_w is not None:
                        debug_w.writerow([ctrl_step, *np.round(tel["target"], 4), *np.round(action[:6], 4), round(float(tel["gripper"]), 4)])
                    step += 1
                if state["discard_episode"]:
                    discard = True
                    ep_done = True
                if state["new_episode"]:
                    ep_done = True

            if discard:
                print(f"✗ 第 {saved+1} 条已丢弃（失败数据不保存）")
            elif ep_frames:
                for frame in ep_frames:
                    ds.add_frame(frame)
                ds.save_episode()
                saved += 1
                print(f"✓ 已保存 {saved}/{args.num_episodes} | 帧数 {len(ep_frames)}")
            state["new_episode"] = False
            state["discard_episode"] = False
    finally:
        state["stop"] = True
        ds.finalize()
        try:
            listener.stop()
        except Exception:
            pass
        if debug_f is not None:
            debug_f.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
    print(f"采集完成 → {args.repo_id}")


if __name__ == "__main__":
    main()
