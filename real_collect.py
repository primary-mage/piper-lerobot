#!/usr/bin/env python3
"""真机 Piper 遥操作采集 LeRobot 数据集（笛卡尔 IK + cv2 控制面板 UI）。

UI（鼠标点击）：
  [START]   复位到 home → 开始录制（每轮起点一致）
  [END]     保存本条 → 回到录制空闲
  [HOME]    手臂回 home（随时可用）
  [DISCARD] 丢弃本条 → 回到录制空闲
  [QUIT]    退出（示教悬停）

手柄：A=夹爪 B=丢弃。键盘：g 夹爪 q 退出。
流程：空闲（可自由遥控不记录）↔ 录制中
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

# 真机 home 姿态（6 关节角，弧度）
HOME_QPOS_REAL = np.array([0.0, 0.6, -1.0, 0.0, 1.15, 0.0], dtype=np.float32)

WIN = "Piper Teleop (Camera + Control)"
IMG_W = 480   # 预览宽度（保持 4:3）
IMG_H = 360   # 预览高度
BAR = 90

# 按钮区域 (x, y, w, h)
BUTTONS = {
    "start":   (5, IMG_H + 10, 90, 70),
    "end":     (100, IMG_H + 10, 90, 70),
    "home":    (195, IMG_H + 10, 90, 70),
    "discard": (290, IMG_H + 10, 90, 70),
    "quit":    (385, IMG_H + 10, 90, 70),
}
LABELS = {"start": "START", "end": "END", "home": "HOME", "discard": "DISCARD", "quit": "QUIT"}


def smooth_move_to_home(env, ctrl_fps, duration=2.5):
    """平滑移动到 home 姿态。"""
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
    """一步遥操：更新目标、解 IK、发指令。"""
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
    arm_q = env.solve_ik(tel["target"])
    # EMA 平滑 IK 输出，减少冗余自由度（J4/J6）的零空间抖动
    if tel.get("prev_arm_q") is not None:
        arm_q = args.ik_smooth * arm_q + (1.0 - args.ik_smooth) * tel["prev_arm_q"]
    tel["prev_arm_q"] = arm_q
    action = np.concatenate([arm_q, [tel["gripper"]]]).astype(np.float32)
    env.send_action(action)
    time.sleep(1.0 / args.ctrl_fps)
    return action


def draw_ui(img_rgb, mode, saved, total):
    """把相机图 + 按钮栏拼成显示图（BGR）。"""
    canvas = np.zeros((IMG_H + BAR, IMG_W, 3), dtype=np.uint8)
    canvas[:IMG_H] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, f"MODE: {mode}   SAVED: {saved}/{total}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for name, (x, y, w, h) in BUTTONS.items():
        active = (name == "start" and mode == "idle") or (name == "end" and mode == "recording")
        color = (0, 140, 0) if active else (70, 70, 70)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(canvas, LABELS[name], (x + 5, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="local/piper_real")
    ap.add_argument("--can_interface", default="can0")
    ap.add_argument("--num_episodes", type=int, default=30, help="显示用目标条数(不再自动退出，只能手动 QUIT)")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--ctrl_fps", type=int, default=50)
    ap.add_argument("--pos_delta", type=float, default=0.01)
    ap.add_argument("--controller", choices=["keyboard", "gamepad"], default="keyboard")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--move_spd_rate", type=int, default=40)
    ap.add_argument("--stick_scale", type=float, default=0.002)
    ap.add_argument("--ik_smooth", type=float, default=0.3, help="IK输出EMA平滑0~1(1=不平滑)，减少J4/J6零空间抖动")
    ap.add_argument("--debug_log", default=None)
    args = ap.parse_args()

    # 数据集目录处理：空壳（只有 meta、无 data/videos）自动清理；已有数据则要求 --overwrite
    ds_dir = os.path.expanduser(f"~/.cache/huggingface/lerobot/{args.repo_id}")
    if os.path.exists(ds_dir):
        if args.overwrite:
            backup = f"{ds_dir}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(ds_dir, backup)
            print(f"[提示] 旧数据集已备份（未删除）→ {backup}")
        else:
            has_data = os.path.isdir(os.path.join(ds_dir, "data")) or os.path.isdir(
                os.path.join(ds_dir, "videos")
            )
            if not has_data:
                shutil.rmtree(ds_dir)
                print(f"[提示] 检测到空壳数据集（只有 meta、无数据），已自动清理：{ds_dir}")
            else:
                print(f"[错误] 数据集已存在且含数据：{ds_dir}")
                print("       如需重新采集请加 --overwrite（会先备份旧数据）")
                raise SystemExit(1)

    env = RealPiperEnv(can_interface=args.can_interface, image_size=96, move_spd_rate=args.move_spd_rate)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=args.fps, features=build_features(), use_videos=True)

    # 明确告知当前输入模式（手柄/键盘），避免"遥控器没反应"的困惑
    if args.controller == "gamepad":
        print("[输入] 模式 = gamepad（手柄）：左摇杆平移、右摇杆升降、A=夹爪、B=丢弃")
        print("       若摇杆无反应，先跑：python test_gamepad.py")
    else:
        print("[输入] 模式 = keyboard（键盘）：w/s/a/d 平移、r/f 升降、g 夹爪、x 丢弃")
        print("       想用手柄请加参数：--controller gamepad")

    state = {"pressed": set(), "toggle_count": 0, "last_g": 0.0, "quit": False, "discard_episode": False}
    ui = {"start": False, "end": False, "home": False, "discard": False, "quit": False}

    def on_press(key):
        state["pressed"].add(key)
        try:
            ch = getattr(key, "char", None)
            if ch == "g":
                now = time.time()
                if now - state["last_g"] > 0.3:
                    state["toggle_count"] += 1
                    state["last_g"] = now
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
                            if code == "BTN_SOUTH":   # A 夹爪
                                state["toggle_count"] += 1
                            elif code == "BTN_EAST":  # B 丢弃
                                state["discard_episode"] = True
                except Exception:
                    time.sleep(0.02)

        threading.Thread(target=gamepad_thread, daemon=True).start()
        print("[提示] 手柄：A=夹爪 B=丢弃；开始/结束/复位请点 UI 按钮")

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (bx, by, bw, bh) in BUTTONS.items():
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    ui[name] = True

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)

    debug_f, debug_w = None, None
    if args.debug_log:
        import csv
        debug_f = open(args.debug_log, "w", newline="")
        debug_w = csv.writer(debug_f)
        debug_w.writerow(["ctrl_step", "mode"] + [f"tx{i}" for i in range(3)] + [f"raw_j{i}" for i in range(1, 7)] + ["gripper"])

    saved = 0
    mode = "idle"
    tel = {"target": env.get_grasp_pos().copy(), "gripper": 0.0, "gripper_goal": 0.0}
    ep_frames = []
    step = 0
    record_every = max(1, args.ctrl_fps // args.fps)
    ctrl_step = 0
    img_cache = None

    try:
        while not state["quit"] and not ui["quit"]:
            action = do_teleop(env, args, state, tel)
            ctrl_step += 1

            if ctrl_step % 3 == 0 or img_cache is None:
                img_cache = env.render_image(IMG_W, keep_aspect=True)
                cv2.imshow(WIN, draw_ui(img_cache, mode, saved, args.num_episodes))
                cv2.waitKey(1)

            # ---- UI 按钮 ----
            if ui["home"]:
                ui["home"] = False
                print("→ 复位到 home...")
                smooth_move_to_home(env, args.ctrl_fps)
                tel["target"] = env.get_grasp_pos().copy()
                img_cache = None

            if ui["start"]:
                ui["start"] = False
                if mode == "idle":
                    print("→ 开始录制：复位到 home...")
                    smooth_move_to_home(env, args.ctrl_fps)
                    tel["target"] = env.get_grasp_pos().copy()
                    ep_frames = []
                    step = 0
                    state["discard_episode"] = False
                    mode = "recording"
                    img_cache = None

            if mode == "recording":
                if ctrl_step % record_every == 0:
                    if img_cache is None:
                        img_cache = env.render_image(IMG_W, keep_aspect=True)
                    img96 = cv2.resize(img_cache, (96, 96))
                    ep_frames.append({
                        "observation.image": img96,
                        "observation.state": env.get_qpos(),
                        "action": action,
                        "next.reward": np.array([0.0], dtype=np.float32),
                        "next.done": np.array([False]),
                        "next.success": np.array([False]),
                        "task": "grasp the object and lift it",
                    })
                    if debug_w is not None:
                        debug_w.writerow([ctrl_step, "recording", *np.round(tel["target"], 4),
                                          *np.round(action[:6], 4), round(float(tel["gripper"]), 4)])
                    step += 1

                if ui["end"]:
                    ui["end"] = False
                    if ep_frames:
                        for f in ep_frames:
                            ds.add_frame(f)
                        ds.save_episode()
                        saved += 1
                        print(f"✓ 已保存 {saved}/{args.num_episodes} | 帧数 {len(ep_frames)}")
                    ep_frames = []
                    mode = "idle"
                if ui["discard"] or state["discard_episode"]:
                    ui["discard"] = False
                    state["discard_episode"] = False
                    print("✗ 本条已丢弃")
                    ep_frames = []
                    mode = "idle"
            else:
                ui["end"] = False
                ui["discard"] = False
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
