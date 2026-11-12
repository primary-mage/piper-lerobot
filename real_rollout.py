#!/usr/bin/env python3
"""真机 ACT 策略部署（START 按钮控制节奏）。

UI（鼠标点击）：
  空闲：[START]  [EMERGENCY STOP]  [QUIT]
  录制：[END]    [EMERGENCY STOP]  [QUIT]

流程：
  启动 → 空闲（手臂回 home，相机预览）→ 你摆道具 → 点 START
  → 策略推理（max_steps 或点 END）→ 回 home、回到空闲 → 摆道具 → START ...
  → QUIT 退出（拖动示教悬停）

抖动修复：策略按 fps(10Hz) 推理，动作按 ctrl_fps(50Hz) 插值发送。
"""
import argparse
import time

import cv2
import numpy as np
import safetensors.torch as st
import torch

from real_piper import RealPiperEnv
from lerobot.policies.act.modeling_act import ACTPolicy

HOME_QPOS_REAL = np.array([0.0, 0.6, -1.0, 0.0, 1.15, 0.0], dtype=np.float32)

WIN = "Real ACT (Camera + Control)"
IMG_W, IMG_H, BAR = 480, 360, 90
BUTTONS = {
    "start": (5, IMG_H + 10, 150, 70),
    "stop":  (165, IMG_H + 10, 150, 70),
    "quit":  (325, IMG_H + 10, 150, 70),
}


def load_stats(checkpoint: str) -> dict:
    pre = st.load_file(f"{checkpoint}/policy_preprocessor_step_3_normalizer_processor.safetensors")
    post = st.load_file(f"{checkpoint}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    state_std = pre["observation.state.std"].numpy().copy()
    action_std = post["action.std"].numpy().copy()
    state_std = np.where(np.abs(state_std) < 1e-8, 1.0, state_std)
    action_std = np.where(np.abs(action_std) < 1e-8, 0.0, action_std)
    return {
        "img_mean": pre["observation.image.mean"].numpy(),
        "img_std": pre["observation.image.std"].numpy(),
        "state_mean": pre["observation.state.mean"].numpy(),
        "state_std": state_std,
        "action_mean": post["action.mean"].numpy(),
        "action_std": action_std,
    }


def draw_ui(img_rgb, mode, episodes_done, step, max_steps):
    canvas = np.zeros((IMG_H + BAR, IMG_W, 3), dtype=np.uint8)
    canvas[:IMG_H] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, f"MODE: {mode}   DONE: {episodes_done}   STEP: {step}/{max_steps}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    labels = {
        "start": "START" if mode == "idle" else "END",
        "stop": "EMERGENCY STOP",
        "quit": "QUIT",
    }
    colors = {"start": (0, 140, 0), "stop": (0, 0, 200), "quit": (70, 70, 70)}
    for name, (x, y, w, h) in BUTTONS.items():
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colors[name], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(canvas, labels[name], (x + 8, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas


def smooth_move_to_home(env, ui, ctrl_fps=30, duration=2.5):
    q_cur = env.get_qpos()
    q_start = q_cur[:6].copy()
    grip = float(q_cur[6])
    n = max(1, int(duration * ctrl_fps))
    for k in range(n + 1):
        if ui["stop"]:
            return
        t = k / n
        q = (1.0 - t) * q_start + t * HOME_QPOS_REAL
        env.send_action(np.concatenate([q.astype(np.float32), np.array([grip], dtype=np.float32)]))
        time.sleep(1.0 / ctrl_fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max_steps", type=int, default=400, help="单轮最多推理步数")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=10, help="策略推理频率（与训练 fps 一致）")
    ap.add_argument("--ctrl_fps", type=int, default=50, help="动作发送频率（插值），越高越平滑")
    ap.add_argument("--move_spd_rate", type=int, default=30)
    ap.add_argument("--action_smooth", type=float, default=0.5, help="策略动作EMA平滑系数0~1(1=不平滑)")
    ap.add_argument("--debug_log", default=None)
    ap.add_argument("--no_confirm", action="store_true")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    stats = load_stats(args.checkpoint)
    env = RealPiperEnv(can_interface="can0", image_size=96, move_spd_rate=args.move_spd_rate)

    print("[安全提示] 真机策略即将开始运动！确保周围无障碍、随时可断电。")
    if not args.no_confirm:
        input("按回车开始，Ctrl+C 取消...")

    ui = {"start": False, "stop": False, "quit": False}

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
        debug_w.writerow(["ep", "step"] + [f"q{i}" for i in range(7)] + [f"a{i}" for i in range(7)])

    sub = max(1, args.ctrl_fps // args.fps)
    mode = "idle"
    episodes_done = 0
    step = 0
    prev_action = None
    prev_smooth = None

    try:
        while not ui["quit"] and not ui["stop"]:
            # 读取相机 + 显示 UI（空闲/录制都刷新）
            img480 = env.render_image(480, keep_aspect=True)
            cv2.imshow(WIN, draw_ui(img480, mode, episodes_done, step, args.max_steps))
            cv2.waitKey(1)

            if mode == "idle":
                if ui["start"]:
                    ui["start"] = False
                    print(f"\n=== 第 {episodes_done+1} 轮：复位到 home ===")
                    smooth_move_to_home(env, ui)
                    if ui["stop"]:
                        break
                    policy.reset()
                    prev_action = None
                    prev_smooth = None
                    step = 0
                    mode = "recording"
                    print("开始策略推理...")
                else:
                    time.sleep(1.0 / 20.0)  # 空闲轮询 20Hz
                continue

            # ---- 录制中 ----
            if ui["stop"]:
                break
            if ui["start"]:  # 录制中 START 已被当作 END
                ui["start"] = False
                print("→ 正常结束本轮")
                smooth_move_to_home(env, ui)
                episodes_done += 1
                mode = "idle"
                continue
            if step >= args.max_steps:
                print(f"→ 本轮达到 {args.max_steps} 步，结束")
                smooth_move_to_home(env, ui)
                episodes_done += 1
                mode = "idle"
                continue

            qpos = env.get_qpos()
            img = cv2.resize(img480, (96, 96))
            img_f = img.astype(np.float32) / 255.0
            img_chw = img_f.transpose(2, 0, 1)
            img_norm = (img_chw - stats["img_mean"]) / stats["img_std"]
            state_norm = (qpos - stats["state_mean"]) / stats["state_std"]

            batch = {
                "observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(args.device),
                "observation.image": torch.from_numpy(img_norm).float().unsqueeze(0).to(args.device),
            }
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = action * torch.from_numpy(stats["action_std"]).to(args.device) + torch.from_numpy(stats["action_mean"]).to(args.device)
            action_np = action[0].cpu().numpy().astype(np.float32)

            if debug_w is not None:
                debug_w.writerow([episodes_done, step, *np.round(qpos, 4), *np.round(action_np, 4)])

            # EMA 平滑策略输出（抑制策略抖动）
            if prev_smooth is None:
                smooth_a = action_np
            else:
                smooth_a = args.action_smooth * action_np + (1.0 - args.action_smooth) * prev_smooth
            prev_smooth = smooth_a

            # 50Hz 插值发送
            a_prev = smooth_a if prev_action is None else prev_action
            for k in range(sub):
                if ui["stop"] or ui["quit"]:
                    break
                alpha = (k + 1) / sub
                interp = (1.0 - alpha) * a_prev + alpha * smooth_a
                env.send_action(interp)
                time.sleep(1.0 / args.ctrl_fps)
            prev_action = smooth_a
            step += 1

        if ui["stop"]:
            print("\n[紧急停止] 部署已中止")
    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl+C")
    finally:
        if debug_f is not None:
            debug_f.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
    print(f"真机部署结束（完成 {episodes_done} 轮）")


if __name__ == "__main__":
    main()
