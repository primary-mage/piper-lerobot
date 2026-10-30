#!/usr/bin/env python3
"""真机 ACT 策略部署：加载训练好的 ACT，读真机关节角 + D435i 相机，输出动作发真机。

UI：[EMERGENCY STOP] 立即中止；[NORMAL END] 结束当前轮。

抖动修复：策略按 fps(10Hz) 推理，但动作按 ctrl_fps(50Hz) 线性插值发送，
避免 10Hz 台阶导致手臂顿挫（与遥操 50Hz 同理）。

用法：bash real_rollout.sh [检查点] [--num_episodes 3] [--debug_log /tmp/roll.csv]
"""
import argparse
import time

import cv2
import numpy as np
import safetensors.torch as st
import torch

from real_piper import RealPiperEnv
from lerobot.policies.act.modeling_act import ACTPolicy

HOME_QPOS_REAL = np.array([0.0, 0.8, -1.2, 0.0, 0.0, 0.0], dtype=np.float32)

WIN = "Real ACT (Camera + Control)"
IMG_W, IMG_H, BAR = 480, 360, 90
BUTTONS = {
    "stop": (5, IMG_H + 10, 230, 70),
    "end":  (245, IMG_H + 10, 230, 70),
}
LABELS = {"stop": "EMERGENCY STOP", "end": "NORMAL END"}


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


def draw_ui(img_rgb, ep, num_ep, step, max_steps):
    canvas = np.zeros((IMG_H + BAR, IMG_W, 3), dtype=np.uint8)
    canvas[:IMG_H] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, f"EP {ep}/{num_ep}   STEP {step}/{max_steps}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for name, (x, y, w, h) in BUTTONS.items():
        color = (0, 0, 200) if name == "stop" else (0, 140, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 1)
        cv2.putText(canvas, LABELS[name], (x + 30, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def move_to_home(env, ui, ctrl_fps=30, duration=2.5):
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
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=10, help="策略推理频率（与训练 fps 一致）")
    ap.add_argument("--ctrl_fps", type=int, default=50, help="动作发送频率（插值），越高越平滑")
    ap.add_argument("--move_spd_rate", type=int, default=30)
    ap.add_argument("--debug_log", default=None, help="调试日志 CSV 路径")
    ap.add_argument("--no_confirm", action="store_true")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    stats = load_stats(args.checkpoint)
    env = RealPiperEnv(can_interface="can0", image_size=96, move_spd_rate=args.move_spd_rate)

    print("[安全提示] 真机策略即将开始运动！确保周围无障碍、随时可断电。")
    if not args.no_confirm:
        input("按回车开始，Ctrl+C 取消...")

    ui = {"stop": False, "end": False}

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
        debug_w.writerow(["step", "ep"] + [f"q{i}" for i in range(7)] + [f"a{i}" for i in range(7)])

    sub = max(1, args.ctrl_fps // args.fps)  # 每个策略步插值成 sub 个控制帧
    prev_action = None

    try:
        for ep in range(args.num_episodes):
            if ui["stop"]:
                break
            print(f"\n=== 部署第 {ep+1}/{args.num_episodes} 轮 ===")
            policy.reset()
            move_to_home(env, ui)
            if ui["stop"]:
                break
            prev_action = None
            print("已复位到 home，开始策略推理...")
            for t in range(args.max_steps):
                if ui["stop"]:
                    break
                if ui["end"]:
                    ui["end"] = False
                    print("→ 正常结束本轮")
                    break

                qpos = env.get_qpos()
                img480 = env.render_image(480, keep_aspect=True)
                img = cv2.resize(img480, (96, 96))
                cv2.imshow(WIN, draw_ui(img480, ep + 1, args.num_episodes, t, args.max_steps))
                cv2.waitKey(1)

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
                    debug_w.writerow([t, ep, *np.round(qpos, 4), *np.round(action_np, 4)])

                # 动作按 ctrl_fps 插值发送（消除 10Hz 台阶顿挫）
                a_prev = action_np if prev_action is None else prev_action
                for k in range(sub):
                    if ui["stop"]:
                        break
                    alpha = (k + 1) / sub
                    interp = (1.0 - alpha) * a_prev + alpha * action_np
                    env.send_action(interp)
                    time.sleep(1.0 / args.ctrl_fps)
                prev_action = action_np
            print(f"第 {ep+1} 轮结束")
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
    print("真机部署结束")


if __name__ == "__main__":
    main()
