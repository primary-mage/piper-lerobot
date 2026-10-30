#!/usr/bin/env python3
"""真机 ACT 策略部署：加载训练好的 ACT，读真机关节角 + D435i 相机，输出动作发真机。

UI（鼠标点击）：
  [EMERGENCY STOP]  立即中止整个部署（手臂进入拖动示教悬停）
  [NORMAL END]      结束当前这一轮，进入下一轮（或结束全部）

用法：
  bash real_rollout.sh [检查点] [--num_episodes 3] [--max_steps 400]

安全：每轮先复位 home、低速、开始前回车确认。
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
    # 防护：std=0 的维度（采集时从没动过的关节）除以 0 会产生 NaN
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
    """平滑复位到 home 姿态（期间响应紧急停止）。"""
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
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--move_spd_rate", type=int, default=30, help="运动速度百分比，越低越安全")
    ap.add_argument("--no_confirm", action="store_true")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    stats = load_stats(args.checkpoint)
    env = RealPiperEnv(can_interface="can0", image_size=96, move_spd_rate=args.move_spd_rate)

    print("[安全提示] 真机策略即将开始运动！请确保手臂周围无障碍物、随时可断电。")
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

    try:
        for ep in range(args.num_episodes):
            if ui["stop"]:
                break
            print(f"\n=== 部署第 {ep+1}/{args.num_episodes} 轮 ===")
            policy.reset()
            move_to_home(env, ui)
            if ui["stop"]:
                break
            print("已复位到 home，开始策略推理...")
            for t in range(args.max_steps):
                if ui["stop"]:
                    break
                if ui["end"]:
                    ui["end"] = False
                    print("→ 正常结束本轮")
                    break

                qpos = env.get_qpos()
                # 与采集一致：先 480x360 (4:3) 再缩到 96x96
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
                env.send_action(action[0].cpu().numpy().astype(np.float32))
                time.sleep(1.0 / args.fps)
            print(f"第 {ep+1} 轮结束")
        if ui["stop"]:
            print("\n[紧急停止] 部署已中止")
    except KeyboardInterrupt:
        print("\n[中断] 用户 Ctrl+C")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
    print("真机部署结束")


if __name__ == "__main__":
    main()
