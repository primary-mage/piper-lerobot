#!/usr/bin/env python3
"""真机 ACT 策略部署：加载训练好的 ACT，读真机关节角 + D435i 相机，输出动作发真机。

用法：
  python real_rollout.py --checkpoint ~/lerobot_outputs/act_piper_real/checkpoints/last/pretrained_model --num_episodes 3

安全：
  - 每轮先复位到 home 姿态
  - 限制最大步数、低速运动（--move_spd_rate）
  - 开始前有回车确认
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


def load_stats(checkpoint: str) -> dict:
    pre = st.load_file(f"{checkpoint}/policy_preprocessor_step_3_normalizer_processor.safetensors")
    post = st.load_file(f"{checkpoint}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    state_std = pre["observation.state.std"].numpy().copy()
    action_std = post["action.std"].numpy().copy()
    # 防护：std=0 的维度（采集时从没动过的关节）除以 0 会产生 NaN，用 1.0 代替
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


def move_to_home(env, ctrl_fps=30, duration=2.5):
    """平滑复位到 home 姿态。"""
    q_cur = env.get_qpos()
    q_start = q_cur[:6].copy()
    grip = float(q_cur[6])
    n = max(1, int(duration * ctrl_fps))
    for k in range(n + 1):
        t = k / n
        q = (1.0 - t) * q_start + t * HOME_QPOS_REAL
        env.send_action(np.concatenate([q.astype(np.float32), np.array([grip], dtype=np.float32)]))
        time.sleep(1.0 / ctrl_fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="ACT 检查点目录（.../pretrained_model）")
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--move_spd_rate", type=int, default=30, help="运动速度百分比，越低越安全")
    ap.add_argument("--no_confirm", action="store_true", help="跳过开始前人工确认")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    stats = load_stats(args.checkpoint)
    env = RealPiperEnv(can_interface="can0", image_size=96, move_spd_rate=args.move_spd_rate)

    print("[安全提示] 真机策略即将开始运动，请确保手臂周围无障碍物、随时可断电。")
    if not args.no_confirm:
        input("按回车开始，Ctrl+C 取消...")

    win = "Real ACT (policy view)"
    try:
        for ep in range(args.num_episodes):
            print(f"\n=== 部署第 {ep+1}/{args.num_episodes} 轮 ===")
            policy.reset()
            move_to_home(env)
            print("已复位到 home，开始策略推理...")
            for t in range(args.max_steps):
                qpos = env.get_qpos()
                # 与采集一致：先 480x360 (4:3) 再缩到 96x96
                img480 = env.render_image(480, keep_aspect=True)
                img = cv2.resize(img480, (96, 96))
                cv2.imshow(win, cv2.cvtColor(img480, cv2.COLOR_RGB2BGR))
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
            print(f"第 {ep+1} 轮完成（{args.max_steps} 步）")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        env.close()
    print("真机部署结束")


if __name__ == "__main__":
    main()
