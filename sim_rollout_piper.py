#!/usr/bin/env python3
"""在 MuJoCo 里部署训练好的 ACT 策略，做 sim2sim 闭环测试。

直接读取检查点里的归一化统计量，手动归一化观测 / 反归一化动作，
绕开预处理器管线的 key/dtype 约定差异。

用法：
  python sim_rollout_piper.py --checkpoint <检查点路径> --num_episodes 3
"""
import argparse

import numpy as np
import safetensors.torch as st
import torch

from piper_sim import PiperEnv
from lerobot.policies.act.modeling_act import ACTPolicy


def load_stats(checkpoint: str) -> dict:
    pre = st.load_file(f"{checkpoint}/policy_preprocessor_step_3_normalizer_processor.safetensors")
    post = st.load_file(f"{checkpoint}/policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    return {
        "img_mean": pre["observation.image.mean"].numpy(),   # (3,1,1)
        "img_std": pre["observation.image.std"].numpy(),     # (3,1,1)
        "state_mean": pre["observation.state.mean"].numpy(),  # (7,)
        "state_std": pre["observation.state.std"].numpy(),    # (7,)
        "action_mean": post["action.mean"].numpy(),           # (7,)
        "action_std": post["action.std"].numpy(),             # (7,)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="ACT 检查点目录（.../pretrained_model）")
    ap.add_argument("--num_episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    stats = load_stats(args.checkpoint)
    env = PiperEnv(image_size=96)
    substeps = max(1, round(0.1 / env.model.opt.timestep))  # 与采集端一致（10fps）

    for ep in range(args.num_episodes):
        policy.reset()
        qpos = env.reset(seed=100 + ep)
        max_h = 0.0
        for _ in range(args.max_steps):
            img = env.render_image()  # (96,96,3) uint8

            # 图像：/255 到 [0,1] → CHW → ImageNet 归一化
            img_f = img.astype(np.float32) / 255.0
            img_chw = img_f.transpose(2, 0, 1)
            img_norm = (img_chw - stats["img_mean"]) / stats["img_std"]
            # 状态：MEAN_STD 归一化
            state_norm = (qpos - stats["state_mean"]) / stats["state_std"]

            batch = {
                "observation.state": torch.from_numpy(state_norm).float().unsqueeze(0).to(device),
                "observation.image": torch.from_numpy(img_norm).float().unsqueeze(0).to(device),
            }
            with torch.inference_mode():
                action = policy.select_action(batch)
            # 反归一化动作
            action = action * torch.from_numpy(stats["action_std"]).to(device) + torch.from_numpy(stats["action_mean"]).to(device)
            action_np = action[0].cpu().numpy().astype(np.float32)

            qpos, _reward = env.step(action_np, substeps)
            max_h = max(max_h, env.cube_height())

        print(f"episode {ep + 1}: 方块最高高度 = {max_h:.3f} m (初始 ~0.105，>0.16 视为抓起)")

    env.close()
    print("sim2sim rollout 完成")


if __name__ == "__main__":
    main()
