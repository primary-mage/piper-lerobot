#!/usr/bin/env python3
"""后处理：把 piper_real 的 action 列做 EMA 平滑，生成 piper_real_smooth（图像/state 不变）。

用法：python postprocess_smooth.py --src local/piper_real --dst local/piper_real_smooth --alpha 0.3
"""
import argparse
import glob
import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from sim_collect_piper import build_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="local/piper_real")
    ap.add_argument("--dst", default="local/piper_real_smooth")
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA 系数（越小越平滑）")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base = os.path.expanduser("~/.cache/huggingface/lerobot")
    src_root = f"{base}/{args.src}"
    dst_root = f"{base}/{args.dst}"
    if args.overwrite and os.path.exists(dst_root):
        bak = f"{dst_root}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.move(dst_root, bak)
        print(f"[提示] 旧目标已备份 → {bak}")

    df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{src_root}/data/**/*.parquet", recursive=True)]).reset_index(drop=True)
    vid = glob.glob(f"{src_root}/videos/**/*.mp4", recursive=True)[0]
    print(f"源数据: {len(df)} 帧, {df['episode_index'].nunique()} 集")

    # 平滑 action（每集独立 EMA）
    actions = np.stack(df['action'].values).astype(np.float32)
    smooth = actions.copy()
    for ep, g in df.groupby('episode_index'):
        idx = g.index.values
        a = actions[idx]
        s = a.copy()
        for i in range(1, len(a)):
            s[i] = args.alpha * a[i] + (1.0 - args.alpha) * s[i - 1]
        smooth[idx] = s
    flip_before = np.sum(np.abs(np.diff(np.sign(np.diff(actions[:, 3])))) > 0)
    flip_after = np.sum(np.abs(np.diff(np.sign(np.diff(smooth[:, 3])))) > 0)
    print(f"action 平滑完成（alpha={args.alpha}）J4 符号翻转: {flip_before} → {flip_after}")

    # 解码视频
    cap = cv2.VideoCapture(vid)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"视频解码完成: {len(frames)} 帧")

    dst = LeRobotDataset.create(repo_id=args.dst, fps=args.fps, features=build_features(), use_videos=True)
    for ep, g in df.groupby('episode_index'):
        for pos, row in g.iterrows():
            dst.add_frame({
                "observation.image": frames[pos],
                "observation.state": np.asarray(row['observation.state'], dtype=np.float32),
                "action": smooth[pos].astype(np.float32),
                "next.reward": np.array([float(row['next.reward'])], dtype=np.float32),
                "next.done": np.array([bool(row['next.done'])]),
                "next.success": np.array([bool(row['next.success'])]),
                "task": "grasp the object and lift it",
            })
        dst.save_episode()
        print(f"  episode {ep} 已保存（{len(g)} 帧）")
    dst.finalize()
    print(f"完成 → {args.dst}")


if __name__ == "__main__":
    main()
