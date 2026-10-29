#!/usr/bin/env python3
"""把 LeRobot 训练日志画成 loss 曲线。

用法：
  bash train_piper.sh 20000 2>&1 | tee train.log    # 训练并保存日志
  python plot_training.py train.log                 # 画曲线

会画出：loss（总损失）、l1_loss（动作重建误差）、kld_loss（VAE KL散度）、lr（学习率）。
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    steps, loss, l1, kld, grdn, lr = [], [], [], [], [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        if "step:" not in line or "loss:" not in line:
            continue
        m = re.search(r"step:(\d+)", line)
        if not m:
            continue

        def g(name):
            x = re.search(name + r":([0-9.eE+-]+)", line)
            return float(x.group(1)) if x else None

        steps.append(int(m.group(1)))
        loss.append(g("loss"))
        l1.append(g("l1_loss"))
        kld.append(g("kld_loss"))
        grdn.append(g("grdn"))
        lr.append(g("lr"))
    return steps, loss, l1, kld, grdn, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    args = ap.parse_args()

    steps, loss, l1, kld, grdn, lr = parse_log(args.logfile)
    if not steps:
        print("未在日志中找到训练指标（确认日志里有 step:/loss: 行）")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, loss, label="loss")
    axes[0, 0].set_title("total loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(steps, l1, color="tab:orange", label="l1_loss")
    axes[0, 1].set_title("action reconstruction error (l1)")
    axes[0, 1].grid(True)

    axes[1, 0].plot(steps, kld, color="tab:green", label="kld_loss")
    axes[1, 0].set_title("VAE KL divergence")
    axes[1, 0].grid(True)

    axes[1, 1].plot(steps, lr, color="tab:purple", label="lr")
    axes[1, 1].set_title("learning rate")
    axes[1, 1].grid(True)

    fig.suptitle(f"ACT training curves ({args.logfile})")
    fig.tight_layout()
    out = args.logfile.rsplit(".", 1)[0] + "_curves.png"
    fig.savefig(out, dpi=120)
    print(f"曲线已保存 → {out}")
    print(f"首尾 loss: {loss[0]:.3f} → {loss[-1]:.3f}（下降 = 在学）")
    if l1 and l1[-1] is not None:
        print(f"首尾 l1_loss: {l1[0]:.3f} → {l1[-1]:.3f}（下降 = 动作预测越来越准）")


if __name__ == "__main__":
    main()
