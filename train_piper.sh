#!/usr/bin/env bash
# 训练 ACT（用键盘遥操采集的数据集）。用法：bash train_piper.sh [训练步数]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env   # 会自动清 LD_LIBRARY_PATH + 代理（避免 cublasLt 崩溃）

STEPS="${1:-20000}"
DATASET="local/piper_keyboard"   # 与 sim_collect_piper.py 的默认 repo_id 一致
OUT="$HOME/lerobot_outputs/act_piper_keyboard"

info "训练 ACT | 数据集: $DATASET | 步数: $STEPS"
rm -rf "$OUT" "$OUT.log"   # $OUT 必须不存在（lerobot-train 会自己创建），日志存独立文件

# tee 同时把日志打到终端 + 存文件，训练完可用 python plot_training.py $OUT.log 画曲线
lerobot-train \
  --policy.type=act \
  --dataset.repo_id="$DATASET" \
  --policy.device=cuda \
  --batch_size=4 \
  --steps="$STEPS" \
  --log_freq=100 \
  --output_dir="$OUT" \
  --job_name=act_piper \
  --wandb.enable=true \
  --wandb.mode=online \
  --wandb.project=piper_grasp \
  --policy.push_to_hub=false 2>&1 | tee "$OUT.log"

info "训练完成 ✓ 检查点: $OUT/checkpoints/last/pretrained_model"
info "loss 曲线：python plot_training.py $OUT.log"
