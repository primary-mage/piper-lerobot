#!/usr/bin/env bash
# 训练 ACT。用法：bash train_piper.sh [步数] [数据集repo_id]
#   默认：20000 步，数据集 local/piper_real（真机数据）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env   # 自动清 LD_LIBRARY_PATH + 代理（避免 cublasLt 崩溃）

STEPS="${1:-20000}"
DATASET="${2:-local/piper_real}"
# W&B 默认关闭（token 失效/网络受限时会让训练直接失败）；需要时：WANDB_ENABLE=true bash train_piper.sh
WANDB_ENABLE="${WANDB_ENABLE:-false}"
OUT="$HOME/lerobot_outputs/act_$(basename "$DATASET")"

info "训练 ACT | 数据集: $DATASET | 步数: $STEPS | 输出: $OUT"
rm -rf "$OUT" "$OUT.log"

lerobot-train \
  --policy.type=act \
  --dataset.repo_id="$DATASET" \
  --policy.device=cuda \
  --batch_size=4 \
  --steps="$STEPS" \
  --log_freq=100 \
  --output_dir="$OUT" \
  --job_name=act_piper \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.mode=offline \
  --wandb.project=piper_grasp \
  --policy.push_to_hub=false 2>&1 | tee "$OUT.log"

info "训练完成 ✓ 检查点: $OUT/checkpoints/last/pretrained_model"
info "loss 曲线：python plot_training.py $OUT.log"
