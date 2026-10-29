#!/usr/bin/env bash
# 训练 ACT 策略（针对 RTX 5060 8GB 显存优化）。
# 前置：04 已采集数据。
# 用法：bash 05_train_act.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env

info "开始 ACT 训练 (batch=${BATCH_SIZE}, steps=${STEPS}, device=cuda)..."
mkdir -p "$OUTPUT_DIR"

lerobot-train \
  --policy.type=act \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.root="$DATASET_ROOT" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --policy.device=cuda \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --wandb.enable=false \
  --policy.push_to_hub=false

info "训练完成 ✓ 输出目录: $OUTPUT_DIR"
info "检查点保存在: $OUTPUT_DIR/checkpoints/last/pretrained_model"
info "下一步：把 config.env 里的 POLICY_PATH 指向最新检查点，然后 bash 06_deploy.sh"
