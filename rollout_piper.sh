#!/usr/bin/env bash
# 部署训练好的 ACT 到 MuJoCo，跑 sim2sim 闭环。用法：bash rollout_piper.sh [检查点路径]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env

CKPT="${1:-$HOME/lerobot_outputs/act_piper_keyboard/checkpoints/last/pretrained_model}"
if [ ! -e "$CKPT" ]; then
  err "未找到检查点: $CKPT"
  exit 1
fi
info "部署策略到 MuJoCo: $CKPT"
python "$SCRIPT_DIR/sim_rollout_piper.py" --checkpoint "$CKPT" --num_episodes 5
