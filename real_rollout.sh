#!/usr/bin/env bash
# 真机 ACT 部署。用法：bash real_rollout.sh [检查点路径] [其他 real_rollout.py 参数]
# 自动清 LD_LIBRARY_PATH + 代理（避免 cublasLt 崩溃）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env

CKPT="${1:-$HOME/lerobot_outputs/act_piper_real/checkpoints/last/pretrained_model}"
shift || true
info "部署检查点: $CKPT"
python "$SCRIPT_DIR/real_rollout.py" --checkpoint "$CKPT" "$@"
