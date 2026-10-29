#!/usr/bin/env bash
# LeRobot 仿真冒烟：在 MuJoCo 里验证「训练 ACT → 仿真评估」整条软件链路，无需真机硬件。
# 用法：bash sim_demo.sh [pusht|aloha|libero]
#   pusht  = 单臂推方块（最快，2 分钟见结果）
#   aloha  = 双机械臂精细操作
#   libero = 抓取放置（pick-and-place，最接近你的抓取目标，但安装最重）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"

SIM_ENV="${1:-pusht}"

case "$SIM_ENV" in
  pusht)
    EXTRA="pusht"
    DATASET="lerobot/pusht"
    ENV_TYPE="pusht"
    EVAL_EXTRA_ARGS=()
    ;;
  aloha)
    EXTRA="aloha"
    DATASET="lerobot/aloha_sim_transfer_cube_human"
    ENV_TYPE="aloha"
    EVAL_EXTRA_ARGS=()
    ;;
  libero)
    EXTRA="libero"
    DATASET="lerobot/libero_object_no_noops"
    ENV_TYPE="libero"
    EVAL_EXTRA_ARGS=(--env.task=libero_object)
    ;;
  *)
    err "未知环境: $SIM_ENV（可选 pusht|aloha|libero）"
    exit 1
    ;;
esac

activate_env

info "仿真环境: $SIM_ENV | 数据集: $DATASET"

# 安装对应环境 + 评估依赖（幂等）
info "安装 lerobot[$EXTRA,evaluation] ..."
python -m pip install "lerobot[$EXTRA,evaluation]"

OUT="$HOME/lerobot_outputs/sim_act_${SIM_ENV}"
CKPT="$OUT/checkpoints/last/pretrained_model"

info "训练 ACT（公开数据集，验证 CUDA 训练链路，冒烟用 2000 步）..."
lerobot-train \
  --policy.type=act \
  --dataset.repo_id="$DATASET" \
  --policy.device=cuda \
  --batch_size=4 \
  --steps=2000 \
  --output_dir="$OUT" \
  --job_name="act_${SIM_ENV}" \
  --wandb.enable=false \
  --policy.push_to_hub=false

info "评估（MuJoCo 里跑 10 个 episode 出成功率）..."
if [ ! -e "$CKPT" ]; then
  warn "未找到检查点 $CKPT，列出实际检查点目录："
  ls -R "$OUT" | head -40
  exit 1
fi

lerobot-eval \
  --policy.type=act \
  --policy.pretrained_path="$CKPT" \
  --env.type="$ENV_TYPE" \
  "${EVAL_EXTRA_ARGS[@]}" \
  --eval.n_episodes=10 \
  --eval.use_async_envs=false \
  --policy.device=cuda

info "仿真冒烟完成 ✓"
info "提示：2000 步只是验证链路能跑通，成功率可能接近 0；真正训练把 --steps 加到 5 万步以上。"
info "想直接看评估出非零成功率：把 --policy.pretrained_path 换成 lerobot/act_pusht 这种预训练模型即可。"
