#!/usr/bin/env bash
# 采集抓取示范数据（遥操作录制数据集）。
# 前置：已激活 CAN(01) + 已安装环境(02) + 03 测试通过。
# 用法：bash 04_record.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env

info "开始采集抓取示范数据..."
info "任务: ${TASK_DESCRIPTION} | 条数: ${NUM_EPISODES} | 遥操作设备: ${TELEOP_TYPE}"
mkdir -p "$DATASET_ROOT"

ROBOT_ARGS=(
  --robot.type=piper
  --robot.can_interface="$CAN_INTERFACE"
  --robot.bitrate="$CAN_BITRATE"
  --robot.include_gripper=true
  --robot.use_degrees=false
  --robot.cameras="$CAMERA_JSON"
)

DATASET_ARGS=(
  --dataset.repo_id="$DATASET_REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.single_task="$TASK_DESCRIPTION"
  --dataset.num_episodes="$NUM_EPISODES"
  --dataset.episode_time_s="$EPISODE_TIME_S"
  --dataset.reset_time_s="$RESET_TIME_S"
  --dataset.video=true
)

if [ -n "${TELEOP_ARGS:-}" ]; then
  # 注意：这里有意让 TELEOP_ARGS 按空格分词展开成多个参数
  # shellcheck disable=SC2086
  lerobot-record "${ROBOT_ARGS[@]}" --teleop.type="$TELEOP_TYPE" $TELEOP_ARGS "${DATASET_ARGS[@]}"
else
  lerobot-record "${ROBOT_ARGS[@]}" --teleop.type="$TELEOP_TYPE" "${DATASET_ARGS[@]}"
fi

info "采集完成，数据集保存在: $DATASET_ROOT/$DATASET_REPO_ID"
info "下一步: bash 05_train_act.sh"
