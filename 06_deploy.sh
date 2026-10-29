#!/usr/bin/env bash
# 部署训练好的策略，让 Piper 自主执行抓取。
# 前置：05 已训练完成，且 config.env 的 POLICY_PATH 指向正确检查点。
# 用法：bash 06_deploy.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"
activate_env

if [ ! -e "$POLICY_PATH" ]; then
  warn "未找到策略: $POLICY_PATH"
  warn "请把 config.env 的 POLICY_PATH 改成实际检查点路径，例如："
  warn "  ls $OUTPUT_DIR/$JOB_NAME/checkpoints/"
  exit 1
fi

info "部署策略并执行抓取..."
info "策略: $POLICY_PATH"
info "任务: $TASK_DESCRIPTION | 运行时长 60s（Ctrl+C 停止）"

lerobot-rollout \
  --strategy.type=base \
  --policy.path="$POLICY_PATH" \
  --robot.type=piper \
  --robot.can_interface="$CAN_INTERFACE" \
  --robot.bitrate="$CAN_BITRATE" \
  --robot.include_gripper=true \
  --robot.use_degrees=false \
  --robot.cameras="$CAMERA_JSON" \
  --display_data=true \
  --task="$TASK_DESCRIPTION" \
  --duration=60

info "部署结束 ✓"
