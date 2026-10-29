#!/usr/bin/env bash
# 配置并激活 Piper 的 CAN 总线（AgileX USB-CAN 模块，gs_usb，默认 can0 @ 1 Mbps）
# 注意：每次重启、重插 USB-CAN 后都需要重新运行本脚本。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"

info "安装 CAN 工具 (can-utils ethtool iproute2)..."
sudo apt update
sudo apt install -y can-utils ethtool iproute2

# 查找 CAN 接口
CAN_IF="$(ip -br link show type can 2>/dev/null | awk '{print $1}' | head -1)"
if [ -z "$CAN_IF" ]; then
  err "未检测到 CAN 接口。请确认：1) USB-CAN 模块已插入 2) 机械臂已上电"
  exit 1
fi
info "检测到 CAN 接口: $CAN_IF"

# 统一重命名为 can0（SDK/插件默认用 can0）
if [ "$CAN_IF" != "$CAN_INTERFACE" ]; then
  info "重命名 $CAN_IF -> $CAN_INTERFACE"
  sudo ip link set "$CAN_IF" down 2>/dev/null || true
  sudo ip link set "$CAN_IF" name "$CAN_INTERFACE"
fi

# 配置波特率并激活
sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true
sudo ip link set "$CAN_INTERFACE" type can bitrate "$CAN_BITRATE"
sudo ip link set "$CAN_INTERFACE" up

info "已激活 $CAN_INTERFACE @ ${CAN_BITRATE} bps"
ip -details link show "$CAN_INTERFACE" | grep -E "can state|bitrate" || true

info "监听 3 秒验证通信（机械臂上电后会周期性发 CAN 帧；无输出也未必是错，可继续用 03_test_piper.py 验证）..."
timeout 3 candump "$CAN_INTERFACE" || true

info "CAN 配置完成 ✓"
