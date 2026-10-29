#!/usr/bin/env bash
# 环境自检：跑一遍，看哪些条件已满足、哪些还缺。只读，不改系统。
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"

echo "==================== 环境自检 ===================="

# --- 系统 / 架构 ---
if [ -f /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  info "系统: $PRETTY_NAME"
  case "$VERSION_ID" in
    20.04|22.04|24.04) info "Ubuntu 版本在推荐范围 (20.04/22.04/24.04) ✓";;
    *) warn "非推荐 Ubuntu 版本，建议 22.04 或 24.04";;
  esac
fi

ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then
  info "架构: x86_64 ✓"
else
  warn "架构: $ARCH（非 x86_64，视频解码会退回 pyav）"
fi

# --- GPU / 驱动 ---
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
  VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
  info "GPU: $GPU | 驱动: $DRV | 显存: ${VRAM} MB"
  DRV_MAJOR="$(echo "$DRV" | cut -d. -f1)"
  if [ "${DRV_MAJOR:-0}" -ge 570 ]; then
    info "驱动 >= 570，支持 cu128/cu130 ✓"
  else
    warn "驱动 < 570，RTX 5060 需要先升级驱动"
  fi
else
  err "未检测到 nvidia-smi，请先安装 NVIDIA 驱动"
fi

# --- Python / conda ---
if command -v conda >/dev/null 2>&1 || [ -x "$HOME/miniforge3/bin/conda" ]; then
  info "conda: 已安装 ✓"
else
  warn "conda: 未安装（运行 02_install_env.sh 会自动装 miniforge）"
fi
info "系统 python3: $(command -v python3 >/dev/null 2>&1 && python3 --version || echo '未找到')"

# --- CAN 工具 ---
for t in candump cansend ip ethtool; do
  if command -v "$t" >/dev/null 2>&1; then info "$t: ✓"; else warn "$t: 缺失（运行 01_setup_can.sh 会自动装）"; fi
done

# --- CAN 接口 ---
CAN_IFS="$(ip -br link show type can 2>/dev/null | awk '{print $1}')"
if [ -n "$CAN_IFS" ]; then
  info "检测到 CAN 接口: $(echo "$CAN_IFS" | tr '\n' ' ')"
else
  warn "未检测到 CAN 接口（检查 USB-CAN 是否插入、机械臂是否上电）"
fi

# --- 相机 ---
if ls /dev/video* >/dev/null 2>&1; then
  info "检测到相机: $(ls /dev/video* 2>/dev/null | tr '\n' ' ')"
else
  warn "未检测到 /dev/video* 相机"
fi

echo "=================================================="
echo "全部 ✓ 即可开始；有 ✗/! 的按提示先补上。"
