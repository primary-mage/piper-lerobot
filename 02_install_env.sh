#!/usr/bin/env bash
# 一键安装：系统依赖 + conda(若无) + 环境 + PyTorch(cu128) + LeRobot + Piper 插件
# 用法：bash 02_install_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1090
source "$SCRIPT_DIR/common.sh"

# --- 1. 系统依赖 ---
info "安装系统依赖 (ffmpeg cmake build-essential ...)"
sudo apt update
sudo apt install -y ffmpeg cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev git curl wget

# --- 2. 定位或安装 conda（兼容已装的 miniconda3 / miniforge3）---
find_conda_base() {
  local exe
  exe="$(command -v conda 2>/dev/null || true)"
  if [ -n "$exe" ]; then
    (cd "$(dirname "$exe")/.." && pwd)
    return 0
  fi
  local base
  for base in "$HOME/miniforge3" "$HOME/miniconda3" "/opt/miniforge3" "/opt/conda"; do
    if [ -x "$base/bin/conda" ]; then echo "$base"; return 0; fi
  done
  return 1
}

CONDA_BASE="$(find_conda_base || true)"
if [ -z "$CONDA_BASE" ]; then
  info "未找到 conda，安装 miniforge ..."
  wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  CONDA_BASE="$HOME/miniforge3"
fi
info "使用 conda: $CONDA_BASE"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# --- 3. 创建 conda 环境 ---
if conda env list | awk '{print $1}' | grep -qx 'lerobot'; then
  info "conda 环境 lerobot 已存在，跳过创建"
else
  info "创建 conda 环境 lerobot (Python 3.12)..."
  conda create -y -n lerobot python=3.12
fi
conda activate lerobot
info "当前环境: $(python --version)"

# --- 4. 安装 PyTorch (cu128，覆盖 RTX 5060 / Blackwell) ---
info "安装 PyTorch cu128 (驱动需 >= 570)..."
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# --- 5. 安装 LeRobot + Piper 插件 ---
info "安装 LeRobot (core_scripts + training)..."
python -m pip install 'lerobot[core_scripts,training]'
info "安装 Piper 插件 (lerobot_robot_piper，会自动带 piper_sdk + python-can)..."
python -m pip install lerobot_robot_piper

# --- 6. 验证 ---
info "验证安装结果..."
python - <<'PY'
import torch
print("PyTorch        :", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU            :", torch.cuda.get_device_name(0))
    print("VRAM (GB)      :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
try:
    import lerobot
    print("LeRobot        :", getattr(lerobot, "__version__", "unknown"))
except Exception as e:
    print("LeRobot import :", e)
try:
    import piper_sdk
    print("piper_sdk      : OK")
except Exception as e:
    print("piper_sdk      :", e)
PY

info "安装完成 ✓"
info "下一步（每次开新终端都要先激活环境）：conda activate lerobot"
info "然后：bash 00_check_env.sh → bash 01_setup_can.sh → python 03_test_piper.py"
