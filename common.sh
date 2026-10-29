#!/usr/bin/env bash
# 公共函数：日志颜色、conda 环境激活。供所有脚本 source。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载全局配置
if [ -f "$SCRIPT_DIR/config.env" ]; then
  # shellcheck disable=SC1090
  source "$SCRIPT_DIR/config.env"
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'
info() { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }

# 清理会干扰 CUDA / 网络的环境变量。
# 本机 LD_LIBRARY_PATH 里混了 libtorch + 系统 CUDA-13.3，会让 torch 加载错误的
# libcublasLt.so（报 cublasLtCreate 崩溃）；socks 代理 + no_proxy 里的 IPv6 会让
# huggingface_hub 的 httpx 报错。LeRobot 脚本不需要这些，直接清掉。
sanitize_env() {
  unset LD_LIBRARY_PATH 2>/dev/null || true
  unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy NO_PROXY no_proxy 2>/dev/null || true
}

# 定位 conda 并激活 lerobot 环境
activate_env() {
  sanitize_env
  local CONDA_EXE
  CONDA_EXE="$(command -v conda 2>/dev/null || true)"
  if [ -z "$CONDA_EXE" ]; then
    for base in "$HOME/miniforge3" "$HOME/miniconda3" "/opt/miniforge3" "/opt/conda"; do
      if [ -x "$base/bin/conda" ]; then
        CONDA_EXE="$base/bin/conda"
        break
      fi
    done
  fi
  if [ -z "$CONDA_EXE" ]; then
    err "找不到 conda，请先运行: bash 02_install_env.sh"
    exit 1
  fi

  local CONDA_BASE
  CONDA_BASE="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
  # shellcheck disable=SC1090
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate lerobot
  info "已激活 conda 环境: lerobot (Python $(python --version 2>&1 | awk '{print $2}'))"
}
