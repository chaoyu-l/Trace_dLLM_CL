#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — 一键安装 TRACE Continual Learning 运行环境
#
#   - 创建 conda 环境 trace_env (默认 Python 3.12)
#   - pip install -r requirements.txt
#   - PyTorch 走 CUDA 12.8 索引（已在 requirements.txt 中配置）
#
# 用法:
#   bash setup_env.sh                       # 默认 env 名 trace_env, python 3.12
#   ENV_NAME=my_env bash setup_env.sh       # 自定义 env 名
#   PYTHON_VERSION=3.11 bash setup_env.sh   # 自定义 Python 版本
# =============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-trace_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="${REPO_ROOT}/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup_env] ERROR: 未找到 conda. 请先安装 Miniconda/Anaconda 并 source 其 profile."
  exit 1
fi

if [ ! -f "$REQ_FILE" ]; then
  echo "[setup_env] ERROR: 找不到 $REQ_FILE"
  exit 1
fi

# ---- 创建或复用 conda 环境 ----
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup_env] conda env '$ENV_NAME' 已存在, 跳过创建. (如需重建请先 conda env remove -n $ENV_NAME)"
else
  echo "[setup_env] 创建 conda env '$ENV_NAME' (python=$PYTHON_VERSION) ..."
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

# ---- 激活环境并安装依赖 ----
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[setup_env] 升级 pip ..."
python -m pip install --upgrade pip

echo "[setup_env] pip install -r requirements.txt ..."
pip install -r "$REQ_FILE"

echo ""
echo "[setup_env] 完成! 验证 PyTorch / CUDA:"
python - <<'PY'
import torch
print(f"  torch          = {torch.__version__}")
print(f"  cuda available = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda version   = {torch.version.cuda}")
    print(f"  device count   = {torch.cuda.device_count()}")
    print(f"  device 0 name  = {torch.cuda.get_device_name(0)}")
PY

# =============================================================================
# TRACE benchmark 数据 (git clone, idempotent)
#   目标: ./data/TRACE-Benchmark/LLM-CL-Benchmark_{500,1000,5000}/
#   兼容 TRACE 仓库的两种可能结构 (root 单层 / TRACE-Benchmark 双层),
#   clone 完成后会把 LLM-CL-Benchmark_* 子目录移到统一位置.
# =============================================================================
TRACE_DST="${REPO_ROOT}/data/TRACE-Benchmark"
echo ""
if [ -d "$TRACE_DST/LLM-CL-Benchmark_5000" ]; then
  echo "[setup_env] TRACE data 已存在: $TRACE_DST/LLM-CL-Benchmark_5000/, 跳过下载"
else
  echo "[setup_env] git clone TRACE benchmark -> $TRACE_DST ..."
  mkdir -p "${REPO_ROOT}/data"
  TMP_CLONE="$(mktemp -d)"
  git clone --depth 1 https://github.com/BeyonderXX/TRACE.git "$TMP_CLONE/TRACE"

  # 找 LLM-CL-Benchmark_5000 在 clone 出的哪一层
  if [ -d "$TMP_CLONE/TRACE/LLM-CL-Benchmark_5000" ]; then
    SRC="$TMP_CLONE/TRACE"
  elif [ -d "$TMP_CLONE/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_5000" ]; then
    SRC="$TMP_CLONE/TRACE/TRACE-Benchmark"
  else
    echo "[setup_env] ERROR: clone 后未找到 LLM-CL-Benchmark_5000"
    find "$TMP_CLONE" -maxdepth 3 -name "LLM-CL-Benchmark_5000" -type d
    rm -rf "$TMP_CLONE"
    exit 1
  fi
  mkdir -p "$TRACE_DST"
  mv "$SRC"/LLM-CL-Benchmark_* "$TRACE_DST/"
  rm -rf "$TMP_CLONE"
  echo "[setup_env] TRACE data 已就绪: $TRACE_DST/"
fi

echo ""
echo "[setup_env] 后续启动方式:"
echo "  conda activate $ENV_NAME"
echo ""
echo "[setup_env] 模型权重: run_<model>_5000_e<N>.sh 首次跑时会自动下载到 ./models/"
echo "           (Llama-3-8B 需先在 hf.co/meta-llama/Meta-Llama-3-8B-Instruct 申请 license"
echo "            + huggingface-cli login, 其他 3 个模型免登录直接下)"
