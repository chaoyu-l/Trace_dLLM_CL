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
# TRACE benchmark 数据 (Google Drive, idempotent)
#   TRACE GitHub repo 里只有代码, 数据托管在 Google Drive:
#     https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV
#   用 gdown 自动下载 + unzip 到 ./data/TRACE-Benchmark/
# =============================================================================
TRACE_DST="${REPO_ROOT}/data/TRACE-Benchmark"
TRACE_GDRIVE_ID="1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"

echo ""
if [ -d "$TRACE_DST/LLM-CL-Benchmark_5000" ]; then
  echo "[setup_env] TRACE data 已存在: $TRACE_DST/LLM-CL-Benchmark_5000/, 跳过下载"
else
  # 确保 gdown 可用 (在 requirements.txt 里, 若用户跳过 pip 装则补一下)
  if ! python -c "import gdown" 2>/dev/null; then
    echo "[setup_env] 安装 gdown ..."
    pip install -q gdown
  fi

  mkdir -p "${REPO_ROOT}/data"
  TMP_DIR="$(mktemp -d)"
  TMP_ZIP="$TMP_DIR/trace_benchmark.zip"

  echo "[setup_env] gdown 下载 TRACE benchmark 从 Google Drive (~80 MB) ..."
  if python -m gdown "https://drive.google.com/uc?id=$TRACE_GDRIVE_ID" -O "$TMP_ZIP"; then
    echo "[setup_env] 解压 ..."
    mkdir -p "$TMP_DIR/extract"
    unzip -q "$TMP_ZIP" -d "$TMP_DIR/extract"
    # zip 是 Mac 打的, 清掉 __MACOSX/ 元数据目录避免后续 find 误命中
    rm -rf "$TMP_DIR/extract/__MACOSX"

    # 找 LLM-CL-Benchmark_5000 在解压出的哪一层
    FOUND=$(find "$TMP_DIR/extract" -maxdepth 4 -name "LLM-CL-Benchmark_5000" -type d -not -path "*__MACOSX*" 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
      PARENT="$(dirname "$FOUND")"
      mkdir -p "$TRACE_DST"
      mv "$PARENT"/LLM-CL-Benchmark_* "$TRACE_DST/"
      rm -rf "$TMP_DIR"
      echo "[setup_env] TRACE data 已就绪: $TRACE_DST/"
    else
      echo "[setup_env] ERROR: 解压后找不到 LLM-CL-Benchmark_5000"
      find "$TMP_DIR/extract" -maxdepth 4 -type d | head -20
      rm -rf "$TMP_DIR"
      exit 1
    fi
  else
    rm -rf "$TMP_DIR"
    cat <<EOF

[setup_env] WARN: gdown 自动下载失败 (Google Drive 配额 / 网络受限 / 需要 cookie 等).
请手动获取 TRACE 数据:
  方案 A — 浏览器下载:
    1. 浏览器打开 https://drive.google.com/file/d/$TRACE_GDRIVE_ID/view
    2. 下载 zip 包, 解压到 $TRACE_DST/
    3. 确认 $TRACE_DST/LLM-CL-Benchmark_5000/{train,eval,test}.json 存在

  方案 B — 软链已有数据 (你本机其他位置已有 TRACE):
    mkdir -p "$(dirname "$TRACE_DST")"
    ln -s /path/to/existing/TRACE-Benchmark "$TRACE_DST"

EOF
    exit 1
  fi
fi

# =============================================================================
# 生成 FOMC_shuffled 数据集 (每样本 ABC 随机置换, 破除原 FOMC C-bias)
#   脚本内部 idempotent: 三个 split 都已存在则 skip
#   各规模在原 FOMC 旁生成 FOMC_shuffled/ 平级目录
# =============================================================================
echo ""
echo "[setup_env] 生成 FOMC_shuffled (idempotent) ..."
for size in 500 1000 5000; do
  size_root="$TRACE_DST/LLM-CL-Benchmark_${size}"
  if [ -d "$size_root/FOMC" ]; then
    python "${REPO_ROOT}/preprocess/shuffle_fomc_labels.py" \
      --src_root "$size_root" --dst_name FOMC_shuffled --seed 1234 \
      2>&1 | sed 's/^/    /'
  else
    echo "  跳过 LLM-CL-Benchmark_${size} (源 FOMC 不存在)"
  fi
done

echo ""
echo "[setup_env] 后续启动方式:"
echo "  conda activate $ENV_NAME"
echo ""
echo "[setup_env] 模型权重: run_<model>_5000_e<N>.sh 首次跑时会自动下载到 ./models/"
echo "           (Llama-3-8B 需先在 hf.co/meta-llama/Meta-Llama-3-8B-Instruct 申请 license"
echo "            + huggingface-cli login, 其他 3 个模型免登录直接下)"
