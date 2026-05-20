#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=h200:1
#SBATCH --output=slurm_logs/%x_%j.log
#SBATCH --error=slurm_logs/%x_%j.error
#SBATCH --qos=neurips-2026
#SBATCH --partition=non-reserved
#SBATCH --constraint=zone-sof1
# sof1-h200-{0,1,2}: MAINTENANCE+RESERVED (其他用户长占), 排除
# 可用节点: sof1-h200-{3,4,5,6,7} (5 节点 × 8 GPU = 40 张 H200, 够 12 个 job)
#SBATCH --exclude=sof1-h200-[0-2]

# =============================================================================
# Parametric TRACE-CL sbatch.
#
# 每个 job 在独立 sandbox cwd (runs/<TAG>/) 里跑 scripts/run_*.sh,
# 让 12 个并发 job 的 ./data_files 和 ./outputs_* 互不干扰.
# models/ 和 data/ 通过 symlink 共享.
#
# Conda env bootstrap:
#   /scratch (miniconda3) 是 node-local, 每个计算节点首次跑会自动跑 setup_env.sh
#   装 trace_env (~10-20 min), 之后同节点直接复用.
#   scripts/run_*.sh 内部自己会 conda activate trace_env (它有 idempotent 检测).
#   我们通过 BASH_ENV 让 sub-bash 自动 source conda profile, 使 sub-shell
#   里 conda function 可用 (否则 'command -v conda' 失败 → run_*.sh 退出).
#
# 用法:
#   sbatch --job-name=llama_8b_5000_e5 --time=24:00:00 \
#       --export=ALL,TAG=llama_8b_5000_e5,SCRIPT=scripts/run_llama_8b_5000_e5.sh \
#       trace.sbatch
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/zhendong_li/Trace_dLLM_CL"

: "${TAG:?TAG required (e.g. llama_8b_5000_e5)}"
: "${SCRIPT:?SCRIPT required (e.g. scripts/run_llama_8b_5000_e5.sh)}"

# ===== Source conda profile =====
# /scratch/zhendong_li/tools/miniconda3 在每个计算节点都有 base install (诊断验证).
CONDA_PROFILE="/scratch/zhendong_li/tools/miniconda3/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_PROFILE" ]; then
    echo "FATAL: conda profile not found at $CONDA_PROFILE" >&2
    exit 1
fi
set +u
source "$CONDA_PROFILE"
set -u

# 让 'bash $SCRIPT' 子进程自动 source conda.sh, 子 shell 里 'command -v conda' OK.
# scripts/run_*.sh 内部 conda activate trace_env 才能继续.
export BASH_ENV="$CONDA_PROFILE"

# DeepSpeed import 时硬性检查 CUDA_HOME (op_builder/builder.installed_cuda_version).
# 集群 CUDA toolkit 装在 /opt/modules/nvidia-cuda-12.8.1, 把它显式加到 PATH/LD.
export CUDA_HOME="/opt/modules/nvidia-cuda-12.8.1"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ===== Verify trace_env health (fail-fast, no auto-install) =====
# trace_env 必须在 sbatch 启动前已经装好 (用 prep_envs.sh 预装所有 sof1 节点).
# 健康 = trace_env/bin/python 能 import torch+deepspeed+transformers.
# 注意: 用绝对路径 python 而非 'conda run' (后者在 sbatch 重定向 + set -e 下
# 出现过 stdin 拦截导致 import 假性失败的 bug).
TRACE_PY="$(conda info --base)/envs/trace_env/bin/python"
if [ ! -x "$TRACE_PY" ]; then
    echo "FATAL: $TRACE_PY 不存在或不可执行 ($(hostname))" >&2
    exit 1
fi
echo "[healthy] 检测 $TRACE_PY ..."
HEALTHY_OUT=$("$TRACE_PY" -c "import torch, deepspeed, transformers; print(torch.__version__)" 2>&1) || HEALTHY_RC=$?
HEALTHY_RC=${HEALTHY_RC:-0}
if [ "$HEALTHY_RC" -ne 0 ]; then
    echo "FATAL: trace_env import 失败 ($(hostname)), exit=$HEALTHY_RC" >&2
    echo "----- python output -----" >&2
    echo "$HEALTHY_OUT" >&2
    echo "-------------------------" >&2
    exit 1
fi
echo "[healthy] OK: $HEALTHY_OUT"

# ===== Build per-job sandbox cwd =====
SANDBOX="$REPO_ROOT/runs/$TAG"
mkdir -p "$SANDBOX"
cd "$SANDBOX"
for d in models data scripts training model evaluations inference utils preprocess sari; do
    if [ -e "$REPO_ROOT/$d" ] && [ ! -e "$d" ]; then
        ln -s "$REPO_ROOT/$d" "$d"
    fi
done
for f in calculate_metrics.py metrics.py requirements.txt setup_env.sh; do
    if [ -e "$REPO_ROOT/$f" ] && [ ! -e "$f" ]; then
        ln -s "$REPO_ROOT/$f" "$f"
    fi
done

# ===== Diagnostics =====
echo "===== TRACE Sandbox Run ====="
echo "  tag        : $TAG"
echo "  script     : $SCRIPT"
echo "  sandbox    : $SANDBOX  (cwd)"
echo "  SLURM_JOB_ID = ${SLURM_JOB_ID:-N/A}"
echo "  Node       : $(hostname)"
echo "  conda envs : $(conda env list | awk '{print $1}' | grep -v '^#' | grep -v '^$' | tr '\n' ' ')"
nvidia-smi -L
echo "============================="

# ===== Run =====
# scripts/run_*.sh 自己会 conda activate trace_env (idempotent).
# BASH_ENV 已 export, 子 bash 里 conda function 可用.
bash "$SCRIPT"

echo "===== DONE: $TAG ====="
