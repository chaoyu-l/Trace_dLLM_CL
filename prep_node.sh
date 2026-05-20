#!/usr/bin/env bash
# =============================================================================
# 在当前节点强制 (重)装 trace_env.
# 由 prep_envs.sh 通过 sbatch --nodelist=<节点> 在每个 sof1-h200 节点上各跑一次.
# /scratch 是 node-local, 每个节点都要装一份, 装完后正式 sbatch 直接复用.
# =============================================================================
set -e
echo "[prep] node=$(hostname) start at $(date '+%H:%M:%S')"

# ===== Step 0: 节点 miniconda3 bootstrap (照搬 user 的 snippet) =====
USERNAME=`whoami`
MINICONDA_DPATH=${MINICONDA_DPATH:-"/scratch/$USERNAME/tools/miniconda3"}
TOOLS_DIR=`dirname $MINICONDA_DPATH`

if [ ! -d "$MINICONDA_DPATH" ]; then
    echo "Installing miniconda3"
    mkdir -p $TOOLS_DIR
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $TOOLS_DIR/miniconda3.sh
    bash $TOOLS_DIR/miniconda3.sh -b -p $MINICONDA_DPATH
    rm $TOOLS_DIR/miniconda3.sh

    $MINICONDA_DPATH/bin/conda update --all -y
    $MINICONDA_DPATH/bin/conda config --set solver libmamba
fi

#conda activate the base environment
$MINICONDA_DPATH/bin/activate
#conda activate the base environment without it being in PATH yet
$MINICONDA_DPATH/bin/conda init

source "$MINICONDA_DPATH/etc/profile.d/conda.sh"
cd /home/zhendong_li/Trace_dLLM_CL

# setup_env.sh 自带 idempotent (trace_env 存在则 skip), 不需要清残留
bash setup_env.sh

# 验证 (with CUDA_HOME, 与 trace.sbatch 一致)
export CUDA_HOME="/opt/modules/nvidia-cuda-12.8.1"
conda run -n trace_env python -c 'import torch, deepspeed, transformers; print(f"[prep] OK: torch {torch.__version__}, deepspeed {deepspeed.__version__}, transformers {transformers.__version__}")'

echo "[prep] node=$(hostname) done at $(date '+%H:%M:%S')"
