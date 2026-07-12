#!/bin/bash
#SBATCH --job-name=director_sft_2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=h200:2
#SBATCH --output=/home/zhendong_li/FrameWorkers/training/director/logs/train_%j.log
#SBATCH --error=/home/zhendong_li/FrameWorkers/training/director/logs/train_%j.error
#SBATCH --qos=neurips-2026
#SBATCH --partition=non-reserved
#SBATCH --constraint=zone-sof1

# Parametric 2-GPU SFT for CoT / templated ablation experiments.
#
# Submit:
#   sbatch --export=ALL,SFT_DATA=...,ADAPTER_DIR=...,EPOCHS=4 \
#          training/director/train_sft_2gpu.sbatch
#
# At 2-GPU with `--per-device-batch 1 --grad-accum 4`, effective batch =
# 1 * 4 * 2 = 8 — matches the 1-GPU canonical SFT (1 * 8 * 1 = 8). For
# 1417 samples * 4 epochs / batch 8, the run lands at ~708 steps (canonical
# `qwen3_v3tmpl_e4_step712` reference).

set -euo pipefail

REPO_ROOT="/home/zhendong_li/FrameWorkers"
cd "$REPO_ROOT"

ENV_DIR="/home/zhendong_li/.conda-envs/director-train"
set +u
source ~/.bashrc
conda activate "$ENV_DIR" 2>/dev/null || conda activate frameworkers
set -u

if [ -z "${CONDA_PREFIX:-}" ] || [ ! -x "$CONDA_PREFIX/bin/python" ]; then
    echo "FATAL: conda activate failed." >&2
    exit 1
fi
echo "python: $(which python) — $(python --version)"
echo "CONDA_PREFIX=$CONDA_PREFIX"
nvidia-smi -L

# Required env vars
: "${SFT_DATA:?SFT_DATA required (e.g. training/director/samples_sft_full.jsonl)}"
: "${ADAPTER_DIR:?ADAPTER_DIR required (output dir; trainer saves to <ADAPTER_DIR>/sft/)}"
EPOCHS="${EPOCHS:-4}"

echo "=== SFT (2-GPU) ==="
echo "  data:    ${SFT_DATA}"
echo "  output:  ${ADAPTER_DIR}/sft"
echo "  epochs:  ${EPOCHS}"
echo "  config:  per_device=1 × grad_accum=4 × 2 GPUs = effective batch 8"

export PYTHONUNBUFFERED=1

PYTHONPATH=. torchrun --nproc_per_node=2 --master_port=29502 \
    training/director/train_sft.py \
        --sft-path "${SFT_DATA}" \
        --adapter-dir "${ADAPTER_DIR}" \
        --sft-epochs "${EPOCHS}" \
        --per-device-batch 1 \
        --grad-accum 4

echo "=== DONE — adapter at ${ADAPTER_DIR}/sft ==="
