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
#SBATCH --exclude=sof1-h200-[0-2]
#SBATCH --time=5-00:00:00

# =============================================================================
# infer_helper.sbatch
#
# Run Phase-2 LoRA inference for ONE round of an existing slow trace job, in
# parallel with that job. infer_single.py has built-in "skip existing"
# (predictions/results-{r}-{t}-{name}.json) so concurrent helpers + original
# never write the same file twice.
#
# Required exports:
#   TAG          — sandbox dir name (e.g. llada_8b_5000_e5)
#   MODEL_FAMILY — "llada" | "dream"
#   TARGET_ROUND — integer 0..7 (only this checkpoint round is processed)
#
# Optional exports (default same as the original run_*.sh):
#   INFERENCE_TASKS — comma-separated task names
#                     (default for llada_*: C-STANCE,FOMC_shuffled,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten)
#                     (override for *_order2: NumGLUE-cm,NumGLUE-ds,FOMC_shuffled,20Minuten,C-STANCE,Py150,MeetingBank,ScienceQA)
#
# Usage:
#   sbatch --job-name=llada_8b_5000_e5_r5 \
#       --export=ALL,TAG=llada_8b_5000_e5,MODEL_FAMILY=llada,TARGET_ROUND=5 \
#       infer_helper.sbatch
# =============================================================================
set -euo pipefail

REPO_ROOT="/home/zhendong_li/Trace_dLLM_CL"

: "${TAG:?TAG required (e.g. llada_8b_5000_e5)}"
: "${MODEL_FAMILY:?MODEL_FAMILY required (llada|dream)}"
: "${TARGET_ROUND:?TARGET_ROUND required (0..7)}"

# ===== source conda =====
CONDA_PROFILE="/scratch/zhendong_li/tools/miniconda3/etc/profile.d/conda.sh"
set +u
source "$CONDA_PROFILE"
conda activate trace_env
set -u
export BASH_ENV="$CONDA_PROFILE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA toolkit needed by deepspeed import-time op compatibility check.
export CUDA_HOME="/opt/modules/nvidia-cuda-12.8.1"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# ===== cd into sandbox =====
SANDBOX="$REPO_ROOT/runs/$TAG"
if [ ! -d "$SANDBOX" ]; then
    echo "FATAL: sandbox $SANDBOX not found" >&2
    exit 1
fi
cd "$SANDBOX"

# ===== resolve paths from MODEL_FAMILY =====
case "$MODEL_FAMILY" in
  llada)
    MODEL_PATH="./models/LLaDA-8B-Instruct"
    OUT_DIR=$(ls -d ./outputs_LLaDA-8b-LoRA_*/lora_8b 2>/dev/null | head -1)
    EXTRA_ARGS="--remasking_strategy low_confidence --inference_batch 64"
    ;;
  dream)
    MODEL_PATH="./models/Dream-7b"
    OUT_DIR=$(ls -d ./outputs_dream_7b_*/lora_7b 2>/dev/null | head -1)
    EXTRA_ARGS="--dream_alg maskgit_plus --inference_batch 128"
    ;;
  *)
    echo "FATAL: unknown MODEL_FAMILY=$MODEL_FAMILY" >&2
    exit 1
    ;;
esac

if [ ! -d "$OUT_DIR" ]; then
    echo "FATAL: out_dir $OUT_DIR not found inside $SANDBOX" >&2
    exit 1
fi
PRED_OUTPUT="${OUT_DIR}/predictions"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_5000"
DATA_CACHE_PATH="./data_files_helper_r${TARGET_ROUND}"  # isolated cache to avoid races

# Default task order; override via INFERENCE_TASKS for order2 variants.
# IMPORTANT: SLURM --export uses "," as delimiter for the var list, so passing
# a comma-separated task string corrupts it (only first token survives).
# Workaround: caller uses ":" as separator; we translate back to "," here.
INFERENCE_TASKS="${INFERENCE_TASKS:-C-STANCE,FOMC_shuffled,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten}"
INFERENCE_TASKS="${INFERENCE_TASKS//:/,}"

mkdir -p "$PRED_OUTPUT" "$DATA_CACHE_PATH"

# Adapter for this round must exist (training already finished there).
ADAPTER_DIR="$OUT_DIR/$TARGET_ROUND"
if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "FATAL: adapter for round $TARGET_ROUND not found at $ADAPTER_DIR" >&2
    exit 1
fi

# SLURM GPU UUID -> integer 0 for diffusion code (same trick as oc.sbatch)
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
    export CUDA_VISIBLE_DEVICES=0
fi

echo "===== infer_helper Run ====="
echo "  tag           : $TAG"
echo "  family        : $MODEL_FAMILY"
echo "  target_round  : $TARGET_ROUND"
echo "  out_dir       : $SANDBOX/$OUT_DIR"
echo "  pred_output   : $SANDBOX/$PRED_OUTPUT"
echo "  tasks         : $INFERENCE_TASKS"
echo "  SLURM_JOB_ID  : ${SLURM_JOB_ID:-N/A}"
echo "  Node          : $(hostname)"
nvidia-smi -L
echo "============================"

python inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "$DATA_CACHE_PATH" \
    --inference_tasks "$INFERENCE_TASKS" \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$OUT_DIR" \
    --model_type diffusion \
    --sampling_steps 0 \
    --sampling_temperature 0.0 \
    $EXTRA_ARGS \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method lora \
    --inference_output_path "$PRED_OUTPUT" \
    --target_round "$TARGET_ROUND" \
    2>&1 | tee "${PRED_OUTPUT}/infer_helper_r${TARGET_ROUND}_${SLURM_JOB_ID:-noslurm}.log"

echo "===== DONE: $TAG round=$TARGET_ROUND ====="
