#!/usr/bin/env bash
# =============================================================================
# run_llada_8b_5000_e5_order2.sh
#   LLaDA-8B-Instruct 一站式 pipeline (配置 A: 5 epoch, 无 in-training eval):
#     任务顺序 (order2): NumGLUE-cm -> NumGLUE-ds -> FOMC_shuffled -> 20Minuten
#                       -> C-STANCE -> Py150 -> MeetingBank -> ScienceQA
#     Phase 0: base 推理 (无 LoRA, idempotent skip; FWT baseline, order2 专属)
#     Phase 1: 训练 (LoRA, 8 任务, 每任务 5 epoch, dynamic canvas group=4)
#     Phase 2: 推理 (test 集完整, low_confidence remasking, sampling_steps=0)
#
#   注意: LLaDA 不能加 --gradient_checkpointing (架构限制)
# =============================================================================
set -eo pipefail

# ===== Conda env auto-activate (skip if already active) =====
TRACE_ENV_NAME="${TRACE_ENV_NAME:-trace_env}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$TRACE_ENV_NAME" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[run] ERROR: conda not in PATH. Please source your conda profile first." >&2
    exit 1
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$TRACE_ENV_NAME"; then
    echo "[run] ERROR: conda env '$TRACE_ENV_NAME' missing." >&2
    echo "       Run: bash setup_env.sh" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$TRACE_ENV_NAME"
  echo "[run] activated conda env: $TRACE_ENV_NAME"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="./models/LLaDA-8B-Instruct"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_5000"
DATA_CACHE_PATH="./data_files"
out_dir="./outputs_LLaDA-8b-LoRA_5000_e5_order2/lora_8b"
PRED_OUTPUT="${out_dir}/predictions"

# 任务顺序 (order2)
TASK_ORDER="NumGLUE-cm,NumGLUE-ds,FOMC_shuffled,20Minuten,C-STANCE,Py150,MeetingBank,ScienceQA"

cl_method="lora"
port=$(shuf -i25000-30000 -n1)

mkdir -p "$out_dir" "$PRED_OUTPUT"
rm -rf "$DATA_CACHE_PATH"/*
mkdir -p "$DATA_CACHE_PATH"

# ===== Model download (idempotent, first-time only) =====
MODEL_REPO_ID="GSAI-ML/LLaDA-8B-Instruct"
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[download] $MODEL_REPO_ID -> $MODEL_PATH"
  mkdir -p "$(dirname "$MODEL_PATH")"
  huggingface-cli download "$MODEL_REPO_ID" --local-dir "$MODEL_PATH" || {
    echo ""
    echo "[ERROR] Download failed for $MODEL_REPO_ID."
    echo "  Gated repo (e.g. Llama-3) requires:"
    echo "    1. Accept license at https://huggingface.co/$MODEL_REPO_ID"
    echo "    2. Run: huggingface-cli login (paste your HF token)"
    exit 1
  }
fi

# =============================================================================
# Phase 0/3: BASE INFERENCE (no LoRA; idempotent; order2 专属)
#   ScienceQA 在新顺序里位置 idx=7
# =============================================================================
BASE_OUT_DIR="./outputs_LLaDA-8b_5000_base_order2/base_metrics"
if [ -f "$BASE_OUT_DIR/results-0-7-ScienceQA.json" ]; then
  echo "[llada_5000_e5_order2] Phase 0 SKIP: base baseline already at $BASE_OUT_DIR/"
else
  mkdir -p "$BASE_OUT_DIR"
  echo "=========================================================================="
  echo "[llada_5000_e5_order2] Phase 0/3: BASE INFERENCE (no LoRA, for FWT baseline)"
  echo "=========================================================================="

  export OMP_NUM_THREADS=4
  export TOKENIZERS_PARALLELISM=false

  python inference/infer_single.py \
      --data_path "$DATA_PATH" \
      --data_output_path "$DATA_CACHE_PATH" \
      --inference_tasks "$TASK_ORDER" \
      --model_name_or_path "$MODEL_PATH" \
      --inference_model_path "$MODEL_PATH" \
      --model_type diffusion \
      --sampling_steps 0 \
      --sampling_temperature 0.0 \
      --remasking_strategy low_confidence \
      --inference_batch 64 \
      --max_prompt_len 1024 \
      --max_ans_len 512 \
      --seed 1234 \
      --CL_method base \
      --inference_output_path "$BASE_OUT_DIR" \
      --target_round 0 \
      2>&1 | tee "$BASE_OUT_DIR/infer_base.log"
fi

echo "=========================================================================="
echo "[run_llada_5000_e5_order2] Phase 1/2: TRAINING (LLaDA-8B, 5 epoch x 8 tasks, order2)"
echo "=========================================================================="

deepspeed --num_gpus=1 --master_port "$port" training/main.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_CACHE_PATH" \
  --dataset_name "$TASK_ORDER" \
  --model_name_or_path "$MODEL_PATH" \
  --per_device_train_batch_size 8 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --num_train_epochs 5,5,5,5,5,5,5,5 \
  --gradient_accumulation_steps 16 \
  --seed 1234 \
  --zero_stage 2 \
  --bf16 \
  --deepspeed \
  --print_loss \
  --CL_method "$cl_method" \
  --output_dir "$out_dir" \
  --model_type diffusion \
  --diffusion_canvas_group_size 4 \
  2>&1 | tee "$out_dir/train_full.log"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=========================================================================="
echo "[run_llada_5000_e5_order2] Phase 2/2: INFERENCE (test split, all rounds)"
echo "=========================================================================="

python inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "$DATA_CACHE_PATH" \
    --inference_tasks "$TASK_ORDER" \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$out_dir" \
    --model_type diffusion \
    --sampling_steps 0 \
    --sampling_temperature 0.0 \
    --remasking_strategy low_confidence \
    --inference_batch 64 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method lora \
    --inference_output_path "$PRED_OUTPUT" \
    --all_rounds \
    2>&1 | tee "$PRED_OUTPUT/infer_llada.log"

echo ""
echo "[run_llada_5000_e5_order2] All done. Predictions -> $PRED_OUTPUT"
