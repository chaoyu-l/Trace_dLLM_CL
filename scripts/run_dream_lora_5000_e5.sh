#!/usr/bin/env bash
# =============================================================================
# run_dream_lora_5000_e5.sh
#   Dream-7B 一站式 pipeline (配置 A: 5 epoch, 无 in-training eval):
#     Phase 1: 训练 (LoRA, 8 任务, 每任务 5 epoch, fix_eos pad)
#     Phase 2: 推理 (test 集完整, all_rounds, maskgit_plus, sampling_steps=0)
# =============================================================================
set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====== 路径配置 (repo 根目录相对路径) ========================================
MODEL_PATH="./models/Dream-7b"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_5000"
DATA_CACHE_PATH="./data_files"
out_dir="./outputs_dream_7b_5000_e5/lora_7b"
PRED_OUTPUT="${out_dir}/predictions"

cl_method="lora"
port=$(shuf -i25000-30000 -n1)

mkdir -p "$out_dir" "$PRED_OUTPUT"
rm -rf "$DATA_CACHE_PATH"/*
mkdir -p "$DATA_CACHE_PATH"

# ===== Model download (idempotent, first-time only) =====
MODEL_REPO_ID="Dream-org/Dream-v0-Instruct-7B"
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
# Phase 1/2: TRAINING
# =============================================================================
echo "=========================================================================="
echo "[run_dream_5000_e5] Phase 1/2: TRAINING (Dream-7B, 5 epoch x 8 tasks, no eval)"
echo "=========================================================================="

deepspeed --num_gpus=1 --master_port "$port" training/main.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_CACHE_PATH" \
  --dataset_name C-STANCE,FOMC_shuffled,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path "$MODEL_PATH" \
  --per_device_train_batch_size 2 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --num_train_epochs 5,5,5,5,5,5,5,5 \
  --gradient_accumulation_steps 8 \
  --seed 1234 \
  --zero_stage 2 \
  --bf16 \
  --gradient_checkpointing \
  --deepspeed \
  --print_loss \
  --CL_method "$cl_method" \
  --output_dir "$out_dir" \
  --model_type diffusion \
  --fix_eos \
  2>&1 | tee "$out_dir/train_full.log"

# =============================================================================
# Phase 2/2: INFERENCE
# =============================================================================
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=========================================================================="
echo "[run_dream_5000_e5] Phase 2/2: INFERENCE (test split, all rounds)"
echo "=========================================================================="

python inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "$DATA_CACHE_PATH" \
    --inference_tasks C-STANCE,FOMC_shuffled,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$out_dir" \
    --model_type diffusion \
    --sampling_steps 0 \
    --sampling_temperature 0.0 \
    --dream_alg maskgit_plus \
    --inference_batch 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method lora \
    --inference_output_path "$PRED_OUTPUT" \
    --all_rounds \
    2>&1 | tee "$PRED_OUTPUT/infer_dream.log"

echo ""
echo "[run_dream_5000_e5] All done. Predictions -> $PRED_OUTPUT"
