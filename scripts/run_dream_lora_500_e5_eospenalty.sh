#!/usr/bin/env bash
# =============================================================================
# run_dream_lora_500_e5_eospenalty.sh
#   Dream-7B + LoRA 一站式 pipeline (500 数据集 + Dream-Coder padding penalty):
#     Phase 1: 训练 (LoRA, 8 任务, 每任务 5 epoch, dynamic canvas group=8)
#     Phase 2: 推理 (test 集完整, --eos_penalty 3.0 抑制 PAD/EOS 早终止)
#
#   Dream-Coder padding penalty (arXiv:2509.01142 §4.1):
#     logits[..., pad_token_id] += eos_penalty * log(1 - t + eps)
#     起始 (t=1) 强负偏置压 PAD, 末尾 (t=eps) 退到 0 允许自然终止.
#     3.0 是 Dream-Coder README 推荐值.
#
#   与原 outputs_dream_7b_500 跑过的实验配置一致, 只多了 --eos_penalty.
#
#   使用流程 (远端机器):
#     1) bash setup_env.sh
#     2) bash scripts/run_dream_lora_500_e5_eospenalty.sh
#
#   注意:
#     - Dream 可以加 --gradient_checkpointing (LLaDA 不行, 二者架构差异)
#     - HF 模型 repo Dream-org/Dream-v0-Instruct-7B 是 public, 不需要 license
# =============================================================================
set -eo pipefail

# ===== Conda env auto-activate =====
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

# ===== 路径配置 =====
MODEL_PATH="./models/Dream-7b"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_500"
DATA_CACHE_PATH="./data_files"
out_dir="./outputs_dream_7b_500_e5_eospenalty/lora_7b"
PRED_OUTPUT="${out_dir}/predictions"

cl_method="lora"
port=$(shuf -i25000-30000 -n1)

mkdir -p "$out_dir" "$PRED_OUTPUT"
rm -rf "$DATA_CACHE_PATH"/*
mkdir -p "$DATA_CACHE_PATH"

if [ ! -d "$DATA_PATH" ]; then
  echo "[Error] DATA_PATH not found: $DATA_PATH"
  echo "        请先在 repo 根目录跑 bash setup_env.sh 拉取 TRACE benchmark."
  exit 1
fi

# ===== Model download (idempotent, first-time only) =====
MODEL_REPO_ID="Dream-org/Dream-v0-Instruct-7B"
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[download] $MODEL_REPO_ID -> $MODEL_PATH"
  mkdir -p "$(dirname "$MODEL_PATH")"
  huggingface-cli download "$MODEL_REPO_ID" --local-dir "$MODEL_PATH" || {
    echo ""
    echo "[ERROR] Download failed for $MODEL_REPO_ID."
    echo "  请确认网络可访问 huggingface.co, 或先 huggingface-cli login."
    exit 1
  }
fi

# =============================================================================
# Phase 1/2: TRAINING (Dream-7B + LoRA, 5 epoch x 8 tasks, 500 数据集)
# =============================================================================
echo "=========================================================================="
echo "[run_dream_500_e5_eospenalty] Phase 1/2: TRAINING"
echo "=========================================================================="

deepspeed --num_gpus=1 --master_port "$port" training/main.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_CACHE_PATH" \
  --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path "$MODEL_PATH" \
  --per_device_train_batch_size 1 \
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
  --diffusion_canvas_group_size 8 \
  2>&1 | tee "$out_dir/train_full.log"

# =============================================================================
# Phase 2/2: INFERENCE (test split, all rounds, --eos_penalty 3.0)
# =============================================================================
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=========================================================================="
echo "[run_dream_500_e5_eospenalty] Phase 2/2: INFERENCE (--eos_penalty 3.0)"
echo "=========================================================================="

python inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "$DATA_CACHE_PATH" \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$out_dir" \
    --model_type diffusion \
    --sampling_steps 0 \
    --sampling_temperature 0.0 \
    --inference_batch 128 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method lora \
    --inference_output_path "$PRED_OUTPUT" \
    --all_rounds \
    --eos_penalty 3.0 \
    2>&1 | tee "$PRED_OUTPUT/infer_dream.log"

echo ""
echo "[run_dream_500_e5_eospenalty] All done. Predictions -> $PRED_OUTPUT"
