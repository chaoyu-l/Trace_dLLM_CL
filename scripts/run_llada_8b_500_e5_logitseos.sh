#!/usr/bin/env bash
# =============================================================================
# run_llada_8b_500_e5_logitseos.sh
#   LLaDA-8B-Instruct 一站式 pipeline (500 数据集 + logits_eos_inf 推理):
#     Phase 1: 训练 (LoRA, 8 任务, 每任务 5 epoch, dynamic canvas group=8)
#     Phase 2: 推理 (test 集完整, low_confidence remasking, sampling_steps=0,
#              开启 logits_eos_inf=True 抑制 EOS)
#
#   与 run_llada_8b_5000_e5.sh 的区别:
#     - 数据集: LLM-CL-Benchmark_500 (相比 5000)
#     - 任务名: FOMC (相比 FOMC_shuffled)
#     - 训练 per_device_batch=8, grad_accum=1, canvas_group=8 (相比 32/4/4)
#     - Phase 2 推理打开 --logits_eos_inf
#     - 不跑 Phase 0 base baseline
#
#   使用流程 (远端机器, 两个 bash 脚本):
#     1) bash setup_env.sh
#        - 创建 conda env trace_env, 装好 PyTorch / deepspeed / peft 等依赖
#        - gdown 下载 TRACE benchmark 到 ./data/TRACE-Benchmark/ (含 500/1000/5000)
#        - 生成 FOMC_shuffled (本脚本不用, 用 FOMC 原 split)
#     2) bash scripts/run_llada_8b_500_e5_logitseos.sh
#        - huggingface-cli download GSAI-ML/LLaDA-8B-Instruct -> ./models/ (~16GB, idempotent)
#        - Phase 1 训练, Phase 2 推理 (logits_eos_inf=True), all in one go.
#
#   前置条件 (setup_env.sh 不负责):
#     - 远端机器已装好 conda, 且 conda 在 PATH 里
#     - 远端能访问 huggingface.co (LLaDA-8B-Instruct 是 public 不需要 license)
#     - GPU 显存够 (LLaDA-8B + LoRA 训练 per_device=8, 推理 batch=128, 建议 A100 80GB)
#
#   注意: LLaDA 不能加 --gradient_checkpointing (架构限制, 见 CLAUDE memory)
# =============================================================================
set -eo pipefail

# ===== Conda env auto-activate (与 5000_e5 脚本一致) =====
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
# 远端机器上, setup_env.sh 会把 TRACE 数据下到 ./data/TRACE-Benchmark/, 模型则在
# 本脚本里通过 huggingface-cli 自动下载到 ./models/ (idempotent skip).
MODEL_PATH="./models/LLaDA-8B-Instruct"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_500"
DATA_CACHE_PATH="./data_files"
out_dir="./outputs_LLaDA-8b-LoRA_500_e5_logitseos/lora_8b"
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

# ===== Model download (idempotent, first-time only; 与 run_llada_8b_5000_e5.sh 一致) =====
MODEL_REPO_ID="GSAI-ML/LLaDA-8B-Instruct"
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
# Phase 1/2: TRAINING (LLaDA-8B + LoRA, 5 epoch x 8 tasks, 500 数据集)
# =============================================================================
echo "=========================================================================="
echo "[run_llada_500_e5_logitseos] Phase 1/2: TRAINING"
echo "=========================================================================="

deepspeed --num_gpus=1 --master_port "$port" training/main.py \
  --data_path "$DATA_PATH" \
  --data_output_path "$DATA_CACHE_PATH" \
  --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
  --model_name_or_path "$MODEL_PATH" \
  --per_device_train_batch_size 8 \
  --max_prompt_len 1024 \
  --max_ans_len 512 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --num_train_epochs 5,5,5,5,5,5,5,5 \
  --gradient_accumulation_steps 1 \
  --seed 1234 \
  --zero_stage 2 \
  --bf16 \
  --deepspeed \
  --print_loss \
  --CL_method "$cl_method" \
  --output_dir "$out_dir" \
  --model_type diffusion \
  --diffusion_canvas_group_size 8 \
  2>&1 | tee "$out_dir/train_full.log"

# =============================================================================
# Phase 2/2: INFERENCE (test split, all rounds, logits_eos_inf=True)
# =============================================================================
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo ""
echo "=========================================================================="
echo "[run_llada_500_e5_logitseos] Phase 2/2: INFERENCE (logits_eos_inf=True)"
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
    --remasking_strategy low_confidence \
    --inference_batch 256 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method lora \
    --inference_output_path "$PRED_OUTPUT" \
    --all_rounds \
    --logits_eos_inf \
    2>&1 | tee "$PRED_OUTPUT/infer_llada.log"

echo ""
echo "[run_llada_500_e5_logitseos] All done. Predictions -> $PRED_OUTPUT"
