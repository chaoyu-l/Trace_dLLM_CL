#!/usr/bin/env bash
# =============================================================================
# test_dream_7b_500_e5_eospenalty.sh
#   Dream-7B base-only inference, paired with run_dream_lora_500_e5_eospenalty.sh.
#   仅跑 Phase 0: BASE INFERENCE (no LoRA), 提供 FWT baseline.
#   推理 flag 与 run_dream_lora_500_e5_eospenalty.sh Phase 2 完全一致 (含
#   --eos_penalty 3.0), 这样 base 行和 rounds 1..8 行在同一推理 regime 下可比.
#
#   不再训练: 使用 ./models/Dream-7b/ 已下载好的 base 权重.
#
#   输出: ./outputs_dream_7b_500_base_eospenalty/base_metrics/
#         calculate_metrics.py --base_dir 可直接指向这里, 与 LoRA 那组
#         outputs_dream_7b_500_e5_eospenalty/lora_7b/predictions/ 配套.
# =============================================================================
set -eo pipefail

# ===== Conda env auto-activate =====
TRACE_ENV_NAME="${TRACE_ENV_NAME:-trace_env}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$TRACE_ENV_NAME" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[test_dream_eospenalty] ERROR: conda not in PATH. Please source your conda profile first." >&2
    exit 1
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$TRACE_ENV_NAME"; then
    echo "[test_dream_eospenalty] ERROR: conda env '$TRACE_ENV_NAME' missing." >&2
    echo "       Run: bash setup_env.sh" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$TRACE_ENV_NAME"
  echo "[test_dream_eospenalty] activated conda env: $TRACE_ENV_NAME"
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 路径配置 (与 run_dream_lora_500_e5_eospenalty.sh 一致) =====
MODEL_PATH="./models/Dream-7b"
DATA_PATH="./data/TRACE-Benchmark/LLM-CL-Benchmark_500"
DATA_CACHE_PATH="./data_files"
BASE_OUT_DIR="./outputs_dream_7b_500_base_eospenalty/base_metrics"

mkdir -p "$BASE_OUT_DIR" "$DATA_CACHE_PATH"

if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[test_dream_eospenalty] ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
  echo "       请先跑 bash scripts/run_dream_lora_500_e5_eospenalty.sh, 或" >&2
  echo "       huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir $MODEL_PATH" >&2
  exit 1
fi

if [ ! -d "$DATA_PATH" ]; then
  echo "[test_dream_eospenalty] ERROR: DATA_PATH not found: $DATA_PATH" >&2
  echo "        请先在 repo 根目录跑 bash setup_env.sh 拉取 TRACE benchmark." >&2
  exit 1
fi

# ===== Idempotent skip =====
if [ -f "$BASE_OUT_DIR/results-0-7-20Minuten.json" ]; then
  echo "[test_dream_eospenalty] SKIP: base baseline already at $BASE_OUT_DIR/"
  exit 0
fi

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

echo "=========================================================================="
echo "[test_dream_eospenalty] Phase 0 only: BASE INFERENCE (no LoRA, +eos_penalty 3.0)"
echo "  MODEL_PATH : $MODEL_PATH"
echo "  DATA_PATH  : $DATA_PATH"
echo "  OUTPUT     : $BASE_OUT_DIR"
echo "=========================================================================="

# 推理参数与 run_dream_lora_500_e5_eospenalty.sh Phase 2 一致, 仅以下三处不同:
#   --inference_model_path  -> $MODEL_PATH (base, 无 LoRA)
#   --CL_method             -> base        (vs lora)
#   --target_round 0        (vs --all_rounds)
python inference/infer_single.py \
    --data_path "$DATA_PATH" \
    --data_output_path "$DATA_CACHE_PATH" \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path "$MODEL_PATH" \
    --inference_model_path "$MODEL_PATH" \
    --model_type diffusion \
    --sampling_steps 0 \
    --sampling_temperature 0.0 \
    --inference_batch 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --CL_method base \
    --inference_output_path "$BASE_OUT_DIR" \
    --target_round 0 \
    --eos_penalty 3.0 \
    2>&1 | tee "$BASE_OUT_DIR/infer_base.log"

echo ""
echo "[test_dream_eospenalty] Done. Base predictions -> $BASE_OUT_DIR"
