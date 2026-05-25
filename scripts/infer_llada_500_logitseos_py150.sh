#!/usr/bin/env bash
# =============================================================================
# infer_llada_500_logitseos_py150.sh
#   LLaDA-8B + LoRA(500 数据集训练权重) 推理 Py150。
#   - 非 cache 推理路径 (走 llada_generate, 不开 use_fast_cache / use_dual_cache)
#   - 开启 logits_eos_inf=True (官方 EOS 抑制行为, 强制 canvas 填内容 token)
#   - 只跑 round 0/1/2 三轮 LoRA 适配器
#   - inference_batch = 4
#   - 只推 Py150 这一个任务
# =============================================================================
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

# ====== 路径配置 ==============================================================
BASE_MODEL="/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/llada-8b-instruct"
DATA_PATH="/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_dLLM_CL/data/TRACE-Benchmark/LLM-CL-Benchmark_500"
LORA_PATH="./outputs_LLaDA-8b-LoRA_500/lora_8b"
DATA_CACHE_PATH="./data_files"

PRED_OUTPUT="${LORA_PATH}/predictions_logitseos_py150"
mkdir -p "$PRED_OUTPUT" "$DATA_CACHE_PATH"

if [ ! -d "$BASE_MODEL" ]; then echo "[Error] BASE_MODEL not found: $BASE_MODEL"; exit 1; fi
if [ ! -d "$DATA_PATH" ];  then echo "[Error] DATA_PATH not found: $DATA_PATH";   exit 1; fi
if [ ! -d "$LORA_PATH" ];  then echo "[Error] LORA_PATH not found: $LORA_PATH";   exit 1; fi

LOG_FILE="$PRED_OUTPUT/infer_llada.log"

echo "[infer_llada_500_logitseos_py150]"
echo "  Base Model : $BASE_MODEL"
echo "  Data       : $DATA_PATH"
echo "  LoRA Path  : $LORA_PATH"
echo "  Output     : $PRED_OUTPUT"
echo "  Log        : $LOG_FILE"
echo "==========================================================" | tee "$LOG_FILE"

# ====== 三轮 LoRA 适配器 (rounds 0,1,2) =======================================
# infer_single.py 在 --target_round 单轮模式下会全新加载 base+LoRA, 不会污染权重,
# 所以 bash 里循环 3 次是最稳的做法 (相比加 --rounds 子集功能, 改动小).
for ROUND in 0 1 2; do
    echo "" | tee -a "$LOG_FILE"
    echo ">>> Round ${ROUND} (logits_eos_inf=True, py150 only) <<<" | tee -a "$LOG_FILE"
    echo "==========================================================" | tee -a "$LOG_FILE"

    python inference/infer_single.py \
        --data_path "$DATA_PATH" \
        --data_output_path "$DATA_CACHE_PATH" \
        --inference_tasks Py150 \
        --model_name_or_path "$BASE_MODEL" \
        --inference_model_path "$LORA_PATH" \
        --model_type diffusion \
        --sampling_steps 0 \
        --sampling_temperature 0.0 \
        --remasking_strategy low_confidence \
        --inference_batch 4 \
        --max_prompt_len 1024 \
        --max_ans_len 512 \
        --seed 1234 \
        --CL_method lora \
        --inference_output_path "$PRED_OUTPUT" \
        --target_round "${ROUND}" \
        --logits_eos_inf \
        2>&1 | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "[infer_llada_500_logitseos_py150] Rounds 0/1/2 finished. Output -> $PRED_OUTPUT" | tee -a "$LOG_FILE"
