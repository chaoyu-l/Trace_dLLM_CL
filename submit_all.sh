#!/usr/bin/env bash
# =============================================================================
# 一次提交 12 个 TRACE 实验 (4 模型 × 3 epoch 配置).
# 每个 job 用独立 sandbox cwd (runs/<tag>/) → ./data_files 不再撞车.
#
# 提交顺序: e5 → e10 → e15.
#   e5 内嵌 Phase 0 (base baseline 推理), 是 e10/e15 算 FWT 的依赖;
#   但由于 sandbox 隔离, e10/e15 不会自动复用 e5 的 base 输出。
#   等所有 job 跑完, 跑 calculate_metrics.py 时手动用
#       --base_dir runs/<model>_5000_e5/outputs_<model>_5000_base/base_metrics
#   指即可。
#
# 时间预算 (--time):
#   按实测 RUNNING data + 加 cache + resume 后的合理估算 (含 2x buffer):
#     AR (llama/qwen):    e5=24h   e10=36h   e20=48h    (实际 ~13/22/30h)
#     扩散 (dream/llada):  e5=48h   e10=48h   e20=72h    (实际 ~32/32/40h, Phase 2 with cache 仍 ~30h)
#   不主动 exclude reboot 节点 sof1-h200-3/5: 让 SLURM fair-share 自由调度,
#   万一被 reboot 砍, resume 重提自动接上 (Phase 1 已 saved adapter / Phase 2 已写 cells skip).
#   实际 deadline 5.5 天 = 132h, 单 job 最长 72h 仍有 60h buffer 给排队/重试.
#
# 用法:
#   bash submit_all.sh                       # 全部 12 个
#   bash submit_all.sh llama_8b_5000_e5      # 只提一个 (substring INCLUDE)
#   bash submit_all.sh e5                    # 只提 4 个 e5
#   EXCLUDE=llama bash submit_all.sh         # 排除 llama (跑 9 个; 等 license 通过再补)
#   EXCLUDE=llama bash submit_all.sh e5      # 4 个 e5 排除 llama → 3 个
#   DRY_RUN=1 bash submit_all.sh             # 只打印会提交的命令, 不真提交
# =============================================================================
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
mkdir -p slurm_logs

FILTER="${1:-}"
EXCLUDE="${EXCLUDE:-}"
DRY_RUN="${DRY_RUN:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"

# tag                          time         script-name
# 第 4 列: 额外 sbatch 参数 (e.g. --exclude=...). 留空就是用 trace.sbatch 默认.
EXPERIMENTS=(
    # === Cached (Fast-dLLM cache 启用; 独立 scripts 文件 *_cached.sh) ===
    "llada_8b_5000_e5_cached         120:00:00    run_llada_8b_5000_e5_cached.sh"
    "dream_lora_5000_e5_cached       120:00:00    run_dream_lora_5000_e5_cached.sh"
    "llada_8b_5000_e10_cached        120:00:00    run_llada_8b_5000_e10_cached.sh"
    "dream_lora_5000_e10_cached      120:00:00    run_dream_lora_5000_e10_cached.sh"
    "llada_8b_5000_e20_cached        120:00:00    run_llada_8b_5000_e20_cached.sh"
    "dream_lora_5000_e20_cached      120:00:00    run_dream_lora_5000_e20_cached.sh"
    # === AR (llama/qwen, 没 cache 概念) ===
    "llama_8b_5000_e5         120:00:00    run_llama_8b_5000_e5.sh"
    "qwen_7b_5000_e5          120:00:00    run_qwen_7b_5000_e5.sh"
    "llama_8b_5000_e10        120:00:00    run_llama_8b_5000_e10.sh"
    "qwen_7b_5000_e10         120:00:00    run_qwen_7b_5000_e10.sh"
    "llama_8b_5000_e20        120:00:00    run_llama_8b_5000_e20.sh"
    "qwen_7b_5000_e20         120:00:00    run_qwen_7b_5000_e20.sh"
    # === 扩散 no cache (严格 baseline; 原 scripts) ===
    "llada_8b_5000_e5         120:00:00    run_llada_8b_5000_e5.sh"
    "dream_lora_5000_e5       120:00:00    run_dream_lora_5000_e5.sh"
    "llada_8b_5000_e10        120:00:00    run_llada_8b_5000_e10.sh"
    "dream_lora_5000_e10      120:00:00    run_dream_lora_5000_e10.sh"
    "llada_8b_5000_e20        120:00:00    run_llada_8b_5000_e20.sh"
    "dream_lora_5000_e20      120:00:00    run_dream_lora_5000_e20.sh"
    # === e5_order2 (不同 task 顺序的 e5 消融, 4 backbone) ===
    "llama_8b_5000_e5_order2         120:00:00    run_llama_8b_5000_e5_order2.sh"
    "qwen_7b_5000_e5_order2          120:00:00    run_qwen_7b_5000_e5_order2.sh"
    "llada_8b_5000_e5_order2         120:00:00    run_llada_8b_5000_e5_order2.sh"
    "dream_lora_5000_e5_order2       120:00:00    run_dream_lora_5000_e5_order2.sh"
)

echo "[submit_all] 候选 ${#EXPERIMENTS[@]} 个; INCLUDE='${FILTER:-(any)}' EXCLUDE='${EXCLUDE:-(none)}'"
submitted=0
for row in "${EXPERIMENTS[@]}"; do
    read -r tag time script_name extra <<< "$row"
    if [ -n "$FILTER" ] && ! echo "$tag" | grep -qE "$FILTER"; then
        continue
    fi
    if [ -n "$EXCLUDE" ] && echo "$tag" | grep -qE "$EXCLUDE"; then
        echo "  [excl] $tag — matched EXCLUDE='$EXCLUDE'"
        continue
    fi
    script_path="scripts/$script_name"
    if [ ! -f "$script_path" ]; then
        echo "  [skip] $tag — $script_path 不存在"
        continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry-run] sbatch --job-name=$tag --time=$time $extra --export=ALL,TAG=$tag,SCRIPT=$script_path  trace.sbatch"
    else
        # shellcheck disable=SC2086
        jobid=$(sbatch --parsable \
            --job-name="${tag}${TAG_SUFFIX}" \
            --time="$time" \
            $extra \
            --export=ALL,TAG="${tag}${TAG_SUFFIX}",SCRIPT="$script_path" \
            trace.sbatch)
        printf "  [submit] %-30s  time=%s  jobid=%s%s\n" "$tag" "$time" "$jobid" "${extra:+  extra=$extra}"
    fi
    submitted=$((submitted + 1))
done

echo ""
echo "[submit_all] 共提交 $submitted 个 job"
echo "[submit_all] 监控:  squeue -u \$USER"
echo "[submit_all] 日志:  slurm_logs/<tag>_<jobid>.log  (实时进度)"
echo "[submit_all] 单跑日志: runs/<tag>/{outputs_*/train_full.log,base_metrics/infer_base.log,predictions/infer_*.log}"
