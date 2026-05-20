#!/usr/bin/env bash
# =============================================================================
# 预装 trace_env 到所有 8 个 sof1-h200 节点 (并行 sbatch, 不申请 GPU).
# 装完后再跑 bash submit_all.sh, 12 个真实 job 启动时 trace.sbatch 的
# trace_env_healthy() 直接通过, 不需要现场装.
#
# 用法:
#   bash prep_envs.sh         # 8 个节点并行装
#   bash prep_envs.sh 0 1 2   # 只装指定 index 的节点 (sof1-h200-0/1/2)
# =============================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p slurm_logs

if [ $# -gt 0 ]; then
    NODES=("$@")
else
    # 默认 5 个可用节点; sof1-h200-{0,1,2} 是 MAINTENANCE+RESERVED 状态
    # (其他用户长期占用), 我们的 sbatch 也只会落到 3-7 这 5 个节点上.
    NODES=(3 4 5 6 7)
fi

echo "[prep_envs] 在 ${#NODES[@]} 个节点上并行装 trace_env: ${NODES[*]/#/sof1-h200-}"

for n in "${NODES[@]}"; do
    NODE="sof1-h200-$n"
    jobid=$(sbatch --parsable \
        --job-name="prep_${NODE}" \
        --nodelist="$NODE" \
        --time=30:00 \
        --cpus-per-task=4 \
        --mem=16G \
        --partition=non-reserved \
        --qos=neurips-2026 \
        --output="slurm_logs/prep_${NODE}_%j.log" \
        --error="slurm_logs/prep_${NODE}_%j.error" \
        --wrap="bash $PWD/prep_node.sh")
    echo "  [submit] prep_${NODE}  jobid=$jobid"
done

echo ""
echo "[prep_envs] 共提交 ${#NODES[@]} 个 prep job (无 GPU, 4 CPU, 16G RAM, 30min)"
echo "[prep_envs] 监控:        squeue -u \$USER --name=prep_sof1-h200-0,prep_sof1-h200-1,..."
echo "[prep_envs] 等全部完成:  while squeue -u \$USER --noheader -h --name=\$(...) | grep -q .; do sleep 30; done"
echo "[prep_envs] 装完后:      bash submit_all.sh   # 提 12 个真实 job"
