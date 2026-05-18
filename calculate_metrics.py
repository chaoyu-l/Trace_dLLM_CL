import os
import json
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# ================= 配置区域 =================
# 任务顺序 (必须与训练顺序一致)
TASK_ORDER_TRACE = [
    "C-STANCE", "FOMC", "MeetingBank", "Py150",
    "ScienceQA", "NumGLUE-cm", "NumGLUE-ds", "20Minuten"
]

# SSR (ACL 2024) SuperNI 10 任务顺序（与 training/params.py::AllDatasetName_SSR 对齐）
TASK_ORDER_SSR = [
    "qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"
]

# 向后兼容：未显式指定 --benchmark 时默认用 TRACE
TASK_ORDER = TASK_ORDER_TRACE

# 定义哪些任务原本是 0-1 小数，需要乘以 100 转为百分制（仅 TRACE）
SCALE_TO_100_TASKS = {
    "C-STANCE", "FOMC", "MeetingBank", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"
}

# SSR 分类任务（sa/pos）可在 rouge-l / accuracy 之间切换，由 --sa_pos_metric 控制
SSR_CLASSIFICATION_TASKS = {"sa", "pos"}


# ===========================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Path to the CL predictions folder (e.g. lora_7b/predictions)")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="[Optional] Path to the Base Model predictions folder")
    parser.add_argument("--output_txt", type=str, default=None,
                        help="[Optional] Append metrics report to this txt file path")
    parser.add_argument("--benchmark", type=str, default="trace", choices=["trace", "ssr"],
                        help="Which task suite: trace (8 tasks) or ssr (10 SuperNI tasks)")
    parser.add_argument("--sa_pos_metric", type=str, default="rouge-l",
                        choices=["rouge-l", "accuracy"],
                        help="[SSR only] metric to use for sa/pos classification tasks. "
                             "SSR paper uses rouge-l for all 10 tasks; accuracy is a supplementary view.")
    return parser.parse_args()


def extract_numeric_score(value):
    """递归提取数值，处理 list/dict 嵌套"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return extract_numeric_score(value[0]) if len(value) > 0 else 0.0
    if isinstance(value, dict):
        # 优先查找常用指标键
        for key in ["sari", "rouge-l", "accuracy", "similarity", "f1", "score"]:
            if key in value:
                return float(value[key])
        if value: return extract_numeric_score(list(value.values())[0])
    return 0.0


def get_metric_value_trace(filepath, task_name):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            eval_res = data.get('eval', {})

            if not eval_res:
                return np.nan

            # 针对特定任务的特定字段提取 (与 TRACE 逻辑保持一致)
            raw_score = 0
            if "C-STANCE" in task_name:
                raw_score = eval_res.get("accuracy", 0)
            elif "FOMC" in task_name:
                raw_score = eval_res.get("accuracy", 0)
            elif "MeetingBank" in task_name:
                raw_score = eval_res.get("rouge-L", eval_res.get("rouge-l", 0))
            elif "Py150" in task_name:
                raw_score = eval_res.get("similarity", 0)
            elif "ScienceQA" in task_name:
                raw_score = eval_res.get("accuracy", 0)
            elif "NumGLUE" in task_name:
                raw_score = eval_res.get("accuracy", 0)
            elif "20Minuten" in task_name:
                raw_score = eval_res.get("sari", 0)
            else:
                if eval_res: raw_score = list(eval_res.values())[0]

            score = extract_numeric_score(raw_score)

            # 统一转为百分制
            if task_name in SCALE_TO_100_TASKS:
                if score <= 1.0:  # 双重保险：只有当分数确实小于等于1时才乘
                    score *= 100

            return score

    except Exception as e:
        # print(f"Error reading {filepath}: {e}")
        return np.nan


def get_metric_value_ssr(filepath, task_name, sa_pos_metric="rouge-l"):
    """SSR 统一用 ROUGE-L（eval_ssr.py 已 ×100）；sa/pos 可切到 accuracy。
    eval_ssr.eval 的 rouge-l 已在 [0, 100] 区间；accuracy 在 [0, 1]，需 ×100。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            eval_res = data.get('eval', {})

            if not eval_res:
                return np.nan

            if task_name in SSR_CLASSIFICATION_TASKS and sa_pos_metric == "accuracy":
                raw_score = eval_res.get("accuracy", 0.0)
                score = extract_numeric_score(raw_score)
                if score <= 1.0:
                    score *= 100
                return score

            raw_score = eval_res.get("rouge-l", eval_res.get("rouge-L", 0.0))
            return extract_numeric_score(raw_score)

    except Exception as e:
        return np.nan


def get_metric_value(filepath, task_name, benchmark="trace", sa_pos_metric="rouge-l"):
    if benchmark == "ssr":
        return get_metric_value_ssr(filepath, task_name, sa_pos_metric=sa_pos_metric)
    return get_metric_value_trace(filepath, task_name)


def main():
    args = parse_args()
    pred_dir = args.pred_dir
    base_dir = args.base_dir
    output_txt = args.output_txt
    benchmark = args.benchmark
    sa_pos_metric = args.sa_pos_metric

    task_order = TASK_ORDER_SSR if benchmark == "ssr" else TASK_ORDER_TRACE
    # 向后兼容：让后续代码里裸用的 TASK_ORDER 也跟随 benchmark
    globals()["TASK_ORDER"] = task_order
    n_tasks = len(task_order)
    print(f"[INFO] benchmark={benchmark}  tasks={task_order}"
          + (f"  sa_pos_metric={sa_pos_metric}" if benchmark == "ssr" else ""))

    # 1. 读取 CL 过程矩阵 R
    # R[i, j] 表示: 训练完第 i 个任务后，在第 j 个任务上的分数
    R = np.full((n_tasks, n_tasks), np.nan)

    if os.path.exists(pred_dir):
        print(f"正在读取 CL 结果... {pred_dir}")
        for train_round in range(n_tasks):
            for test_task_id in range(n_tasks):
                task_name = TASK_ORDER[test_task_id]
                filename = f"results-{train_round}-{test_task_id}-{task_name}.json"
                filepath = os.path.join(pred_dir, filename)
                if os.path.exists(filepath):
                    R[train_round, test_task_id] = get_metric_value(filepath, task_name, benchmark=benchmark, sa_pos_metric=sa_pos_metric)
    else:
        print(f"[Warning] CL Predictions directory not found: {pred_dir}")

    # 2. 读取 Base Model 结果 (如果有)
    # Base 结果通常看作 Round -1，但我们脚本里生成的是 results-0-xxx (因为target_round=0)
    R_base = np.full(n_tasks, np.nan)
    has_base = False

    if base_dir and os.path.exists(base_dir):
        print(f"正在读取 Base 结果... {base_dir}")
        for test_task_id in range(n_tasks):
            task_name = TASK_ORDER[test_task_id]
            # 注意：run_infer_base.sh 生成的文件也是 results-0-...
            filename = f"results-0-{test_task_id}-{task_name}.json"
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                R_base[test_task_id] = get_metric_value(filepath, task_name, benchmark=benchmark, sa_pos_metric=sa_pos_metric)

        # 检查是否读到了有效数据
        if not np.all(np.isnan(R_base)):
            has_base = True

    # ================= 展示合并表格 =================
    df_R = pd.DataFrame(R, index=[f"Train {t}" for t in TASK_ORDER], columns=TASK_ORDER)

    if has_base:
        df_base = pd.DataFrame([R_base], index=["Base Model"], columns=TASK_ORDER)
        # 将 Base 行拼接到最上方
        df_all = pd.concat([df_base, df_R])
    else:
        df_all = df_R

    output_lines = []
    output_lines.append("\n" + "=" * 60)
    output_lines.append(" >>> 综合性能表 (Performance Matrix %) <<< ")
    output_lines.append("=" * 60)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.precision', 2)
    output_lines.append(df_all.to_string())
    output_lines.append("=" * 60)

    # ================= 计算论文指标 =================
    output_lines.append("\n[Metrics Report]")

    # 1. OP (Overall Performance)
    op = np.nanmean(R[-1, :])
    output_lines.append(f"1. OP (Overall Performance):  {op:.2f}")

    # 2. BWT (Backward Transfer)
    # 严格 TRACE 论文口径: 分母为 T (n_tasks)
    # 通用 CL 口径: 分母为 T-1
    bwt_sum = 0.0
    valid_bwt_count = 0
    for i in range(n_tasks - 1):
        # R[final, i] - R[i, i]
        peak = R[i, i]
        final = R[-1, i]
        if not np.isnan(peak) and not np.isnan(final):
            bwt_sum += (final - peak)
            valid_bwt_count += 1

    # 这里采用通用口径 T-1，如果想对齐 TRACE 论文表格，可以改除以 n_tasks
    bwt = bwt_sum / (n_tasks - 1)
    output_lines.append(f"2. BWT (Backward Transfer):   {bwt:.2f}  (负值代表遗忘)")
    output_lines.append(f"   Avg Forgetting:            {-bwt:.2f}  (越低越好)")

    # 3. FWT (Forward Transfer)
    fwt_list = []

    # 场景 A: 有 Base Model -> 计算 GEM FWT (Zero-shot - Base)
    if has_base:
        for i in range(1, n_tasks):
            # 模型在学习 Task i 之前的表现 (即 R[i-1, i]) vs Base Model 在 Task i 的表现
            zs_perf = R[i - 1, i]
            base_perf = R_base[i]
            if not np.isnan(zs_perf) and not np.isnan(base_perf):
                fwt_list.append(zs_perf - base_perf)

        fwt = np.mean(fwt_list) if fwt_list else 0.0
        output_lines.append(f"3. FWT (vs Base Model):       {fwt:.2f}  (严谨口径: ZeroShot - Base)")

    # 场景 B: 无 Base Model -> 计算 Average Zero-shot
    else:
        for i in range(1, n_tasks):
            zs_perf = R[i - 1, i]
            if not np.isnan(zs_perf):
                fwt_list.append(zs_perf)

        fwt = np.mean(fwt_list) if fwt_list else 0.0
        output_lines.append(f"3. FWT (Raw Zero-Shot):       {fwt:.2f}  (简易口径: 仅 ZeroShot 均值)")

    output_text = "\n".join(output_lines)
    print(output_text)

    if output_txt:
        output_dir = os.path.dirname(output_txt)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_txt, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== Run @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            f.write(output_text)


if __name__ == "__main__":
    main()


"""

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_old/lora_8b/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_old/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/metrics.txt"

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_old/lora_8b/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_old/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/metrics.txt"

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_test/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/outputs_Llama-8b-LoRA_500_test/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型已经完成的实验/Trace_experiments/metrics.txt"

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_LLaDA-8b-LoRA_500/lora_8b/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_LLaDA-8b-LoRA_500/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/metrics.txt"

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_dream_7b_500/lora_7b/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_dream_7b_500/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/metrics.txt"

python calculate_metrics.py \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_LLaDA-8b-LoRA_500/lora_8b/predictions"


python calculate_metrics.py --benchmark ssr --pred_dir <path> \                                                                                                                                                                                                                                                   
  [--sa_pos_metric rouge-l|accuracy] \                                                                                                                                                                                                                                                                          
  [--base_dir <path>]   # 传 base_metrics 目录才能算 FWT                                                                                                                                                                                                                                                        
  [--output_txt <file>] # 可选，报告 tee 到文件   

python calculate_metrics.py \
    --benchmark ssr \
    --sa_pos_metric rouge-l \
    --pred_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_Llama-8b-LoRA_ssr_200/lora_8b/predictions" \
    --base_dir "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/outputs_Llama-8b-LoRA_ssr_200/base_metrics" \
    --output_txt "/media/chaoyu/BCF8E25947FDB178/打包实验/Trace_Results/metrics.txt"
"""

# windows
"""
python calculate_metrics.py --pred_dir "E:\扩散大语言模型已经完成的实验\Trace_experiments\outputs_Llama-8b-LoRA_500\lora_8b\predictions" --base_dir "E:\扩散大语言模型已经完成的实验\Trace_experiments\outputs_Llama-8b-LoRA_500\base_metrics" --output_txt "E:\扩散大语言模型已经完成的实验\Trace_experiments\metrics.txt"
"""

# llama结果
"""
python calculate_metrics.py --pred_dir "E:\打包实验\Trace_Results\outputs_Llama-8b-LoRA_500\lora_8b\predictions" --base_dir "E:\打包实验\Trace_Results\outputs_Llama-8b-LoRA_500\base_metrics" --output_txt "E:\打包实验\Trace_Results\metrics.txt"
"""

# qwen结果
"""
python calculate_metrics.py --pred_dir "E:\打包实验\Trace_Results\outputs_Qwen-7b-LoRA_500\lora_7b\predictions" --base_dir "E:\打包实验\Trace_Results\outputs_Qwen-7b-LoRA_500\base_metrics" --output_txt "E:\打包实验\Trace_Results\metrics.txt"
"""

# Dream结果
"""
python calculate_metrics.py --pred_dir "E:\打包实验\Trace_Results\outputs_dream_7b_500\lora_7b\predictions" --base_dir "E:\打包实验\Trace_Results\outputs_dream_7b_500\base_metrics" --output_txt "E:\打包实验\Trace_Results\metrics.txt"
"""

# Llada结果

"""
python calculate_metrics.py --pred_dir "E:\打包实验\Trace_Results\outputs_LLaDA-8b-LoRA_500\lora_8b\predictions" --base_dir "E:\打包实验\Trace_Results\outputs_LLaDA-8b-LoRA_500\base_metrics" --output_txt "E:\打包实验\Trace_Results\metrics.txt"
"""
