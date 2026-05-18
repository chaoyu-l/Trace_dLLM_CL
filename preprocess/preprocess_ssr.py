"""
preprocess_ssr.py
=================
把 SuperNI 10 个任务的原始 JSON 处理成 TRACE 风格的 continual-learning 数据集。
严格复刻 SSR 官方 (https://github.com/DeepLearnXMU/SSR) 两步预处理逻辑：
  Step 1 (length_filter)  — 官方 custom/niv2-c012/1_length_fiiter.py
  Step 2 (split_sampling) — 官方 custom/niv2-c012/2_split_and_random_selection.py

用户决定（2026-04-18）：
  * 拼接: prompt = Definition + "\n\n" + input, answer = output[0]
  * 主档 (main): 每任务 train=2000 / eval=500 / test=500（test.json = eval.json 的拷贝）
  * mini 档 (pipeline 调试): train=200 / eval=100 / test=100
        → 嵌套子集：mini.train = main.train[:200]；mini.eval = main.eval[:100]
  * 输出格式: JSON array (和 TRACE 现有 train.json 一致)，每条仅保留 {prompt, answer}
  * 目录名用官方小写短名: qa/qg/sa/sum/trans/dsg/expl/para/pe/pos
  * 不生成 extra.json、不生成 smp01/smp005/smp001
  * sa / pos 为二分类，官方做 label-balanced 采样（pos=1000/neg=1000 for train 2000）

运行方式：
    python3 preprocess/preprocess_ssr.py --step 1        # 只跑长度过滤，写到 _filtered/
    python3 preprocess/preprocess_ssr.py --step 2        # 读 _filtered/ 做切分与输出
    python3 preprocess/preprocess_ssr.py --step all      # 两步都跑
"""

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


# --------------------------------------------------------------------------------------
# 路径 & 常量（全部用绝对路径以和 CLAUDE.md 风格一致，避免 cwd 不同导致找不到文件）
# --------------------------------------------------------------------------------------
SUPERNI_TASK_DIR = "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/SuperNI Dataset/natural-instructions/tasks"

LLAMA2_TOKENIZER_PATH = "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/llama-2-7b-chat-hf"
SIMCSE_TOKENIZER_PATH = "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/sup-simcse-roberta-base"

# 输出根：两档数据集各自一个目录，和 TRACE 现有 LLM-CL-Benchmark_500/5000 并列
OUT_ROOT = "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/TRACE-Benchmark/TRACE-Benchmark"
OUT_MAIN_DIR = os.path.join(OUT_ROOT, "LLM-CL-Benchmark_SSR")
OUT_MINI_DIR = os.path.join(OUT_ROOT, "LLM-CL-Benchmark_SSR_mini")

# Step 1 过滤后的中间文件（每任务一个 jsonl，保留官方字段便于审计）
FILTERED_DIR = "/media/chaoyu/BCF8E25947FDB178/扩散大语言模型实验/TRACE-Benchmark/TRACE-Benchmark/_ssr_filtered"

# 官方过滤阈值（from 1_length_fiiter.py，未做任何调整）
MAX_INSTRUCTION_INPUT_LEN = 500      # instruction + input token 数上限
MAX_INPUT_LEN = 800                  # input 单独上限
MAX_TARGET_LEN = 128                 # output 上限
SIMCSE_MAX_LEN = 510                 # SimCSE 编码长度上限（retrieval 约束）

# 切分规模
NUM_TRAIN_MAIN = 2000
NUM_EVAL_MAIN = 500
NUM_TRAIN_MINI = 200
NUM_EVAL_MINI = 100

# 10 个任务：(category_short_name, superNI_task_filename_stem, is_binary)
TASK_LIST: List[Tuple[str, str, bool]] = [
    ("qa",    "task024_cosmosqa_answer_generation",          False),
    ("qg",    "task074_squad1.1_question_generation",        False),
    ("sa",    "task1312_amazonreview_polarity_classification", True),  # binary: positive/negative
    ("sum",   "task511_reddit_tifu_long_text_summarization", False),
    ("trans", "task1219_ted_translation_en_es",              False),
    ("dsg",   "task574_air_dialogue_sentence_generation",    False),
    ("expl",  "task192_hotpotqa_sentence_generation",        False),
    ("para",  "task177_para-nmt_paraphrasing",               False),
    ("pe",    "task064_all_elements_except_first_i",         False),
    ("pos",   "task346_hybridqa_classification",             True),  # binary: True/False
]

POS_LABELS = ["positive", "true"]
NEG_LABELS = ["negative", "false"]


# --------------------------------------------------------------------------------------
# Step 1: 长度过滤
# --------------------------------------------------------------------------------------
def length_filter(raw_task_json: dict, task_name: str, tok_llama, tok_simcse) -> Tuple[list, dict]:
    """逐条过滤一个 SuperNI 任务文件，返回 (保留样本列表, 统计信息 dict)。
    严格照搬官方 1_length_fiiter.py 的规则，仅字段命名保持一致。"""
    definition = raw_task_json["Definition"][0]
    instruction_len = len(tok_llama.encode(definition))

    kept = []
    skipped = {
        "empty_input_or_output": 0,
        "tok_len_one": 0,
        "instr_plus_input_too_long": 0,
        "input_too_long_800": 0,
        "target_too_long_128": 0,
        "simcse_too_long_510": 0,
    }
    in_lens, out_lens = [], []
    in_lens_kept, out_lens_kept = [], []

    for line in tqdm(raw_task_json["Instances"], total=len(raw_task_json["Instances"]),
                     desc=f"filter[{task_name}]", leave=False):
        inp = line["input"].strip()
        # SuperNI 的 output 是 list，官方取第一个并 strip
        out = line["output"][0].strip() if line["output"] else ""
        if inp == "" or out == "":
            skipped["empty_input_or_output"] += 1
            continue

        in_len = len(tok_llama.encode(inp))
        out_len = len(tok_llama.encode(out))
        in_lens.append(in_len)
        out_lens.append(out_len)

        # 官方规则：input_len==1 或 target_len==1 视为异常
        if in_len == 1 or out_len == 1:
            skipped["tok_len_one"] += 1
            continue
        if in_len + instruction_len > MAX_INSTRUCTION_INPUT_LEN:
            skipped["instr_plus_input_too_long"] += 1
            continue
        if in_len > MAX_INPUT_LEN:
            skipped["input_too_long_800"] += 1
            continue
        if out_len > MAX_TARGET_LEN:
            skipped["target_too_long_128"] += 1
            continue
        # SimCSE 510 过滤（官方仅对 Definition + input 计算，不含 output）
        simcse_len = len(tok_simcse.encode(definition)) + len(tok_simcse.encode(inp))
        if simcse_len > SIMCSE_MAX_LEN:
            skipped["simcse_too_long_510"] += 1
            continue

        # 官方这里还顺手加了 full_prompt 和 task_name 两个字段，方便下游用
        kept.append({
            "id": line.get("id"),
            "input": inp,
            "output": out,                                    # 注意：已经从 list 取第一个并 strip 成 str
            "full_prompt": definition + "\n\n" + inp,         # 官方拼接方式
            "task_name": task_name,
        })
        in_lens_kept.append(in_len)
        out_lens_kept.append(out_len)

    stat = {
        "task_name": task_name,
        "num_raw": len(raw_task_json["Instances"]),
        "num_kept": len(kept),
        "instruction_tok_len": instruction_len,
        "avg_input_tok_raw": round(sum(in_lens) / max(len(in_lens), 1), 2),
        "avg_output_tok_raw": round(sum(out_lens) / max(len(out_lens), 1), 2),
        "avg_input_tok_kept": round(sum(in_lens_kept) / max(len(in_lens_kept), 1), 2),
        "avg_output_tok_kept": round(sum(out_lens_kept) / max(len(out_lens_kept), 1), 2),
        "skipped_reasons": skipped,
    }
    return kept, stat


def run_step1(verbose: bool = True) -> List[dict]:
    """跑 Step 1：加载两个 tokenizer，对每个任务做长度过滤，写到 FILTERED_DIR/<task>.jsonl。
    返回每个任务的统计 dict 列表（也会写 _stat.json）。"""
    os.makedirs(FILTERED_DIR, exist_ok=True)
    print(f"[Step 1] Loading LLaMA-2 tokenizer from: {LLAMA2_TOKENIZER_PATH}")
    tok_llama = AutoTokenizer.from_pretrained(LLAMA2_TOKENIZER_PATH, use_fast=True)
    print(f"[Step 1] Loading SimCSE  tokenizer from: {SIMCSE_TOKENIZER_PATH}")
    tok_simcse = AutoTokenizer.from_pretrained(SIMCSE_TOKENIZER_PATH, use_fast=True)

    all_stats = []
    for cate, task_stem, _is_bin in TASK_LIST:
        raw_path = os.path.join(SUPERNI_TASK_DIR, task_stem + ".json")
        if verbose:
            print(f"\n[Step 1] {cate}  <-  {task_stem}.json")
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        kept, stat = length_filter(raw, task_stem, tok_llama, tok_simcse)
        stat["category"] = cate

        # 写成 jsonl（一行一条）方便大文件增量审计
        out_path = os.path.join(FILTERED_DIR, f"{cate}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        if verbose:
            print(f"  raw={stat['num_raw']}  kept={stat['num_kept']}  "
                  f"(avg_in_tok raw/kept={stat['avg_input_tok_raw']}/{stat['avg_input_tok_kept']}  "
                  f"avg_out_tok raw/kept={stat['avg_output_tok_raw']}/{stat['avg_output_tok_kept']})")
            print(f"  skipped: {stat['skipped_reasons']}")
        all_stats.append(stat)

    with open(os.path.join(FILTERED_DIR, "_stat.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # 最后打一张汇总表给用户看
    print("\n" + "=" * 78)
    print(f"{'cate':6s} {'num_raw':>8s} {'num_kept':>9s} {'need≥2520':>10s} {'need≥100(mini)':>14s}")
    print("-" * 78)
    for s in all_stats:
        need_main = "OK" if s["num_kept"] >= 2520 else f"SHORT by {2520 - s['num_kept']}"
        need_mini = "OK" if s["num_kept"] >= 100 else f"SHORT by {100 - s['num_kept']}"
        print(f"{s['category']:6s} {s['num_raw']:>8d} {s['num_kept']:>9d} {need_main:>10s} {need_mini:>14s}")
    print("=" * 78)
    return all_stats


# --------------------------------------------------------------------------------------
# Step 2: 切分 + 写 TRACE 格式
# --------------------------------------------------------------------------------------
def _read_filtered(cate: str) -> list:
    path = os.path.join(FILTERED_DIR, f"{cate}.jsonl")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _write_trace_json(out_dir: str, split_name: str, records: list):
    """写成 TRACE 现有的 JSON array 格式，每条仅保留 {prompt, answer}。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}.json")
    # 仅保留 prompt 和 answer，和 TRACE 现有文件同构
    minimal = [{"prompt": r["full_prompt"], "answer": r["output"]} for r in records]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(minimal, f, ensure_ascii=False, indent=2)
    return path


def _sample_generic(data: list, seed: int = 0) -> Tuple[list, list]:
    """非二分类任务：按官方逻辑 np.random.seed(0) + np.random.choice 抽 2500 条，
    前 2000 当 train, 后 500 当 eval。
    返回 (main_train_2000, main_eval_500)。
    mini 在外层用嵌套规则 main_train[:200], main_eval[:100] 得到。"""
    rng = np.random.RandomState(seed)
    # 这里我们不要 extra，总采样改成 num_train + num_eval = 2500
    ids = rng.choice(len(data), NUM_TRAIN_MAIN + NUM_EVAL_MAIN, replace=False)
    train_ids = ids[:NUM_TRAIN_MAIN]
    eval_ids = ids[NUM_TRAIN_MAIN:NUM_TRAIN_MAIN + NUM_EVAL_MAIN]
    train = [data[i] for i in train_ids]
    evald = [data[i] for i in eval_ids]
    return train, evald


def _sample_binary(data: list, seed: int = 0) -> Tuple[list, list]:
    """二分类任务 (sa, pos)：按官方逻辑做 label-balanced 采样。
    train 2000 = 1000 positive + 1000 negative（拼接顺序也照搬：先 1000 正，再 1000 负）
    eval  500 = 250 positive + 250 negative。"""
    pos_data, neg_data = [], []
    for line in data:
        lo = line["output"].lower()
        if lo in POS_LABELS:
            pos_data.append(line)
        elif lo in NEG_LABELS:
            neg_data.append(line)
        else:
            print(f"[warn] unknown binary label: {line['output']!r}  (task={line.get('task_name')})")

    rng = np.random.RandomState(seed)
    need_per_class = (NUM_TRAIN_MAIN + NUM_EVAL_MAIN) // 2  # 1250
    pos_ids = rng.choice(len(pos_data), need_per_class, replace=False)
    neg_ids = rng.choice(len(neg_data), need_per_class, replace=False)

    half_train = NUM_TRAIN_MAIN // 2     # 1000
    half_eval = NUM_EVAL_MAIN // 2       # 250

    pos_train = [pos_data[i] for i in pos_ids[:half_train]]
    neg_train = [neg_data[i] for i in neg_ids[:half_train]]
    pos_eval = [pos_data[i] for i in pos_ids[half_train:half_train + half_eval]]
    neg_eval = [neg_data[i] for i in neg_ids[half_train:half_train + half_eval]]

    train = pos_train + neg_train         # 1000 pos 然后 1000 neg（和官方一致）
    evald = pos_eval + neg_eval           # 250 pos 然后 250 neg
    return train, evald


def run_step2():
    if not os.path.isdir(FILTERED_DIR):
        print(f"[Step 2] Filter dir not found: {FILTERED_DIR}. Run --step 1 first.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(OUT_MAIN_DIR, exist_ok=True)
    os.makedirs(OUT_MINI_DIR, exist_ok=True)

    summary = []
    for cate, task_stem, is_bin in TASK_LIST:
        print(f"\n[Step 2] {cate}  ({task_stem}, binary={is_bin})")
        data = _read_filtered(cate)
        if len(data) < NUM_TRAIN_MAIN + NUM_EVAL_MAIN:
            print(f"  [ABORT] {cate} only has {len(data)} filtered samples (<{NUM_TRAIN_MAIN + NUM_EVAL_MAIN}).")
            print(f"          Stopping Step 2 per plan Q2=(b). Re-run after user decision.")
            sys.exit(3)

        if is_bin:
            train, evald = _sample_binary(data, seed=0)
        else:
            train, evald = _sample_generic(data, seed=0)

        # main 档 —— train/eval/test (test = eval 拷贝)
        task_main_dir = os.path.join(OUT_MAIN_DIR, cate)
        train_path = _write_trace_json(task_main_dir, "train", train)
        eval_path = _write_trace_json(task_main_dir, "eval", evald)
        test_path = _write_trace_json(task_main_dir, "test", evald)   # test = eval 内容拷贝

        # mini 档 —— 嵌套子集，Q1=(a)
        # 二分类任务里 train/evald 是 [pos... pos | neg... neg] 顺序拼接，
        # 朴素前缀切片会全是 positive。按类各取前 N/2，保持嵌套子集 + label 平衡。
        if is_bin:
            half_tr, half_ev = NUM_TRAIN_MINI // 2, NUM_EVAL_MINI // 2
            pos_off_tr, pos_off_ev = NUM_TRAIN_MAIN // 2, NUM_EVAL_MAIN // 2
            train_mini = train[:half_tr] + train[pos_off_tr:pos_off_tr + half_tr]
            eval_mini = evald[:half_ev] + evald[pos_off_ev:pos_off_ev + half_ev]
        else:
            train_mini = train[:NUM_TRAIN_MINI]
            eval_mini = evald[:NUM_EVAL_MINI]
        task_mini_dir = os.path.join(OUT_MINI_DIR, cate)
        train_mini_path = _write_trace_json(task_mini_dir, "train", train_mini)
        eval_mini_path = _write_trace_json(task_mini_dir, "eval", eval_mini)
        test_mini_path = _write_trace_json(task_mini_dir, "test", eval_mini)

        print(f"  main: train={len(train)} eval={len(evald)} test={len(evald)}  -> {task_main_dir}")
        print(f"  mini: train={len(train_mini)} eval={len(eval_mini)} test={len(eval_mini)}  -> {task_mini_dir}")

        summary.append({
            "category": cate,
            "task": task_stem,
            "binary": is_bin,
            "num_filtered": len(data),
            "main": {"train": len(train), "eval": len(evald), "test": len(evald)},
            "mini": {"train": len(train_mini), "eval": len(eval_mini), "test": len(eval_mini)},
            "main_dir": task_main_dir,
            "mini_dir": task_mini_dir,
        })

    with open(os.path.join(OUT_MAIN_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_MINI_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[Step 2] Done. Summary saved to _summary.json in both output dirs.")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["1", "2", "all"], required=True)
    args = ap.parse_args()

    if args.step in ("1", "all"):
        run_step1()
    if args.step in ("2", "all"):
        run_step2()


if __name__ == "__main__":
    main()
