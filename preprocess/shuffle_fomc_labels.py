#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
shuffle_fomc_labels.py — 重排 FOMC 数据集每条样本的 ABC 选项分配

为什么:
  原 FOMC 任务 label 分布严重不均 (A=26%, B=25%, C=49%), 模型"不会就猜 C"
  即可得到 ~49% 的 baseline accuracy, 难以区分"真学到语义"vs"利用 C-bias".

  本脚本对每条样本随机置换 {dovish, hawkish, neutral} 在 ABC 三个位置的分配,
  同步改写 prompt 文本和 answer 字段. 置换由 (base_seed, split_name, idx) 决定,
  完全可复现. 新数据的 label 分布会接近均匀 (各 ~33%), 任何固定字母偏好都会
  被 break.

用法:
  python preprocess/shuffle_fomc_labels.py \\
      --src_root ./data/TRACE-Benchmark/LLM-CL-Benchmark_5000 \\
      --dst_name FOMC_shuffled \\
      --seed 1234

输出:
  <src_root>/FOMC_shuffled/{train,eval,test}.json

下游用法:
  scripts/run_<model>_5000_e<N>.sh 里把
    --dataset_name FOMC,...
  改成
    --dataset_name FOMC_shuffled,...
  其余完全不动. dispatch 端 (base_model._eval_task / infer_single) 已经做了
  task 名归一化, 会自动复用 eval_FOMC.eval.
"""
import argparse
import collections
import hashlib
import json
import os
import random
from itertools import permutations


# 原 FOMC prompt 模板里的选项文本 (所有样本一致)
ORIG_OPTIONS_TEXT = "A. dovish, B. hawkish, C. neutral"
ORIG_LETTER_TO_MEANING = {"A": "dovish", "B": "hawkish", "C": "neutral"}
ALL_PERMUTATIONS = list(permutations(["dovish", "hawkish", "neutral"]))


def _seed_for_sample(base_seed: int, split_name: str, idx: int) -> int:
    """Deterministic per-sample seed (independent of Python hash randomization)."""
    key = f"{base_seed}|{split_name}|{idx}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def shuffle_one(sample: dict, sample_seed: int) -> dict:
    rng = random.Random(sample_seed)
    chosen = rng.choice(ALL_PERMUTATIONS)  # tuple e.g. ('neutral','dovish','hawkish')
    new_letter_to_meaning = {"A": chosen[0], "B": chosen[1], "C": chosen[2]}
    meaning_to_new_letter = {m: l for l, m in new_letter_to_meaning.items()}

    new_options_text = (
        f"A. {new_letter_to_meaning['A']}, "
        f"B. {new_letter_to_meaning['B']}, "
        f"C. {new_letter_to_meaning['C']}"
    )

    if ORIG_OPTIONS_TEXT not in sample["prompt"]:
        raise ValueError(
            f"[shuffle_fomc_labels] sample prompt 不含期望的选项文本\n"
            f"  expected substring: {ORIG_OPTIONS_TEXT!r}\n"
            f"  got prompt (first 200 chars): {sample['prompt'][:200]!r}"
        )

    new_prompt = sample["prompt"].replace(ORIG_OPTIONS_TEXT, new_options_text, 1)

    old_letter = sample["answer"].strip()
    if old_letter not in ORIG_LETTER_TO_MEANING:
        raise ValueError(f"[shuffle_fomc_labels] 未识别的 answer: {sample['answer']!r}")
    old_meaning = ORIG_LETTER_TO_MEANING[old_letter]
    new_letter = meaning_to_new_letter[old_meaning]

    out = dict(sample)
    out["prompt"] = new_prompt
    out["answer"] = new_letter
    return out


def process_split(src_path: str, dst_path: str, split_name: str, base_seed: int) -> list:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for i, sample in enumerate(data):
        sample_seed = _seed_for_sample(base_seed, split_name, i)
        new_data.append(shuffle_one(sample, sample_seed))

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    return new_data


def _label_stats(data: list, name: str) -> None:
    cnt = collections.Counter(d["answer"] for d in data)
    total = len(data)
    print(f"  [{name}] N={total}")
    for letter in "ABC":
        c = cnt.get(letter, 0)
        pct = 100.0 * c / total if total else 0.0
        print(f"    {letter}: {c:5d} ({pct:5.2f}%)")


def _all_splits_present(dst_dir: str) -> bool:
    """True iff train.json / eval.json / test.json 都已存在且非空."""
    for split in ("train", "eval", "test"):
        path = os.path.join(dst_dir, f"{split}.json")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True,
                    help="LLM-CL-Benchmark_XXXX 根目录, 下含 FOMC/")
    ap.add_argument("--src_name", default="FOMC",
                    help="源数据子目录名 (默认 FOMC)")
    ap.add_argument("--dst_name", default="FOMC_shuffled",
                    help="输出数据子目录名 (默认 FOMC_shuffled)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--force", action="store_true",
                    help="强制重生成 (即使 dst 下 train/eval/test.json 都已存在)")
    args = ap.parse_args()

    src_dir = os.path.join(args.src_root, args.src_name)
    dst_dir = os.path.join(args.src_root, args.dst_name)
    print(f"src: {src_dir}/")
    print(f"dst: {dst_dir}/")
    print(f"seed: {args.seed}")
    print()

    # Idempotent: 三个 split 全部已生成 -> 跳过 (除非 --force)
    if (not args.force) and _all_splits_present(dst_dir):
        print(f"[skip] {dst_dir} 已存在完整 train/eval/test.json, 跳过. (--force 强制重生成)")
        for split in ("train", "eval", "test"):
            data = json.load(open(os.path.join(dst_dir, f"{split}.json")))
            _label_stats(data, split)
        return

    for split in ("train", "eval", "test"):
        src = os.path.join(src_dir, f"{split}.json")
        dst = os.path.join(dst_dir, f"{split}.json")
        if not os.path.exists(src):
            print(f"skip {split}: {src} 不存在")
            continue
        new_data = process_split(src, dst, split, args.seed)
        _label_stats(new_data, split)

    # Sanity check: 打印 train[0] 原 vs 新
    print()
    print("=== Sanity check: train[0] 原 vs 新 ===")
    orig = json.load(open(os.path.join(src_dir, "train.json")))[0]
    new = json.load(open(os.path.join(dst_dir, "train.json")))[0]
    print(f"  orig answer:  {orig['answer']}")
    print(f"  orig options: {orig['prompt'][orig['prompt'].find('A. '):orig['prompt'].find('A. ') + 60]}")
    print(f"  new  answer:  {new['answer']}")
    print(f"  new  options: {new['prompt'][new['prompt'].find('A. '):new['prompt'].find('A. ') + 60]}")
    # 验证语义一致性: 原 answer 的 meaning == 新 answer 的 meaning
    orig_meaning = ORIG_LETTER_TO_MEANING[orig["answer"]]
    new_opt_text = new["prompt"][new["prompt"].find("A. "):new["prompt"].find("A. ") + 60]
    new_letter_pos = new_opt_text.find(orig_meaning)
    print(f"  原 answer 语义 '{orig_meaning}' 在新 prompt 里出现位置={new_letter_pos} "
          f"(应非负 + 大致对应 A/B/C 之一)")
    print()
    print(f"Done. 在训练脚本里把 --dataset_name FOMC 改成 --dataset_name {args.dst_name}")


if __name__ == "__main__":
    main()
