# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import math
import sys
import copy
from tqdm import tqdm
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import json
import time
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoConfig,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, set_random_seed, load_hf_tokenizer, infer_model_family
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_CStance, eval_Py150, eval_FOMC, \
    eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, eval_ssr

from model.llada_generate import (
    llada_generate,
    llada_generate_with_cache,
    llada_generate_with_dual_cache,
)

from model.dream_generate import (
    dream_generate,
    dream_generate_with_cache,
)


def _fmt_elapsed(secs):
    secs = int(secs)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def _now():
    return datetime.now().strftime("%H:%M:%S")


def _enable_llada_kv_cache(model):
    """Patch LLaDAModel.forward to remove the KV cache assertion.

    The upstream LLaDA model blocks KV cache with an assertion even though the
    infrastructure (past_key_values plumbing, RoPE offset handling, bidirectional
    attention bias slicing) is fully implemented.  Fast-dLLM prefix caching only
    caches immutable prompt tokens whose K,V are deterministic, so this is safe.
    """
    import inspect, textwrap

    inner = getattr(model, "model", None)
    if inner is None:
        return

    cls = type(inner)
    orig_forward = cls.forward

    try:
        src = textwrap.dedent(inspect.getsource(orig_forward))
    except OSError:
        print("[WARN] Cannot auto-patch LLaDA for KV cache: source unavailable")
        return

    target = 'assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."'
    if target not in src:
        return

    src = src.replace(target, "pass")
    local_ns = {}
    exec(compile(src, inspect.getfile(cls), "exec"), orig_forward.__globals__, local_ns)  # noqa: S102
    cls.forward = local_ns["forward"]
    print("[INFO] Patched LLaDAModel to enable KV cache (Fast-dLLM)")

# Diffusion models denoise a fixed-size canvas, so shorter max_ans_len directly
# reduces computation. AR models already stop at EOS, making this unnecessary.
DIFFUSION_TASK_MAX_ANS_LEN = {
    # TRACE
    "C-STANCE":    4,
    "FOMC":        4,
    "NumGLUE-cm":  8,
    "NumGLUE-ds":  8,
    "Py150":       32,
    "MeetingBank": 224,   # 7·32，与 LLADA_TASK_BLOCK_LENGTH=32 对齐（原 216 不整除会触发回退）
    "20Minuten":   160,   # 5·32，与 LLADA_TASK_BLOCK_LENGTH=32 对齐（原 152 不整除会触发回退）
    "ScienceQA":   480,   # 15·32，与 LLADA_TASK_BLOCK_LENGTH=32 对齐（原 464 不整除会触发回退）
    # SSR (ACL 2024) — P95 answer token lengths on 2000-sample train set,
    # ceil to multiple of 8. sa/pos are strictly 1-token binary labels
    # (positive/negative, True/False), so we use canvas=4 to match the TRACE
    # binary-classification convention (C-STANCE/FOMC).
    "qa":          24,
    "qg":          24,
    "sa":          4,
    "sum":         48,
    "trans":       72,
    "dsg":         32,
    "expl":        80,
    "para":        32,
    "pe":          96,
    "pos":         4,
}

# Per-task semi-autoregressive block size for LLaDA inference.
# Derived from LLaDA paper's reported best configs:
#   - GSM8K/MATH (numerical):    gen=256, block=8
#   - HumanEval/MBPP (code/long): gen=512, block=32
# Rule of thumb: keep block size at the absolute scale the paper validated
# (8 for short-numeric, 32 for code/long), not a fixed ratio of gen_length.
# For very short canvases (gen<=16), use a single block (block=gen_length)
# since semi-AR splitting is meaningless on label-style outputs.
# Dream has no native semi-AR decoding, so this dict only applies to LLaDA.
LLADA_TASK_BLOCK_LENGTH = {
    # TRACE
    "C-STANCE":    16,   # 1 block  — short label, single-shot
    "FOMC":        16,   # 1 block  — short label, single-shot
    "NumGLUE-cm":  8,    # 4 blocks — numeric reasoning, mirrors GSM8K block=8
    "NumGLUE-ds":  8,    # 4 blocks — numeric reasoning, mirrors GSM8K block=8
    "Py150":       32,   # 4 blocks — code, mirrors HumanEval block=32
    "MeetingBank": 32,   # 16 blocks — long-form, mirrors MBPP/HumanEval
    "20Minuten":   32,   # 16 blocks — long-form summarization
    "ScienceQA":   32,   # 16 blocks — long-form QA
    # SSR — block only takes LLaDA-paper-validated values {8, 16, 32, 64, 128}
    # (EVAL.md: GSM8K=8/16, GPQA/IFEval=16, HumanEval/MBPP=32, Math=64/128).
    # canvas % block == 0 required by LLaDA generate.py:68.
    "qa":          8,    # 3 blocks — canvas=24, only 8 ∈ paper-set divides 24
    "qg":          8,    # 3 blocks — canvas=24, same
    "sa":          4,    # 1 block  — canvas=4 forces block=canvas (<8)
    "sum":         16,   # 3 blocks — canvas=48, 16 closer to paper default
    "trans":       8,    # 9 blocks — canvas=72, only 8 ∈ paper-set divides 72
    "dsg":         16,   # 2 blocks — canvas=32, 16 is mid option
    "expl":        16,   # 5 blocks — canvas=80
    "para":        16,   # 2 blocks — canvas=32, same as dsg
    "pe":          32,   # 3 blocks — canvas=96, 32 = HumanEval/MBPP paper block
    "pos":         4,    # 1 block  — canvas=4 forces block=canvas (<8)
}
LLADA_DEFAULT_BLOCK_LENGTH = 128  # fallback for unknown tasks


def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path', type=str, default='Dahoas/rm-static')
    parser.add_argument('--data_output_path', type=str, default='/tmp/data_files/')
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--inference_model_path", type=str, required=True)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--max_ans_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--inference_batch", type=int, default=4)
    parser.add_argument("--inference_tasks", type=list_of_strings, default='all')
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--inference_output_path', type=str, default=None)
    parser.add_argument('--CL_method', default=None)

    parser.add_argument("--model_type", type=str, default="causal", choices=["causal", "diffusion"])
    parser.add_argument("--sampling_steps", type=int, default=64)
    parser.add_argument("--sampling_temperature", type=float, default=0.0)
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="LLaDA remasking strategy.")
    parser.add_argument("--dream_alg", type=str, default="origin",
                        choices=["origin", "maskgit_plus", "topk_margin", "entropy"],
                        help="Dream sampling algorithm. 'origin' (default, random transfer) "
                             "matches Dream's stock behavior; 'maskgit_plus' aligns with LLaDA's "
                             "low_confidence (commit highest-confidence positions) for "
                             "mechanism-level fairness ablations.")
    parser.add_argument("--block_length", type=int, default=None,
                        help="Block length for block-wise decoding. "
                             "Used by LLaDA and Dream (Fast-dLLM). "
                             "Default: None (single block = full generation length).")

    # Fast-dLLM acceleration options (inference only, does not affect training)
    parser.add_argument("--use_fast_cache", action="store_true",
                        help="Enable KV cache reuse across denoising steps (Fast-dLLM). "
                             "Supports both LLaDA (prefix cache) and Dream (prompt KV cache).")
    parser.add_argument("--use_dual_cache", action="store_true",
                        help="Enable dual KV cache (prefix+suffix) for LLaDA (Fast-dLLM). Implies --use_fast_cache.")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Confidence threshold for adaptive unmasking (Fast-dLLM). "
                             "Supports both LLaDA and Dream. "
                             "Recommended: 0.9 for quality, 0.5 for speed.")

    parser.add_argument("--target_round", type=int, default=0)
    parser.add_argument("--all_rounds", action="store_true",
                        help="Run all rounds in one process (avoid reloading base model)")
    parser.add_argument("--max_inference_samples", type=int, default=None,
                        help="If set, randomly subsample this many samples from each task's "
                             "test set (seeded by --seed). Useful for quick verification.")

    parser.add_argument("--deepspeed", action="store_true", help="(ignored, kept for CLI compat)")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="(ignored)")

    args = parser.parse_args()
    return args


def load_base_model(args, tokenizer):
    """Load base model once, with Flash Attention 2 when available."""
    model_family = getattr(args, "model_family", None)
    trust_remote = model_family in ("llada", "dream")

    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=trust_remote
    )

    if model_family == "dream":
        model_class = AutoModel
    else:
        model_class = AutoModelForCausalLM

    load_kwargs = dict(
        pretrained_model_name_or_path=args.model_name_or_path,
        config=model_config,
        trust_remote_code=trust_remote,
        torch_dtype=torch.bfloat16,
    )

    # Dream/LLaDA custom models may not support attn_implementation kwarg
    try:
        model = model_class.from_pretrained(**load_kwargs, attn_implementation="flash_attention_2")
        print("[INFO] Loaded model with Flash Attention 2")
    except (ValueError, ImportError, TypeError):
        model = model_class.from_pretrained(**load_kwargs)
        print("[INFO] Flash Attention 2 not available, using default attention")

    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    if tokenizer.eos_token_id is not None:
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id

    return model


def load_adapter_for_round(base_model, args, round_idx, device):
    """Load LoRA adapter for a specific round, merge weights, and move to GPU."""
    inference_model_path = os.path.join(args.inference_model_path, str(round_idx))
    print(f"[INFO] Loading adapter from: {inference_model_path}")
    t0 = time.time()

    if args.CL_method in ["base", "base_model"]:
        print(f"[INFO] CL_method='{args.CL_method}', skipping adapter loading.")
        model = base_model

    elif args.CL_method == "lora":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, inference_model_path)
        model = model.merge_and_unload()
        print("[INFO] LoRA merged and unloaded — no per-forward adapter overhead")

    elif args.CL_method not in ["O-LoRA", "LFPT5", "OGD", "PP", "L2P"]:
        state_dict = torch.load(
            os.path.join(inference_model_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        for name, param in base_model.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name])
        del state_dict
        model = base_model
    else:
        model = base_model

    model.to(device)
    model.eval()

    model_family = getattr(args, "model_family", None)
    if model_family == "llada":
        print("[INFO] torch.compile skipped for LLaDA (incompatible with CUDA graphs)")
    elif torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default")
            print("[INFO] torch.compile applied (default mode, no CUDA graphs)")
        except Exception as e:
            print(f"[INFO] torch.compile skipped: {e}")

    print(f"[INFO] Adapter loaded and model ready in {time.time() - t0:.1f}s")
    return model


def main():
    args = parse_args()
    set_random_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    # ======================== Setup tokenizer ========================
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    model_family = infer_model_family(args.model_name_or_path)
    args.model_family = model_family
    print(f"[INFO] Inferred model_family = {model_family}")

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inference_tasks = args.inference_tasks
    task_num = len(inference_tasks)

    # ======================== Prediction function ========================
    def prediction(model, infer_dataloader, progress_bar, max_ans_len=None, block_length=None):
        effective_max_ans_len = max_ans_len if max_ans_len is not None else args.max_ans_len
        # LLaDA: per-task block_length when CLI not set; CLI value forces a global override.
        effective_block_length = (
            args.block_length
            if args.block_length is not None
            else (block_length if block_length is not None else LLADA_DEFAULT_BLOCK_LENGTH)
        )
        if args.model_type == "diffusion" and args.sampling_steps <= 0:
            effective_steps = effective_max_ans_len
        else:
            effective_steps = args.sampling_steps
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()

        for step, batch in enumerate(infer_dataloader):
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']

            batch = to_device(batch, device)

            progress_bar.update(1)
            progress_bar.set_description(f"Step {step}", refresh=False)

            with torch.no_grad():
                if args.model_type == "diffusion":
                    if getattr(args, "model_family", None) == "dream":
                        if getattr(args, "use_fast_cache", False):
                            generate_ids = dream_generate_with_cache(
                                model,
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                max_new_tokens=effective_max_ans_len,
                                steps=effective_steps,
                                temperature=args.sampling_temperature,
                                threshold=getattr(args, "confidence_threshold", None),
                                block_length=getattr(args, "block_length", None),
                                alg=getattr(args, "dream_alg", "origin"),
                            )
                        else:
                            out = model.diffusion_generate(
                                batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                max_new_tokens=effective_max_ans_len,
                                steps=effective_steps,
                                temperature=args.sampling_temperature,
                                alg=getattr(args, "dream_alg", "origin"),
                                return_dict_in_generate=True,
                                output_history=False,
                            )
                            generate_ids = out.sequences if hasattr(out, "sequences") else out
                    else:
                        llada_kwargs = dict(
                            model=model, tokenizer=tokenizer,
                            prompt_input_ids=batch['input_ids'],
                            attention_mask=batch["attention_mask"],
                            max_new_tokens=effective_max_ans_len,
                            steps=effective_steps,
                            temperature=args.sampling_temperature,
                            remasking=args.remasking_strategy,
                            block_length=effective_block_length,
                        )
                        if getattr(args, "use_dual_cache", False):
                            llada_kwargs["threshold"] = getattr(args, "confidence_threshold", None)
                            generate_ids = llada_generate_with_dual_cache(**llada_kwargs)
                        elif getattr(args, "use_fast_cache", False):
                            llada_kwargs["threshold"] = getattr(args, "confidence_threshold", None)
                            generate_ids = llada_generate_with_cache(**llada_kwargs)
                        else:
                            generate_ids = llada_generate(**llada_kwargs)
                else:
                    do_sample = args.temperature > 0
                    generation_kwargs = dict(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_new_tokens=effective_max_ans_len,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                        do_sample=do_sample,
                        num_return_sequences=1,
                        use_cache=True
                    )
                    if do_sample:
                        generation_kwargs["temperature"] = args.temperature
                    generate_ids = model.generate(**generation_kwargs)

            if args.model_type == "diffusion" and getattr(args, "model_family", None) == "dream":
                full_input_len = batch["input_ids"].shape[1]
                sequences = []
                for i in range(len(batch["input_ids"])):
                    gen_ids = generate_ids[i, full_input_len:]
                    txt = tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    stop_list = []
                    if tokenizer.eos_token:
                        stop_list.append(tokenizer.eos_token)
                    stop_list += ["<|im_end|>", "<|endoftext|>"]
                    for st in stop_list:
                        if st and st in txt:
                            txt = txt.split(st)[0]
                    sequences.append(txt.strip())
                predicted_sequences += sequences

            elif args.model_type == "diffusion" and getattr(args, "model_family", None) == "llada":
                full_input_len = batch["input_ids"].shape[1]
                sequences = []
                for i in range(len(batch["input_ids"])):
                    gen_ids = generate_ids[i, full_input_len:]
                    txt = tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    stop_list = ["<|eot_id|>"]
                    if tokenizer.eos_token:
                        stop_list.append(tokenizer.eos_token)
                    for st in stop_list:
                        if st and st in txt:
                            txt = txt.split(st)[0]
                    sequences.append(txt.strip())
                predicted_sequences += sequences

            else:
                prompt_len = batch["input_ids"].shape[1]
                sequences = tokenizer.batch_decode(
                    generate_ids[:, prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                predicted_sequences += sequences

        return sources_sequences, predicted_sequences, ground_truths

    def save_inference_results(evaluation_result, sources_sequences, predicted_sequences,
                               ground_truths, round_idx, i_task, task, eval_details=None):
        if eval_details is None:
            eval_details = {}
        df = {
            "eval": evaluation_result,
            "prompts": sources_sequences,
            "results": predicted_sequences,
            "labels": ground_truths,
            "eval_details": eval_details
        }
        os.makedirs(args.inference_output_path, exist_ok=True)
        fname = f"results-{round_idx}-{i_task}-{task}.json"
        with open(os.path.join(args.inference_output_path, fname), "w+", encoding='utf-8') as f:
            json.dump(df, f, ensure_ascii=False)

    def _normalize_task_name(task):
        """剥掉诊断后缀 (_shuffled / _perm / _swap), 让派生数据集复用基类 task 的 evaluator/block-length."""
        for suffix in ("_shuffled", "_perm", "_swap"):
            if task.endswith(suffix):
                return task[: -len(suffix)]
        return task

    def evaluate_task(inference_task, sources_sequences, predicted_sequences, ground_truths):
        inference_task = _normalize_task_name(inference_task)
        dispatch = {
            # TRACE
            "ScienceQA": lambda: eval_ScienceQA.eval(predicted_sequences, ground_truths, return_details=True),
            "MeetingBank": lambda: eval_MeetingBank.eval(predicted_sequences, ground_truths, return_details=True),
            "C-STANCE": lambda: eval_CStance.eval(predicted_sequences, ground_truths, return_details=True),
            "Py150": lambda: eval_Py150.eval(predicted_sequences, ground_truths, return_details=True),
            "FOMC": lambda: eval_FOMC.eval(predicted_sequences, ground_truths, return_details=True),
            "NumGLUE-cm": lambda: eval_NumGLUE_cm.eval(predicted_sequences, ground_truths, return_details=True),
            "NumGLUE-ds": lambda: eval_NumGLUE_ds.eval(predicted_sequences, ground_truths, return_details=True),
            "20Minuten": lambda: eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths, return_details=True),
            # SSR — all 10 tasks share LLaMA-Factory ComputeMetrics (ROUGE-L primary).
            # sa/pos additionally record exact-match accuracy for downstream selection.
            "qa":    lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="qa",    return_details=True),
            "qg":    lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="qg",    return_details=True),
            "sa":    lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="sa",    return_details=True),
            "sum":   lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="sum",   return_details=True),
            "trans": lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="trans", return_details=True),
            "dsg":   lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="dsg",   return_details=True),
            "expl":  lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="expl",  return_details=True),
            "para":  lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="para",  return_details=True),
            "pe":    lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="pe",    return_details=True),
            "pos":   lambda: eval_ssr.eval(predicted_sequences, ground_truths, task="pos",   return_details=True),
        }
        if inference_task in dispatch:
            return dispatch[inference_task]()
        return {}, {}


    def run_single_round(model, round_idx):
        """Run inference for all tasks in a single round."""
        t_round = time.time()
        print(f"[{_now()}] ===== Round {round_idx} start =====")

        for inference_task_id in range(task_num):
            inference_task = inference_tasks[inference_task_id]
            t_task = time.time()
            task_status = "[Past/Current]" if inference_task_id <= round_idx else "[Future/Zero-shot]"
            print(f"[{_now()}] ***** Inference {task_status} Task: {inference_task} (Train Round {round_idx}) *****")

            # Per-task max_ans_len for diffusion models to avoid wasting
            # compute on oversized canvas for short-answer tasks.
            # 归一化, 让 FOMC_shuffled 等派生数据集命中 FOMC 的 per-task max_ans_len.
            _mal_task = _normalize_task_name(inference_task)
            if args.model_type == "diffusion" and _mal_task in DIFFUSION_TASK_MAX_ANS_LEN:
                task_max_ans_len = DIFFUSION_TASK_MAX_ANS_LEN[_mal_task]
            else:
                task_max_ans_len = args.max_ans_len
            task_steps = task_max_ans_len if (args.model_type == "diffusion" and args.sampling_steps <= 0) else args.sampling_steps

            # Per-task block_length for LLaDA semi-AR decoding. CLI override wins.
            task_block_length = None
            if (args.model_type == "diffusion"
                    and getattr(args, "model_family", None) == "llada"
                    and args.block_length is None):
                # 归一化, 让 FOMC_shuffled 等派生数据集命中 FOMC 的 block_length
                _bl_task = _normalize_task_name(inference_task)
                task_block_length = LLADA_TASK_BLOCK_LENGTH.get(
                    _bl_task, LLADA_DEFAULT_BLOCK_LENGTH
                )

            block_log = ""
            if task_block_length is not None:
                block_log = f", block_length={task_block_length} (per-task)"
            elif args.block_length is not None:
                block_log = f", block_length={args.block_length} (CLI override)"
            print(f"[INFO] max_ans_len for {inference_task}: {task_max_ans_len}"
                  f"{' (adaptive)' if task_max_ans_len != args.max_ans_len else ''}"
                  f", steps={task_steps}{block_log}")

            dataset_path = os.path.join(args.data_path, inference_task)
            _, _, infer_dataset = create_prompt_dataset(
                args.local_rank, dataset_path, args.data_output_path, args.seed, distributed=False
            )

            if args.max_inference_samples and args.max_inference_samples < len(infer_dataset):
                g = torch.Generator().manual_seed(args.seed)
                idx = torch.randperm(len(infer_dataset), generator=g)[:args.max_inference_samples].tolist()
                infer_dataset = torch.utils.data.Subset(infer_dataset, idx)
                print(f"[INFO] Subsampled {inference_task} test set to "
                      f"{len(infer_dataset)} samples (seed={args.seed})")

            is_diffusion_mode = (args.model_type == "diffusion")
            mf = getattr(args, "model_family", None)
            if mf is None:
                raise RuntimeError("args.model_family is not set")

            # Diffusion 路径：推理分支实际未用 pad_to_multiple_of（left-pad 到 max_len），
            # 但为风格一致也显式设 1，避免以后有人复用导致 canvas 与训练不一致。
            pad_mult = 1 if is_diffusion_mode else 8

            inf_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=task_max_ans_len,
                pad_to_multiple_of=pad_mult,
                inference=True,
                is_diffusion=is_diffusion_mode,
                model_family=mf
            )

            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(
                infer_dataset,
                collate_fn=inf_data_collator,
                sampler=infer_sampler,
                batch_size=args.inference_batch
            )

            progress_bar = tqdm(total=len(infer_dataloader), leave=True,
                                desc=f"Round {round_idx} - {inference_task}")

            sources_sequences, predicted_sequences, ground_truths = prediction(
                model, infer_dataloader, progress_bar,
                max_ans_len=task_max_ans_len, block_length=task_block_length,
            )

            evaluation_result, eval_details = evaluate_task(
                inference_task, sources_sequences, predicted_sequences, ground_truths
            )

            print(f"***** Saving inference results for {inference_task} *****")
            save_inference_results(
                evaluation_result, sources_sequences, predicted_sequences, ground_truths,
                round_idx, inference_task_id, inference_task, eval_details=eval_details
            )

            del sources_sequences, predicted_sequences, ground_truths
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[{_now()}] Task {inference_task} done in {_fmt_elapsed(time.time() - t_task)}")

        print(f"[{_now()}] ===== Round {round_idx} finished in {_fmt_elapsed(time.time() - t_round)} =====")

    # ======================== Load base model ONCE ========================
    t_load = time.time()
    base_model = load_base_model(args, tokenizer)

    if getattr(args, "model_family", None) == "llada":
        if not hasattr(base_model, "prepare_inputs_for_generation"):
            base_model.prepare_inputs_for_generation = lambda input_ids, **kwargs: {"input_ids": input_ids}
        if getattr(args, "use_fast_cache", False) or getattr(args, "use_dual_cache", False):
            _enable_llada_kv_cache(base_model)

    print(f"[INFO] Base model loaded in {time.time() - t_load:.1f}s")

    # ======================== Run rounds ========================
    t_all = time.time()
    if args.all_rounds:
        num_rounds = task_num
        print("=" * 60)
        print(f"FAST MODE: Running ALL {num_rounds} rounds in one process")
        print("=" * 60)

        for round_idx in range(num_rounds):
            model_copy = copy.deepcopy(base_model)
            model = load_adapter_for_round(model_copy, args, round_idx, device)
            run_single_round(model, round_idx)

            del model, model_copy
            gc.collect()
            torch.cuda.empty_cache()
    else:
        round_idx = args.target_round
        print("=" * 60)
        print(f"Running Inference for Round {round_idx}")
        print("=" * 60)

        model = load_adapter_for_round(base_model, args, round_idx, device)
        run_single_round(model, round_idx)

    print(f"[{_now()}] All done in {_fmt_elapsed(time.time() - t_all)}")


if __name__ == "__main__":
    main()
