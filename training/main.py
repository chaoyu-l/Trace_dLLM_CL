#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys

sys.dont_write_bytecode = True

import argparse
import os
import math
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)
"""
导入deepspeed相关参数
"""
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import (
    create_prompt_dataset,
    IndexedPromptDataset,
    FixedOrderSampler,
    precompute_group_max_ans_len,
    materialize_dataset_order,
)

from utils.data.data_collator import DataCollator
from utils.utils import (
    print_rank_0, to_device, save_hf_format, set_random_seed,
    get_all_reduce_mean, get_optimizer_grouped_parameters, load_hf_tokenizer,
    infer_model_family,
)
from utils.ds_utils import get_train_ds_config

from utils.model.model_utils import create_hf_model


from params import Method2Class, AllDatasetName

# [新增] 严格可复现性设置
def seed_worker(worker_id):
    # 将 worker_seed 与全局 seed 和 worker_id 绑定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def parse_args():
    def list_of_strings(arg):
        return arg.split(',')

    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=('Where to store the data-related files such as shuffle index. '
              'This needs to be on a local storage of a node (not on a shared storage)')
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum prompt length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum answer length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps to accumulate gradients.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of warmup steps.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")

    # local_rank: set by deepspeed/torch launcher
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')

    # Precision flags (optional)
    parser.add_argument('--bf16', action='store_true', help='Enable bf16 in DeepSpeed config (optional).')
    parser.add_argument('--fp16', action='store_true', help='Enable fp16 in DeepSpeed config (optional).')

    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    # Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")

    # Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')

    # Evaluation and loss plotting
    parser.add_argument('--do_eval',
                        action='store_true',
                        help='Run generation-based evaluation after each training epoch.')
    parser.add_argument('--eval_max_samples',
                        type=int,
                        default=50,
                        help='Per-epoch in-training eval 样本上限 (dev split 顺序前 N 条). '
                             '<=0 表示用全部 dev. 默认 50, 用于找每任务收敛 epoch.')
    parser.add_argument('--eval_every_n_epochs',
                        type=int,
                        default=1,
                        help='In-training eval 间隔. =1 表示每 epoch 评一次 (默认); =2 表示每 2 '
                             'epoch 评一次 (在 epoch 2,4,6,... 末尾); 仅在 --do_eval 开启时生效.')
    parser.add_argument('--plot_loss',
                        action='store_true',
                        help='Plot and save the training loss curve to output_dir.')

    # CL method
    parser.add_argument('--CL_method',
                        default=None,
                        help='continual learning method used')

    # ========================== [新增] 扩散模型相关参数 ==========================
    parser.add_argument("--model_type",
                        type=str,
                        default="causal",
                        choices=["causal", "diffusion"],
                        help="Model type: 'causal' for AR models, 'diffusion' for LLaDA/DLMs.")

    parser.add_argument("--sampling_steps",
                        type=int,
                        default=64,
                        help="[Diffusion Only] Number of denoising steps for inference.")

    parser.add_argument("--sampling_temperature",
                        type=float,
                        default=0.0,
                        help="[Diffusion Only] Temperature for sampling (0.0 = argmax).")

    parser.add_argument("--remasking_strategy",
                        type=str,
                        default="low_confidence",
                        choices=["low_confidence", "random"],
                        help="[Diffusion Only] LLaDA strategy for re-masking.")

    parser.add_argument("--block_length",
                        type=int,
                        default=None,
                        help="[Diffusion Only] Block length for block-wise decoding. "
                             "Used by LLaDA and Dream (Fast-dLLM). "
                             "Default: None (single block = full generation length).")

    # Fast-dLLM acceleration options (inference/eval only, does not affect training)
    parser.add_argument("--use_fast_cache",
                        action="store_true",
                        help="[Eval Only] Enable KV cache reuse across denoising steps (Fast-dLLM). "
                             "Supports both LLaDA (prefix cache) and Dream (prompt KV cache).")
    parser.add_argument("--use_dual_cache",
                        action="store_true",
                        help="[Eval Only] Enable dual KV cache for LLaDA (Fast-dLLM). Implies --use_fast_cache.")
    parser.add_argument("--confidence_threshold",
                        type=float,
                        default=None,
                        help="[Eval Only] Confidence threshold for adaptive unmasking (Fast-dLLM). "
                             "Supports both LLaDA and Dream. "
                             "Recommended: 0.9 for quality, 0.5 for speed.")

    # Token Reweighting Arguments
    parser.add_argument(
        "--diffusion_token_reweighting",
        action="store_true",
        help="Whether to enable focal loss-style token reweighting (alpha * (1-exp(-loss))^gamma * loss)."
    )

    parser.add_argument(
        "--diffusion_alpha",
        type=float,
        default=1.0,
        help="The alpha parameter for focal loss token reweighting (default: 1.0)."
    )

    parser.add_argument(
        "--diffusion_gamma",
        type=float,
        default=1.0,
        help="The gamma parameter for focal loss token reweighting (default: 1.0)."
    )

    # Time Reweighting Arguments
    parser.add_argument(
        "--diffusion_time_reweighting",
        type=str,
        default="cart",
        choices=["original", "linear", "cart", "none"],
        help="Time reweighting strategy: 'original' (DDPM style), 'linear', 'cart' (Cartesian), or 'none'."
    )

    parser.add_argument(
        "--diffusion_cart_p",
        type=float,
        default=0.1,
        help="The probability threshold p for the CART time reweighting strategy (default: 0.1)."
    )

    # ========================== Dynamic Canvas (EOS overflow 修复) ==========================
    # 仅作用于 diffusion 训练（LLaDA / Dream），AR 路径完全无影响。
    # 开启后：每个 window（大小由本参数决定）内的样本，共享 canvas = max(answer_token_len
    # in window) + 1（EOS 终止位）。这是 LLaDA 官方 GUIDELINES.md 描述的 "pad to batch
    # max" SFT 行为；与本项目原本的 "pad to per-task fixed canvas" 不同。
    # 同时把 RandomSampler 换成 FixedOrderSampler（独立 RNG，跨模型同序）。
    parser.add_argument(
        "--diffusion_canvas_group_size",
        type=int,
        default=0,
        help="[Diffusion Only] Window size for dynamic-canvas SFT padding. "
             "0 (default) = OFF, 保留原 task-fixed canvas 行为, 现有 outputs_*_500 可复现. "
             "N > 0 = 启用, 每 N 个连续样本共享一个 canvas = max(ans_len in window) + 1. "
             "推荐值 = per_device_batch * grad_accum_steps (本项目默认 1*8=8), 与 DeepSpeed "
             "一次 optimizer.step() 消费的样本数对齐, 模拟 bs=N 的 LLaDA 官方训练。"
    )

    # ===== Fix-EOS: per-task fixed-N EOS pad 策略 (EOS overflow 缓解) =====
    # 开启后每条样本 canvas 严格等于:
    #   prompt_len + ans_len + K   (K = 答案区尾部 EOS 总数, 由 DIFFUSION_TASK_EOS_PAD 决定)
    # 与 dynamic_canvas/task-fixed canvas 的根本区别: 完全 per-sample 决定 EOS 数量,
    # 不受同窗口 outlier 牵连。EOS 占 answer region 比例 = K / (ans_len + K)。
    # Per-task K (LLM-CL-Benchmark_500/5000 实测验证):
    #   短任务 (C-STANCE/FOMC/NumGLUE-cm/NumGLUE-ds, ans_len <= 2): K=1
    #   长任务 (Py150/MeetingBank/ScienceQA/20Minuten, ans_len >= 9): K=2
    # 任务未登记入 DIFFUSION_TASK_EOS_PAD 时, 用安全下限 K=1 兜底 + WARN 日志。
    # 开启时完全覆盖 --diffusion_canvas_group_size。
    parser.add_argument(
        "--fix_eos",
        action="store_true",
        help="[Diffusion Only] 开启 per-task fixed-N EOS pad (EOS overflow 缓解). "
             "K 值严格由 utils/data/data_collator.py: DIFFUSION_TASK_EOS_PAD 决定, "
             "默认关闭, 走 --diffusion_canvas_group_size / task-fixed canvas。"
    )

    # ========================== CL 诊断参数 ==========================
    parser.add_argument(
        "--num_probe_batches",
        type=int,
        default=None,
        help=(
            "Enable CL diagnostics (gradient conflict, representation drift) and "
            "set the number of eval batches used as fixed probes. "
            "When not set, diagnostics are disabled and training runs without overhead. "
            "Actual probe sample count = this × per_device_eval_batch_size. "
            "Recommended: 32 for 500-sample datasets, 64 for 5000-sample datasets."
        ),
    )
    # =========================================================================

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # ------------------ distributed-safe init ------------------
    if args.local_rank == -1:
        # Single GPU (no distributed process group)
        device = torch.device("cuda")
        args.global_rank = 0
        world_size = 1
    else:
        # Launched by deepspeed/torchrun
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
        args.global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    # -----------------------------------------------------------

    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        bf16=args.bf16,
        fp16=args.fp16,
        enable_tensorboard=args.enable_tensorboard,
        tb_path=args.tensorboard_path,
        tb_name="v2_sft"
    )

    # set batch size
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = (
            args.per_device_train_batch_size
            * world_size
            * args.gradient_accumulation_steps
    )

    # seed + barrier (barrier only if distributed initialized)
    set_random_seed(args.seed)
    if args.local_rank != -1:
        torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)

    model_family = infer_model_family(args.model_name_or_path)
    args.model_family = model_family
    print_rank_0(f"[INFO] Inferred model_family = {model_family}", args.global_rank)

    # Diffusion (dream/llada): right padding for training; AR (llama/qwen): left padding
    if model_family in ("dream", "llada"):
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"
    # 左截断：超长 prompt 砍头保尾部，保证 chat-format 的 generation header 留下。
    # collator 用手动切片（_truncate_ids / ids[-max_len:]），这里只是语义一致。
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token is None:
        print_rank_0(f"[INFO] Tokenizer has no pad_token. Setting pad_token = eos_token: {tokenizer.eos_token}",
                     args.global_rank)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if model_family == "dream":
        model_class = AutoModel
    else:
        model_class = AutoModelForCausalLM

    model = create_hf_model(
        model_class,  # <--- 关键修改
        args.model_name_or_path,
        tokenizer,
        ds_config=ds_config,
        disable_dropout=args.disable_dropout
    )
    # ==========================================================================================

    # 再次确认模型配置中的 pad_token_id (这对 model.generate 至关重要)
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        print_rank_0(f"[INFO] Setting model.config.pad_token_id to {tokenizer.pad_token_id}", args.global_rank)
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.CL_method == "lora":
        from peft import get_peft_model, LoraConfig, TaskType

        lora_targets = ["q_proj", "v_proj"]
        lora_r = 8

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=lora_targets
        )

        if model_family == "llada" and not hasattr(model, "prepare_inputs_for_generation"):
            print_rank_0("[INFO] Patching missing 'prepare_inputs_for_generation' for LLaDA...", args.global_rank)

            def _dummy_prepare_inputs(input_ids, **kwargs):
                return {"input_ids": input_ids}

            model.prepare_inputs_for_generation = _dummy_prepare_inputs
        # ========================================================

        model = get_peft_model(model, peft_config)

        # 打印可训练参数，确认 LoRA 已经正确挂载
        print_rank_0("[INFO] Checking Trainable Parameters:", args.global_rank)
        if args.global_rank == 0:
            model.print_trainable_parameters()

        # PEFT 通常已经正确设置 requires_grad，这段可以保留也可以删
        for n, p in model.named_parameters():
            if "lora" in n:
                p.requires_grad = True

    train_task_list, eval_task_list = {}, {}
    infer_eval_task_list = {}

    Datasets = AllDatasetName if args.dataset_name[0] == "all" else args.dataset_name
    # Dynamic-canvas 仅在 diffusion 训练上有意义；group_size==0 时保持 RandomSampler/原
    # collator 的行为完全不变，确保 AR 旧产物 (outputs_Llama-* / outputs_Qwen-*) 与现有
    # outputs_LLaDA-*_500 / outputs_dream_*_500 仍可复现。
    canvas_group_size = int(getattr(args, "diffusion_canvas_group_size", 0) or 0)
    fix_eos = bool(getattr(args, "fix_eos", False))
    use_fixed_eos_pad = fix_eos and (args.model_type == "diffusion")
    if fix_eos and not use_fixed_eos_pad:
        print_rank_0(
            f"[WARN] --fix_eos 仅在 model_type=diffusion 下生效；当前 "
            f"model_type={args.model_type}, 未启用。",
            args.global_rank,
        )
    # 互斥：fix-eos 完全覆盖 dynamic_canvas
    if use_fixed_eos_pad and canvas_group_size > 0:
        print_rank_0(
            f"[INFO] --fix_eos 已开启，完全覆盖 --diffusion_canvas_group_size={canvas_group_size}。"
            f"走 DIFFUSION_TASK_EOS_PAD 的 per-task K，不再使用 group_max。",
            args.global_rank,
        )
    use_dynamic_canvas = (canvas_group_size > 0) \
        and (args.model_type == "diffusion") \
        and (world_size == 1) \
        and (not use_fixed_eos_pad)
    if canvas_group_size > 0 and (not use_dynamic_canvas) and (not use_fixed_eos_pad):
        print_rank_0(
            f"[WARN] --diffusion_canvas_group_size {canvas_group_size} 仅在 "
            f"model_type=diffusion 且单进程 (world_size == 1) 下生效；当前 "
            f"model_type={args.model_type}, world_size={world_size}, 未启用，"
            f"回退到原 task-fixed canvas 行为。",
            args.global_rank,
        )
    if use_fixed_eos_pad:
        # 把每个任务实际生效的 K 也 dump 出来（dict 命中 vs 兜底 K=1）
        try:
            from utils.data.data_collator import DIFFUSION_TASK_EOS_PAD as _PER_TASK_EOS
        except Exception:
            _PER_TASK_EOS = {}
        # 归一化: 让 FOMC_shuffled / *_perm / *_swap 命中 base task 的 K 表
        def _norm(t):
            for _suf in ("_shuffled", "_perm", "_swap"):
                if t.endswith(_suf):
                    return t[: -len(_suf)]
            return t
        per_task_lines = []
        missing = []
        for _t in Datasets:
            _base = _norm(_t)
            if _base in _PER_TASK_EOS:
                _n = _PER_TASK_EOS[_base]
                _src = "dict" if _base == _t else f"dict-via-{_base}"
            else:
                _n = 0  # fallback: K=1 (safe minimum)
                _src = "FALLBACK"
                missing.append(_t)
            per_task_lines.append(f"{_t}=K{_n+1}({_src})")
        print_rank_0(
            f"[INFO] --fix_eos 已开启。per-task K: " + " | ".join(per_task_lines),
            args.global_rank,
        )
        if missing:
            print_rank_0(
                f"[WARN] 任务未登记入 DIFFUSION_TASK_EOS_PAD: {missing} → 用安全下限 K=1 兜底。"
                f"建议在 data_collator.py 加上对应条目。",
                args.global_rank,
            )

    for i_task, dataset in enumerate(Datasets):
        dataset_path = os.path.join(args.data_path, dataset)
        train_dataset, eval_dataset, _ = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed
        )

        # =============== Dynamic-canvas: dataset / sampler / group_max 表 ===============
        # 共享 mutable dict，在 sampler 每个 epoch 开始时由回调 in-place 刷新。
        # collator 持引用 → 看到的永远是当前 epoch 的 group_max。
        idx_to_group_max_table: dict = {}

        if use_dynamic_canvas:
            # 1) 用 IndexedPromptDataset 包装，让 __getitem__(idx) 返回的 dict 携带 _idx
            train_dataset = IndexedPromptDataset(train_dataset)

            # 2) per-task base_seed：避免 8 个任务用相同的 randperm
            #    用一个大质数偏移让 i_task 之间充分不相干
            base_seed = int(args.seed) + i_task * 10007

            # 3) Window size 来自 CLI；推荐 = per_device_batch * grad_accum_steps，但用户可独立指定
            window = canvas_group_size

            # 4) 回调：每个 epoch 用新 order 重算 group_max 表（in-place 更新）
            def _make_group_max_refresh(_train_ds=train_dataset, _window=window, _table=idx_to_group_max_table):
                def _refresh(order):
                    new_tbl = precompute_group_max_ans_len(
                        _train_ds, order,
                        group_size=_window,
                        tokenizer=tokenizer,
                        apply_chat_template=False,
                    )
                    _table.clear()
                    _table.update(new_tbl)
                return _refresh

            refresh_group_max = _make_group_max_refresh()

            # 5) 预热：先按 epoch 0 的 order 算一次 group_max，保证 collator 第 1 个 batch 之前表非空
            #    （sampler.__iter__ 也会再算一次同样的 order，多一次小开销但语义安全）
            refresh_group_max(materialize_dataset_order(len(train_dataset), base_seed))

            # 6) FixedOrderSampler 替代 RandomSampler，回调挂载
            train_sampler = FixedOrderSampler(
                num_samples=len(train_dataset),
                base_seed=base_seed,
                on_epoch_start=refresh_group_max,
            )
            eval_sampler = SequentialSampler(eval_dataset)

            print_rank_0(
                f"[INFO] dynamic_canvas ON  task={dataset}  base_seed={base_seed}  "
                f"window={window}  preheat_groups={(len(idx_to_group_max_table) + window - 1) // window}",
                args.global_rank,
            )
        else:
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_dataset)
                eval_sampler = SequentialSampler(eval_dataset)
            else:
                train_sampler = DistributedSampler(train_dataset)
                eval_sampler = DistributedSampler(eval_dataset)
        # ================================================================================

        # ========================== [修改] 传入 is_diffusion 参数 ==========================
        is_diffusion_mode = (args.model_type == "diffusion")

        # Diffusion 路径（LLaDA/Dream）关闭 pad_to_multiple_of：训练 canvas 必须严格等于
        # [prompt_len | task_ans_len] 以匹配推理窗口；8 倍数圆整会多出 1–7 个 EOS 进入
        # forward_process/loss，造成训练/推理 canvas 轻微不一致。
        # AR 路径（Qwen/LLaMA）保留 8，命中 Tensor Core，padding 不污染因果 LM。
        pad_mult = 1 if is_diffusion_mode else 8

        # Dynamic-canvas: 把共享的 group_max 表传给 collator；标志关闭时为 None，
        # collator 完全退化回 task-fixed canvas（_get_effective_ans_len()）。
        collator_group_max = idx_to_group_max_table if use_dynamic_canvas else None

        data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=pad_mult,
            inference=False,
            is_diffusion=is_diffusion_mode,
            model_family=model_family,
            task=dataset,
            idx_to_group_max=collator_group_max,
            fix_eos=use_fixed_eos_pad,
        )
        # =================================================================================

        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            sampler=train_sampler,
            batch_size=args.per_device_train_batch_size,
            # [新增] 以下两行确保多进程下的 Masking 严格可复现
            worker_init_fn=seed_worker,
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=data_collator,
            sampler=eval_sampler,
            batch_size=args.per_device_eval_batch_size
        )
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader

        if args.do_eval:
            infer_data_collator = DataCollator(
                tokenizer,
                model=model,
                padding="longest",
                max_prompt_len=args.max_prompt_len,
                max_ans_len=args.max_ans_len,
                pad_to_multiple_of=pad_mult,
                inference=True,
                is_diffusion=is_diffusion_mode,
                model_family=model_family,
                task=dataset,
            )
            # 顺序前 N 条 (dev split) 做 per-epoch 收敛评测; <=0 / 超长退化为全部
            cap = getattr(args, 'eval_max_samples', 50)
            if cap is None or cap <= 0 or cap >= len(eval_dataset):
                infer_eval_subset = eval_dataset
            else:
                infer_eval_subset = torch.utils.data.Subset(eval_dataset, list(range(cap)))
            infer_eval_dataloader = DataLoader(
                infer_eval_subset,
                collate_fn=infer_data_collator,
                sampler=SequentialSampler(infer_eval_subset),
                batch_size=args.per_device_eval_batch_size
            )
            infer_eval_task_list[dataset] = infer_eval_dataloader

    args.infer_eval_task_list = infer_eval_task_list if args.do_eval else None

    def get_optimizer(model):
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        return optimizer, lr_scheduler



    optimizer, lr_scheduler = get_optimizer(model)

    # ---------- BEFORE deepspeed.initialize: HF model only ----------
    if args.gradient_checkpointing:
        # 必须关闭 KV cache，否则 HF + checkpointing 会出问题
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True
    )

    # ---------- AFTER deepspeed.initialize: enable checkpointing on HF model ----------
    if args.gradient_checkpointing:
        hf_model = model.module if hasattr(model, "module") else model

        # 1. 强制关闭 KV cache (必须)
        if hasattr(hf_model, "config") and hf_model.config is not None:
            hf_model.config.use_cache = False

        # 2. 必须开启 input_require_grads，否则 LoRA + Checkpointing 会报错
        if hasattr(hf_model, "enable_input_require_grads"):
            hf_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            hf_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # 3. 开启 Checkpointing
        hf_model.gradient_checkpointing_enable()

        print_rank_0("[INFO] Gradient Checkpointing & Input Grads Enabled!", args.global_rank)

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](
            model, tokenizer, optimizer,
            train_task_list, eval_task_list, args
        )
        CL_Trainer.train_continual()


if __name__ == "__main__":
    main()