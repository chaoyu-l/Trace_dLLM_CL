import torch  # 导入 PyTorch 主库：张量、GPU、自动求导、模型训练都会用到
import copy  # Python 深拷贝库：用于复制对象，避免原对象被联动修改
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, infer_model_family
# 从你项目自己的 utils.utils 中导入多个工具函数：
# - print_rank_0: 只在主进程打印日志，避免多卡重复输出
# - to_device: 把 batch 中的张量统一搬到 GPU/指定设备
# - save_hf_format: 按 HuggingFace 格式保存模型
# - set_random_seed: 设置随机种子，保证可复现
# - get_all_reduce_mean: 多卡下做 all-reduce 再求平均
# - get_optimizer_grouped_parameters: 构建优化器参数组
# - save_zero_three_model: DeepSpeed ZeRO-3 专用保存函数
# - load_hf_tokenizer: 加载 tokenizer

from utils.data.data_utils import create_prompt_dataset
# 导入创建 prompt 数据集的函数，通常负责读取/预处理 train/eval/test 数据

from utils.data.data_collator import DataCollator
# 导入 DataCollator：DataLoader 在拼 batch 时会调用它，负责 padding、labels 构建等

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# PyTorch 数据加载相关组件：
# - DataLoader: 迭代 batch
# - RandomSampler: 随机采样训练集
# - SequentialSampler: 顺序采样验证/测试集

from tqdm import tqdm
# 进度条工具，用于训练时显示 step 进度

import torch.distributed as dist
# PyTorch 分布式训练接口，支持多卡/多机通信

import torch.nn.functional as F
# PyTorch 的函数式 API，例如 softmax、cross_entropy 等

import json
# Python 处理 JSON 数据的标准库

import os
# Python 操作文件路径、目录、环境变量的标准库

import time
from datetime import datetime
# Python 时间相关标准库，例如计时、sleep 等
import importlib

import inspect
# Python 反射/检查工具，这里用于查看 model.forward 的参数列表

import gc
import re
import itertools

from evaluations import eval_ScienceQA, eval_MeetingBank, eval_CStance, eval_Py150, eval_FOMC, \
    eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten, eval_ssr
# 导入多个任务的评测函数：
# ScienceQA、MeetingBank、C-STANCE、Py150、FOMC、NumGLUE 等

from transformers import GenerationConfig
# HuggingFace 的生成配置类，用来统一控制 generate 的采样参数

from utils.diffusion.dream_gen_utils import q_sample, context_adaptive_reweight
from utils.diffusion.decode_config import (
    resolve_task_canvas,
    resolve_llada_block_length,
)
from utils.cl_metrics import (
    linear_cka,
    linear_cka_unbiased,
    avg_cosine_similarity,
    magnitude_weighted_conflict_ratio,
)


generation_config = GenerationConfig(
    temperature=0.1,  # 生成温度，越小越保守
    do_sample=True,  # 是否采样；True 表示不是纯贪心/beam，而是随机采样
    num_return_sequences=1  # 每个输入只返回 1 条生成结果
)


# ================= [修改] LLaDA 专用生成器 (支持 Left Padding + RoPE 修正) =================


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


class CL_Base_Model:
    # 定义一个 Continual Learning 基础模型类
    # 负责：
    # 1. 保存 model / tokenizer / optimizer
    # 2. 根据模型类型决定 AR 还是 diffusion
    # 3. 执行 generate
    # 4. 训练单个任务
    # 5. 顺序训练多个任务（continual learning）
    # 6. 保存各轮模型

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, args):

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_task_list = train_task_list
        self.eval_task_list = eval_task_list
        self.args = args
        # 保存超参数和配置对象 args

        if hasattr(model, "module"):
            # 如果 model 有 .module 属性
            # 通常说明它被 DataParallel / DDP / DeepSpeed 等封装过
            raw_model = model.module
            # 取出最底层的原始模型
        else:
            raw_model = model
            # 否则当前 model 本身就是原始模型

        forward_params = inspect.signature(raw_model.forward).parameters
        # 用 inspect 读取 raw_model.forward 的函数签名
        # 得到它接收的参数列表，例如 input_ids / attention_mask / use_cache / position_ids 等

        self.supports_use_cache = "use_cache" in forward_params
        # 判断这个模型的 forward 是否支持 use_cache 参数
        # 有些模型支持 forward(..., use_cache=False)，有些不支持
        # 后面会根据这个标志安全地传参

        # 默认类型处理
        if not hasattr(self.args, "model_type"):
            # 如果 args 里没有 model_type 这个字段
            self.args.model_type = "causal"
            # 默认设为 "causal"，表示自回归模型（AR）

        self.is_diffusion = (self.args.model_type == "diffusion")
        # 判断当前是否为 diffusion language model
        # True 表示扩散模型，False 表示 AR 模型

        self.model_family = self._resolve_model_family()
        # 解析模型家族类型，例如：
        # llada / qwen / dream / llama2 / llama3

        # --- Dream diffusion CART cache (optional) ---
        self._cart_weight_matrix = None
        # 缓存 Dream 的 context-adaptive reweight 矩阵
        # 这样不用每个 batch 都重新构建

        self._cart_p = float(getattr(self.args, "diffusion_cart_p", 0.1))
        # 读取 Dream 的 CART 超参数 p
        # fallback 0.1 与 training/main.py:308 argparse default 及 line 358 处一致;
        # argparse 注册过该字段, 所以正常路径下 fallback 是死代码, 但保持三处统一避免误导.

        self._max_seq_len = int(getattr(self.args, "max_prompt_len", 512)) + int(getattr(self.args, "max_ans_len", 512))
        # 记录最大序列长度 = prompt 最大长度 + answer 最大长度
        # 如果 args 中没有这两个参数，则默认各取 512，总长 1024

    def _resolve_model_family(self) -> str:
        """Read args.model_family (set by main.py / infer_single.py).
        Fallback: infer from model_name_or_path via the shared utility."""
        mf = getattr(self.args, "model_family", None)
        if mf and str(mf).strip():
            return str(mf).strip().lower()

        mpath = getattr(self.args, "model_name_or_path", None)
        if not mpath:
            raise ValueError(
                "args.model_family not set and args.model_name_or_path missing; "
                "cannot infer model family.")
        mf = infer_model_family(mpath)
        self.args.model_family = mf
        return mf

    def generate(self, batch, max_new_tokens=64, task=None):
        # 定义统一的生成接口
        # 对不同模型家族，内部走不同的生成实现
        # task: 当前任务名 (in-training eval 调用方传入)。LLaDA 用它查 per-task
        #       block_length, 让 in-training eval 与 Phase 2 inference 解码轨迹一致。
        #       None 时退化到 LLADA_DEFAULT_BLOCK_LENGTH (=128).

        input_ids = batch["input_ids"]
        # 从 batch 里取 input_ids
        # shape 一般是 [B, L]

        attention_mask = batch.get("attention_mask", None)
        # 尝试从 batch 里取 attention_mask
        # 如果没有，就用 None

        if self.is_diffusion:
            # 如果是 diffusion 模型，走扩散生成逻辑

            model_family = self.model_family
            # 取出当前模型家族名称，例如 dream / llada

            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            # 拿到底层原始模型

            # ===== Dream =====
            if model_family == "dream":
                use_cache = getattr(self.args, "use_fast_cache", False)
                conf_threshold = getattr(self.args, "confidence_threshold", None)

                if use_cache:
                    return dream_generate_with_cache(
                        raw_model,
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        steps=getattr(self.args, "sampling_steps", 64),
                        temperature=getattr(self.args, "sampling_temperature", 0.0),
                        threshold=conf_threshold,
                        block_length=getattr(self.args, "block_length", None),
                    )

                out = raw_model.diffusion_generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    steps=getattr(self.args, "sampling_steps", 64),
                    temperature=getattr(self.args, "sampling_temperature", 0.0),
                    return_dict_in_generate=True,
                    output_history=False,
                )
                return out.sequences if hasattr(out, "sequences") else out

            # LLaDA block_length: CLI > per-task 表 > 128.
            # resolver 内部 is None 判断 + 任务名归一化, 与 Phase 2 inference 走同一份表。
            _bl = resolve_llada_block_length(task, cli_block_length=getattr(self.args, "block_length", None))
            llada_kwargs = dict(
                model=raw_model,
                tokenizer=self.tokenizer,
                prompt_input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                steps=getattr(self.args, "sampling_steps", 64),
                temperature=getattr(self.args, "sampling_temperature", 0.0),
                block_length=_bl,
                remasking=getattr(self.args, "remasking_strategy", "low_confidence"),
            )

            use_cache = getattr(self.args, "use_fast_cache", False)
            dual_cache = getattr(self.args, "use_dual_cache", False)
            conf_threshold = getattr(self.args, "confidence_threshold", None)

            if dual_cache:
                llada_kwargs["threshold"] = conf_threshold
                return llada_generate_with_dual_cache(**llada_kwargs)
            elif use_cache:
                llada_kwargs["threshold"] = conf_threshold
                return llada_generate_with_cache(**llada_kwargs)
            else:
                return llada_generate(**llada_kwargs)

        # ===== AR (Llama/Qwen 等) =====
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

    # ------------------------------------------------------------------
    #  CL diagnostic helpers
    # ------------------------------------------------------------------

    def _get_device(self):
        if self.args.local_rank == -1:
            return torch.device("cuda")
        return torch.device("cuda", self.args.local_rank)

    def _compute_loss(self, batch):
        """Forward + loss for a single batch (on device, sources/gts already removed).

        Pops diffusion-specific keys (p_mask, answer_lens, …) from *batch*.
        Covers Dream-diffusion, LLaDA-diffusion, and AR branches.
        """
        p_mask = batch.pop("p_mask", None)
        answer_lens = batch.pop("answer_lens", None)

        # ---- Dream diffusion ----
        if self.is_diffusion and self.model_family == "dream":
            input_ids = batch.pop("input_ids", None)
            attention_mask_2d = batch.pop("attention_mask", None)
            position_ids = batch.pop("position_ids", None)
            loss_mask = batch.pop("loss_mask", None)
            batch.pop("labels", None)

            if input_ids is None or attention_mask_2d is None \
                    or position_ids is None or loss_mask is None:
                raise ValueError(
                    "[Dream diffusion] batch must contain input_ids, "
                    "attention_mask, position_ids, loss_mask"
                )
            if getattr(self.tokenizer, "mask_token_id", None) is None:
                raise ValueError(
                    "[Dream diffusion] tokenizer.mask_token_id is None"
                )

            labels = input_ids.contiguous()
            treat_eos_as_one = bool(getattr(self.args, "treat_eos_as_one", False))
            eos_id = getattr(self.tokenizer, "eos_token_id", None)

            x_t, t, t_mask = q_sample(
                input_ids=labels,
                maskable_mask=loss_mask.bool(),
                mask_token_id=self.tokenizer.mask_token_id,
                eos_token_id=(eos_id if treat_eos_as_one else None),
            )

            attn_bool = attention_mask_2d.bool()
            attention_mask_4d = (
                attn_bool.unsqueeze(1).unsqueeze(-2)
                & attn_bool.unsqueeze(1).unsqueeze(-1)
            ).contiguous()

            forward_kwargs = {}
            if self.supports_use_cache:
                forward_kwargs["use_cache"] = False

            outputs = self.model(
                input_ids=x_t,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                **forward_kwargs,
            )

            logits = outputs.logits.float()
            shift_logits = torch.cat(
                (logits[:, :1], logits[:, :-1]), dim=1
            )

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                labels.reshape(-1),
            )

            masked_flat = t_mask.reshape(-1).bool()
            loss = loss.masked_fill(~masked_flat, 0.0)

            if bool(getattr(self.args, "diffusion_token_reweighting", False)):
                alpha = float(getattr(self.args, "diffusion_alpha", 1.0))
                gamma = float(getattr(self.args, "diffusion_gamma", 1.0))
                loss = alpha * torch.pow(1.0 - torch.exp(-loss), gamma) * loss

            time_rw = str(
                getattr(self.args, "diffusion_time_reweighting", "cart")
            ).lower()
            cart_p = float(getattr(self.args, "diffusion_cart_p", 0.1))

            if time_rw == "cart":
                seq_len = labels.size(1)
                if self._cart_weight_matrix is None \
                        or self._cart_weight_matrix.size(0) < seq_len:
                    cache_len = max(seq_len, int(self._max_seq_len))
                    self._cart_weight_matrix = context_adaptive_reweight(
                        cache_len, cart_p=cart_p
                    )
                wm = self._cart_weight_matrix[:seq_len, :seq_len].to(
                    loss.device
                )
                non_mask = (~t_mask.bool()) & attn_bool
                ctx_weight = (
                    non_mask.to(wm.dtype).matmul(wm).masked_fill(non_mask, 0)
                )
                loss = loss * ctx_weight.reshape(-1)
            else:
                if time_rw == "original":
                    w = 1.0 / t
                elif time_rw == "linear":
                    w = 1.0 - t
                elif time_rw in {"none", "off"}:
                    w = t.new_ones(t.size())
                else:
                    raise ValueError(
                        f"Unknown diffusion_time_reweighting={time_rw}"
                    )
                w = w.view(-1, 1).float().expand(labels.size()).reshape(-1)
                loss = loss * w

            valid = masked_flat.long().sum()
            return loss.sum() / valid.clamp_min(1)

        # ---- LLaDA diffusion or AR ----
        if self.is_diffusion:
            labels = batch.pop("labels", None)
            if labels is None:
                raise ValueError(
                    "Diffusion training requires 'labels' from the collator."
                )
        else:
            labels = batch.get("labels", None)

        forward_kwargs = {}
        if self.supports_use_cache:
            forward_kwargs["use_cache"] = False

        outputs = self.model(**batch, **forward_kwargs)

        if self.is_diffusion:
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            raw_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            ).view_as(labels)
            if p_mask is not None:
                raw_loss = raw_loss / p_mask
            sample_loss = raw_loss.sum(dim=1)
            if answer_lens is not None:
                if answer_lens.device != sample_loss.device:
                    answer_lens = answer_lens.to(sample_loss.device)
                sample_loss = sample_loss / answer_lens
            return sample_loss.mean()

        # AR: use HF-provided loss when available
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    # ------------------------------------------------------------------
    #  Gradient-level CL metrics (incremental, CPU-offloaded)
    # ------------------------------------------------------------------

    def _accumulate_grads(self, dataloader, params, device, max_batches=4):
        """Average gradients over *max_batches* batches.  Returns list[Tensor] on CPU.

        Uses torch.autograd.grad (instead of loss.backward()) so that
        DeepSpeed ZeRO's AccumulateGrad hooks are never triggered.  Those
        hooks track gradient-reduction state; bypassing them keeps the
        engine in a clean state for subsequent training backward passes.
        """
        acc = [torch.zeros(p.shape, dtype=torch.float32) for p in params]
        n = 0
        none_counts = 0
        total_params = len(params)
        for batch in itertools.islice(dataloader, max_batches):
            batch.pop("sources", None)
            batch.pop("gts", None)
            batch = to_device(batch, device)

            loss = self._compute_loss(batch)
            grads = torch.autograd.grad(loss, params, allow_unused=True)

            batch_nones = 0
            for i, g in enumerate(grads):
                if g is not None:
                    acc[i].add_(g.detach().float().cpu())
                else:
                    batch_nones += 1
            none_counts += batch_nones

            del loss, batch, grads
            torch.cuda.empty_cache()
            n += 1

        if n > 0 and none_counts > 0:
            none_ratio = none_counts / (total_params * n)
            print_rank_0(
                f"[CL Diagnostics] WARNING: _accumulate_grads got None grads "
                f"for {none_counts}/{total_params * n} (param x batch) entries "
                f"({none_ratio:.1%}). Likely cause: gradient checkpointing "
                f"(use_reentrant=True) breaks the autograd graph for "
                f"parameters inside checkpointed regions.",
                self.args.global_rank,
            )
        if n > 1:
            for a in acc:
                a.div_(n)
        return acc

    def _compute_gradient_metrics_pair(
        self, old_dataloader, new_dataloader, device, max_batches=4,
        save_dir=None, pair_key="",
    ):
        """Gradient diagnostics between old-task and new-task losses.

        Gradients are offloaded to CPU and compared parameter-by-parameter so
        that GPU memory stays bounded.

        Returns a *dict* (or ``None`` on OOM) with keys:

        * ``cosine``  – global gradient cosine similarity
                        (PCGrad, Yu et al., NeurIPS 2020)
        * ``conflict_ratio`` – element-wise negative-sign fraction
        * ``mw_conflict_ratio`` – magnitude-weighted conflict ratio;
          high-magnitude conflicts count more, following the finding in
          PS-LoRA (Zhou et al., 2025) that large opposite-sign updates
          disproportionately drive forgetting
        * ``per_layer`` – ``{layer_idx: {cosine, conflict_ratio}}``
          per-layer gradient cosine, following the layer-wise analysis in
          TreeLoRA (Qian et al., ICML 2025)
        """
        hf_model = self.model.module if hasattr(self.model, "module") else self.model
        gc_was_enabled = getattr(hf_model, "is_gradient_checkpointing", False)
        try:
            self.model.train()

            if gc_was_enabled:
                hf_model.gradient_checkpointing_disable()

            named_params = [
                (n, p)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            ]
            params = [p for _, p in named_params]
            names = [n for n, _ in named_params]

            trainable_numel = sum(p.numel() for p in params)
            est_gb = trainable_numel * 4 * 2 / (1024 ** 3)
            if est_gb > 50:
                print_rank_0(
                    f"[CL Diagnostics] gradient metrics need ~{est_gb:.1f} GB "
                    f"CPU RAM ({trainable_numel / 1e9:.2f}B trainable params)",
                    self.args.global_rank,
                )

            old_grads = self._accumulate_grads(
                old_dataloader, params, device, max_batches
            )
            new_grads = self._accumulate_grads(
                new_dataloader, params, device, max_batches
            )

            # ── Global accumulators ──────────────────────────
            dot_prod = 0.0
            norm_old_sq = 0.0
            norm_new_sq = 0.0
            n_conflict = 0
            n_total = 0
            mw_conflict_num = 0.0
            mw_conflict_den = 0.0

            # ── Per-layer accumulators (TreeLoRA, ICML 2025) ─
            _layer_re = re.compile(r"layers\.(\d+)\.")
            layer_acc = {}

            for name, g_o, g_n in zip(names, old_grads, new_grads):
                prod = g_o * g_n
                abs_prod = prod.abs()
                conflict_mask = prod < 0

                dot_prod += prod.sum().item()
                norm_old_sq += g_o.pow(2).sum().item()
                norm_new_sq += g_n.pow(2).sum().item()
                n_conflict += int(conflict_mask.sum().item())
                n_total += g_o.numel()
                mw_conflict_num += abs_prod[conflict_mask].sum().item()
                mw_conflict_den += abs_prod.sum().item()

                m = _layer_re.search(name)
                if m:
                    l_idx = int(m.group(1))
                    s = layer_acc.setdefault(
                        l_idx,
                        dict(dot=0.0, no_sq=0.0, nn_sq=0.0, nc=0, nt=0),
                    )
                    s["dot"] += prod.sum().item()
                    s["no_sq"] += g_o.pow(2).sum().item()
                    s["nn_sq"] += g_n.pow(2).sum().item()
                    s["nc"] += int(conflict_mask.sum().item())
                    s["nt"] += g_o.numel()

            if save_dir and self.args.global_rank == 0:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(
                    {n: g for n, g in zip(names, old_grads)},
                    os.path.join(save_dir, f"grads_{pair_key}_old.pt"),
                )
                torch.save(
                    {n: g for n, g in zip(names, new_grads)},
                    os.path.join(save_dir, f"grads_{pair_key}_new.pt"),
                )
                print_rank_0(
                    f"[CL Intermediates] saved grads_{pair_key}_old/new.pt",
                    self.args.global_rank,
                )

            del old_grads, new_grads
            gc.collect()

            denom = max((norm_old_sq ** 0.5) * (norm_new_sq ** 0.5), 1e-12)
            per_layer = {}
            for l_idx in sorted(layer_acc):
                s = layer_acc[l_idx]
                ld = max((s["no_sq"] ** 0.5) * (s["nn_sq"] ** 0.5), 1e-12)
                per_layer[f"layer_{l_idx}"] = {
                    "cosine": s["dot"] / ld,
                    "conflict_ratio": s["nc"] / max(s["nt"], 1),
                }

            result = {
                "cosine": dot_prod / denom,
                "conflict_ratio": n_conflict / max(n_total, 1),
                "mw_conflict_ratio": (
                    mw_conflict_num / max(mw_conflict_den, 1e-30)
                ),
                "per_layer": per_layer,
            }
            return result

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                gc.collect()
                print_rank_0(
                    "[CL Diagnostics] OOM during gradient metrics – skipped.",
                    self.args.global_rank,
                )
                return None
            raise
        finally:
            if gc_was_enabled:
                hf_model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    #  Representation-drift helpers
    # ------------------------------------------------------------------

    def _prepare_probe_batches(self, device, num_batches=None):
        """Sample batches from the first eval task as fixed probe inputs.

        *num_batches* defaults to ``args.num_probe_batches`` (CLI flag) so that
        it can be tuned from shell scripts without editing Python code.
        """
        if num_batches is None:
            # argparse 注册的 --num_probe_batches default=None, 所以 getattr fallback 32 永远不命中;
            # 直接 int(None) 会 TypeError, 必须先 is None 检查. 调用方 (train_continual) 通过
            # diagnostics_enabled gate 保证只有 CLI 显式传了才进来, 但仍 defensively 兜底以防未来
            # 有人直接调用 _prepare_probe_batches(num_batches=None).
            _np = getattr(self.args, "num_probe_batches", None)
            num_batches = int(_np) if _np is not None else 32
        first_task = list(self.eval_task_list.keys())[0]
        loader = self.eval_task_list[first_task]
        probes = []
        for batch in itertools.islice(loader, num_batches):
            probe = {}
            for k in ("input_ids", "attention_mask", "position_ids"):
                if k in batch:
                    probe[k] = batch[k].clone()
            if probe:
                probes.append(probe)
        return probes

    def _extract_layer_representations(self, probe_batches, device):
        """Feed probe batches through the model and collect per-layer hidden
        states (mean-pooled over sequence).

        Returns ``{layer_idx: Tensor[N, D]}`` on CPU.
        """
        if not probe_batches:
            return {}

        raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )

        # Unwrap PEFT (LoRA/Adapter) wrapper if present
        if hasattr(raw_model, "base_model") and hasattr(raw_model.base_model, "model"):
            raw_model = raw_model.base_model.model

        # Locate transformer decoder layers
        if hasattr(raw_model, "model") and hasattr(raw_model.model, "layers"):
            layers = raw_model.model.layers
        elif hasattr(raw_model, "layers"):
            layers = raw_model.layers
        elif hasattr(raw_model, "transformer") and \
                hasattr(raw_model.transformer, "h"):
            layers = raw_model.transformer.h
        else:
            print_rank_0(
                "[CL Diagnostics] Cannot locate transformer layers – "
                "skipping representation drift.",
                self.args.global_rank,
            )
            return {}

        hooks = []
        activations: dict = {}

        for idx in range(len(layers)):
            def _hook(module, _input, output, _idx=idx):
                h = output[0] if isinstance(output, tuple) else output
                activations.setdefault(_idx, []).append(
                    h.detach().float().mean(dim=1).cpu()
                )
            hooks.append(layers[idx].register_forward_hook(_hook))

        self.model.eval()
        with torch.no_grad():
            for batch in probe_batches:
                fwd = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                fwd_kwargs = {}
                if self.supports_use_cache:
                    fwd_kwargs["use_cache"] = False
                self.model(**fwd, **fwd_kwargs)

        for h in hooks:
            h.remove()

        return {
            idx: torch.cat(acts, dim=0)
            for idx, acts in activations.items()
        }

    # ------------------------------------------------------------------
    #  Diagnostics I/O
    # ------------------------------------------------------------------

    def _save_diagnostics(self, diagnostics):
        """Persist the diagnostics dict to *<output_dir>/cl_diagnostics.json*."""
        if not getattr(self.args, "output_dir", None):
            return
        path = os.path.join(self.args.output_dir, "cl_diagnostics.json")
        with open(path, "w") as f:
            json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    def _get_intermediates_dir(self, sub=""):
        """Return ``<output_dir>/cl_intermediates[/sub]``, creating it on rank 0."""
        if not getattr(self.args, "output_dir", None):
            return None
        d = os.path.join(self.args.output_dir, "cl_intermediates")
        if sub:
            d = os.path.join(d, sub)
        if self.args.global_rank == 0:
            os.makedirs(d, exist_ok=True)
        return d

    # ------------------------------------------------------------------
    #  Evaluation and loss plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_task_name(task):
        """剥掉诊断后缀 (_shuffled / _perm / _swap), 让派生数据集复用基类 task 的 evaluator."""
        for suffix in ("_shuffled", "_perm", "_swap"):
            if task.endswith(suffix):
                return task[: -len(suffix)]
        return task

    def _eval_task(self, task, sources, predictions, ground_truths):
        """Run the task-specific evaluation function and return the result dict."""
        task = self._normalize_task_name(task)
        if task == "ScienceQA":
            return eval_ScienceQA.eval(predictions, ground_truths)
        elif task == "MeetingBank":
            return eval_MeetingBank.eval(predictions, ground_truths)
        elif task == "C-STANCE":
            return eval_CStance.eval(predictions, ground_truths)
        elif task == "Py150":
            return eval_Py150.eval(predictions, ground_truths)
        elif task == "FOMC":
            return eval_FOMC.eval(predictions, ground_truths)
        elif task == "NumGLUE-cm":
            return eval_NumGLUE_cm.eval(predictions, ground_truths)
        elif task == "NumGLUE-ds":
            return eval_NumGLUE_ds.eval(predictions, ground_truths)
        elif task == "20Minuten":
            return eval_20Minuten.eval(sources, predictions, ground_truths)
        elif task in {"qa", "qg", "sa", "sum", "trans", "dsg", "expl", "para", "pe", "pos"}:
            return eval_ssr.eval(predictions, ground_truths, task=task)
        else:
            print_rank_0(f"[Eval] Unknown task '{task}', skipping metric computation.", self.args.global_rank)
            return {}

    def _decode_predictions(self, generate_ids, batch_input_ids):
        """Decode generated token ids into text strings based on model family."""
        sequences = []
        prompt_len = batch_input_ids.shape[1]

        if self.is_diffusion and self.model_family == "dream":
            for i in range(batch_input_ids.shape[0]):
                gen_ids = generate_ids[i, prompt_len:]
                txt = self.tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                stop_list = []
                if self.tokenizer.eos_token:
                    stop_list.append(self.tokenizer.eos_token)
                stop_list += ["<|im_end|>", "<|endoftext|>"]
                for st in stop_list:
                    if st and st in txt:
                        txt = txt.split(st)[0]
                sequences.append(txt.strip())

        elif self.is_diffusion and self.model_family == "llada":
            for i in range(batch_input_ids.shape[0]):
                gen_ids = generate_ids[i, prompt_len:]
                txt = self.tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                stop_list = ["<|eot_id|>"]
                if self.tokenizer.eos_token:
                    stop_list.append(self.tokenizer.eos_token)
                for st in stop_list:
                    if st and st in txt:
                        txt = txt.split(st)[0]
                sequences.append(txt.strip())

        else:
            sequences = self.tokenizer.batch_decode(
                generate_ids[:, prompt_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        return sequences

    def _run_epoch_eval(self, task, i_task, epoch):
        """Generate predictions on eval data and compute task-specific metrics after an epoch.

        Saves results in TRACE format: results-{i_task}-{i_task}-{task}.json
        """
        infer_eval_task_list = getattr(self.args, 'infer_eval_task_list', None)
        if infer_eval_task_list is None or task not in infer_eval_task_list:
            return None

        device = self._get_device()
        eval_dataloader = infer_eval_task_list[task]

        self.model.eval()

        # --- Temporarily disable gradient checkpointing for generation ---
        hf_model = self.model.module if hasattr(self.model, "module") else self.model
        gc_was_enabled = getattr(hf_model, "is_gradient_checkpointing", False)
        old_use_cache = getattr(hf_model.config, "use_cache", None) if hasattr(hf_model, "config") else None

        if gc_was_enabled:
            hf_model.gradient_checkpointing_disable()
        # AR 推理需要 KV cache 加速; 但 LLaDA 的 modeling.py 显式 assert (past_key_values is None
        # and not use_cache), Dream 的 diffusion_generate 也自带 cache 管理。只对 AR 开 use_cache,
        # 对 diffusion 路径保持 False, 否则 LLaDA forward 立即崩在 "kvcache is not suppotred for MDM"。
        if hasattr(hf_model, "config") and hf_model.config is not None and not self.is_diffusion:
            hf_model.config.use_cache = True

        predicted_sequences = []
        sources_sequences = []
        ground_truths = []

        print_rank_0(
            f"[Eval] Starting evaluation for Task {i_task} ({task}) Epoch {epoch + 1} ...",
            self.args.global_rank,
        )

        # Per-task canvas: diffusion 走 DIFFUSION_TASK_MAX_ANS_LEN[task], AR 走 args.max_ans_len。
        # 与 Phase 2 inference (infer_single.py: resolve_task_canvas) 完全一致 -> in-training
        # eval 出的指标和 final inference 同口径。
        if self.is_diffusion:
            eval_max_new_tokens = resolve_task_canvas(task, self.args.max_ans_len)
        else:
            eval_max_new_tokens = self.args.max_ans_len

        for step, batch in enumerate(eval_dataloader):
            sources_sequences += batch.pop('sources', [])
            ground_truths += batch.pop('gts', [])

            gen_batch = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
            gen_batch = to_device(gen_batch, device)

            with torch.no_grad():
                generate_ids = self.generate(gen_batch, max_new_tokens=eval_max_new_tokens, task=task)

            decoded = self._decode_predictions(generate_ids, gen_batch['input_ids'])
            predicted_sequences += decoded

            del generate_ids, gen_batch
            torch.cuda.empty_cache()

        # --- Restore gradient checkpointing and use_cache ---
        if gc_was_enabled:
            hf_model.gradient_checkpointing_enable()
        if old_use_cache is not None and hasattr(hf_model, "config"):
            hf_model.config.use_cache = old_use_cache

        evaluation_result = self._eval_task(task, sources_sequences, predicted_sequences, ground_truths)

        if self.args.global_rank == 0 and getattr(self.args, 'output_dir', None):
            # Per-epoch dev 50 条评测产物按 task 分目录:
            #   epoch_predictions/<i_task>/results-<i>-<i>-<task>-ep<N>.json
            # 与 Phase 2 inference 完整 test 集的 predictions/ 物理隔离.
            pred_dir = os.path.join(self.args.output_dir, "epoch_predictions", str(i_task))
            os.makedirs(pred_dir, exist_ok=True)
            # 文件名带 -ep{epoch} 后缀, 防止同 task 的多个 epoch 互相覆盖
            result_file = os.path.join(
                pred_dir,
                f"results-{i_task}-{i_task}-{task}-ep{epoch}.json"
            )
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump({
                    "eval": evaluation_result,
                    "prompts": sources_sequences,
                    "results": predicted_sequences,
                    "labels": ground_truths,
                }, f, ensure_ascii=False)

            # 累计 metrics 到 epoch_eval.json (per-task per-epoch 汇总表, 用于画收敛曲线)
            summary_path = os.path.join(self.args.output_dir, "epoch_eval.json")
            summary = {}
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except (json.JSONDecodeError, OSError):
                    summary = {}
            summary.setdefault(task, {})[str(epoch)] = evaluation_result
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        print_rank_0(
            f"[Eval] Task {i_task} ({task}) Epoch {epoch + 1}: {evaluation_result}",
            self.args.global_rank,
        )

        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()
        return evaluation_result

    def _plot_single_task_loss_curve(self, task_name, losses):
        """Plot and save a single task loss curve."""
        if not losses or self.args.global_rank != 0:
            return
        if not getattr(self.args, 'output_dir', None):
            return
        try:
            matplotlib = importlib.import_module("matplotlib")
            matplotlib.use("Agg")
            plt = importlib.import_module("matplotlib.pyplot")

            fig, ax = plt.subplots(figsize=(10, 5))
            steps = list(range(len(losses)))
            ax.plot(steps, losses, linewidth=1.0, alpha=0.8)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss Curve ({task_name})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            safe_task_name = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(task_name))
            save_dir = os.path.join(self.args.output_dir, "loss_curves")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{safe_task_name}_loss.png")
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

            json_path = os.path.join(save_dir, f"{safe_task_name}_loss.json")
            with open(json_path, "w") as f:
                json.dump(losses, f)

            print_rank_0(f"[Plot] Single-task loss curve saved to {save_path}", self.args.global_rank)
        except ImportError:
            print_rank_0("[Plot] matplotlib not available, skipping single-task loss plot.", self.args.global_rank)

    def _plot_loss_curve(self, all_losses):
        """Plot training loss curve for all tasks and save to output_dir."""
        if not all_losses or self.args.global_rank != 0:
            return
        if not getattr(self.args, 'output_dir', None):
            return
        try:
            matplotlib = importlib.import_module("matplotlib")
            matplotlib.use("Agg")
            plt = importlib.import_module("matplotlib.pyplot")

            fig, ax = plt.subplots(figsize=(14, 6))
            global_step = 0
            colors = plt.cm.tab10.colors
            task_boundaries = []

            for idx, (task_name, losses) in enumerate(all_losses.items()):
                if not losses:
                    continue
                steps = list(range(global_step, global_step + len(losses)))
                color = colors[idx % len(colors)]
                ax.plot(steps, losses, color=color, label=task_name, alpha=0.7, linewidth=0.8)
                task_boundaries.append(global_step)
                global_step += len(losses)

            for bs in task_boundaries[1:]:
                ax.axvline(x=bs, color='gray', linestyle='--', alpha=0.5)

            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Curve (Continual Learning)")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(self.args.output_dir, "training_loss.png")
            fig.savefig(save_path, dpi=150)
            plt.close(fig)

            json_path = os.path.join(self.args.output_dir, "training_loss.json")
            with open(json_path, "w") as f:
                json.dump(all_losses, f)

            print_rank_0(f"[Plot] Loss curve saved to {save_path}", self.args.global_rank)
        except ImportError:
            print_rank_0("[Plot] matplotlib not available, skipping loss plot.", self.args.global_rank)

    # ------------------------------------------------------------------
    #  RNG state isolation — prevents diagnostics from altering training
    # ------------------------------------------------------------------

    def _save_rng_state(self):
        """Save torch CPU + CUDA RNG states before running diagnostics."""
        state = {'torch_cpu': torch.random.get_rng_state()}
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state()
        return state

    def _restore_rng_state(self, state):
        """Restore previously saved RNG states after diagnostics finish."""
        torch.random.set_rng_state(state['torch_cpu'])
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state(state['torch_cuda'])

    # ------------------------------------------------------------------

    def train_one_task(self, task, i_task, epochs):

        if self.args.local_rank == -1:
            # 如果 local_rank == -1，说明不是分布式训练
            device = torch.device("cuda")
            # 直接使用默认 CUDA 设备
        else:
            torch.cuda.set_device(self.args.local_rank)
            # 分布式模式下，把当前进程绑定到对应 GPU

            device = torch.device("cuda", self.args.local_rank)
            # 构造当前进程对应的 device

            self.device = device
            # 保存到 self.device 中，方便后续使用

        train_dataloader = self.train_task_list[task]

        loss_history = []

        total_steps = epochs * len(train_dataloader)

        progress_bar = tqdm(total=total_steps, leave=True, disable=(self.args.global_rank != 0))
        # 创建进度条
        # - total=总步数
        # - leave=True：结束后保留进度条
        # - disable=(global_rank != 0)：只有主进程显示进度条，其他进程不显示

        for epoch in range(epochs):
            # 开始 epoch 循环

            print_rank_0(
                f"Beginning of Epoch {epoch + 1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                self.args.global_rank
            )
            # 只在 rank 0 打印当前 epoch 开始的信息

            self.model.train()
            # 把模型切换到训练模式
            # 会启用 dropout、BN 的训练行为等

            for step, batch in enumerate(train_dataloader):
                batch.pop("sources", None)
                batch.pop("gts", None)
                batch = to_device(batch, device)

                loss = self._compute_loss(batch)

                if self.args.global_rank == 0:
                    progress_bar.update(1)

                    if getattr(self.args, "print_loss", False):
                        progress_bar.set_description(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}",
                                                     refresh=False)

                    if getattr(self.args, 'plot_loss', False):
                        loss_history.append(loss.item())

                self.model.backward(loss)
                self.model.step()

            if getattr(self.args, 'do_eval', False):
                eval_every = max(1, int(getattr(self.args, 'eval_every_n_epochs', 1) or 1))
                # epoch 是 0-indexed; (epoch+1) % N == 0 -> 1-indexed 第 N, 2N, 3N... epoch 末尾触发
                if (epoch + 1) % eval_every == 0:
                    _eval_rng = self._save_rng_state()
                    self._run_epoch_eval(task, i_task, epoch)
                    self._restore_rng_state(_eval_rng)
                    self.model.train()

        return loss_history

    def train_continual(self):
        t_all_train = time.time()
        print_rank_0(
            f"[{_now()}] ===== Continual training start =====",
            self.args.global_rank,
        )
        device = self._get_device()
        task_names = list(self.train_task_list.keys())
        diagnostics = {}
        all_losses = {}

        diagnostics_enabled = getattr(self.args, 'num_probe_batches', None) is not None

        probe_batches = []
        if diagnostics_enabled:
            probe_batches = self._prepare_probe_batches(device)
            print_rank_0(
                f"[CL Diagnostics] prepared {len(probe_batches)} probe batches "
                f"(num_probe_batches={self.args.num_probe_batches}) "
                f"for representation drift",
                self.args.global_rank,
            )

            intermediates_root = self._get_intermediates_dir()
            if intermediates_root and self.args.global_rank == 0:
                torch.save(probe_batches, os.path.join(intermediates_root, "probe_batches.pt"))
                print_rank_0(
                    f"[CL Intermediates] saved probe_batches ({len(probe_batches)} batches) "
                    f"-> {intermediates_root}/probe_batches.pt",
                    self.args.global_rank,
                )
        else:
            print_rank_0(
                "[INFO] CL diagnostics disabled (--num_probe_batches not set). "
                "Pass --num_probe_batches <N> to enable gradient/representation metrics.",
                self.args.global_rank,
            )

        # Fix 4b: Resume from last saved adapter (idempotent for fresh runs)
        # 扫描 output_dir/<round>/, 找到最大已 saved round R, 加载 LoRA 状态,
        # 跳过 task 0..R 训练循环. fresh first run 无 saved adapter -> -1 -> 不 skip.
        _resume_from = -1
        for _r in range(len(task_names)):
            _bin = os.path.join(self.args.output_dir, str(_r), "pytorch_model.bin")
            _safe = os.path.join(self.args.output_dir, str(_r), "adapter_model.safetensors")
            if os.path.isfile(_bin) or os.path.isfile(_safe):
                _resume_from = _r
        if _resume_from >= 0:
            _last_dir = os.path.join(self.args.output_dir, str(_resume_from))
            _bin = os.path.join(_last_dir, "pytorch_model.bin")
            _safe = os.path.join(_last_dir, "adapter_model.safetensors")
            try:
                if os.path.isfile(_safe):
                    from safetensors.torch import load_file
                    _state_dict = load_file(_safe)
                else:
                    _state_dict = torch.load(_bin, map_location="cpu", weights_only=False)
                _raw = self.model.module if hasattr(self.model, "module") else self.model
                _missing, _unexpected = _raw.load_state_dict(_state_dict, strict=False)
                print_rank_0(
                    f"[resume] loaded LoRA from {_last_dir} "
                    f"(missing={len(_missing)}, unexpected={len(_unexpected)}). "
                    f"Skipping tasks 0..{_resume_from}, resume from task {_resume_from + 1}.",
                    self.args.global_rank,
                )
            except Exception as _e:
                print_rank_0(
                    f"[resume] WARN: load_state_dict from {_last_dir} failed ({_e}). "
                    f"Falling back to train-from-scratch.",
                    self.args.global_rank,
                )
                _resume_from = -1

        for i_task, task in enumerate(self.train_task_list):
            if i_task <= _resume_from:
                print_rank_0(
                    f"[resume] skip task {i_task + 1}/{len(self.train_task_list)} ({task}) "
                    f"-- adapter already saved",
                    self.args.global_rank,
                )
                continue
            t_task = time.time()
            print_rank_0(
                f"[{_now()}] ----- Task {i_task + 1}/{len(self.train_task_list)} "
                f"({task}) start -----",
                self.args.global_rank,
            )

            grad_metrics = {}
            repr_before = {}

            if diagnostics_enabled:
                _diag_rng = self._save_rng_state()

                if i_task > 0:
                    try:
                        task_sub = f"task_{i_task}_{task}"
                        grad_save_dir = self._get_intermediates_dir(task_sub)
                        for prev_idx in range(i_task):
                            prev_task = task_names[prev_idx]
                            pair_key = f"{prev_task}_vs_{task}"
                            result = self._compute_gradient_metrics_pair(
                                self.train_task_list[prev_task],
                                self.train_task_list[task],
                                device,
                                save_dir=grad_save_dir,
                                pair_key=pair_key,
                            )
                            if result is not None:
                                grad_metrics[pair_key] = result
                                print_rank_0(
                                    f"[CL Diagnostics] grad pair {pair_key}: "
                                    f"cos={result['cosine']:.4f}  "
                                    f"conflict={result['conflict_ratio']:.4f}  "
                                    f"mw_conflict={result['mw_conflict_ratio']:.4f}",
                                    self.args.global_rank,
                                )
                        if grad_metrics:
                            gm_vals = list(grad_metrics.values())
                            mean_cos = sum(v["cosine"] for v in gm_vals) / len(gm_vals)
                            mean_conflict = sum(
                                v["conflict_ratio"] for v in gm_vals
                            ) / len(gm_vals)
                            mean_mw_conflict = sum(
                                v["mw_conflict_ratio"] for v in gm_vals
                            ) / len(gm_vals)
                            print_rank_0(
                                f"[CL Diagnostics] grad task {i_task} ({task}): "
                                f"mean_cos={mean_cos:.4f}  "
                                f"mean_conflict={mean_conflict:.4f}  "
                                f"mean_mw_conflict={mean_mw_conflict:.4f}  "
                                f"(pairs={len(grad_metrics)})",
                                self.args.global_rank,
                            )
                        else:
                            print_rank_0(
                                f"[CL Diagnostics] grad task {i_task} ({task}): "
                                "no valid gradient pairs.",
                                self.args.global_rank,
                            )
                    except Exception as exc:
                        print_rank_0(
                            f"[CL Diagnostics] gradient metrics failed (fatal): {exc}",
                            self.args.global_rank,
                        )
                        raise

                repr_before = self._extract_layer_representations(
                    probe_batches, device
                )

                self._restore_rng_state(_diag_rng)
                self.model.zero_grad()
                raw_tmp = self.model.module if hasattr(self.model, "module") else None
                if raw_tmp is not None:
                    raw_tmp.zero_grad()
                self.model.train()
                torch.cuda.empty_cache()

            # ================ train current task ================
            loss_history = self.train_one_task(
                task, i_task, int(self.args.num_train_epochs[i_task])
            )
            if getattr(self.args, 'plot_loss', False) and loss_history:
                task_key = f"task_{i_task}_{task}"
                all_losses[task_key] = loss_history
                self._plot_single_task_loss_curve(task_key, loss_history)

            if diagnostics_enabled:
                _diag_rng_post = self._save_rng_state()

                repr_after = self._extract_layer_representations(
                    probe_batches, device
                )

                repr_drift = {}
                cka_ub_vals = []
                cos_vals = []
                sorted_idxs = sorted(
                    set(repr_before.keys()) & set(repr_after.keys())
                )
                for layer_idx in sorted_idxs:
                    X = repr_before[layer_idx]
                    Y = repr_after[layer_idx]
                    cka_val = linear_cka(X, Y)
                    cka_ub = linear_cka_unbiased(X, Y)
                    cos_val = avg_cosine_similarity(X, Y)
                    repr_drift[f"layer_{layer_idx}"] = {
                        "cka": cka_val,
                        "cka_unbiased": cka_ub,
                        "cosine": cos_val,
                    }
                    cka_ub_vals.append(cka_ub)
                    cos_vals.append(cos_val)

                boundary_key = f"task_{i_task}_{task}"
                repr_save_dir = self._get_intermediates_dir(boundary_key)
                if repr_save_dir and self.args.global_rank == 0:
                    torch.save(repr_before, os.path.join(repr_save_dir, "repr_before.pt"))
                    torch.save(repr_after, os.path.join(repr_save_dir, "repr_after.pt"))
                    print_rank_0(
                        f"[CL Intermediates] saved repr_before/repr_after "
                        f"({len(sorted_idxs)} layers) -> {repr_save_dir}",
                        self.args.global_rank,
                    )
                del repr_before, repr_after

                n_layers = len(cka_ub_vals)
                if n_layers > 0:
                    third = max(n_layers // 3, 1)
                    repr_drift["_summary"] = {
                        "mean_cka_unbiased": sum(cka_ub_vals) / n_layers,
                        "mean_cosine": sum(cos_vals) / n_layers,
                        "shallow_cka_unbiased": (
                            sum(cka_ub_vals[:third]) / third
                        ),
                        "middle_cka_unbiased": (
                            sum(cka_ub_vals[third:-third]) / max(n_layers - 2 * third, 1)
                            if n_layers > 2 * third else
                            sum(cka_ub_vals) / n_layers
                        ),
                        "deep_cka_unbiased": (
                            sum(cka_ub_vals[-third:]) / third
                        ),
                    }

                diagnostics[boundary_key] = {
                    "grad_metrics": grad_metrics,
                    "repr_drift": repr_drift,
                }

                if self.args.global_rank == 0:
                    self._save_diagnostics(diagnostics)

                summary_str = ""
                if "_summary" in repr_drift:
                    s = repr_drift["_summary"]
                    summary_str = (
                        f"  drift: mean_cka_ub={s['mean_cka_unbiased']:.4f} "
                        f"shallow={s['shallow_cka_unbiased']:.4f} "
                        f"deep={s['deep_cka_unbiased']:.4f}"
                    )
                print_rank_0(
                    f"[CL Diagnostics] {boundary_key}: "
                    f"repr_layers={len(sorted_idxs)}, "
                    f"grad_pairs={len(grad_metrics)}"
                    f"{summary_str}",
                    self.args.global_rank,
                )

                self._restore_rng_state(_diag_rng_post)

            # ===== Task boundary cleanup =====
            self.model.zero_grad()
            raw = self.model.module if hasattr(self.model, "module") else None
            if raw is not None:
                raw.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

            self.save_model(i_task)

            torch.cuda.empty_cache()
            gc.collect()

            print_rank_0(
                f"[{_now()}] Task {i_task + 1} ({task}) done in "
                f"{_fmt_elapsed(time.time() - t_task)}",
                self.args.global_rank,
            )

        if getattr(self.args, 'plot_loss', False):
            self._plot_loss_curve(all_losses)

        print_rank_0(
            f"[{_now()}] ===== Continual training finished in "
            f"{_fmt_elapsed(time.time() - t_all_train)} =====",
            self.args.global_rank,
        )

    def save_model(self, round):
        # 保存当前轮次/任务结束后的模型
        # 参数 round 表示第几轮任务

        if self.args.output_dir is not None:
            # 如果指定了输出目录
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)
            # 只在主进程打印保存路径

        if self.args.global_rank == 0:
            # 只有主进程负责保存标准 HF 格式模型
            save_hf_format(self.model, self.tokenizer, self.args, sub_folder=str(round))
            # 保存 HuggingFace 格式模型和 tokenizer
            # 子目录名使用当前 round，例如 0、1、2 ...

        if self.args.zero_stage == 3:
            # 如果当前使用的是 DeepSpeed ZeRO-3
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
            # 调用 ZeRO-3 专用保存函数
            # 因为 ZeRO-3 下参数被分片，普通保存方式不够

        print_rank_0('Sucessful saving model after round {}'.format(round), self.args.global_rank)
        # 打印保存成功信息（只在 rank 0 输出）