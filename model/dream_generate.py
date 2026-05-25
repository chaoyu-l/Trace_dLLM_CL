"""
Fast-dLLM acceleration for Dream diffusion language model.

Implements block-wise KV cache reuse and confidence-aware parallel decoding,
adapted from NVlabs/Fast-dLLM (Wu et al., ICLR 2026).

Two public functions:
    dream_generate            -- standalone baseline (matches Dream._sample)
    dream_generate_with_cache -- block-wise KV cache + adaptive unmasking

Optional EOS/PAD penalty (Dream-Coder §4.1 padding penalty, arXiv:2509.01142):
    logits[..., pad_token_id] += eos_penalty * log(1 - t + eps)
    where t goes from 1 -> eps linearly, so the penalty starts strong
    (~ eos_penalty * log(eps)) and anneals to 0 in the last step.
"""

import torch
import torch.nn.functional as F
import torch.distributions as dists
from typing import Optional
from transformers.cache_utils import DynamicCache


def _apply_pad_penalty(logits: torch.Tensor,
                       pad_token_id: int,
                       eos_penalty: float,
                       t: torch.Tensor,
                       eps: float) -> torch.Tensor:
    """In-place Dream-Coder padding penalty on the pad column of `logits`.

    Negative bias on the pad/EOS column that anneals to 0 as t -> eps,
    so the model is discouraged from collapsing to PAD early in denoising
    but is free to terminate naturally near the end.
    """
    bias = float(eos_penalty) * torch.log(1.0 - t + eps)
    logits[..., pad_token_id] = logits[..., pad_token_id] + bias
    return logits


def make_dream_eos_penalty_hook(eos_penalty: float,
                                pad_token_id: int,
                                steps: int,
                                eps: float = 1e-3):
    """Build a `generation_logits_hook_func` for Dream's stock `_sample`.

    Mirrors the schedule used inside `dream_generate` (timesteps =
    linspace(1, eps, steps + 1), penalty = eos_penalty * log(1 - t + eps)),
    so the no-cache path (Dream HF `diffusion_generate`) and the fast-cache
    path produce the same penalty trajectory.
    """
    timesteps = torch.linspace(1.0, eps, steps + 1)

    def hook(step, x, logits):
        if eos_penalty == 0.0 or pad_token_id is None:
            return logits
        idx = min(int(step), len(timesteps) - 2)
        t = timesteps[idx].to(logits.device)
        return _apply_pad_penalty(logits, pad_token_id, eos_penalty, t, eps)

    return hook


# ---------------------------------------------------------------------------
#  Sampling helpers (match Dream generation_utils.py exactly)
# ---------------------------------------------------------------------------

def _top_p_logits(logits, top_p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits, top_k):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def _sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                   margin_confidence=False, neg_entropy=False):
    """Sample tokens and return (confidence, x0).  Matches Dream exactly."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _get_mask_token_id(model):
    """Resolve mask_token_id from a (possibly PEFT-wrapped) Dream model."""
    for obj in (model, getattr(model, "base_model", None)):
        if obj is None:
            continue
        for attr in ("generation_config", "config"):
            cfg = getattr(obj, attr, None)
            if cfg is not None and getattr(cfg, "mask_token_id", None) is not None:
                return int(cfg.mask_token_id)
    raise ValueError("Cannot find mask_token_id in Dream model config")


def _prepare_dream_inputs(input_ids, attention_mask, max_new_tokens, mask_token_id):
    """Pad input_ids with [MASK] and build 4D bidirectional attention mask.

    Returns (x, attn_mask_4d, tok_idx) matching Dream's _sample preprocessing.
    """
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attn_padded = F.pad(attention_mask, (0, max_new_tokens), value=1.0)
        tok_idx = attn_padded.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attn_padded == 0, 1)
        attn_mask_4d = torch.logical_and(
            attn_padded.unsqueeze(1).unsqueeze(-2),
            attn_padded.unsqueeze(1).unsqueeze(-1),
        )
    else:
        tok_idx = None
        attn_mask_4d = None

    return x, attn_mask_4d, tok_idx


# ---------------------------------------------------------------------------
#  Denoising step (used by baseline dream_generate only)
# ---------------------------------------------------------------------------

def _denoise_step(x, gen_logits, mask_index, prompt_len,
                  mask_token_id, i, steps, t, s,
                  temperature, top_p, top_k, alg, alg_temp, threshold):
    """Apply one Dream denoising step.  Modifies *x* in-place."""
    device = x.device
    B, total_len = x.shape
    gen_mask = mask_index[:, prompt_len:]
    mask_logits = gen_logits[gen_mask]

    if mask_logits.shape[0] == 0:
        return

    if threshold is not None:
        confidence, x0 = _sample_tokens(
            mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
        )
        reveal = confidence >= threshold
        if not reveal.any():
            reveal[confidence.argmax()] = True
        new_vals = torch.full((mask_logits.shape[0],), mask_token_id,
                              device=device, dtype=torch.long)
        new_vals[reveal] = x0[reveal]
        x[mask_index] = new_vals
        return

    if alg == "origin":
        p_transfer = (1 - s / t) if i < steps - 1 else 1
        x0 = torch.full((mask_logits.shape[0],), mask_token_id,
                         device=device, dtype=torch.long)
        transfer = torch.rand(x0.shape, device=device) < p_transfer
        if transfer.any():
            _, x0[transfer] = _sample_tokens(
                mask_logits[transfer],
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
        x[mask_index] = x0.clone()
        return

    kw = dict(temperature=temperature, top_p=top_p, top_k=top_k)
    if alg == "topk_margin":
        kw["margin_confidence"] = True
    elif alg == "entropy":
        kw["neg_entropy"] = True
    elif alg != "maskgit_plus":
        raise RuntimeError(f"Unknown alg: {alg}")

    confidence, x0 = _sample_tokens(mask_logits, **kw)

    num_mask_token = mask_index.sum() / mask_index.shape[0]
    n_transfer = (int(num_mask_token * (1 - s / t))
                  if i < steps - 1
                  else int(num_mask_token))

    full_conf = torch.full((B, total_len), -torch.inf,
                           device=device, dtype=gen_logits.dtype)
    full_conf[mask_index] = confidence

    if n_transfer > 0:
        if alg_temp is None or alg_temp == 0:
            _, tidx = torch.topk(full_conf, n_transfer)
        else:
            fc = F.softmax(full_conf / alg_temp, dim=-1)
            tidx = torch.multinomial(fc, num_samples=n_transfer)

        x_ = torch.full_like(x, mask_token_id)
        x_[mask_index] = x0.clone()
        ridx = torch.arange(B, device=device).unsqueeze(1).expand_as(tidx)
        x[ridx, tidx] = x_[ridx, tidx]


# =========================================================================
#  Baseline Dream generation (standalone, matches Dream._sample exactly)
# =========================================================================

@torch.no_grad()
def dream_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: int = 128,
    steps: int = 64,
    temperature: float = 0.0,
    alg: str = "origin",
    alg_temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    eps: float = 1e-3,
    threshold: Optional[float] = None,
    eos_penalty: float = 0.0,
    pad_token_id: Optional[int] = None,
):
    """Dream diffusion generation -- standalone version of ``Dream._sample``.

    Returns
    -------
    x : LongTensor [B, prompt_len + max_new_tokens]
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    mask_token_id = _get_mask_token_id(model)

    x, attn_mask_4d, tok_idx = _prepare_dream_inputs(
        input_ids, attention_mask, max_new_tokens, mask_token_id,
    )

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    use_pad_penalty = (eos_penalty != 0.0) and (pad_token_id is not None)

    for i in range(steps):
        mask_index = (x == mask_token_id)
        if mask_index.sum() == 0:
            break

        logits = model(
            input_ids=x,
            attention_mask=attn_mask_4d,
            position_ids=tok_idx,
        ).logits

        shifted = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if use_pad_penalty:
            _apply_pad_penalty(shifted, pad_token_id, eos_penalty, timesteps[i], eps)
        gen_logits = shifted[:, prompt_len:]

        _denoise_step(
            x, gen_logits, mask_index, prompt_len,
            mask_token_id, i, steps, timesteps[i], timesteps[i + 1],
            temperature, top_p, top_k, alg, alg_temp, threshold,
        )

    return x


# =========================================================================
#  Fast-dLLM: Block-wise KV cache + confidence-aware parallel decoding
#  Reference: NVlabs/Fast-dLLM, generation_utils_block.py
# =========================================================================

@torch.no_grad()
def dream_generate_with_cache(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    max_new_tokens: int = 128,
    steps: int = 64,
    temperature: float = 0.0,
    alg: str = "origin",
    alg_temp: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    eps: float = 1e-3,
    threshold: Optional[float] = None,
    block_length: Optional[int] = None,
    eos_penalty: float = 0.0,
    pad_token_id: Optional[int] = None,
):
    """Dream generation with block-wise KV cache (Fast-dLLM).

    The generation canvas is divided into blocks of ``block_length`` tokens.
    Each block is processed with ``steps // num_blocks`` denoising steps:

    **Block init**: Full forward over the entire sequence.  The first token
    of the current block is immediately revealed.  KV states for positions
    before the block (the *prefix*) are cached.

    **Block steps 1..N**: Only positions from ``block_start`` onward are
    forwarded, reusing the cached prefix KV.  Unmasking is restricted to
    the current block.

    When ``threshold`` is set, confidence-aware parallel decoding is used:
    tokens above ``threshold`` are revealed each iteration until the block
    is fully unmasked (variable number of steps).  Otherwise, fixed-step
    denoising with entropy-based confidence ranking is used.

    Returns
    -------
    x : LongTensor [B, prompt_len + max_new_tokens]
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    mask_token_id = _get_mask_token_id(model)

    if prompt_len < 2:
        return dream_generate(
            model, input_ids, attention_mask, max_new_tokens, steps,
            temperature, alg, alg_temp, top_p, top_k, eps, threshold,
            eos_penalty=eos_penalty, pad_token_id=pad_token_id,
        )

    gen_length = max_new_tokens

    # --- Block configuration (match generation_utils_block.py) ---
    if block_length is None or block_length >= gen_length:
        block_length = gen_length
    if gen_length % block_length != 0:
        block_length = gen_length
    num_blocks = gen_length // block_length
    if steps % num_blocks != 0:
        block_length = gen_length
        num_blocks = 1
    steps_per_block = steps // num_blocks

    x, attn_mask_4d, tok_idx = _prepare_dream_inputs(
        input_ids, attention_mask, max_new_tokens, mask_token_id,
    )
    total_len = x.shape[1]

    timesteps = torch.linspace(1, eps, steps_per_block + 1, device=device)
    use_threshold = (threshold is not None)
    use_pad_penalty = (eos_penalty != 0.0) and (pad_token_id is not None)

    for blk in range(num_blocks):
        block_start = prompt_len + blk * block_length
        block_end = block_start + block_length

        # ---- Block init: full forward, reveal first token, build cache ----
        cache = DynamicCache()
        output = model(
            input_ids=x,
            attention_mask=attn_mask_4d,
            position_ids=tok_idx,
            past_key_values=cache,
            use_cache=True,
        )
        logits = output.logits
        # Only need the single slice; avoid allocating a full shifted copy.
        block_start_logits = logits[:, block_start - 1, :]       # [B, V]
        if use_pad_penalty:
            _apply_pad_penalty(block_start_logits, pad_token_id, eos_penalty, timesteps[0], eps)
        _, first_x0 = _sample_tokens(
            block_start_logits, temperature=temperature,
            top_p=top_p, top_k=top_k,
        )
        x[:, block_start] = first_x0

        num_layers = len(cache.key_cache)
        prefix_keys = [
            cache.key_cache[l][:, :, :block_start, :].clone()
            for l in range(num_layers)
        ]
        prefix_values = [
            cache.value_cache[l][:, :, :block_start, :].clone()
            for l in range(num_layers)
        ]
        del cache, output, logits

        suffix_attn = (attn_mask_4d[:, :, block_start:, :]
                       if attn_mask_4d is not None else None)
        suffix_pos = (tok_idx[:, block_start:]
                      if tok_idx is not None else None)

        # ---- Block denoising loop (cached forward, steps 1+) ----
        i = 1
        while True:
            if (x[:, block_start:block_end] == mask_token_id).sum() == 0:
                break
            if not use_threshold and i >= steps_per_block:
                break

            step_cache = DynamicCache()
            step_cache.key_cache = list(prefix_keys)
            step_cache.value_cache = list(prefix_values)

            output = model(
                input_ids=x[:, block_start:],
                attention_mask=suffix_attn,
                position_ids=suffix_pos,
                past_key_values=step_cache,
                use_cache=True,
            )
            local_logits = output.logits
            local_shifted = torch.empty_like(local_logits)
            local_shifted[:, 0] = local_logits[:, 0]
            local_shifted[:, 1:] = local_logits[:, :-1]
            del step_cache, output, local_logits

            if use_pad_penalty:
                t_idx = min(i, steps_per_block - 1)
                _apply_pad_penalty(local_shifted, pad_token_id, eos_penalty, timesteps[t_idx], eps)

            suffix_len = total_len - block_start

            if use_threshold:
                # --- Confidence-aware parallel decoding ---
                # (generation_utils_block.py confidence_threshold branch)
                mask_index = (x[:, block_start:] == mask_token_id)
                mask_logits = local_shifted[mask_index]

                if mask_logits.shape[0] == 0:
                    del local_shifted
                    break

                confidence, x0 = _sample_tokens(
                    mask_logits, temperature=temperature,
                    top_p=top_p, top_k=top_k,
                )

                x_ = torch.full((B, suffix_len), mask_token_id,
                                device=device, dtype=torch.long)
                full_confidence = torch.full(
                    (B, suffix_len), -torch.inf,
                    device=device, dtype=local_shifted.dtype,
                )
                x_[mask_index] = x0.clone()
                full_confidence[mask_index] = confidence
                full_confidence[:, block_length:] = -torch.inf

                n_candidates = (
                    x[:, block_start:block_end] == mask_token_id
                ).sum().item()
                if n_candidates == 0:
                    del local_shifted
                    break

                k_select = min(n_candidates, full_confidence.shape[1])
                selected_conf, select_idx = torch.topk(
                    full_confidence, k_select,
                )

                transfer = torch.zeros(
                    (B, suffix_len), device=device, dtype=torch.bool,
                )
                for b in range(B):
                    transfer[b, select_idx[b, 0]] = True
                    for k in range(1, k_select):
                        if selected_conf[b, k] >= threshold:
                            transfer[b, select_idx[b, k]] = True

                x[:, block_start:][transfer] = x_[transfer]
            else:
                # --- Fixed-step denoising with entropy confidence ---
                # (generation_utils_block.py non-threshold branch)
                t = timesteps[i]
                s = timesteps[i + 1]

                mask_index = (x[:, block_start:] == mask_token_id)
                mask_index[:, block_length:] = False

                mask_logits = local_shifted[mask_index]
                if mask_logits.shape[0] == 0:
                    del local_shifted
                    break

                confidence, x0 = _sample_tokens(
                    mask_logits, temperature=temperature,
                    top_p=top_p, top_k=top_k, neg_entropy=True,
                )

                num_mask = mask_index.sum() / B
                n_transfer = (int(num_mask * (1 - s / t))
                              if i < steps_per_block - 1
                              else int(num_mask))

                full_conf = torch.full(
                    (B, suffix_len), -torch.inf,
                    device=device, dtype=local_shifted.dtype,
                )
                full_conf[mask_index] = confidence
                full_conf[:, block_length:] = -torch.inf

                if n_transfer > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, tidx = torch.topk(full_conf, n_transfer)
                    else:
                        fc = F.softmax(full_conf / alg_temp, dim=-1)
                        tidx = torch.multinomial(fc, num_samples=n_transfer)

                    x_ = torch.full(
                        (B, suffix_len), mask_token_id,
                        device=device, dtype=torch.long,
                    )
                    x_[mask_index] = x0.clone()
                    ridx = torch.arange(B, device=device) \
                        .unsqueeze(1).expand_as(tidx)
                    x[:, block_start:][ridx, tidx] = x_[ridx, tidx]

            del local_shifted
            i += 1

    return x
