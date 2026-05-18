import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
#  LLaDA-8B-Instruct special token IDs
#  Mirrors the hard-coded constants in official ML-GSAI/LLaDA generate.py.
#  Source: huggingface.co/GSAI-ML/LLaDA-8B-Instruct/tokenizer_config.json
# ---------------------------------------------------------------------------
_LLADA_EOS_TOKEN_ID = 126081  # <|endoftext|>
_LLADA_EOT_TOKEN_ID = 126348  # <|eot_id|>


# ---------------------------------------------------------------------------
#  Shared helpers
# ---------------------------------------------------------------------------

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise.clamp_min(1e-10))) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    dtype = torch.long
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode='floor')
    rem = total - base * steps
    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).to(dtype).clone()
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    num_transfer_tokens = num_transfer_tokens + add_mask.to(dtype)
    return num_transfer_tokens


def _get_llada_mask_id(model, tokenizer) -> int:
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)
    if getattr(model.config, "mask_token_id", None) is not None:
        return int(model.config.mask_token_id)
    raise ValueError(
        f"Critical Error: Could not find a valid 'mask_token_id' in tokenizer or model config. "
        f"Tokenizer attributes: {tokenizer.special_tokens_map}"
    )


def _logsumexp_f64_chunked(logits, chunk_size=16384):
    """logsumexp along dim=-1 in float64, chunked over the vocab dimension
    to avoid materialising the full [B, S, V] tensor in float64."""
    V = logits.shape[-1]
    out_shape = logits.shape[:-1]
    device = logits.device

    max_val = torch.full(out_shape, torch.finfo(torch.float64).min,
                         dtype=torch.float64, device=device)
    for vi in range(0, V, chunk_size):
        chunk = logits[..., vi:vi + chunk_size].to(torch.float64)
        torch.maximum(max_val, chunk.max(dim=-1).values, out=max_val)
        del chunk

    sum_exp = torch.zeros(out_shape, dtype=torch.float64, device=device)
    for vi in range(0, V, chunk_size):
        chunk = logits[..., vi:vi + chunk_size].to(torch.float64)
        sum_exp += chunk.sub_(max_val.unsqueeze(-1)).exp_().sum(dim=-1)
        del chunk

    return max_val + sum_exp.log()


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens,
    threshold: float = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
):
    """Vectorized token selection -- replaces the per-batch Python loop."""
    # Block A — port of official `if logits_eos_inf: logits[:, :, 126081] = -torch.inf`.
    # Applied BEFORE add_gumbel_noise so that argmax cannot pick EOS.
    if logits_eos_inf:
        logits[:, :, _LLADA_EOS_TOKEN_ID] = -torch.inf

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    del logits_with_noise

    # Block B — port of the *effective* half of official's chained assignment
    # `logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf`.
    # Per Python's left-to-right assignment, the first write hits logits_with_noise
    # which is post-argmax + del'd here -> no-op. The second write hits logits
    # before the confidence path (logsumexp/softmax), zeroing EoT confidence.
    if confidence_eos_eot_inf:
        logits[:, :, _LLADA_EOT_TOKEN_ID] = -torch.inf

    if remasking == "low_confidence":
        # softmax(logits)[x0] == exp(logits[x0] - logsumexp(logits))
        # Uses chunked float64 logsumexp to avoid full [B, S, V] float64 allocation.
        x0_logits = torch.gather(logits, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(torch.float64)
        lse = _logsumexp_f64_chunked(logits)
        x0_p = (x0_logits - lse).exp()
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided when threshold is None.")

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device)
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    B, L = confidence.shape
    _values, idx = torch.sort(confidence, dim=1, descending=True)
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


# =========================================================================
#  Original llada_generate  (unchanged, kept as baseline / fallback)
# =========================================================================

@torch.no_grad()
def llada_generate(
        model,
        tokenizer,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        max_new_tokens: int = 128,
        steps: int = 64,
        temperature: float = 0.0,
        block_length: int = 128,
        remasking: str = 'low_confidence',
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
):
    device = prompt_input_ids.device
    B, P = prompt_input_ids.shape
    gen_length = max_new_tokens

    mask_id = _get_llada_mask_id(model, tokenizer)

    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_input_ids.clone()

    full_attention_mask = None
    if attention_mask is not None:
        gen_mask = torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=-1)

    if gen_length % block_length != 0:
        block_length = gen_length

    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks
        if steps == 0:
            steps = num_blocks

    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        start_idx = P + num_block * block_length
        end_idx = P + (num_block + 1) * block_length

        block_mask_index = (x[:, start_idx:end_idx] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)

            outputs = model(x, attention_mask=full_attention_mask)
            logits = outputs.logits

            # Block A — port of official `if logits_eos_inf: logits[:, :, 126081] = -torch.inf`.
            if logits_eos_inf:
                logits[:, :, _LLADA_EOS_TOKEN_ID] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            del logits_with_noise

            # Block B — port of the effective half of official's chained assignment
            # `logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf`.
            # First write is post-argmax + logits_with_noise just del'd -> no-op in official too;
            # only `logits[:, :, 126348] = -torch.inf` actually flows into the confidence path.
            if confidence_eos_eot_inf:
                logits[:, :, _LLADA_EOT_TOKEN_ID] = -torch.inf

            if remasking == 'low_confidence':
                x0_logits = torch.gather(logits, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1).to(torch.float64)
                lse = _logsumexp_f64_chunked(logits)
                x0_p = (x0_logits - lse).exp()
            elif remasking == 'random':
                x0_p = torch.rand((B, x.shape[1]), device=device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, end_idx:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)

            for j in range(B):
                current_k = num_transfer_tokens[j, i].item()
                if current_k > 0:
                    _, select_index = torch.topk(confidence[j], k=current_k)
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

    return x


# =========================================================================
#  Fast-dLLM Optimization 1: KV Cache for Block-Wise Decoding
#  Reference: Wu et al., "Fast-dLLM", ICLR 2026
# =========================================================================

@torch.no_grad()
def llada_generate_with_cache(
        model,
        tokenizer,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        max_new_tokens: int = 128,
        steps: int = 64,
        temperature: float = 0.0,
        block_length: int = 128,
        remasking: str = 'low_confidence',
        threshold: float = None,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
):
    """LLaDA generation with prefix KV cache reuse across denoising steps.

    At each block, the first forward caches KV for the prompt prefix.
    Subsequent steps only process tokens from the current block onward,
    reusing the cached prefix KV -- reducing per-step cost from
    O(prompt+gen_length) to O(current_block + future_blocks).
    """
    device = prompt_input_ids.device
    B, P = prompt_input_ids.shape
    gen_length = max_new_tokens
    mask_id = _get_llada_mask_id(model, tokenizer)

    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_input_ids.clone()

    full_attention_mask = None
    if attention_mask is not None:
        gen_mask = torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=-1)

    if gen_length % block_length != 0:
        block_length = gen_length
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks
        if steps == 0:
            steps = num_blocks
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        current_block_start = P + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Step 0: full forward to build KV cache
        output = model(x, attention_mask=full_attention_mask, use_cache=True)
        past_key_values = output.past_key_values
        logits = output.logits
        del output

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = False
        x0, transfer_index = _get_transfer_index(
            logits, temperature, remasking, mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None,
            threshold,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )
        del logits
        x[transfer_index] = x0[transfer_index]

        # Trim KV cache to only the prefix (before current block)
        new_past_key_values = []
        for layer_kv in past_key_values:
            trimmed_layer = ()
            for tensor in layer_kv:
                trimmed_layer += (tensor[:, :, :current_block_start, ...],)
            new_past_key_values.append(trimmed_layer)
        past_key_values = new_past_key_values

        # Steps 1..N: only process current_block_start onward, with cached prefix
        # Matches Fast-dLLM generate_with_prefix_cache while-loop structure
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

            # With past_key_values, HF expects attention_mask of shape
            # [B, past_len + current_len]. Since past_len = current_block_start
            # and current_len = total_len - current_block_start, the full
            # attention_mask (shape [B, total_len]) is correct.
            cache_out = model(
                x[:, current_block_start:],
                attention_mask=full_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = cache_out.logits
            del cache_out

            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = False

            x0, transfer_index = _get_transfer_index(
                logits, temperature, remasking, mask_index,
                x[:, current_block_start:],
                num_transfer_tokens[:, i] if (threshold is None and i < steps_per_block) else None,
                threshold,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
            )
            del logits
            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            i += 1
            if threshold is None and i >= steps_per_block:
                break

    return x


# =========================================================================
#  Fast-dLLM Optimization 2: Dual Cache (prefix + suffix)
#  Reference: Wu et al., "Fast-dLLM", ICLR 2026
# =========================================================================

@torch.no_grad()
def llada_generate_with_dual_cache(
        model,
        tokenizer,
        prompt_input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        max_new_tokens: int = 128,
        steps: int = 64,
        temperature: float = 0.0,
        block_length: int = 128,
        remasking: str = 'low_confidence',
        threshold: float = None,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
):
    """LLaDA generation with dual KV cache (prefix + suffix).

    In addition to caching the prefix, also caches the masked suffix tokens.
    Only the current block is re-computed at each step, achieving greater
    speedup with negligible accuracy loss.
    """
    device = prompt_input_ids.device
    B, P = prompt_input_ids.shape
    gen_length = max_new_tokens
    mask_id = _get_llada_mask_id(model, tokenizer)

    x = torch.full((B, P + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_input_ids.clone()

    full_attention_mask = None
    if attention_mask is not None:
        gen_mask = torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)
        full_attention_mask = torch.cat([attention_mask, gen_mask], dim=-1)

    if gen_length % block_length != 0:
        block_length = gen_length
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        steps = (steps // num_blocks) * num_blocks
        if steps == 0:
            steps = num_blocks
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        s = P + num_block * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # Step 0: full forward to build KV cache
        output = model(x, attention_mask=full_attention_mask, use_cache=True)
        past_key_values = output.past_key_values

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        global_mask_index = (x == mask_id)
        global_mask_index[:, e:] = False

        logits = output.logits
        del output

        x0, transfer_index = _get_transfer_index(
            logits, temperature, remasking, global_mask_index, x,
            num_transfer_tokens[:, 0] if threshold is None else None,
            threshold,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
        )
        del logits
        x = torch.where(transfer_index, x0, x)

        # Steps 1..N: only process current block with dual cache
        # Matches Fast-dLLM generate_with_dual_cache inner loop
        for i in range(1, steps_per_block):
            if (x[:, s:e] == mask_id).sum() == 0:
                break

            cache_out = model(
                x[:, s:e],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
            )

            logits = cache_out.logits
            del cache_out
            mask_blk = (x[:, s:e] == mask_id)

            x0_blk, transfer_idx_blk = _get_transfer_index(
                logits, temperature, remasking, mask_blk, x[:, s:e],
                num_transfer_tokens[:, i] if threshold is None else None,
                threshold,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
            )
            del logits
            blk_new = torch.where(transfer_idx_blk, x0_blk, x[:, s:e])
            x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

    return x
