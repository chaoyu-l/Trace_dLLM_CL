# utils/diffusion/dream_loss.py
import torch
import torch.nn.functional as F

from utils.diffusion.dream_gen_utils import (
    q_sample,
    token_reweight_loss,
    context_adaptive_reweight,
)


def compute_dream_diffusion_loss(
    model,
    batch,
    *,
    vocab_size: int,
    mask_token_id: int,
    pad_token_id: int,
    eos_token_id: int,
    time_reweighting: str = "linear",     # match official default config: "linear"
    token_reweighting: bool = False,
    alpha: float = 1.0,                  # match Dream config diffusion.alpha
    gamma: float = 2.0,                  # match Dream config diffusion.gamma
    cart_p: float = 0.8,                 # match Dream config diffusion.cart_p
    treat_eos_as_one: bool = False,      # match trainer default behavior (False unless你明确开启)
):
    """
    Align to Dream-main/src/trainer/fsdp_sft_trainer.py diffusion loss path.

    Batch must contain:
      input_ids      [B, L]
      attention_mask [B, L]
      position_ids   [B, L]
      loss_mask      [B, L]   (prompt区域0，其它1)  —— 这里要求是 bool 或 0/1 都行
    """
    input_ids = batch["input_ids"]
    attention_mask_2d = batch["attention_mask"]
    position_ids = batch["position_ids"]
    loss_mask = batch["loss_mask"]

    batch_size, seq_len = input_ids.shape

    # attention_mask: 2d -> 4d（与官方一致）
    attn_bool = attention_mask_2d.bool()
    attention_mask_4d = torch.logical_and(
        attn_bool.unsqueeze(1).unsqueeze(-2),
        attn_bool.unsqueeze(1).unsqueeze(-1),
    )

    # diffusion corruption（与官方一致：maskable_mask = loss_mask）
    masked_input_ids, t, loss_mask_nonflatten = q_sample(
        input_ids,
        maskable_mask=loss_mask.bool(),
        mask_token_id=mask_token_id,
        eos_token_id=(eos_token_id if treat_eos_as_one else None),
    )

    # forward（官方 use_cache=False）
    outputs = model(
        input_ids=masked_input_ids,
        attention_mask=attention_mask_4d,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = outputs.logits  # [B, L, V]

    # official "shift": shift_logits = [logits[:,0], logits[:,:-1]]
    shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
    shift_labels = input_ids.contiguous()

    # flatten
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(shift_logits.device)

    # CE per-token (reduction='none') —— 官方是 CrossEntropyLoss(reduction='none')
    ce_flat = F.cross_entropy(shift_logits, shift_labels, reduction="none")  # [B*L]

    # loss_mask flatten：注意这里是 diffusion 的 “被mask位置” (u < t) & maskable_mask
    loss_mask_flat = loss_mask_nonflatten.reshape(-1).to(ce_flat.device)

    # masked_fill(~mask, 0) —— 与官方一致（不直接 index 取子集）
    loss_flat = ce_flat.masked_fill(~loss_mask_flat, 0.0)

    # token reweighting（官方的 focal-like 变换）
    if token_reweighting:
        loss_flat = token_reweight_loss(loss_flat, alpha=alpha, gamma=gamma)

    # time reweighting（与官方一致）
    if time_reweighting == "original":
        weight = 1.0 / t[:, None].float().expand(batch_size, seq_len)
    elif time_reweighting == "linear":
        weight = 1.0 - t[:, None].float().expand(batch_size, seq_len)
    elif time_reweighting == "cart":
        weight_matrix = context_adaptive_reweight(
            seq_len, cart_p=cart_p, device=loss_flat.device, dtype=loss_flat.dtype
        )  # [L, L]
        non_mask = ~loss_mask_nonflatten.to(loss_flat.device)  # True: not masked positions
        weight = (
            non_mask.type_as(weight_matrix)
            .matmul(weight_matrix)
            .masked_fill(non_mask, 0)
        )  # [B, L]
    else:
        weight = t.new_ones((batch_size, 1)).float().expand(batch_size, seq_len)

    # apply time weights（flatten 对齐）
    loss_flat = loss_flat * weight.reshape(-1)

    denom = torch.sum(loss_mask_flat).to(loss_flat.dtype)
    loss = torch.sum(loss_flat) / denom

    stats = {
        "denom_masked_tokens": int(denom.item()),
        "t_mean": float(t.mean().item()),
    }
    return loss, stats
