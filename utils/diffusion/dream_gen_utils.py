# utils/diffusion/dream_gen_utils.py
import torch


def q_sample(
    input_ids,
    maskable_mask,
    mask_token_id,
    t_min=0.0,
    t_max=1.0,
    eos_token_id=None,
    t=None,
    t_mask=None,
):
    """
    Match Dream-main/src/diffllm/gen_utils.py::q_sample
    """
    x_0 = input_ids

    if t_mask is None:
        if t is None:
            t = torch.rand((x_0.shape[0],), dtype=torch.float, device=input_ids.device)
            t = t_min + (t_max - t_min) * t
        u = torch.rand_like(x_0, dtype=torch.float)
        t_mask = (u < t[:, None]) & maskable_mask

    x_t = x_0.masked_fill(t_mask, mask_token_id)

    # NOTE: 官方只有在 eos_token_id != None 时才做这段特殊处理（通常 treat_eos_as_one 才开）
    if eos_token_id is not None:
        last_non_eos_token_idx = ((input_ids != eos_token_id) | (~maskable_mask)).sum(dim=-1) - 1
        seq_len = x_0.shape[1]
        for i in range(x_0.shape[0]):
            if last_non_eos_token_idx[i] < seq_len - 1:
                t_mask_at_eos = t_mask[i, last_non_eos_token_idx[i] + 1]
                if t_mask_at_eos:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = mask_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = True
                else:
                    x_t[i, last_non_eos_token_idx[i] + 1 :] = eos_token_id
                    t_mask[i, last_non_eos_token_idx[i] + 1 :] = False

    return x_t, t, t_mask


def token_reweight_loss(loss_flat: torch.Tensor, *, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    """
    Match Dream-main/src/trainer/fsdp_sft_trainer.py token reweighting:

        loss = alpha * (1 - exp(-loss))**gamma * loss

    Inputs:
      loss_flat: shape [B*L], already masked_fill(~loss_mask, 0)
    """
    return alpha * (1.0 - torch.exp(-loss_flat)).pow(gamma) * loss_flat


def context_adaptive_reweight(seq_len: int, cart_p: float = 0.8, device=None, dtype=None) -> torch.Tensor:
    """
    Match Dream-main/src/trainer/fsdp_sft_trainer.py::context_adaptive_reweight

        w(d) = 0                                  if d == 0
             = 0.5 * cart_p * (1-cart_p)^(d-1)    otherwise

    Returns:
      weight_matrix: [L, L]
    """
    device = device if device is not None else "cpu"
    dtype = dtype if dtype is not None else torch.float32

    pos = torch.arange(seq_len, device=device)
    dist = pos.unsqueeze(0) - pos.unsqueeze(1)          # [L, L]
    abs_dist = torch.abs(dist).to(dtype)                # [L, L]

    # 0.5 * exp((d-1)*log(1-p) + log(p))
    weight = torch.exp((abs_dist - 1) * torch.log(torch.tensor(1 - cart_p, device=device, dtype=dtype))
                       + torch.log(torch.tensor(cart_p, device=device, dtype=dtype))) * 0.5
    weight = weight.masked_fill(abs_dist == 0, 0.0)
    return weight
