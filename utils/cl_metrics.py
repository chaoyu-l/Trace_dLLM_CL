"""Continual-learning diagnostic metrics.

Vector-level (gradient):
  gradient_cosine_similarity        – PCGrad [1]
  gradient_conflict_ratio           – PCGrad [1]
  magnitude_weighted_conflict_ratio – PS-LoRA [2]

Representation-level:
  linear_cka            – biased HSIC, Kornblith et al. [3]
  linear_cka_unbiased   – unbiased HSIC, Nguyen et al. [4]
  avg_cosine_similarity – per-sample cosine mean

References
----------
[1] Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.
[2] Zhou, Wu & Wei, "Resolving Conflicts in Lifelong Learning via Aligning
    Updates in Subspaces", arXiv 2512.08960, 2025.  (PS-LoRA)
[3] Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019.
[4] Nguyen, Raghu & Kornblith, "Do Wide and Deep Networks Learn the Same
    Things? Uncovering How Neural Network Representations Vary with Width
    and Depth", ICLR 2021.
[5] Qian et al., "TreeLoRA: Efficient Continual Learning via Layer-Wise
    LoRAs Guided by a Hierarchical Gradient-Similarity Tree", ICML 2025.
[6] Kim, Kim & Sohn, "Measuring Representational Shifts in Continual
    Learning: A Linear Transformation Perspective", ICML 2025.
"""

import torch
import torch.nn.functional as F


# ── Vector-level gradient metrics ─────────────────────────────────

def gradient_cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """Cosine similarity between two 1-D gradient vectors.

    Standard definition from PCGrad [1]: conflict iff cos < 0.
    """
    return F.cosine_similarity(g1.unsqueeze(0), g2.unsqueeze(0)).item()


def gradient_conflict_ratio(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """Fraction of elements where the two gradients have opposing signs."""
    return ((g1 * g2) < 0).float().mean().item()


def magnitude_weighted_conflict_ratio(
    g1: torch.Tensor, g2: torch.Tensor,
) -> float:
    """Conflict ratio weighted by element-wise gradient magnitude product.

    PS-LoRA [2] showed that *large* opposite-sign updates are
    disproportionately responsible for catastrophic forgetting.  This metric
    weights each parameter position by ``|g1_j * g2_j|`` so that
    high-magnitude conflicts dominate:

        MWCR = sum_{j: conflict} |g1_j * g2_j|  /  sum_j |g1_j * g2_j|

    A value close to 1 means almost all gradient energy is in conflict;
    close to 0 means conflicts only affect near-zero gradient entries.
    """
    prod = g1 * g2
    abs_prod = prod.abs()
    total = abs_prod.sum().item()
    if total < 1e-30:
        return 0.0
    return abs_prod[prod < 0].sum().item() / total


# ── Representation-level metrics ──────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA with the *biased* HSIC estimator [3].

    Fast, but systematically over-estimates similarity when the probe
    set is small.  Prefer :func:`linear_cka_unbiased` for CL diagnostics.

    Uses the Gram-matrix Frobenius inner product formulation
    (``<K_c, L_c>_F``), which is the canonical definition from
    Kornblith et al. and is O(N²D) — much cheaper than the feature-space
    ``||X^T Y||_F²`` route when D >> N.
    """
    X = X.float()
    Y = Y.float()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T  # [N, N] centred Gram matrix
    L = Y @ Y.T  # [N, N]

    hsic_xy = (K * L).sum()      # <K_c, L_c>_F  (Frobenius inner product)
    hsic_xx = (K * K).sum()      # ||K_c||_F^2
    hsic_yy = (L * L).sum()      # ||L_c||_F^2

    denom = hsic_xx.sqrt() * hsic_yy.sqrt()
    if denom < 1e-12:
        return 0.0
    return (hsic_xy / denom).item()


def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC_1 estimator (Song et al., 2012; used in [4]).

    Expects Gram matrices K, L of shape ``[m, m]``.
    Zeroes diagonals to remove self-similarity bias.
    """
    m = K.shape[0]
    K = K.clone()
    L = L.clone()
    K.fill_diagonal_(0)
    L.fill_diagonal_(0)
    kl = K @ L
    score = (
        kl.trace()
        + K.sum() * L.sum() / ((m - 1) * (m - 2))
        - 2.0 * kl.sum() / (m - 2)
    )
    return score / (m * (m - 3))


def linear_cka_unbiased(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA with the *unbiased* HSIC estimator from [4].

    Requires ``N >= 4`` samples.  Falls back to biased CKA otherwise.
    More reliable than :func:`linear_cka` when the probe set is small
    (the typical case in CL diagnostics).
    """
    X = X.float()
    Y = Y.float()
    if X.shape[0] < 4:
        return linear_cka(X, Y)
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = _unbiased_hsic(K, L)
    hsic_xx = _unbiased_hsic(K, K)
    hsic_yy = _unbiased_hsic(L, L)
    denom = (hsic_xx * hsic_yy).clamp(min=0).sqrt()
    if denom < 1e-12:
        return 0.0
    return (hsic_xy / denom).clamp(-1.0, 1.0).item()


def avg_cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Average per-sample cosine similarity between two ``[N, D]`` matrices."""
    return F.cosine_similarity(X.float(), Y.float(), dim=1).mean().item()
