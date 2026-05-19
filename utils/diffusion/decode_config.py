"""Single source of truth for per-task diffusion decode config.

之前散在 3 个文件:
  - inference/infer_single.py: DIFFUSION_TASK_MAX_ANS_LEN, LLADA_TASK_BLOCK_LENGTH,
                                LLADA_DEFAULT_BLOCK_LENGTH (+ 每任务 block 的查表逻辑)
  - utils/data/data_collator.py: 复制了一份 DIFFUSION_TASK_MAX_ANS_LEN
  - model/base_model.py: 写死字面量 128

统一到这里之后:
  - 加新任务/改 canvas/改 block 只改本文件
  - Phase 2 inference 和 in-training eval 走同一份表 -> 解码轨迹一致
  - resolver 内做 task name normalization (剥 _shuffled/_perm/_swap 后缀)
"""

# =============================================================================
# Per-task canvas (max_new_tokens) for diffusion generation.
# Diffusion models denoise a fixed-size canvas, so shorter max_ans_len directly
# reduces computation. AR models already stop at EOS, making this unnecessary.
# =============================================================================
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

# =============================================================================
# Per-task semi-autoregressive block size for LLaDA inference.
# Derived from LLaDA paper's reported best configs:
#   - GSM8K/MATH (numerical):    gen=256, block=8
#   - HumanEval/MBPP (code/long): gen=512, block=32
# Rule of thumb: keep block size at the absolute scale the paper validated
# (8 for short-numeric, 32 for code/long), not a fixed ratio of gen_length.
# For very short canvases (gen<=16), use a single block (block=gen_length)
# since semi-AR splitting is meaningless on label-style outputs.
# Dream has no native semi-AR decoding, so this dict only applies to LLaDA.
# =============================================================================
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

LLADA_DEFAULT_BLOCK_LENGTH = 128  # fallback for tasks not in LLADA_TASK_BLOCK_LENGTH


_TASK_SUFFIXES = ("_shuffled", "_perm", "_swap")


def normalize_task_name(task: str) -> str:
    """剥掉诊断后缀 (_shuffled / _perm / _swap)，让派生数据集命中基类任务的表项。"""
    if task is None:
        return task
    for suffix in _TASK_SUFFIXES:
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def resolve_task_canvas(task: str, fallback: int) -> int:
    """Per-task diffusion canvas (max_new_tokens). 命中表内返回表值, 否则用 fallback."""
    return int(DIFFUSION_TASK_MAX_ANS_LEN.get(normalize_task_name(task), fallback))


def resolve_llada_block_length(task: str, cli_block_length=None) -> int:
    """LLaDA semi-AR block_length.

    优先级:
      1. CLI --block_length (non-None) 一票否决 -> 直接返回 (全局 override)
      2. 否则查 per-task 表
      3. 否则 LLADA_DEFAULT_BLOCK_LENGTH

    最终消费者 model/llada_generate.py:178 还有一层安全网:
      `if gen_length % block_length != 0: block_length = gen_length`
    所以即使 task 不在表里、又传了不能整除 canvas 的值, 也只是塌成单块, 不会崩。
    """
    if cli_block_length is not None:
        return int(cli_block_length)
    return int(LLADA_TASK_BLOCK_LENGTH.get(normalize_task_name(task), LLADA_DEFAULT_BLOCK_LENGTH))
