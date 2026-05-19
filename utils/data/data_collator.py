import logging  # Python 标准库：日志系统，用于打印/记录运行信息
import torch  # PyTorch：张量、模型输入构造等
import sys

from typing import Optional, Any, Union, List, Dict  # 类型标注：让代码更清晰、IDE 更好提示
from dataclasses import dataclass  # dataclass：用“字段 + 默认值”方式定义类（更像配置对象）
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy  # HF tokenizer 基类 & padding 策略枚举

from utils.data.data_utils import DEFAULT_SYSTEM_PROMPT  # 你项目内部：默认 system prompt（系统提示词）

B_INST, E_INST = "[INST]", "[/INST]"  # LLaMA2 Chat 模板中的指令开始/结束标记
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"  # LLaMA2 Chat 模板中的 system 段包裹标记

logger = logging.getLogger(__name__)  # 获取当前模块的 logger（可用于 debug/info/warn 等）

# Diffusion 训练窗口应与推理窗口对齐：模型在固定长度窗口上 denoise，
# 训练时每个样本 pad 到 (prompt_len + task_ans_len)，让尾部 pad/EOS 获得监督信号。
# 单一来源位于 utils/diffusion/decode_config.py, in-training eval / Phase 2 inference
# / training-canvas 共享同一份表 -> 三处解码窗口口径强制一致。
from utils.diffusion.decode_config import DIFFUSION_TASK_MAX_ANS_LEN  # noqa: F401

# Per-task fixed EOS pad count N (K = N+1 = 答案区尾部 EOS 总数)。
# 基于 LLM-CL-Benchmark_500 真实 ans_len 分布 (collator 实测验证 4000/4000):
#   短任务 (ans_len 中位数 <= 2): K=1 (N=0). 物理下限, 再加 EOS frac 反而 >50%.
#   长任务 (ans_len 中位数 >= 9): K=2 (N=1). EOS frac <=18%, 给 forward_process
#     bernoulli mask 一个冗余位 (单样本至少 67% 概率把 EOS 学到, vs K=1 的 50%).
# 仅在 CLI --fix_eos 开启时生效;
# 该 dict 未命中的任务用安全下限 K=1 兜底 (任意 ans_len 下 EOS frac 最低)。
DIFFUSION_TASK_EOS_PAD = {
    # TRACE — short-answer (med ans_len <= 2): K=1 (N=0)
    "C-STANCE":    0,
    "FOMC":        0,
    "NumGLUE-cm":  0,
    "NumGLUE-ds":  0,
    # TRACE — long-answer (med ans_len >= 9): K=2 (N=1)
    "Py150":       1,
    "MeetingBank": 1,
    "ScienceQA":   1,
    "20Minuten":   1,
}


@dataclass  # dataclass：自动生成 __init__ / repr 等
class DataCollator:
    tokenizer: PreTrainedTokenizerBase  # 必需：HF tokenizer（负责把文本转 token id）
    model: Optional[Any] = None  # 可选：模型对象（用于从 config 里找 mask_token_id 等）
    padding: Union[bool, str, PaddingStrategy] = True  # padding 配置；True 常表示 pad 到 batch 最长
    max_prompt_len: Optional[int] = None  # prompt 最大长度（token 数）
    max_ans_len: Optional[int] = None  # answer 最大长度（token 数）
    pad_to_multiple_of: Optional[int] = 1  # pad 到多少的倍数（便于 Tensor Core / 加速）
    label_pad_token_id: int = -100  # labels 的 pad 值（CrossEntropy 会忽略 -100）
    return_tensors: str = "pt"  # 返回张量格式：PyTorch
    inference: bool = False  # 是否推理模式（推理不需要 labels）
    demonstrations: Optional[Any] = None  # 可选：few-shot demonstrations（这里未使用）
    task: str = None  # 当前任务名（Dream/LLaDA 训练用来查 DIFFUSION_TASK_MAX_ANS_LEN）
    is_diffusion: bool = False  # 是否扩散模型路径（Dream/LLaDA 的 diffusion collator）
    model_family: Optional[str] = None  # 必须显式传入：qwen|llama2|llama3|llada|dream
    # Dynamic-canvas（仅 diffusion 训练用，AR 路径无影响）：
    # 当传入 idx_to_group_max 时，每条样本的 canvas 改为 prompt_len + group_max_ans_len，
    # 替代原本的 prompt_len + DIFFUSION_TASK_MAX_ANS_LEN[task]。
    # group_max 由 training/main.py 在每个任务训练前用 K=grad_accum_steps 窗口预计算。
    # 上界 clamp 到 self.max_ans_len（CLI --max_ans_len, 默认 512），不是 task_canvas
    # —— 详见 _get_per_sample_ans_len 的 docstring。
    idx_to_group_max: Optional[Dict[int, int]] = None
    # Fix-EOS（仅 diffusion 训练；与 idx_to_group_max 互斥，优先级更高）：
    # True 时启用：每条样本 canvas = prompt_len + ans_len + K，
    #   K 严格由 DIFFUSION_TASK_EOS_PAD[task] 决定 (N+1)，未登记任务 K=1 兜底。
    # False = OFF，走 dynamic_canvas 或 task-fixed canvas（兼容旧产物）。
    fix_eos: bool = False

    def _validate_model_routing(self):
        ar_families = {"qwen", "llama2", "llama3"}
        diffusion_families = {"llada", "dream"}
        all_families = ar_families | diffusion_families

        mf = (self.model_family or "").strip().lower()
        if mf not in all_families:
            raise ValueError(
                f"Invalid model_family={self.model_family}. "
                f"Expected one of: {sorted(all_families)}"
            )

        if self.is_diffusion:
            if mf not in diffusion_families:
                raise ValueError(
                    f"Routing mismatch: is_diffusion=True but model_family={mf}. "
                    f"Expected one of: {sorted(diffusion_families)}"
                )
        else:
            if mf not in ar_families:
                raise ValueError(
                    f"Routing mismatch: is_diffusion=False but model_family={mf}. "
                    f"Expected one of: {sorted(ar_families)}"
                )

        # chat-template is mandatory for AR chat models and diffusion llada/dream paths.
        if mf in {"qwen", "llama2", "llama3", "llada", "dream"} and not hasattr(self.tokenizer, "apply_chat_template"):
            raise ValueError(
                f"model_family={mf} requires tokenizer.apply_chat_template(), "
                "but tokenizer does not provide it."
            )

        # normalize once; downstream branches can rely on canonical lowercase value
        self.model_family = mf

    def __call__(self, batch, return_tensors=None):
        self._validate_model_routing()
        if return_tensors is None:  # 如果外部没传，就用 dataclass 字段里的默认
            return_tensors = self.return_tensors  # 一般是 "pt"

        if self.is_diffusion:  # 扩散模型（Dream/LLaDA）走 diffusion_call
            return self.diffusion_call(batch, return_tensors)
        else:  # 普通 decoder-only（AR，比如 LLaMA/Qwen）走 decoder_call
            return self.decoder_call(batch, return_tensors)

    # Helper 1: 安全获取 Pad ID (Strict)
    def _get_pad_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:  # tokenizer 自带 pad_token_id 最优先
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:  # 没有 pad，就用 eos 当 pad（很多 decoder-only 常见做法）
            return self.tokenizer.eos_token_id
        raise ValueError(  # 两个都没有就直接报错：必须显式设置 tokenizer.pad_token
            "Tokenizer has no pad_token_id and no eos_token_id. "
            "Please set tokenizer.pad_token explicitly."
        )

    def _get_mask_id(self) -> int:
        """
        获取 Mask ID。
        """
        # [Priority 1] Tokenizer 显式定义的 mask_token_id (最准确)
        if getattr(self.tokenizer, "mask_token_id", None) is not None:  # tokenizer 有 mask_token_id 就直接用
            return int(self.tokenizer.mask_token_id)

        # [Priority 2] 通过 mask_token 字符串反查
        if getattr(self.tokenizer, "mask_token", None) is not None:  # tokenizer 有 mask_token 字符串，就把它转成 id
            return self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # [Priority 3] 尝试从 Model Config 获取 (如果你传了 model 进来)
        if self.model is not None:  # 如果传了 model，就从 model.config 查 mask_token_id
            if getattr(self.model.config, "mask_token_id", None) is not None:
                return int(self.model.config.mask_token_id)

        # 如果以上都找不到，说明 Tokenizer 初始化有问题。
        # 必须在 external (main.py) 显式指定 mask_token_id。
        raise ValueError(  # 直接抛“必须修复”的错误，避免 silent bug
            "CRITICAL ERROR: DataCollator cannot determine 'mask_token_id'.\n"
            "Solution: In your main.py/infer_single.py, after loading tokenizer, force set it:\n"
            "   tokenizer.mask_token_id = 156895  # (Use your actual mask id)\n"
            "   # Or if using a special token string:\n"
            "   # tokenizer.add_special_tokens({'mask_token': '<|mask|>'})"
        )

    def _get_effective_ans_len(self) -> int:
        """
        Diffusion 训练用：返回与推理窗口对齐的 answer 长度。
        有 self.task 且命中 DIFFUSION_TASK_MAX_ANS_LEN 时用 per-task；否则回退到 self.max_ans_len。
        """
        if self.task and self.task in DIFFUSION_TASK_MAX_ANS_LEN:
            return int(DIFFUSION_TASK_MAX_ANS_LEN[self.task])
        if self.max_ans_len is None:
            raise ValueError(
                "DataCollator._get_effective_ans_len: self.task 未命中 DIFFUSION_TASK_MAX_ANS_LEN "
                "且 self.max_ans_len 为 None。请在构造时传入 task= 或 max_ans_len=。"
            )
        return int(self.max_ans_len)

    def _get_eos_pad_count_for_task(self) -> int:
        """返回当前 task 的 fixed EOS pad 数 N (K = N+1 = 答案区尾部 EOS 总数)。

        优先级:
          1. self.fix_eos == False  -> 返回 -1, fix-eos 路径关闭, 走 dynamic_canvas / task-fixed canvas;
          2. self.task 命中 DIFFUSION_TASK_EOS_PAD -> 用该 dict 值 (per-task 推荐);
          3. 任务未登记 -> 安全下限 K=1 (N=0) 兜底 (任意 ans_len 下 EOS frac 最低)。
             启动日志已经 WARN 过这条情况, 这里不再重复。
        """
        if not self.fix_eos:
            return -1
        # 归一化: 剥掉诊断后缀 (_shuffled / _perm / _swap), 让派生数据集命中基类 task 的 K 表
        task = self.task or ""
        for suffix in ("_shuffled", "_perm", "_swap"):
            if task.endswith(suffix):
                task = task[: -len(suffix)]
                break
        if task and task in DIFFUSION_TASK_EOS_PAD:
            return int(DIFFUSION_TASK_EOS_PAD[task])
        return 0  # safe fallback: K=1

    def _get_per_sample_ans_len(self, sample_idx: Optional[int]) -> int:
        """Dynamic-canvas: 当传入 idx_to_group_max 且能查到 _idx 时，用 group_max 替代
        task-fixed canvas（实现 LLaDA 官方 SFT "pad to batch max" 行为，详见 GUIDELINES.md）。
        否则退化为 _get_effective_ans_len() —— 即原 task-fixed canvas。

        上界 clamp 到 self.max_ans_len（CLI 参数 --max_ans_len, 默认 512），不是
        task_canvas。这一层 clamp 只防极端 outlier 把 canvas 撑出 max_seq_len 之外
        （正常情况不触发，因为左截断已经把 len(ids) 卡在 max_seq_len 之内）。

        关键设计点（上版 bug）：上界必须用 max_ans_len 而非 task_canvas
        (DIFFUSION_TASK_MAX_ANS_LEN[task])。如果用 task_canvas, group_max > task_canvas
        时 dynamic 就退化回 task-fixed，window 内的长样本反而被截短，破坏"pad to window
        max" 的语义；同时短样本也享受不到 dynamic 收益（因为 clamp 几乎总是触发）。

        下界至少 1（防 0 长 answer 导致 canvas 等于 prompt_len）。
        """
        if self.idx_to_group_max is None or sample_idx is None:
            return int(self._get_effective_ans_len())
        gm = self.idx_to_group_max.get(int(sample_idx))
        if gm is None:
            return int(self._get_effective_ans_len())
        gm = max(1, int(gm))
        if self.max_ans_len is not None:
            gm = min(gm, int(self.max_ans_len))
        return gm

    def _split_system_user(self, prompt_text: str):
        """
        data_utils 当前做法：
          - add_sys_prefix=True:  prompt = DEFAULT_SYSTEM_PROMPT + "\n\n" + raw_prompt
          - add_sys_prefix=False: prompt = raw_prompt
        这里把它拆成 (system, user) 两段，供 Qwen/LLaDA 使用。
        """
        p = prompt_text  # 原始 prompt 文本
        prefix = DEFAULT_SYSTEM_PROMPT.strip()  # 默认 system prompt（去掉首尾空格用于匹配）
        if p.strip().startswith(prefix):  # 如果 prompt 以默认 system prompt 开头
            # 尝试按 "\n\n" 分割出 user 内容
            parts = p.split("\n\n", 1)  # 只分割一次：前面当 system，后面当 user
            if len(parts) == 2:  # 正常情况：确实被分成两段
                system_txt = parts[0].strip()  # system 内容
                user_txt = parts[1].strip()  # user 内容
                return system_txt, user_txt  # 返回(system, user)
            else:
                raise ValueError("No User Input")
        return None, p.strip()  # 没有 system 前缀：system=None，user=整个 prompt（去空格）

    def _strip_trailing_assistant(self, text: str) -> str:
        """
        去掉末尾的assistant
        """
        t = (text or "").rstrip()  # 把 None 变成空串，然后去掉右侧空白
        for suf in ["Assistant:", "assistant:", "ASSISTANT:"]:  # 可能的数据里会把 assistant 标签拼在末尾
            if t.endswith(suf):  # 如果以这些后缀结尾
                return t[:-len(suf)].rstrip()  # 去掉后缀再 rstrip
        return t  # 否则原样返回

    def _format_llama2_inst(self, system_txt: Optional[str], user_txt: str) -> str:
        """
        把 (system,user) 转成 LLaMA2 chat 的 INST 模板。
        仅用于 LLaDA 路线（以及你明确想用 LLaMA2-chat 时）。
        """
        user_txt = (user_txt or "").strip()  # user 文本去空格
        if system_txt:  # 如果有 system 段
            system_txt = system_txt.strip()  # system 去空格
            return f"{B_INST} {B_SYS}{system_txt}{E_SYS}{user_txt} {E_INST}"  # [INST] <<SYS>>...<</SYS>> user [/INST]
        else:
            return f"{B_INST} {user_txt} {E_INST}"  # 没 system 就纯 [INST] user [/INST]

    def _compute_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # equivalent to Dream's compute_position_id_with_mask when mask is 0/1
        pos = attention_mask.long().cumsum(-1) - 1  # 对 mask 做累计和：mask=1 的位置变成 0..n-1；mask=0 会停住
        return pos.clamp(min=0)  # 把 padding 区域可能的 -1 clamp 到 0（避免负 position id）

    def _truncate_ids(self, ids: List[int], max_len: int) -> List[int]:
        if max_len is None:  # max_len 不限制就原样返回
            return ids
        if len(ids) <= max_len:  # 长度没超过限制
            return ids
        # 左截断：保留尾部（含最新轮 user 内容和 assistant generation header），
        # 与训练侧 full_ids_raw[-max_seq_len:] 对齐，避免推理分布偏移
        return ids[-max_len:]

    # only support left padding for now (AR Models)
    def tokenize(self, sentence, cutoff_len, add_bos_token=True, add_eos_token=True):
        result = self.tokenizer(  # 调用 tokenizer 编码
            sentence,  # 输入文本
            truncation=True,  # 允许截断
            max_length=cutoff_len,  # 截断到 cutoff_len
            add_special_tokens=False,  # 这里不自动加 special tokens（后面手动加 bos/eos）
            padding=False,  # 单条不 padding（后面统一 pad）
            return_tensors=None,  # 返回 python list（不是 torch tensor）
        )

        if (  # 如果长度还没到 cutoff_len 且需要 eos
                len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)  # 手动追加 eos
            result["attention_mask"].append(1)  # eos 位置也算有效 token

        if (  # 如果长度还没到 cutoff_len 且需要 bos
                len(result["input_ids"]) < cutoff_len
                and add_bos_token
        ):
            result["input_ids"] = [self.tokenizer.bos_token_id] + result["input_ids"]  # 在开头插 bos
            result["attention_mask"] = [1] + result["attention_mask"]  # bos 也算有效 token

        result["labels"] = result["input_ids"].copy()  # labels 默认等于 input_ids（SFT 时会再 mask）
        return result  # 返回 dict：input_ids/attention_mask/labels

    # support decoder-only models for left padding (AR Models)
    def decoder_call(self, batch, return_tensors):
        sources = []  # 保存 prompt 文本（debug/评估用）
        gts = []  # 保存 ground-truth answer（推理评估用）
        tokenized_sources = []  # 保存每条样本 tokenized 后的 dict
        label_lens = []  # 训练时：每条 label 的有效长度（用于 pad labels）
        actual_max_len = 0  # 本 batch 内最长序列长度（用于决定 pad 到多长）
        limit_len = self.max_prompt_len + self.max_ans_len if not self.inference else self.max_prompt_len
        # ↑ 训练：prompt+answer 的总上限；推理：只限制 prompt 上限

        for instance in batch:  # 遍历 batch 中的每个样本
            instruction = instance['prompt']  # 取 prompt（你的数据字段名：prompt）
            label = instance['answer']  # 取 answer（你的数据字段名：answer）

            # ===== AR chat models: 用 chat_template 构造输入 =====
            if self.model_family in {"qwen", "llama2", "llama3"}:
                sys_txt, user_txt = self._split_system_user(instruction)  # 拆成 system/user
                sources.append(instruction)  # 记录原始 prompt
                gts.append(label)  # 记录答案（推理会用）

                # -------------------------
                # Train (SFT)
                # -------------------------
                if not self.inference:  # 训练模式：需要 full ids 和 prompt ids
                    # 1) full sequence: system + user + assistant(label)
                    full_msgs = []  # 构造完整对话消息列表（含 assistant 内容）
                    if sys_txt:  # 有 system 就加上
                        full_msgs.append({"role": "system", "content": sys_txt})
                    full_msgs.append({"role": "user", "content": user_txt})  # user 消息
                    full_msgs.append({"role": "assistant", "content": label})  # assistant 消息（就是要预测的答案）

                    full_ids_raw = self.tokenizer.apply_chat_template(  # 用 tokenizer 自带 chat_template 得到 token ids
                        full_msgs,
                        tokenize=True,  # 直接返回 token id 列表
                        add_generation_prompt=False,  # 因为 assistant content 已经给了，不需要 generation header
                    )

                    # 2) prompt sequence: system + user (+ assistant header)
                    prompt_msgs = []  # 构造“只到 assistant header”为止的 prompt（用于 mask labels）
                    if sys_txt:
                        prompt_msgs.append({"role": "system", "content": sys_txt})
                    prompt_msgs.append({"role": "user", "content": user_txt})

                    prompt_ids_raw = self.tokenizer.apply_chat_template(
                        prompt_msgs,
                        tokenize=True,  # 返回 token ids
                        add_generation_prompt=True,  # 关键：加 assistant header，确保 labels mask 掉 header
                    )

                    # 3) 兜底：如果 prompt_ids_raw 不是 full_ids_raw 的严格前缀，用最长公共前缀长度
                    if (len(prompt_ids_raw) > len(full_ids_raw)) or (
                            full_ids_raw[:len(prompt_ids_raw)] != prompt_ids_raw):
                        lcp = 0  # lcp = longest common prefix length
                        for a, b in zip(full_ids_raw, prompt_ids_raw):  # 逐 token 对齐比较
                            if a != b:
                                break
                            lcp += 1
                        prompt_len_raw = lcp  # 用最长公共前缀作为 prompt_len
                    else:
                        prompt_len_raw = len(prompt_ids_raw)  # 正常情况：prompt_ids 就是前缀

                    # 4) 左截断：保留尾部（答案完整保留，超长时截掉 prompt 头部）
                    if limit_len is not None and len(full_ids_raw) > limit_len:
                        tokens_removed = len(full_ids_raw) - limit_len
                        full_ids = full_ids_raw[-limit_len:]
                        prompt_len = max(0, prompt_len_raw - tokens_removed)
                    else:
                        full_ids = full_ids_raw
                        prompt_len = prompt_len_raw

                    # 5) labels：prompt 区域 -100，其余预测
                    labels = [-100] * prompt_len + full_ids[prompt_len:]  # prompt 不算 loss，答案部分算 loss

                    tokenized_sources.append({  # 保存该样本 tokenized 结果
                        "input_ids": full_ids,  # 输入 token ids（prompt+answer）
                        "attention_mask": [1] * len(full_ids),  # 全 1（因为这里暂不 pad）
                        "labels": labels,  # labels（prompt masked）
                    })

                    # max len & label len（注意：这里 label_len 是“未被 mask 的 token 数”，不等同于答案纯文本长度）
                    if len(full_ids) > actual_max_len:  # 更新 batch 内最大长度
                        actual_max_len = len(full_ids)
                    label_lens.append(len(full_ids) - prompt_len)  # 记录有效 label 长度（未被 -100 的长度）

                # -------------------------
                # Inference
                # -------------------------
                else:  # 推理模式：只构造 prompt（不拼接答案）
                    infer_msgs = []  # 推理对话消息列表（不含 assistant content）
                    if sys_txt:
                        infer_msgs.append({"role": "system", "content": sys_txt})
                    infer_msgs.append({"role": "user", "content": user_txt})

                    prompt_ids = self.tokenizer.apply_chat_template(
                        infer_msgs,
                        tokenize=True,  # 返回 token ids
                        add_generation_prompt=True,  # 推理需要 assistant header 作为 generation 起点
                    )

                    prompt_ids = self._truncate_ids(prompt_ids, limit_len)  # 推理也截断 prompt（右截断：保留左侧）

                    tokenized_sources.append({
                        "input_ids": prompt_ids,  # 推理输入 ids
                        "attention_mask": [1] * len(prompt_ids),  # 全 1（未 pad）
                    })
                    if len(prompt_ids) > actual_max_len:  # 更新 max len
                        actual_max_len = len(prompt_ids)

                # 这一条样本结束，进入下一条
                continue

            raise RuntimeError(
                f"Unreachable decoder branch with model_family={self.model_family}. "
                "Only qwen/llama2/llama3 are valid in AR mode."
            )

        actual_pad_len = (
                (actual_max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of)
        # ↑ 把 max_len 向上取整到 pad_to_multiple_of 的倍数（例如 8/16/128）

        for idx in range(len(tokenized_sources)):  # 对 batch 内每条样本执行 padding
            pad_len = actual_pad_len - len(tokenized_sources[idx]["input_ids"])  # 需要 pad 的长度
            assert sum(tokenized_sources[idx]["attention_mask"]) == len(tokenized_sources[idx]["input_ids"])
            # ↑ 断言：attention_mask 里 1 的个数应该等于 input_ids 长度（因为还没 pad）

            tokenized_sources[idx]["input_ids"] = [self.tokenizer.pad_token_id] * pad_len + tokenized_sources[idx]["input_ids"]
            # ↑ LEFT padding：在左侧补 pad_token_id

            tokenized_sources[idx]["attention_mask"] = [0] * pad_len + tokenized_sources[idx]["attention_mask"]
            # ↑ LEFT padding：左侧补 0（padding），右侧原本是有效 token（1）

            if not self.inference:  # 训练：也要对 labels 做对应 padding
                label_len = label_lens[idx]  # 该样本 label 的有效长度（未 pad 前）
                label_mask_len = actual_pad_len - label_len  # labels 左侧要补多少 -100

                tokenized_sources[idx]["labels"] = [-100] * label_mask_len + tokenized_sources[idx]["labels"][-label_len:]

                # ↑ labels LEFT padding：左补 -100；右侧保留最后 label_len 个 labels
                assert len(tokenized_sources[idx]["input_ids"]) == len(tokenized_sources[idx]["attention_mask"]) == len(
                    tokenized_sources[idx]["labels"]) == actual_pad_len
                # ↑ 断言：三者长度一致，且等于 pad 后长度

        model_inputs = {  # 最终拼成 PyTorch tensor batch
            'input_ids': torch.tensor([source["input_ids"] for source in tokenized_sources]),
            'attention_mask': torch.tensor([source["attention_mask"] for source in tokenized_sources])
        }

        if not self.inference:  # 训练才返回 labels
            model_inputs['labels'] = torch.tensor([source["labels"] for source in tokenized_sources])

        model_inputs['sources'] = sources  # 附加字段：原始 prompt 文本（调试/可视化）
        if self.inference:  # 推理才返回 gts（用来算指标）
            model_inputs['gts'] = gts

        return model_inputs  # 返回给 Trainer / DataLoader

    # ==============================================================
    #  Diffusion Collator (LLaDA-style SFT & Inference)
    # ==============================================================
    def diffusion_call(self, batch, return_tensors):
        """
        LLaDA-style SFT collation
        """
        if self.model_family not in {"llada", "dream"}:
            raise ValueError(
                f"Invalid diffusion model_family={self.model_family}. Expected llada or dream."
            )

        # ==================== 推理阶段 ====================
        if self.inference:  # diffusion 的推理逻辑
            if self.model_family == 'llada':  # LLaDA 推理
                # LLaDA inference now uses batch collation with attention_mask
                input_ids_list = []
                sources, gts = [], []

                for ins in batch:
                    p = (ins["prompt"] or "").strip()
                    sys_txt, user_txt = self._split_system_user(p)

                    infer_msgs = []
                    if sys_txt:
                        infer_msgs.append({"role": "system", "content": sys_txt})
                    infer_msgs.append({"role": "user", "content": user_txt})
                    prompt_ids = self.tokenizer.apply_chat_template(
                        infer_msgs,
                        tokenize=True,
                        add_generation_prompt=True,
                    )

                    if self.max_prompt_len is not None and len(prompt_ids) > self.max_prompt_len:
                        # 左截断：保留尾部（含 assistant generation header），与训练一致
                        prompt_ids = prompt_ids[-self.max_prompt_len:]

                    input_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))
                    sources.append(p)
                    gts.append(ins.get("answer", ""))

                pad_id = self._get_pad_id()
                max_len = max(x.numel() for x in input_ids_list)
                B = len(input_ids_list)

                # Left padding: align token tails for generation.
                padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
                padded_attn = torch.zeros((B, max_len), dtype=torch.long)
                for i, ids_t in enumerate(input_ids_list):
                    l = ids_t.numel()
                    padded_ids[i, max_len - l:] = ids_t
                    padded_attn[i, max_len - l:] = 1

                return {
                    "input_ids": padded_ids,
                    "attention_mask": padded_attn,
                    "sources": sources,
                    "gts": gts,
                }

            # ===== [AFTER] Dream diffusion inference: LEFT padding + keep prefix (official demo style) =====
            if self.model_family == "dream":  # Dream 推理：允许 batch>1，用 attention_mask + left padding
                input_ids_list = []  # 保存每条 prompt 的 tensor ids
                sources, gts = [], []  # 保存文本和答案

                for ins in batch:  # 遍历 batch
                    p = ins["prompt"].strip()  # prompt 去空格
                    sys_txt, user_txt = self._split_system_user(p)  # 拆 system/user
                    user_txt = self._strip_trailing_assistant(user_txt)  # 去掉末尾的 Assistant: 标签（防止污染）

                    messages = []  # 构造 chat messages
                    if sys_txt:
                        messages.append({"role": "system", "content": sys_txt})
                    messages.append({"role": "user", "content": user_txt})

                    # Same as Dream demo: chat_template + generation prompt
                    prompt_ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                    # ↑ Dream 用 chat_template，且推理要加 generation prompt（assistant header）

                    # Truncation: keep TAIL (truncate left) —— 与训练侧左截断对齐：
                    # 保留尾部（含最新轮 user 内容 + assistant generation header），
                    # 否则 Dream canvas 紧贴 prompt 末尾 denoise 时会 OOD。
                    if self.max_prompt_len is not None and len(prompt_ids) > self.max_prompt_len:
                        prompt_ids = prompt_ids[-self.max_prompt_len:]

                    input_ids_list.append(torch.tensor(prompt_ids, dtype=torch.long))  # 转 tensor 保存
                    sources.append(p)  # 保存原 prompt
                    gts.append(ins.get("answer", ""))  # 保存答案（可能没有则空串）

                pad_id = self._get_pad_id()  # 获取 pad id
                max_len = max(x.numel() for x in input_ids_list)  # batch 内最大长度
                B = len(input_ids_list)  # batch size

                # LEFT padding
                padded_ids = torch.full((B, max_len), pad_id, dtype=torch.long)  # 先全填 pad
                padded_attn = torch.zeros((B, max_len), dtype=torch.long)  # attention_mask 先全 0

                for i, ids_t in enumerate(input_ids_list):  # 把每条样本贴到右侧（left pad）
                    l = ids_t.numel()  # 当前样本长度
                    padded_ids[i, max_len - l:] = ids_t  # 右对齐填入真实 token
                    padded_attn[i, max_len - l:] = 1  # 对应位置置 1（有效 token）

                return {
                    "input_ids": padded_ids,  # [B, max_len]
                    "attention_mask": padded_attn,  # [B, max_len]
                    "sources": sources,  # 文本
                    "gts": gts,  # 答案
                }
            # ======================================================================

            raise ValueError(f"Unknown diffusion model_family={self.model_family} in inference.")
            # ↑ 推理阶段：既不是 llada 也不是 dream，就报错（避免 silent bug）

        # ========================================================================
        # -------- 以下是训练逻辑 (SFT Training) --------
        # ========================================================================

        prompts = [x["prompt"] for x in batch]  # batch 所有 prompt 文本
        answers = [x["answer"] for x in batch]  # batch 所有 answer 文本
        # Dynamic-canvas: 取每条样本在原数据集里的 idx，用于查 group_max。
        # 没有 _idx (旧 PromptDataset 路径) 或没有 idx_to_group_max 表时 → None，
        # 后续退化回 task-fixed canvas 行为。
        sample_idxs = [x.get("_idx") if isinstance(x, dict) else None for x in batch]

        pad_id = self._get_pad_id()  # pad token id
        mask_id = self._get_mask_id()  # mask token id（LLaDA 训练会用）
        bos_id = self.tokenizer.bos_token_id  # bos id
        eos_id = self.tokenizer.eos_token_id  # eos id

        if self.max_prompt_len is None or self.max_ans_len is None:  # diffusion 训练必须同时指定 prompt/ans 上限
            raise ValueError("max_prompt_len and max_ans_len must be set for diffusion_call().")

        max_seq_len = int(self.max_prompt_len + self.max_ans_len)  # 总序列最大长度（prompt+answer）

        # ============================
        # Dream: official fsdp_sft_trainer-style collation
        #   -> return input_ids/attention_mask/position_ids/loss_mask
        #   -> NO labels/p_mask/answer_lens here
        # ============================
        if self.model_family == "dream":  # Dream 训练路径
            input_ids_list = []
            attn_list = []
            loss_mask_list = []
            prompt_len_list = []

            for prompt, answer in zip(prompts, answers):  # 遍历样本
                sys_txt, user_txt = self._split_system_user((prompt or "").strip())  # 拆 system/user

                user_txt = self._strip_trailing_assistant(user_txt)  # 去掉末尾 assistant 标签

                prompt_chat = []  # prompt 对话 messages（不含 assistant content）
                if sys_txt:
                    prompt_chat.append({"role": "system", "content": sys_txt})
                prompt_chat.append({"role": "user", "content": user_txt})

                # prompt string with generation prompt (assistant header)
                prompt_chat_str = self.tokenizer.apply_chat_template(
                    prompt_chat, add_generation_prompt=True, tokenize=False
                )
                # ↑ tokenize=False：返回字符串（包含 special tokens 的文本模板），后面再 tokenizer(...) 成 ids

                # response with EOS appended (official SFTDataset does this)
                response_chat_str = (answer or "") + (self.tokenizer.eos_token or "")
                # ↑ Dream 官方数据集通常会在 answer 后面拼 eos token 字符串（确保训练结束符）

                prompt_out = self.tokenizer(
                    prompt_chat_str, return_tensors="pt", add_special_tokens=False
                )
                # ↑ 把 prompt_chat_str 编成 ids（不额外加 special tokens，因为模板里已经有）

                resp_out = self.tokenizer(
                    response_chat_str, return_tensors="pt", add_special_tokens=False
                )
                # ↑ 把 answer(+eos) 编成 ids


                prompt_ids = prompt_out["input_ids"][0]  # [Lp]：取出 1D ids
                resp_ids = resp_out["input_ids"][0]  # [Lr]：取出 1D ids

                prompt_len_raw = int(prompt_ids.numel())  # prompt 原始 token 长度（用于做 loss_mask）

                # 1) full concat first (Dream-main SFTDataset)
                full_ids_raw = torch.cat([prompt_ids, resp_ids], dim=0)  # 拼接成完整序列 [L_raw]

                # 2) 左截断：保留尾部（答案完整保留，超长时截掉 prompt 头部）
                if full_ids_raw.numel() > max_seq_len:
                    tokens_removed = int(full_ids_raw.numel()) - max_seq_len
                    ids = full_ids_raw[-max_seq_len:]
                    prompt_len = max(0, prompt_len_raw - tokens_removed)
                else:
                    ids = full_ids_raw
                    prompt_len = prompt_len_raw

                # 3) attention_mask = ones (Dream-main NOTE: we use 1 here)
                attn = torch.ones_like(ids, dtype=torch.long)

                # loss_mask: prompt 区域置 0，answer 区域置 1
                loss_mask = torch.ones_like(ids, dtype=torch.long)
                if prompt_len > 0:
                    loss_mask[:min(prompt_len, loss_mask.size(0))] = 0

                # 不再 pad 到 max_seq_len，只记录真实长度的 ids/attn/loss_mask/prompt_len
                input_ids_list.append(ids)
                attn_list.append(attn)
                loss_mask_list.append(loss_mask)
                prompt_len_list.append(prompt_len)

            # Diffusion 推理在固定长度窗口上 denoise；训练必须把每个样本 pad 到
            # (prompt_len + task_ans_len)，并在 pad 区给 attn=1 / loss_mask=1，
            # 让尾部 pad/EOS 也进入 q_sample 的加噪候选。否则模型从未学过"答完就吐 EOS"，
            # 推理时固定窗口的尾部必然 OOD（表现为复读/输出崩溃，Py150 最敏感）。
            # 官方 Dream SFT 走的就是这个逻辑（pad 到固定 max_length，attn=1, loss_mask=1）。
            #
            # Dynamic-canvas: 当 idx_to_group_max 提供时，eff_ans_len 改为 per-sample
            # group_max（梯度累积窗口内最长 answer），缓解 PAD/EOS overflow。
            # 当未提供时退化为 task-fixed canvas，行为与改动前完全一致。
            # Fix-EOS（fix_eos=True）：完全覆盖上述两条路径，每条样本
            # canvas = ids.numel() + N（ids 已含 1 EOS 终止符，见 response_chat_str
            # 末尾的 eos_token），EOS frac 与同窗口其他样本完全无关。
            # 实际 N 来自 DIFFUSION_TASK_EOS_PAD[task]，命中不到走 K=1 兜底。
            _n_pad_task = self._get_eos_pad_count_for_task()
            if _n_pad_task >= 0:
                per_sample_targets = [ids.numel() + _n_pad_task for ids in input_ids_list]
            else:
                per_sample_targets = [
                    max(plen + self._get_per_sample_ans_len(sidx), ids.numel())
                    for plen, ids, sidx in zip(prompt_len_list, input_ids_list, sample_idxs)
                ]

            max_len_in_batch = max(per_sample_targets)
            m = self.pad_to_multiple_of if self.pad_to_multiple_of and self.pad_to_multiple_of > 1 else 1
            if max_len_in_batch % m != 0:
                max_len_in_batch = ((max_len_in_batch + m - 1) // m) * m

            padded_ids, padded_attn, padded_loss_mask = [], [], []
            for ids, attn, lm in zip(input_ids_list, attn_list, loss_mask_list):
                pad_len = max_len_in_batch - ids.numel()
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                    attn = torch.cat([attn, torch.ones((pad_len,), dtype=torch.long)])
                    lm = torch.cat([lm, torch.ones((pad_len,), dtype=torch.long)])
                padded_ids.append(ids)
                padded_attn.append(attn)
                padded_loss_mask.append(lm)

            input_ids = torch.stack(padded_ids, dim=0)
            attention_mask = torch.stack(padded_attn, dim=0)
            loss_mask = torch.stack(padded_loss_mask, dim=0)
            position_ids = self._compute_position_ids(attention_mask)

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
                "sources": prompts,
            }

        # ============================
        # LLaDA (official-style SFT, aligned with ML-GSAI/LLaDA GUIDELINES.md)
        #   - each sample padded with <EOS> up to prompt_len + max_ans_len
        #     (real tokens that participate in forward_process AND loss)
        #   - forward_process runs on the whole [B, L]; prompt positions
        #     are then restored (masked_indices & ~prompt_mask)
        #   - answer_lens = L - prompt_len  (INCLUDES EOS padding, per official)
        #   - attention_mask = all ones (official training does not pass mask)
        #   - returns {input_ids, attention_mask, labels, p_mask, answer_lens, sources}
        #     (same keys/shapes as before; base_model.py untouched)
        # ============================

        if self.model_family != "llada":
            raise RuntimeError(
                f"Unreachable diffusion training branch with model_family={self.model_family}."
            )

        # Pad with EOS (not pad_id) — EOS participates in forward_process and loss.
        llada_pad_token = eos_id if eos_id is not None else pad_id
        if llada_pad_token == mask_id:
            raise ValueError(
                f"LLaDA training requires pad/eos token ({llada_pad_token}) != mask_id ({mask_id}). "
                "Set tokenizer.eos_token to a non-mask id before training."
            )

        input_ids_list = []   # 每条样本的 ids（prompt + answer + EOS padding；还未对齐到 batch 最长）
        prompt_len_list = []  # 每条样本的 prompt_len（截断后）

        for prompt, answer, sample_idx in zip(prompts, answers, sample_idxs):
            sys_txt, user_txt = self._split_system_user(prompt)

            # ===== 严格按 LLaDA-main/GUIDELINES.md 第 60-62 行 SFT 数据格式构造 =====
            # 官方规范:
            #   <BOS><start_id>user<end_id>\n[user]<eot_id><start_id>assistant<end_id>\n[answer]<EOS><EOS>...
            #
            # 拆解为四段:
            #   [P] = <BOS><start_id>user<end_id>\n[user]<eot_id><start_id>assistant<end_id>\n
            #         (prompt 部分, 含 assistant header 但不含答案)
            #   [A] = [answer]
            #         (纯 answer 文本 token, 无任何 chat-template wrapping)
            #   [E] = <EOS>
            #         (答案后第一个 EOS, 作为终止符)
            #   [P_pad] = <EOS><EOS>...
            #         (剩余 EOS pad 由 dynamic canvas / task_canvas 决定)
            #
            # 注: 项目原写法 apply_chat_template(full_msgs, add_generation_prompt=False)
            # 会因 LLaDA tokenizer 的 chat_template "无条件追加 assistant header" 缺陷,
            # 在 [A] 之后多塞 5 个 token (<eot_id><start_id>assistant<end_id>\n\n), 与
            # GUIDELINES.md 不符。这里改为 prompt-only chat_template + 手动拼答案+EOS。

            # ----- 段 [P]: prompt-only chat_template (add_generation_prompt=True) -----
            # 在 prompt-only 输入下, LLaDA tokenizer 的 chat_template 末尾的"无条件追加"
            # 正好是我们想要的 assistant header, 输出与 GUIDELINES.md 的 [P] 段一致。
            prompt_msgs = []
            if sys_txt:
                prompt_msgs.append({"role": "system", "content": sys_txt})
            prompt_msgs.append({"role": "user", "content": user_txt})
            prompt_ids_raw = self.tokenizer.apply_chat_template(
                prompt_msgs, tokenize=True, add_generation_prompt=True,
            )

            # ----- 段 [A]: 纯 answer token, 不走 chat_template wrapping -----
            answer_ids_raw = self.tokenizer(
                answer or "", add_special_tokens=False
            )["input_ids"]

            # ----- 段 [E]: 答案后追加 1 个 EOS -----
            # GUIDELINES.md "Paris.<EOS>..." — 答案紧跟 EOS, 中间不应有任何分隔符
            ids = list(prompt_ids_raw) + list(answer_ids_raw) + [eos_id]
            prompt_len = len(prompt_ids_raw)

            # ----- 左截断: 超长时砍 prompt 头部, 保留完整 [A]+[E] -----
            # 等价于原项目左截断, 但因 [A]+[E] 比原"chat_template wrapped answer"短,
            # 触发截断的概率更低。
            if len(ids) > max_seq_len:
                tokens_removed = len(ids) - max_seq_len
                ids = ids[-max_seq_len:]
                prompt_len = max(0, prompt_len - tokens_removed)

            # ----- 安全检查: 至少保留 1 个 answer-region token (= 至少那个 EOS 终止符) -----
            prompt_len = min(prompt_len, max(0, len(ids) - 1))
            if len(ids) - prompt_len <= 0:
                raise ValueError(
                    f"Invalid resp_len: len(ids)={len(ids)}, prompt_len={prompt_len}, max_seq_len={max_seq_len}."
                )

            # ----- 段 [P_pad]: 由 dynamic canvas 决定补多少 EOS -----
            # OFF (group_size=0):    per_sample_ans_len = task_canvas (DIFFUSION_TASK_MAX_ANS_LEN[task])
            # ON  (group_size>0):    per_sample_ans_len = group_max (LLaDA 官方"pad to batch max")
            # fix_eos=True (覆盖前两者): 严格 per-sample, target_len = len(ids) + N (K-1)。
            #
            # 注: 由于上面新格式的 [A]+[E] 比原写法短得多, 短答案任务下短样本会真的被
            # pad 到 task_canvas, 不会再被原写法的"5 个无意义 token"提前撑过 task_canvas。
            _n_pad_task = self._get_eos_pad_count_for_task()
            if _n_pad_task >= 0:
                # Fix-EOS: ids 已含 1 个 EOS 终止符 (line 752), 再补 N 个 EOS pad。
                # 实际 N 来自 DIFFUSION_TASK_EOS_PAD[task] (per-task 表)；命中不到走 K=1 兜底。
                # EOS frac = (N+1) / (ans_len + N + 1)，与同窗口其他样本完全无关。
                target_len = len(ids) + _n_pad_task
            else:
                target_len = prompt_len + int(self._get_per_sample_ans_len(sample_idx))
            if target_len < len(ids):  # 超长样本兜底保护(防截断后 prompt_len 反而变大)
                target_len = len(ids)
            if len(ids) < target_len:
                ids = ids + [eos_id] * (target_len - len(ids))

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            prompt_len_list.append(prompt_len)

        # Batch padding 到 batch 内最长（仍用 <EOS>）
        max_len_in_batch = max(t.numel() for t in input_ids_list)
        if self.pad_to_multiple_of and self.pad_to_multiple_of > 1:
            m = self.pad_to_multiple_of
            if max_len_in_batch % m != 0:
                max_len_in_batch = ((max_len_in_batch + m - 1) // m) * m

        B = len(input_ids_list)
        input_ids_clean = torch.full((B, max_len_in_batch), llada_pad_token, dtype=torch.long)
        for i, t in enumerate(input_ids_list):
            input_ids_clean[i, :t.numel()] = t

        prompt_lens_t = torch.tensor(prompt_len_list, dtype=torch.long)

        # ---- forward_process (official): per-sample t, Bernoulli over full [B, L] ----
        eps = 1e-3
        t_sample = torch.rand(B)                                          # [B]
        p_mask_sample = (1.0 - eps) * t_sample + eps                      # [B]   ∈ (eps, 1]
        p_mask_full = p_mask_sample.unsqueeze(1).expand(B, max_len_in_batch)  # [B, L]

        rand_mat = torch.rand((B, max_len_in_batch))
        masked_indices = rand_mat < p_mask_full                           # [B, L]

        # prompt 区不 mask（等价于官方 noisy_batch[prompt_mask] = input_ids[prompt_mask]）
        token_positions = torch.arange(max_len_in_batch).unsqueeze(0).expand(B, -1)
        prompt_mask = token_positions < prompt_lens_t.unsqueeze(1)        # [B, L] True = prompt 区
        masked_indices = masked_indices & (~prompt_mask)

        # labels：-100 at prompt / 未 mask 位置；真 token id at mask 位置（含 EOS padding）
        labels = torch.full_like(input_ids_clean, self.label_pad_token_id)
        labels[masked_indices] = input_ids_clean[masked_indices]

        # input_ids：被 mask 的位置替换成 mask_id
        noisy_input_ids = input_ids_clean.clone()
        noisy_input_ids[masked_indices] = mask_id

        # p_mask：未 mask 处 1.0（CE 本就是 0，不影响）；mask 处写入该样本的 p_mask
        p_mask_tensor = torch.ones_like(input_ids_clean, dtype=torch.float32)
        p_mask_tensor[masked_indices] = p_mask_full[masked_indices].float()

        # attention_mask：官方训练根本不传，这里给全 1 保接口一致
        attention_mask = torch.ones_like(input_ids_clean, dtype=torch.long)

        # answer_lens = L - prompt_len（含 EOS padding），官方 loss 用此做 per-sample 归一
        answer_lens_tensor = torch.tensor(
            [max_len_in_batch - pl for pl in prompt_len_list],
            dtype=torch.float32,
        ).clamp(min=1.0)

        return {
            "input_ids": noisy_input_ids,       # [B, L]
            "attention_mask": attention_mask,   # [B, L]  全 1
            "labels": labels,                   # [B, L]  -100 / real
            "p_mask": p_mask_tensor,            # [B, L]  1.0 / per-sample p
            "answer_lens": answer_lens_tensor,  # [B]     L - prompt_len
            "sources": prompts,
        }