## 默认模型在测试时输出的顺序和测试集顺序相同
## The inputs must not be shuffled during evaluation.


import re
from rouge import Rouge
from fuzzywuzzy import fuzz
import evaluate
from nltk.translate.bleu_score import sentence_bleu


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def caculate_bleu(results, data, gram):
    bleus = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)
    avg_bleu = sum(bleus) / len(results)
    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    # 1. 严格检查：如果是纯空格或空字符串，直接返回 0
    if not str1.strip() or not str2.strip():
        return 0.0

    try:
        rouge = Rouge(metrics=["rouge-l"])
        scores = rouge.get_scores(str1, str2, avg=True)
        rouge_l = scores['rouge-l']['f']
        return rouge_l
    except Exception:
        # 2. 兜底保护：如果 rouge 内部处理后为空（例如只剩标点符号），返回 0
        return 0.0


def caculate_rouge(results, data):
    rouges = []
    count = len(results)
    if count == 0:
        return 0.0

    for output_id in range(count):
        prediction = results[output_id]
        target = data[output_id]

        # 直接调用增加过保护的 score_rouge
        # 即使是空字符串，它现在会返回 0.0，而不会报错
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    # 计算平均分
    # 原逻辑：sum(有效分数) / 总样本数 (等价于把错误的当0分)
    # 现逻辑：sum(所有分数，含0) / 总样本数
    avg_rouge = sum(rouges) / count
    return avg_rouge


########################
## Accuracy (EM)
########################
def caculate_accuracy(results, data):
    """Exact-match accuracy -- identical to TRACE upstream (BeyonderXX/TRACE metrics.py)."""
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        if prediction == target:
            scores += 1
    avg_score = scores / len(results)
    return avg_score


########################
## Answer Extraction for Classification Tasks (A/B/C/D/E)
########################
def extract_choice(text, valid_options="ABC"):
    """Extract a single choice letter from LLM output.

    Extraction pipeline based on three established reference implementations:

    Ref 1 -- OpenCompass (open-compass/opencompass, NeurIPS 2024 D&B Track)
        opencompass/utils/text_postprocessors.py
        first_option_postprocess(): pattern-based extraction with 50+ regex
        first_capital_postprocess(): scan for first uppercase letter

    Ref 2 -- lm-evaluation-harness (EleutherAI/lm-evaluation-harness)
        lm_eval/filters/extraction.py
        MultiChoiceRegexFilter: r':[\s]*({options})'

    Ref 3 -- TRACE upstream (BeyonderXX/TRACE, NeurIPS 2024)
        evaluations/eval_ScienceQA.py line 10:
        answers.append(datium[0])   # first character as fallback
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""

    opts = valid_options.upper()
    opt_re = f"[{opts}]"

    if len(text) == 1 and text.upper() in opts:
        return text.upper()

    # --- Ref 1 & 2: Pattern-based extraction (OpenCompass + lm-eval-harness) ---
    patterns = [
        rf"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*({opt_re})\b",
        rf"(?:the\s+)?(?:correct\s+)?answer\s+is\s+option\s*:?\s*({opt_re})\b",
        rf"ANSWER\s*:\s*({opt_re})",
        rf"[Aa]nswer\s*:\s*({opt_re})",
        rf"答案是?\s*[:：]?\s*({opt_re})",
        rf"答案选项?[为是]?\s*[:：]?\s*({opt_re})",
        rf"选择?\s*[:：]?\s*({opt_re})",
        rf"故选?\s*({opt_re})",
        rf"态度[是为]?\s*[:：]?\s*({opt_re})",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m and m.group(1).upper() in opts:
            return m.group(1).upper()

    # --- Ref 3 + Ref 1 (first_capital_postprocess): first valid option char ---
    first_char = text[0].upper()
    if first_char in opts:
        return first_char

    for char in text:
        if char.upper() in opts:
            return char.upper()

    return first_char


########################
## Answer Extraction for Free-form Tasks (NumGLUE etc.)
########################
def extract_freeform_answer(text):
    """Extract the core answer from a free-form model response.

    Ref 1 -- OpenCompass (open-compass/opencompass)
        opencompass/utils/text_postprocessors.py
        general_postprocess(): truncate at first newline / period / comma
        first_number_postprocess(): extract first number via regex

    Ref 2 -- TRACE upstream (BeyonderXX/TRACE)
        Exact-match on raw output (our baseline / fallback).
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""

    first_line = text.split("\n")[0].strip()

    # Ref 1 (first_number_postprocess): leading number
    m = re.match(r"^(-?\d+(?:[.,]\d+)?)", first_line)
    if m:
        return m.group(1).replace(",", "")

    # Ref 1 (general_postprocess): truncate at first sentence-ending delimiter
    truncated = re.split(r"[.\n,;!?]", first_line, 1)[0].strip()
    if truncated:
        return truncated

    return first_line


########################
## F1-micro
########################
def f1_score(list1, list2):
    # TP: item in list1 and list2
    # FP: item in list1 but not in list2
    # TN: item not in list1 and list2
    # FN: item in list2 but not in list1
    num_TP = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                num_TP += 1
                break
    precision = num_TP / len(list1)
    recall = num_TP / len(list2)
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall / (precision + recall))


def caculate_f1(results, data):
    scores = []
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if len(prediction) == 0 or len(target) == 0:
            continue
        score = f1_score(target, prediction)
        scores.append(score)
    avg_score = sum(scores) / len(results)
    return avg_score


########################
## fuzzywuzzy
########################
def caculate_fuzz(results, data):
    scores = 0
    for output_id in range(len(results)):
        prediction = results[output_id]
        target = data[output_id]
        if prediction == "" or target == "":
            continue
        scores += fuzz.ratio(prediction, target)
    avg_score = scores / len(results)
    return avg_score


########################
## SARI
########################
import os
import evaluate
def caculate_sari(inputs, results, data):
    # 1. 获取本地 sari 文件夹的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sari_local_path = os.path.join(current_dir, "sari")

    # 2. 验证路径是否存在
    if not os.path.exists(os.path.join(sari_local_path, "sari.py")):
        raise FileNotFoundError(f"找不到本地 SARI 脚本: {sari_local_path}/sari.py")

    print(f"[INFO] Loading SARI metric from local path: {sari_local_path}")

    # 3. [重要] 强制加载本地，不要写 try-except 自动降级到 "sari"
    # 因为远程/缓存里的 "sari" 是坏的，如果不报错直接用坏的，你永远不知道为什么错了
    sari = evaluate.load(sari_local_path)

    # 4. 格式化数据
    formatted_refs = [[label] for label in data]

    # 5. 计算
    translation_result = sari.compute(sources=inputs, predictions=results, references=formatted_refs)

    return translation_result