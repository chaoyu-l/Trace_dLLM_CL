"""SSR (ACL 2024) per-task evaluator.

Mirrors LLaMA-Factory's ComputeMetrics used in the official SSR repo:
  DeepLearnXMU/SSR -> src/llmtuner/tuner/sft/metric.py

SSR paper Sec 5.1 adopts ROUGE-L for every task (10 SuperNI tasks including
sa / pos classification), scaled to percentage [0, 100]. The upstream
implementation also reports rouge-1/rouge-2/bleu-4 alongside rouge-l, so
we surface all four for completeness.

For sa (sentiment: positive/negative) and pos (True/False) we additionally
record lowercased exact-match accuracy as a supplementary metric; the SSR
paper itself uses ROUGE-L, accuracy is opt-in via calculate_metrics.py.

Dependencies: jieba, rouge-chinese, nltk (all installed manually).
"""

import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

_CLASSIFICATION_TASKS = {"sa", "pos"}


def _score_one(pred: str, label: str):
    hypothesis = list(jieba.cut(pred))
    reference = list(jieba.cut(label))

    if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
        result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
    else:
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]

    rouges = {k: round(result[k]["f"] * 100, 4) for k in ("rouge-1", "rouge-2", "rouge-l")}
    bleu_4 = round(
        sentence_bleu(
            [list(label)], list(pred),
            smoothing_function=SmoothingFunction().method3,
        ) * 100,
        4,
    )
    return rouges, bleu_4


def eval(predicted_sequences, ground_truths, task=None, return_details=False):
    n = len(predicted_sequences)
    assert n == len(ground_truths), "pred / gt length mismatch"

    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
    for pred, gt in zip(predicted_sequences, ground_truths):
        rouges, bleu_4 = _score_one(pred, gt)
        for k, v in rouges.items():
            score_dict[k].append(v)
        score_dict["bleu-4"].append(bleu_4)

    result = {k: (sum(v) / len(v)) if v else 0.0 for k, v in score_dict.items()}

    if task in _CLASSIFICATION_TASKS:
        correct = sum(
            1 for p, g in zip(predicted_sequences, ground_truths)
            if p.strip().lower() == g.strip().lower()
        )
        result["accuracy"] = correct / n if n else 0.0

    if return_details:
        details = {
            "predictions_for_eval": list(predicted_sequences),
            "labels_for_eval": list(ground_truths),
            "task": task,
        }
        return result, details
    return result
