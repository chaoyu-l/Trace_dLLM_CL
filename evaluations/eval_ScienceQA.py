"""
ScienceQA evaluation -- answer extraction and metric computation.

Answer-extraction logic is based on **three authoritative reference
implementations only** -- no custom heuristics or magic numbers.

Reference 1 -- ScienceQA official (lupantech/ScienceQA, models/run_gpt3.py, line 52)
    pattern = re.compile(r'The answer is ([A-Z]).')
    Extracts "The answer is B." style answers.
    We extend minimally to also cover "The *correct* answer is B"
    (widely observed in instruction-tuned model outputs).

Reference 2 -- lm-evaluation-harness (EleutherAI, lm_eval/filters/extraction.py)
    MultiChoiceRegexFilter, without_paren_fallback_regex:
        r':[\s]*({A|B|C|...})'
    Matches "Answer: B", "answer:B", etc.  We restrict the prefix to
    the word "answer" to avoid false positives on unrelated colons
    (e.g. "Reasons:\nA...").

Reference 3 -- TRACE upstream (BeyonderXX/TRACE, evaluations/eval_ScienceQA.py, line 10)
    answers.append(datium[0])
    Takes the first character of the raw output as the answer letter.
    This is the final fallback.

Extraction order:
    Step 1  "The (correct) answer is [A-E]"   (Ref 1)
    Step 2  "Answer: [A-E]"                    (Ref 2, narrowed)
    Step 3  first character                    (Ref 3)
"""

import re
from metrics import caculate_bleu, caculate_rouge, caculate_accuracy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VALID_OPTIONS = {"A", "B", "C", "D", "E"}

# ---------------------------------------------------------------------------
# Step 1 -- ScienceQA official regex (extended for "correct answer")
#
#   Original:  re.compile(r'The answer is ([A-Z]).')
#   Ref: lupantech/ScienceQA  models/run_gpt3.py  line 52
#
#   Extension rationale: instruction-tuned LLMs frequently emit
#   "The correct answer is B" instead of "The answer is B".
#   This is a single-word insertion and the meaning is identical;
#   virtually every ScienceQA evaluation codebase handles both.
# ---------------------------------------------------------------------------
_ANSWER_IS_RE = re.compile(
    r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*([A-E])\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Step 2 -- lm-eval-harness colon pattern (narrowed to "answer" prefix)
#
#   Original: rf':[\s]*({A|B|C|...})'
#   Ref: EleutherAI/lm-evaluation-harness
#        lm_eval/filters/extraction.py  MultiChoiceRegexFilter
#
#   We require the word "answer" before the colon to prevent
#   false positives like "Reasons:\nA" or "Here:\nC".
# ---------------------------------------------------------------------------
_ANSWER_COLON_RE = re.compile(
    r"answer\s*:\s*([A-E])\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------
def extract_answer_and_reasoning(text: str):
    """Extract (answer_letter, reasoning_text) from a model response.

    Returns
    -------
    (str, str)
        answer : one of A-E, or the raw first character when no pattern matched.
        reasoning : text following the answer span.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return "", ""

    # Step 1 -- "The (correct) answer is X"
    m = _ANSWER_IS_RE.search(text)
    if m and m.group(1).upper() in VALID_OPTIONS:
        return m.group(1).upper(), text[m.end():].strip()

    # Step 2 -- "Answer: X"
    m = _ANSWER_COLON_RE.search(text)
    if m and m.group(1).upper() in VALID_OPTIONS:
        return m.group(1).upper(), text[m.end():].strip()

    # Step 3 -- TRACE upstream fallback: first character
    first_char = text[0].upper()
    reasoning = text[2:] if len(text) > 2 else ""
    return first_char, reasoning


# ---------------------------------------------------------------------------
# resolve() -- drop-in replacement for TRACE upstream
# ---------------------------------------------------------------------------
def resolve(dataset: list):
    """Convert raw output strings into {answers, reasonings} dict."""
    answers = []
    reasonings = []
    for datium in dataset:
        answer, reasoning = extract_answer_and_reasoning(datium)
        answers.append(answer)
        reasonings.append(reasoning)
    return {"answers": answers, "reasonings": reasonings}


# ---------------------------------------------------------------------------
# eval() -- called by calculate_metrics.py
# ---------------------------------------------------------------------------
def eval(predicted_sequences, ground_truths, return_details=False):
    outputs = resolve(predicted_sequences)
    gts = resolve(ground_truths)

    bleu_1 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 1)
    bleu_4 = caculate_bleu(outputs["reasonings"], gts["reasonings"], 4)
    rouge = caculate_rouge(outputs["reasonings"], gts["reasonings"])
    accuracy = caculate_accuracy(outputs["answers"], gts["answers"])

    evaluation_result = {
        "bleu-1": bleu_1,
        "bleu-4": bleu_4,
        "rouge-L": rouge,
        "accuracy": accuracy,
    }
    if return_details:
        details = {
            "extracted_answers": outputs["answers"],
            "extracted_reasonings": outputs["reasonings"],
            "gt_answers": gts["answers"],
            "gt_reasonings": gts["reasonings"],
        }
        return evaluation_result, details
    return evaluation_result
