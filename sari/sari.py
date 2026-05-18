# Copyright 2020 The HuggingFace Datasets Authors.
# Licensed under the Apache License, Version 2.0.
"""
SARI metric standalone implementation.
Adapted from the original implementation by Wei Xu et al. (2016).
"""

import datasets
import evaluate
from collections import Counter

_CITATION = """\
@inproceedings{xu-etal-2016-optimizing,
    title = "Optimizing Statistical Machine Translation for Text Simplification",
    author = "Xu, Wei  and Napoles, Courtney  and Pavlick, Ellie  and Chen, Quanze  and Callison-Burch, Chris",
    booktitle = "Transactions of the Association for Computational Linguistics",
    year = "2016",
}
"""

_DESCRIPTION = """\
SARI is a metric used for evaluating automatic text simplification systems.
It explicitly measures the goodness of words that are added, kept and deleted by the system.
"""

_KWARGS_DESCRIPTION = """
Calculates SARI scores.
Args:
    sources (list of str): The source sentences.
    predictions (list of str): The predicted sentences.
    references (list of list of str): The reference sentences.
Returns:
    sari (float): The SARI score.
"""


def SARIngram(sgrams, cgrams, rgrams_list, numref):
    rgramsall = [item for sublist in rgrams_list for item in sublist]
    rgramscount = Counter(rgramsall)
    sgramscount = Counter(sgrams)
    cgramscount = Counter(cgrams)
    combined = {}

    for gram in sgramscount:
        combined[gram] = sgramscount[gram]
    for gram in cgramscount:
        combined[gram] = combined.get(gram, 0) + cgramscount[gram]
    for gram in rgramscount:
        combined[gram] = combined.get(gram, 0) + rgramscount[gram]

    keepgramscount = 0
    keepgramscount_total = 0
    addgramscount = 0
    addgramscount_total = 0
    delgramscount = 0
    delgramscount_total = 0

    for gram in combined:
        gramcount = combined[gram]
        sgramcount = sgramscount.get(gram, 0)
        cgramcount = cgramscount.get(gram, 0)

        # calculate max references count for this n-gram
        rgramcount = 0
        for rgrams in rgrams_list:
            rcount = Counter(rgrams).get(gram, 0)
            if rcount > rgramcount:
                rgramcount = rcount

        # KEEP operation
        if cgramcount > 0 and sgramcount > 0 and rgramcount > 0:
            keepgramscount += min(cgramcount, min(sgramcount, rgramcount))
            keepgramscount_total += max(cgramcount, sgramcount)

        # ADD operation
        if cgramcount > 0 and sgramcount == 0:
            addgramscount += min(cgramcount, rgramcount)
            addgramscount_total += cgramcount

        # DEL operation
        if cgramcount == 0 and sgramcount > 0:
            delgramscount += min(sgramcount, max(0, sgramcount - rgramcount))
            delgramscount_total += sgramcount

    keepscore = 0
    if keepgramscount_total > 0:
        keepscore = keepgramscount / keepgramscount_total

    addscore = 0
    if addgramscount_total > 0:
        addscore = addgramscount / addgramscount_total

    delscore = 0
    if delgramscount_total > 0:
        delscore = delgramscount / delgramscount_total

    return keepscore, delscore, addscore


def SARIsent(ssent, csent, rsents):
    numref = len(rsents)

    # Simple tokenization by splitting on whitespace
    # This is standard for SARI if inputs are already tokenized
    s1grams = ssent.split(" ")
    c1grams = csent.split(" ")

    s2grams, c2grams, s3grams, c3grams, s4grams, c4grams = [], [], [], [], [], []
    r1grams_list, r2grams_list, r3grams_list, r4grams_list = [], [], [], []

    # Generate n-grams for Source (s) and Candidate (c)
    for i in range(len(s1grams) - 1): s2grams.append(s1grams[i] + " " + s1grams[i + 1])
    for i in range(len(s1grams) - 2): s3grams.append(s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2])
    for i in range(len(s1grams) - 3): s4grams.append(
        s1grams[i] + " " + s1grams[i + 1] + " " + s1grams[i + 2] + " " + s1grams[i + 3])

    for i in range(len(c1grams) - 1): c2grams.append(c1grams[i] + " " + c1grams[i + 1])
    for i in range(len(c1grams) - 2): c3grams.append(c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2])
    for i in range(len(c1grams) - 3): c4grams.append(
        c1grams[i] + " " + c1grams[i + 1] + " " + c1grams[i + 2] + " " + c1grams[i + 3])

    # Generate n-grams for References (r)
    for rsent in rsents:
        r1grams = rsent.split(" ")
        r2grams, r3grams, r4grams = [], [], []
        for i in range(len(r1grams) - 1): r2grams.append(r1grams[i] + " " + r1grams[i + 1])
        for i in range(len(r1grams) - 2): r3grams.append(r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2])
        for i in range(len(r1grams) - 3): r4grams.append(
            r1grams[i] + " " + r1grams[i + 1] + " " + r1grams[i + 2] + " " + r1grams[i + 3])
        r1grams_list.append(r1grams)
        r2grams_list.append(r2grams)
        r3grams_list.append(r3grams)
        r4grams_list.append(r4grams)

    (keep1score, del1score, add1score) = SARIngram(s1grams, c1grams, r1grams_list, numref)
    (keep2score, del2score, add2score) = SARIngram(s2grams, c2grams, r2grams_list, numref)
    (keep3score, del3score, add3score) = SARIngram(s3grams, c3grams, r3grams_list, numref)
    (keep4score, del4score, add4score) = SARIngram(s4grams, c4grams, r4grams_list, numref)

    avgkeepscore = (keep1score + keep2score + keep3score + keep4score) / 4
    avgdelscore = (del1score + del2score + del3score + del4score) / 4
    avgaddscore = (add1score + add2score + add3score + add4score) / 4

    finalscore = (avgkeepscore + avgdelscore + avgaddscore) / 3
    return finalscore


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Sari(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "sources": datasets.Value("string", id="sequence"),
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Sequence(datasets.Value("string", id="sequence"), id="references"),
                }
            ),
        )

    def _compute(self, sources, predictions, references):
        sari_sum = 0.0
        # TRACE benchmark 的数据通常 references 是 [[ref1], [ref2]] 这种格式
        # 原始 SARI 算法要求 references 是一个 list of strings (多个参考)
        for src, pred, refs in zip(sources, predictions, references):
            sari_sum += SARIsent(src, pred, refs)

        # 返回平均分 (0-100)
        return {"sari": 100.0 * sari_sum / len(sources)}