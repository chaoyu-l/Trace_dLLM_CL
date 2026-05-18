import json
from metrics import caculate_accuracy, extract_choice

VALID_OPTIONS = "ABC"   # A=dovish, B=hawkish, C=neutral


def eval(predicted_sequences, ground_truths, return_details=False):
    extracted = [extract_choice(p, VALID_OPTIONS) for p in predicted_sequences]
    gt_extracted = [extract_choice(g, VALID_OPTIONS) for g in ground_truths]
    accuracy = caculate_accuracy(extracted, gt_extracted)
    evaluation_result = {"accuracy": accuracy}
    if return_details:
        details = {
            "predictions_for_eval": extracted,
            "labels_for_eval": gt_extracted
        }
        return evaluation_result, details
    return evaluation_result
