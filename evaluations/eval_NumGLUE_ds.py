import json
from metrics import caculate_accuracy, extract_freeform_answer


def eval(predicted_sequences, ground_truths, return_details=False):
    extracted = [extract_freeform_answer(p) for p in predicted_sequences]
    gt_clean = [g.strip() for g in ground_truths]
    accuracy = caculate_accuracy(extracted, gt_clean)
    evaluation_result = {"accuracy": accuracy}
    if return_details:
        details = {
            "predictions_for_eval": extracted,
            "labels_for_eval": gt_clean
        }
        return evaluation_result, details
    return evaluation_result
