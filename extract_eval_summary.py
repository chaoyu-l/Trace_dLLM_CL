#!/usr/bin/env python3
"""
Walk runs/<tag>/ and emit eval_summary.json with just the metric scores.

Each results-{round}-{task_id}-{task}.json file has a heavy 'prompts'/'results'/
'labels' payload (~1.6 GB total across 22 experiments). For computing FWT/BWT
/AvgAcc we only need the 'eval' field (a few bytes per file). This script
strips everything else into a single compact summary that's git-friendly.

Usage:
    python extract_eval_summary.py
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_PATH = REPO_ROOT / "eval_summary.json"

RESULT_PATTERN = re.compile(r"results-(\d+)-(\d+)-(.+)\.json$")


def extract_eval(path):
    try:
        with path.open() as f:
            data = json.load(f)
        return data.get("eval")
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def main():
    summary = {"experiments": {}}

    for exp_dir in sorted(RUNS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        tag = exp_dir.name
        exp = {"base_metrics": {}, "predictions": {}, "epoch_eval": None}

        # Match both *_base (e5) and *_base_order2 (order2) layouts
        for base_dir in exp_dir.glob("outputs_*_base*/base_metrics"):
            for f in sorted(base_dir.glob("results-*.json")):
                m = RESULT_PATTERN.match(f.name)
                if m:
                    _, _, task = m.groups()
                    exp["base_metrics"][task] = extract_eval(f)

        for pred_dir in exp_dir.glob("outputs_*/lora_*/predictions"):
            for f in sorted(pred_dir.glob("results-*.json")):
                m = RESULT_PATTERN.match(f.name)
                if m:
                    rnd, _, task = m.groups()
                    exp["predictions"].setdefault(rnd, {})[task] = extract_eval(f)

        for f in exp_dir.glob("outputs_*/lora_*/epoch_eval.json"):
            try:
                with f.open() as fh:
                    exp["epoch_eval"] = json.load(fh)
            except Exception as e:
                exp["epoch_eval"] = {"_error": f"{type(e).__name__}: {e}"}

        if exp["base_metrics"] or exp["predictions"] or exp["epoch_eval"]:
            summary["experiments"][tag] = exp

    with OUT_PATH.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"Wrote {OUT_PATH}")
    print(f"Experiments: {len(summary['experiments'])}")
    print()
    print(f"{'Experiment':<36}  {'baseline':>10}  {'predictions':>14}  {'epoch_eval':>10}")
    print("-" * 78)
    for tag, exp in sorted(summary["experiments"].items()):
        nb = len(exp["base_metrics"])
        np_total = sum(len(d) for d in exp["predictions"].values())
        nr = len(exp["predictions"])
        ee = "yes" if exp["epoch_eval"] else "-"
        print(f"{tag:<36}  {f'{nb}/8':>10}  {f'{np_total} ({nr} rounds)':>14}  {ee:>10}")


if __name__ == "__main__":
    main()
