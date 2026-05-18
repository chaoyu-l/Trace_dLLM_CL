#!/usr/bin/env python
import argparse
import glob
import hashlib
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

JUDGE_VERSION = "v2_task_specific_prompts"

CLASSIFICATION_OPTIONS = {
    "C-STANCE": "ABC",
    "FOMC": "ABC",
    "ScienceQA": "ABCDE",
}

DEFAULT_TASKS = ["C-STANCE", "FOMC", "ScienceQA", "NumGLUE-cm", "NumGLUE-ds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate prediction files with GPT API."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single results-*.json file, or a directory containing them.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="results-*.json",
        help="Glob pattern when --input is a directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name for evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to same folder as input file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".gpt_eval",
        help="Output file suffix before .json.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only first N samples (for debugging/cost control).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API calls.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for one sample.",
    )
    parser.add_argument(
        "--cache_file",
        type=str,
        default=".gpt_eval_cache.json",
        help="Cache file to avoid repeated API calls.",
    )
    parser.add_argument(
        "--api_key_env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for OpenAI API key.",
    )
    parser.add_argument(
        "--only_tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names to evaluate.",
    )
    parser.add_argument(
        "--include_unsupported",
        action="store_true",
        help="Also evaluate files whose task is not in --only_tasks.",
    )
    parser.add_argument(
        "--mode",
        choices=["sync", "batch"],
        default="sync",
        help="sync: live HTTP per sample. batch: OpenAI Batch API (~50%% cheaper, async).",
    )
    parser.add_argument(
        "--poll_interval",
        type=float,
        default=30.0,
        help="Batch poll interval in seconds (only used in --mode batch).",
    )
    parser.add_argument(
        "--completion_window",
        type=str,
        default="24h",
        help="Batch completion window. OpenAI currently accepts '24h'.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .gpt_eval_batch_*.json state file to resume a submitted batch.",
    )
    parser.add_argument(
        "--batch_state_dir",
        type=str,
        default=None,
        help="Where to write batch JSONL + state files. Default: input dir (or its parent if --input is a file).",
    )
    parser.add_argument(
        "--no_poll",
        action="store_true",
        help="In --mode batch: submit and exit without polling. Use --resume later to fetch results.",
    )
    return parser.parse_args()


def infer_task_name_from_filename(path: str) -> str:
    filename = os.path.basename(path)
    m = re.match(r"results-\d+-\d+-(.+)\.json$", filename)
    if m:
        return m.group(1)
    return "UnknownTask"


def get_eval_files(input_path: str, pattern: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = sorted(glob.glob(os.path.join(input_path, pattern)))
    return [p for p in files if os.path.isfile(p)]


def safe_json_load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_json_dump(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(path: str, cache: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_cache_key(
    model: str,
    task_name: str,
    prompt: str,
    label: str,
    prediction: str,
) -> str:
    raw = json.dumps(
        {
            "judge_version": JUDGE_VERSION,
            "model": model,
            "task": task_name,
            "prompt": prompt,
            "label": label,
            "prediction": prediction,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def build_messages(
    task_name: str,
    prompt: str,
    label: str,
    prediction: str,
) -> List[Dict[str, str]]:
    options = CLASSIFICATION_OPTIONS.get(task_name)
    if task_name in ("C-STANCE", "FOMC", "ScienceQA"):
        task_rule = (
            "Task type: multiple-choice classification.\n"
            f"Valid options: {', '.join(list(options))}.\n"
            "Judge by final choice only. Ignore extra explanation text.\n"
            "If prediction has no valid final option, mark incorrect.\n"
            "If prediction and reference choose same option, mark correct."
        )
    elif task_name in ("NumGLUE-cm", "NumGLUE-ds"):
        task_rule = (
            "Task type: free-form numeric/short-answer.\n"
            "Extract final concise answer from prediction.\n"
            "Mark correct when mathematically/semantically equivalent.\n"
            "Allow trivial format differences: commas, spaces, trailing punctuation,\n"
            "fraction vs decimal equivalence, and sign-preserving numeric forms.\n"
            "If prediction does not give a clear final answer, mark incorrect."
        )
    else:
        task_rule = (
            "Task type: generic QA.\n"
            "Judge by final answer correctness and semantic equivalence.\n"
            "If no clear answer is present, mark incorrect."
        )

    user_content = (
        "You are grading one model output.\n"
        "Decide whether prediction is correct relative to reference.\n\n"
        f"{task_rule}\n\n"
        f"Task: {task_name}\n"
        f"[Prompt]\n{prompt}\n\n"
        f"[Reference Answer]\n{label}\n\n"
        f"[Model Prediction]\n{prediction}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "is_correct": 0 or 1,\n'
        '  "predicted_answer": "short extracted final answer",\n'
        '  "reference_answer": "short normalized reference answer",\n'
        '  "reason": "one short sentence"\n'
        "}"
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a strict but fair evaluator. "
                "Always output valid JSON only, without markdown."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def call_gpt_judge(
    client: Any,
    model: str,
    task_name: str,
    prompt: str,
    label: str,
    prediction: str,
    max_retries: int,
) -> Dict[str, Any]:
    messages = build_messages(task_name, prompt, label, prediction)

    last_err = None
    for retry_idx in range(max_retries):
        try:
            response = client.responses.create(
                model=model,
                input=messages,
                temperature=0,
                max_output_tokens=400,
            )
            text = getattr(response, "output_text", "") or ""
            parsed = extract_json_block(text)
            if not parsed:
                raise ValueError(f"Cannot parse JSON from model output: {text}")

            is_correct = int(parsed.get("is_correct", 0))
            is_correct = 1 if is_correct == 1 else 0
            return {
                "is_correct": is_correct,
                "predicted_answer": str(parsed.get("predicted_answer", "")).strip(),
                "reference_answer": str(parsed.get("reference_answer", "")).strip(),
                "reason": str(parsed.get("reason", "")).strip(),
            }
        except Exception as err:
            last_err = err
            sleep_s = min(2 ** retry_idx, 8)
            time.sleep(sleep_s)

    raise RuntimeError(f"GPT judge failed after retries: {last_err}")


def evaluate_single_file(
    client: Any,
    file_path: str,
    model: str,
    max_samples: Optional[int],
    sleep_seconds: float,
    max_retries: int,
    cache: Dict[str, Any],
) -> Dict[str, Any]:
    obj = safe_json_load(file_path)
    prompts = obj.get("prompts", [])
    preds = obj.get("results", [])
    labels = obj.get("labels", [])

    n = min(len(prompts), len(preds), len(labels))
    if max_samples is not None:
        n = min(n, max_samples)
    task_name = infer_task_name_from_filename(file_path)

    details = []
    correct = 0
    errors = 0
    n_empty = 0

    for i in range(n):
        prompt = str(prompts[i])
        pred = str(preds[i])
        label = str(labels[i])

        if not pred.strip():
            judge = {
                "is_correct": 0,
                "predicted_answer": "",
                "reference_answer": str(label).strip(),
                "reason": "empty_prediction",
            }
            n_empty += 1
        else:
            cache_key = build_cache_key(model, task_name, prompt, label, pred)
            if cache_key in cache:
                judge = cache[cache_key]
            else:
                try:
                    judge = call_gpt_judge(
                        client=client,
                        model=model,
                        task_name=task_name,
                        prompt=prompt,
                        label=label,
                        prediction=pred,
                        max_retries=max_retries,
                    )
                    cache[cache_key] = judge
                except Exception as err:
                    judge = {
                        "is_correct": 0,
                        "predicted_answer": "",
                        "reference_answer": "",
                        "reason": f"API_ERROR: {err}",
                    }
                    errors += 1

        correct += int(judge.get("is_correct", 0) == 1)
        details.append(
            {
                "idx": i,
                "is_correct": judge.get("is_correct", 0),
                "predicted_answer": judge.get("predicted_answer", ""),
                "reference_answer": judge.get("reference_answer", ""),
                "reason": judge.get("reason", ""),
                "raw_prediction": pred,
                "raw_label": label,
            }
        )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    acc = (correct / n) if n > 0 else 0.0
    result = {
        "task": task_name,
        "file": file_path,
        "model": model,
        "n_samples": n,
        "n_correct": correct,
        "accuracy": acc,
        "n_api_errors": errors,
        "n_empty": n_empty,
    }
    return {"summary": result, "details": details, "original": obj}


def build_output_path(input_file: str, output_dir: Optional[str], suffix: str) -> str:
    base_name = os.path.basename(input_file)
    stem, ext = os.path.splitext(base_name)
    new_name = f"{stem}{suffix}{ext}"
    if output_dir:
        return os.path.join(output_dir, new_name)
    return os.path.join(os.path.dirname(input_file), new_name)


# ===========================================================================
# Batch API helpers
# ===========================================================================
def build_batch_request(
    custom_id: str,
    model: str,
    task_name: str,
    prompt: str,
    label: str,
    prediction: str,
) -> Dict[str, Any]:
    messages = build_messages(task_name, prompt, label, prediction)
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": messages,
            "temperature": 0,
            "max_output_tokens": 400,
        },
    }


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def collect_pending_for_file(
    file_path: str,
    model: str,
    max_samples: Optional[int],
    cache: Dict[str, Any],
    request_counter: List[int],
) -> Dict[str, Any]:
    """Walk one results-*.json and classify each sample as
    {empty, cache_hit, pending_api}. Pending samples produce batch request
    dicts with stable custom_ids; non-pending fill `details` directly.
    """
    obj = safe_json_load(file_path)
    prompts = obj.get("prompts", [])
    preds = obj.get("results", [])
    labels = obj.get("labels", [])

    n = min(len(prompts), len(preds), len(labels))
    if max_samples is not None:
        n = min(n, max_samples)
    task_name = infer_task_name_from_filename(file_path)

    details: List[Optional[Dict[str, Any]]] = [None] * n
    pending: List[Dict[str, Any]] = []
    request_meta: Dict[str, Dict[str, Any]] = {}
    n_empty = 0
    cache_hits = 0

    for i in range(n):
        prompt = str(prompts[i])
        pred = str(preds[i])
        label = str(labels[i])

        if not pred.strip():
            details[i] = {
                "idx": i,
                "is_correct": 0,
                "predicted_answer": "",
                "reference_answer": str(label).strip(),
                "reason": "empty_prediction",
                "raw_prediction": pred,
                "raw_label": label,
            }
            n_empty += 1
            continue

        cache_key = build_cache_key(model, task_name, prompt, label, pred)
        if cache_key in cache:
            judge = cache[cache_key]
            details[i] = {
                "idx": i,
                "is_correct": judge.get("is_correct", 0),
                "predicted_answer": judge.get("predicted_answer", ""),
                "reference_answer": judge.get("reference_answer", ""),
                "reason": judge.get("reason", ""),
                "raw_prediction": pred,
                "raw_label": label,
            }
            cache_hits += 1
            continue

        custom_id = f"req_{request_counter[0]:08d}"
        request_counter[0] += 1
        pending.append(
            build_batch_request(custom_id, model, task_name, prompt, label, pred)
        )
        request_meta[custom_id] = {
            "file_path": file_path,
            "sample_idx": i,
            "task": task_name,
            "cache_key": cache_key,
        }

    return {
        "task": task_name,
        "n_samples": n,
        "n_empty": n_empty,
        "cache_hits": cache_hits,
        "details": details,
        "pending_requests": pending,
        "request_meta": request_meta,
        "obj": obj,
        "file_path": file_path,
    }


def submit_batch_job(
    client: Any,
    jsonl_path: str,
    completion_window: str,
) -> Dict[str, str]:
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    return {"batch_id": batch.id, "input_file_id": upload.id}


def poll_batch_status(
    client: Any,
    batch_id: str,
    poll_interval: float,
) -> Any:
    last_seen = (None, -1, -1)
    while True:
        b = client.batches.retrieve(batch_id)
        status = b.status
        counts = getattr(b, "request_counts", None)
        n_total = getattr(counts, "total", 0) if counts else 0
        n_done = getattr(counts, "completed", 0) if counts else 0
        n_failed = getattr(counts, "failed", 0) if counts else 0
        snapshot = (status, n_done, n_failed)
        if snapshot != last_seen:
            print(
                f"[BATCH] {batch_id} status={status} "
                f"done={n_done}/{n_total} failed={n_failed}"
            )
            last_seen = snapshot
        if status in ("completed", "failed", "cancelled", "expired"):
            return b
        time.sleep(poll_interval)


def fetch_batch_output(client: Any, output_file_id: str) -> List[Dict[str, Any]]:
    content = client.files.content(output_file_id)
    text = content.text if hasattr(content, "text") else str(content)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def parse_batch_response_row(
    row: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if row.get("error"):
        return None, f"BATCH_ERROR: {row['error']}"
    resp = row.get("response") or {}
    status_code = resp.get("status_code")
    if status_code and status_code != 200:
        return None, f"HTTP_ERROR: {status_code}"
    body = resp.get("body") or {}

    output_text = body.get("output_text") or ""
    if not output_text:
        chunks = []
        for item in body.get("output", []) or []:
            if item.get("type") == "message":
                for c in item.get("content", []) or []:
                    if c.get("type") == "output_text":
                        chunks.append(c.get("text", ""))
        output_text = "".join(chunks)

    if not output_text:
        return None, "PARSE_ERROR: no output text"

    parsed = extract_json_block(output_text)
    if not parsed:
        return None, f"PARSE_ERROR: {output_text[:200]}"

    is_correct = int(parsed.get("is_correct", 0))
    is_correct = 1 if is_correct == 1 else 0
    return {
        "is_correct": is_correct,
        "predicted_answer": str(parsed.get("predicted_answer", "")).strip(),
        "reference_answer": str(parsed.get("reference_answer", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
    }, None


def save_batch_state(path: str, state: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_batch_state(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_batch_outputs(
    args: Any,
    per_file_state: Dict[str, Dict[str, Any]],
    judge_by_cid: Dict[str, Dict[str, Any]],
    request_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    error_prefixes = (
        "API_ERROR",
        "BATCH_ERROR",
        "HTTP_ERROR",
        "PARSE_ERROR",
        "NO_RESPONSE",
    )

    # Fill in pending details from batch results.
    for cid, meta in request_map.items():
        fp = meta["file_path"]
        if fp not in per_file_state:
            continue
        partial = per_file_state[fp]
        details = partial["details"]
        i = meta["sample_idx"]
        if i >= len(details) or details[i] is not None:
            continue
        obj = partial["obj"]
        raw_pred = str(obj["results"][i]) if i < len(obj.get("results", [])) else ""
        raw_label = str(obj["labels"][i]) if i < len(obj.get("labels", [])) else ""
        judge = judge_by_cid.get(cid) or {
            "is_correct": 0,
            "predicted_answer": "",
            "reference_answer": "",
            "reason": "NO_RESPONSE",
        }
        details[i] = {
            "idx": i,
            **judge,
            "raw_prediction": raw_pred,
            "raw_label": raw_label,
        }

    all_summaries: List[Dict[str, Any]] = []
    for fp, partial in per_file_state.items():
        details = partial["details"]
        for i, d in enumerate(details):
            if d is None:
                details[i] = {
                    "idx": i,
                    "is_correct": 0,
                    "predicted_answer": "",
                    "reference_answer": "",
                    "reason": "NO_RESPONSE",
                    "raw_prediction": "",
                    "raw_label": "",
                }
        n = len(details)
        correct = sum(1 for d in details if d.get("is_correct") == 1)
        n_errors = sum(
            1 for d in details if str(d.get("reason", "")).startswith(error_prefixes)
        )
        acc = correct / n if n > 0 else 0.0
        summary = {
            "task": partial["task"],
            "file": fp,
            "model": args.model,
            "n_samples": n,
            "n_correct": correct,
            "accuracy": acc,
            "n_api_errors": n_errors,
            "n_empty": partial["n_empty"],
            "judge_version": JUDGE_VERSION,
        }
        all_summaries.append(summary)
        out_path = build_output_path(fp, args.output_dir, args.suffix)
        output_obj = partial["obj"]
        output_obj["gpt_eval"] = summary
        output_obj["gpt_eval_details"] = details
        safe_json_dump(out_path, output_obj)
        print(
            f"[GPT-EVAL] Done: acc={acc:.4f} "
            f"({correct}/{n}) -> {out_path}"
        )
    return all_summaries


def run_batch_mode(
    args: Any,
    client: Any,
    files: List[str],
    cache: Dict[str, Any],
) -> None:
    only_tasks = {t.strip() for t in args.only_tasks.split(",") if t.strip()}

    per_file_state: Dict[str, Dict[str, Any]] = {}
    request_map: Dict[str, Dict[str, Any]] = {}
    batch_id: Optional[str] = None

    if args.resume:
        state = load_batch_state(args.resume)
        batch_id = state["batch_id"]
        saved_files = state.get("files", files)
        request_map = state.get("request_map", {})
        request_counter = [1]
        for fp in saved_files:
            task_name = infer_task_name_from_filename(fp)
            if (task_name not in only_tasks) and (not args.include_unsupported):
                continue
            partial = collect_pending_for_file(
                file_path=fp,
                model=args.model,
                max_samples=args.max_samples,
                cache=cache,
                request_counter=request_counter,
            )
            per_file_state[fp] = partial
        print(f"[BATCH] Resuming batch_id={batch_id}")
    else:
        all_pending: List[Dict[str, Any]] = []
        request_counter = [1]

        for fp in files:
            task_name = infer_task_name_from_filename(fp)
            if (task_name not in only_tasks) and (not args.include_unsupported):
                print(f"[BATCH] Skip unsupported task: {fp}")
                continue
            partial = collect_pending_for_file(
                file_path=fp,
                model=args.model,
                max_samples=args.max_samples,
                cache=cache,
                request_counter=request_counter,
            )
            per_file_state[fp] = partial
            all_pending.extend(partial["pending_requests"])
            request_map.update(partial["request_meta"])
            print(
                f"[BATCH] {os.path.basename(fp)}: "
                f"empty={partial['n_empty']}, cache_hits={partial['cache_hits']}, "
                f"pending={len(partial['pending_requests'])}"
            )

        if not all_pending:
            print("[BATCH] No pending requests (all empty or cache hits). Writing outputs.")
            write_batch_outputs(args, per_file_state, {}, request_map)
            save_cache(args.cache_file, cache)
            print("\n===== GPT Re-evaluation Summary =====")
            return

        if args.batch_state_dir:
            state_dir = args.batch_state_dir
        elif os.path.isdir(args.input):
            state_dir = args.input
        else:
            state_dir = os.path.dirname(args.input) or "."
        os.makedirs(state_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = os.path.join(state_dir, f".gpt_eval_batch_{ts}.jsonl")
        state_path = os.path.join(state_dir, f".gpt_eval_batch_{ts}.json")
        write_jsonl(jsonl_path, all_pending)
        print(f"[BATCH] Wrote {len(all_pending)} requests -> {jsonl_path}")

        info = submit_batch_job(client, jsonl_path, args.completion_window)
        batch_id = info["batch_id"]
        save_batch_state(
            state_path,
            {
                "batch_id": batch_id,
                "input_file_id": info["input_file_id"],
                "model": args.model,
                "submitted_at": ts,
                "jsonl_path": jsonl_path,
                "files": list(per_file_state.keys()),
                "request_map": request_map,
                "n_pending": len(all_pending),
                "judge_version": JUDGE_VERSION,
            },
        )
        print(f"[BATCH] Submitted batch_id={batch_id}")
        print(f"[BATCH] State file: {state_path}")
        print(
            f"[BATCH] To resume later: "
            f"python {os.path.basename(__file__)} "
            f"--mode batch --input <same> --resume {state_path}"
        )

        if args.no_poll:
            print("[BATCH] --no_poll set; exiting before polling.")
            return

    if not batch_id:
        return

    batch_obj = poll_batch_status(client, batch_id, args.poll_interval)
    if batch_obj.status != "completed":
        err_id = getattr(batch_obj, "error_file_id", None)
        print(f"[BATCH] Batch did not complete cleanly: status={batch_obj.status}")
        if err_id:
            print(f"[BATCH] error_file_id={err_id}")
        return
    output_file_id = getattr(batch_obj, "output_file_id", None)
    if not output_file_id:
        print("[BATCH] Completed but no output_file_id; aborting.")
        return

    rows = fetch_batch_output(client, output_file_id)
    print(f"[BATCH] Fetched {len(rows)} output rows.")

    judge_by_cid: Dict[str, Dict[str, Any]] = {}
    n_parse_err = 0
    for row in rows:
        cid = row.get("custom_id")
        if not cid:
            continue
        judge, err = parse_batch_response_row(row)
        if judge is None:
            judge = {
                "is_correct": 0,
                "predicted_answer": "",
                "reference_answer": "",
                "reason": err or "PARSE_ERROR",
            }
            n_parse_err += 1
        else:
            meta = request_map.get(cid)
            if meta:
                cache[meta["cache_key"]] = judge
        judge_by_cid[cid] = judge
    if n_parse_err:
        print(f"[BATCH] {n_parse_err} responses failed to parse.")

    all_summaries = write_batch_outputs(args, per_file_state, judge_by_cid, request_map)
    save_cache(args.cache_file, cache)

    print("\n===== GPT Re-evaluation Summary =====")
    for s in all_summaries:
        print(
            f"{os.path.basename(s['file'])}: "
            f"acc={s['accuracy']:.4f}, n={s['n_samples']}, "
            f"empty={s['n_empty']}, api_errors={s['n_api_errors']}"
        )


def main() -> None:
    args = parse_args()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: openai. Please run `pip install openai` first."
        ) from exc

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Please set environment variable: {args.api_key_env}"
        )

    try:
        import httpx
        client = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(300.0, connect=30.0),
            max_retries=5,
        )
    except ImportError:
        client = OpenAI(api_key=api_key, timeout=300.0, max_retries=5)

    if args.mode == "batch" and args.resume:
        files: List[str] = []
    else:
        files = get_eval_files(args.input, args.pattern)
        if not files:
            raise FileNotFoundError(
                f"No files found under {args.input} with pattern {args.pattern}"
            )

    cache = load_cache(args.cache_file)

    if args.mode == "batch":
        run_batch_mode(args, client, files, cache)
        return

    all_summaries = []
    only_tasks = {t.strip() for t in args.only_tasks.split(",") if t.strip()}

    for fp in files:
        task_name = infer_task_name_from_filename(fp)
        if (task_name not in only_tasks) and (not args.include_unsupported):
            print(f"[GPT-EVAL] Skip unsupported task file: {fp}")
            continue

        print(f"[GPT-EVAL] Processing: {fp}")
        evaluated = evaluate_single_file(
            client=client,
            file_path=fp,
            model=args.model,
            max_samples=args.max_samples,
            sleep_seconds=args.sleep,
            max_retries=args.max_retries,
            cache=cache,
        )
        summary = evaluated["summary"]
        summary["judge_version"] = JUDGE_VERSION
        all_summaries.append(summary)

        out_path = build_output_path(fp, args.output_dir, args.suffix)
        output_obj = evaluated["original"]
        output_obj["gpt_eval"] = summary
        output_obj["gpt_eval_details"] = evaluated["details"]
        safe_json_dump(out_path, output_obj)

        print(
            f"[GPT-EVAL] Done: acc={summary['accuracy']:.4f} "
            f"({summary['n_correct']}/{summary['n_samples']}) -> {out_path}"
        )

    save_cache(args.cache_file, cache)

    print("\n===== GPT Re-evaluation Summary =====")
    for s in all_summaries:
        print(
            f"{os.path.basename(s['file'])}: "
            f"acc={s['accuracy']:.4f}, n={s['n_samples']}, "
            f"empty={s.get('n_empty', 0)}, api_errors={s['n_api_errors']}"
        )


if __name__ == "__main__":
    main()
