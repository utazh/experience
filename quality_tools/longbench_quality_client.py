#!/usr/bin/env python3
import argparse
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from transformers import AutoTokenizer


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = {}
    for token in prediction_tokens:
        common[token] = min(prediction_tokens.count(token), ground_truth_tokens.count(token))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(1, len(prediction_tokens))
    recall = num_same / max(1, len(ground_truth_tokens))
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, answers: List[str]) -> float:
    return max((qa_f1_score(prediction, answer) for answer in answers), default=0.0)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path, limit: int, start: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start:
                continue
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def truncate_middle(tokenizer, text: str, max_tokens: int) -> Tuple[str, int, bool]:
    ids = tokenizer(text, truncation=False, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text, len(ids), False
    half = max_tokens // 2
    kept = ids[:half] + ids[-(max_tokens - half):]
    return tokenizer.decode(kept, skip_special_tokens=True), len(ids), True


def apply_chat_template(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_records(
    datasets: List[str],
    samples: int,
    data_root: Path,
    official_root: Path,
    tokenizer,
    max_model_len: int,
    reserve_tokens: int,
    use_chat_template: bool,
    start_index: int = 0,
) -> List[Dict[str, Any]]:
    prompt_map = load_json(official_root / "config" / "dataset2prompt.json")
    max_new_map = load_json(official_root / "config" / "dataset2maxlen.json")
    records = []
    for dataset in datasets:
        rows = load_jsonl(data_root / f"{dataset}.jsonl", samples, start=start_index)
        prompt_format = prompt_map[dataset]
        max_new_tokens = int(max_new_map[dataset])
        max_prompt_tokens = max_model_len - max_new_tokens - reserve_tokens
        if max_prompt_tokens <= 0:
            raise ValueError(f"max_prompt_tokens <= 0 for {dataset}")
        for offset, row in enumerate(rows):
            idx = start_index + offset
            raw_prompt = prompt_format.format(**row)
            truncated_prompt, raw_token_count, truncated = truncate_middle(
                tokenizer, raw_prompt, max_prompt_tokens - 64
            )
            prompt = apply_chat_template(tokenizer, truncated_prompt, use_chat_template)
            prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
            if len(prompt_ids) > max_prompt_tokens:
                prompt = tokenizer.decode(prompt_ids[:max_prompt_tokens], skip_special_tokens=False)
                prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
            records.append({
                "uid": f"{dataset}-{idx}",
                "dataset": dataset,
                "sample_index": idx,
                "prompt": prompt,
                "answers": row.get("answers", []),
                "all_classes": row.get("all_classes", []),
                "length": row.get("length"),
                "max_new_tokens": max_new_tokens,
                "raw_prompt_tokens": raw_token_count,
                "prompt_tokens": len(prompt_ids),
                "truncated": truncated,
            })
    return records


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    pieces: List[str] = []
    with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choice = chunk.get("choices", [{}])[0]
            text = choice.get("text", "") or ""
            if text and first_token_time is None:
                first_token_time = time.perf_counter()
            pieces.append(text)
    end = time.perf_counter()
    return {
        "text": "".join(pieces),
        "ttft_ms": None if first_token_time is None else (first_token_time - start) * 1000,
        "latency_ms": (end - start) * 1000,
    }


def wait_server(base_url: str, timeout_s: float) -> None:
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return
            last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as exc:
            last_err = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server not ready at {url}: {last_err}")


def mean(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="qasper,multifieldqa_en")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--data-root", default="/data1/datasets/LongBench/data")
    parser.add_argument("--official-root", default="/home/panzihang/src/experience/LongBench-official/LongBench")
    parser.add_argument("--model-path", default="/data1/llm/Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--served-model", default="qwen2-7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--reserve-tokens", type=int, default=32)
    parser.add_argument("--warm-max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    records = build_records(
        datasets=datasets,
        samples=args.samples,
        data_root=Path(args.data_root),
        official_root=Path(args.official_root),
        tokenizer=tokenizer,
        max_model_len=args.max_model_len,
        reserve_tokens=args.reserve_tokens,
        use_chat_template=not args.no_chat_template,
        start_index=args.start_index,
    )
    with (out_dir / "prepared_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "prompt"}, ensure_ascii=False) + "\n")

    wait_server(args.base_url, args.server_timeout)
    warm_records = []
    for i, rec in enumerate(records, start=1):
        result = stream_completion(
            args.base_url,
            args.served_model,
            rec["prompt"],
            args.warm_max_tokens,
            args.temperature,
            args.request_timeout,
        )
        warm_records.append({"uid": rec["uid"], **result})
        print(f"warm {i}/{len(records)} {rec['uid']} ttft_ms={result['ttft_ms']}", flush=True)

    scored_records = []
    pred_files: Dict[str, Any] = {}
    for dataset in datasets:
        pred_files[dataset] = (out_dir / f"{dataset}.jsonl").open("w", encoding="utf-8")
    try:
        for i, rec in enumerate(records, start=1):
            result = stream_completion(
                args.base_url,
                args.served_model,
                rec["prompt"],
                int(rec["max_new_tokens"]),
                args.temperature,
                args.request_timeout,
            )
            pred = result["text"].strip()
            score = best_f1(pred, rec["answers"])
            scored = {
                "uid": rec["uid"],
                "dataset": rec["dataset"],
                "sample_index": rec["sample_index"],
                "pred": pred,
                "answers": rec["answers"],
                "all_classes": rec["all_classes"],
                "length": rec["length"],
                "f1": score,
                "ttft_ms": result["ttft_ms"],
                "latency_ms": result["latency_ms"],
                "prompt_tokens": rec["prompt_tokens"],
                "raw_prompt_tokens": rec["raw_prompt_tokens"],
                "truncated": rec["truncated"],
            }
            scored_records.append(scored)
            pred_files[rec["dataset"]].write(json.dumps(scored, ensure_ascii=False) + "\n")
            pred_files[rec["dataset"]].flush()
            print(
                f"score {i}/{len(records)} {rec['uid']} f1={score*100:.2f} ttft_ms={result['ttft_ms']}",
                flush=True,
            )
    finally:
        for f in pred_files.values():
            f.close()

    by_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        subset = [r for r in scored_records if r["dataset"] == dataset]
        by_dataset[dataset] = {
            "samples": len(subset),
            "f1": round(100 * mean(r["f1"] for r in subset), 4),
            "mean_ttft_ms": round(mean(r["ttft_ms"] for r in subset), 4),
            "mean_latency_ms": round(mean(r["latency_ms"] for r in subset), 4),
            "mean_prompt_tokens": round(mean(r["prompt_tokens"] for r in subset), 2),
            "truncated": sum(1 for r in subset if r["truncated"]),
        }
    summary = {
        "policy": args.policy,
        "model_path": args.model_path,
        "served_model": args.served_model,
        "datasets": by_dataset,
        "overall": {
            "samples": len(scored_records),
            "f1": round(100 * mean(r["f1"] for r in scored_records), 4),
            "mean_ttft_ms": round(mean(r["ttft_ms"] for r in scored_records), 4),
            "mean_latency_ms": round(mean(r["latency_ms"] for r in scored_records), 4),
        },
        "notes": {
            "prompt_source": str(Path(args.official_root) / "config" / "dataset2prompt.json"),
            "metric": "LongBench qa_f1_score formula for qasper and multifieldqa_en",
            "cache_use": "first pass warms LMCache; second identical prompt is scored",
        },
    }
    with (out_dir / "warm_records.jsonl").open("w", encoding="utf-8") as f:
        for rec in warm_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
