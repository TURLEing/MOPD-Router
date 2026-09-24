#!/usr/bin/env python3
"""Run math generation with one independent TP=1 vLLM replica per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data-parallel math evaluation")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True, help="per GPU replica")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-token-ids")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def count_records(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def balanced_ranges(size: int, workers: int) -> list[tuple[int, int]]:
    quotient, remainder = divmod(size, workers)
    ranges = []
    begin = 0
    for index in range(workers):
        width = quotient + int(index < remainder)
        ranges.append((begin, begin + width))
        begin += width
    return ranges


def main() -> int:
    args = parse_args()
    input_file = Path(args.input_file).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    metrics_file = Path(args.metrics_file).expanduser().resolve()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"--gpus contains duplicate GPU IDs: {args.gpus}")

    problem_count = count_records(input_file)
    if problem_count == 0:
        raise ValueError(f"input dataset is empty: {input_file}")
    gpus = gpus[:problem_count]
    ranges = balanced_ranges(problem_count, len(gpus))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = output_file.parent / ".dp_runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    workers = []
    for shard_index, (gpu, (begin, end)) in enumerate(zip(gpus, ranges)):
        shard_output = run_root / f"shard_{shard_index:02d}.jsonl"
        shard_metrics = run_root / f"shard_{shard_index:02d}_metrics.json"
        shard_log = run_root / f"shard_{shard_index:02d}.log"
        command = [
            sys.executable,
            str(ROOT / "benchmarks/math_eval/eval_math.py"),
            "--input_file", str(input_file),
            "--model_path", args.model_path,
            "--output_file", str(shard_output),
            "--metrics_file", str(shard_metrics),
            "--max_tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--top_p", str(args.top_p),
            "--top_k", str(args.top_k),
            "--max_num_seqs", str(args.max_num_seqs),
            "--n", str(args.n),
            "--begin_idx", str(begin),
            "--end_idx", str(end),
            "--seed", str(args.seed),
        ]
        if args.stop_token_ids:
            command.extend(["--stop_token_ids", args.stop_token_ids])
        if args.enable_thinking:
            command.append("--enable_thinking")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        log_handle = shard_log.open("w", encoding="utf-8")
        started_at = now_iso()
        started_epoch = time.time()
        print(
            f"[math-dp] shard {shard_index + 1}/{len(gpus)}: "
            f"problems [{begin}, {end}) on GPU {gpu}; log: {shard_log}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        workers.append({
            "shard_index": shard_index,
            "gpu": gpu,
            "begin": begin,
            "end": end,
            "output": shard_output,
            "metrics": shard_metrics,
            "log": shard_log,
            "log_handle": log_handle,
            "process": process,
            "started_at": started_at,
            "started_epoch": started_epoch,
        })

    failed = False
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        futures = {executor.submit(worker["process"].wait): worker for worker in workers}
        for future in as_completed(futures):
            worker = futures[future]
            return_code = future.result()
            worker["log_handle"].close()
            worker["return_code"] = return_code
            worker["finished_at"] = now_iso()
            worker["duration_seconds"] = round(time.time() - worker["started_epoch"], 3)
            failed = failed or return_code != 0
            print(
                f"[math-dp] shard {worker['shard_index'] + 1}/{len(gpus)} "
                f"finished in {worker['duration_seconds']:.1f}s (exit {return_code})",
                flush=True,
            )

    if failed:
        print(f"[math-dp] one or more shards failed; inspect {run_root}", file=sys.stderr)
        return 1

    records = []
    shard_metrics_payloads = []
    for worker in workers:
        if not worker["output"].is_file() or not worker["metrics"].is_file():
            raise FileNotFoundError(f"missing output from shard {worker['shard_index']}")
        with worker["output"].open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
        shard_metrics_payloads.append(json.loads(worker["metrics"].read_text(encoding="utf-8")))
    if len(records) != problem_count:
        raise RuntimeError(
            f"incomplete math shards: expected {problem_count} records, got {len(records)}"
        )

    temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_output.replace(output_file)

    total_predictions = sum(item["total_predictions"] for item in shard_metrics_payloads)
    correct_predictions = sum(item["correct_predictions"] for item in shard_metrics_payloads)
    passed_problems = sum(
        item.get("passed_problems", round(item[f"pass@{args.n}"] * item["num_problems"]))
        for item in shard_metrics_payloads
    )
    average_output_tokens = sum(
        item["average_output_tokens"] * item["num_problems"]
        for item in shard_metrics_payloads
    ) / problem_count
    public_workers = [
        {
            "shard_index": worker["shard_index"],
            "gpu": worker["gpu"],
            "begin_idx": worker["begin"],
            "end_idx": worker["end"],
            "started_at": worker["started_at"],
            "finished_at": worker["finished_at"],
            "duration_seconds": worker["duration_seconds"],
            "return_code": worker["return_code"],
            "log": str(worker["log"]),
        }
        for worker in workers
    ]
    metrics = {
        "dataset": str(input_file),
        "num_problems": problem_count,
        "samples_per_problem": args.n,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy": correct_predictions / total_predictions if total_predictions else 0.0,
        "passed_problems": passed_problems,
        f"pass@{args.n}": passed_problems / problem_count,
        "average_output_tokens": average_output_tokens,
        "data_parallel_size": len(gpus),
        "tensor_parallel_size_per_replica": 1,
        "max_num_seqs_per_replica": args.max_num_seqs,
        "workers": public_workers,
    }
    write_json(metrics_file, metrics)
    print(
        f"[math-dp] merged {problem_count} problems / {total_predictions} generations "
        f"into {output_file}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
