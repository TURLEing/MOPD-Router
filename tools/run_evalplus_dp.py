#!/usr/bin/env python3
"""Run EvalPlus generation with one independent vLLM replica per GPU."""

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
EVALPLUS_SOURCE = ROOT / "benchmarks/code_eval/coding/evalplus"
EVALPLUS_DATASETS = {
    "humaneval": ROOT / "benchmarks/code_eval/data/HumanEvalPlus.jsonl",
    "mbpp": ROOT / "benchmarks/code_eval/data/MbppPlus.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data-parallel EvalPlus generation followed by one evaluation pass."
    )
    parser.add_argument("--dataset", required=True, choices=("humaneval", "mbpp"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated physical GPU IDs")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64, help="per GPU replica")
    parser.add_argument("--output-root", default=str(ROOT / "evalplus_results"))
    parser.add_argument("--timing-file")
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


def output_identifier(model: str, temperature: float) -> str:
    return model.strip("./").replace("/", "--") + f"_vllm_temp_{temperature}"


def merge_jsonl(inputs: list[Path], output: Path, task_order: dict[str, int]) -> None:
    records: list[tuple[int, int, dict]] = []
    for input_path in inputs:
        task_sample_index: dict[str, int] = {}
        with input_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record["task_id"]
                sample_index = task_sample_index.get(task_id, 0)
                task_sample_index[task_id] = sample_index + 1
                records.append((task_order[task_id], sample_index, record))
    records.sort(key=lambda item: (item[0], item[1]))
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for _, _, record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)


def load_task_ids(dataset: str) -> list[str]:
    """Load task order from the vendored dataset without network access."""
    dataset_path = EVALPLUS_DATASETS[dataset]
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing vendored EvalPlus dataset: {dataset_path}")

    task_ids = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(
                    f"missing task_id in {dataset_path} at line {line_number}"
                )
            task_ids.append(task_id)

    if not task_ids:
        raise ValueError(f"vendored EvalPlus dataset is empty: {dataset_path}")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task_id values in {dataset_path}")
    return task_ids


def main() -> int:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"--gpus contains duplicate GPU IDs: {args.gpus}")
    if args.n_samples < 1:
        raise ValueError("--n-samples must be at least 1")

    output_root = Path(args.output_root).expanduser().resolve()
    final_dir = output_root / args.dataset
    final_dir.mkdir(parents=True, exist_ok=True)
    identifier = output_identifier(args.model, args.temperature)
    final_samples = final_dir / f"{identifier}.jsonl"
    final_raw = final_dir / f"{identifier}.raw.jsonl"
    final_all = final_dir / f"{identifier}.all_solutions.jsonl"
    result_path = final_dir / f"{Path(args.model.rstrip('/')).name}_eval_results.json"

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = output_root / ".dp_runs" / args.dataset / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    timing_path = (
        Path(args.timing_file).expanduser().resolve()
        if args.timing_file
        else run_root / "timing.json"
    )
    timing = {
        "dataset": args.dataset,
        "model": str(Path(args.model).expanduser().resolve()),
        "started_at": now_iso(),
        "finished_at": None,
        "status": "running",
        "data_parallel_size": len(gpus),
        "tensor_parallel_size_per_replica": 1,
        "max_num_seqs_per_replica": args.max_num_seqs,
        "workers": [],
        "evaluation": None,
        "files": {
            "samples": str(final_samples),
            "raw_samples": str(final_raw),
            "all_solutions": str(final_all),
            "evaluation": str(result_path),
            "run_root": str(run_root),
        },
    }
    write_json(timing_path, timing)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = base_env.get("PYTHONPATH")
    base_env["PYTHONPATH"] = (
        f"{EVALPLUS_SOURCE}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(EVALPLUS_SOURCE)
    )
    base_env["HUMANEVAL_OVERRIDE_PATH"] = str(EVALPLUS_DATASETS["humaneval"])
    base_env["MBPP_OVERRIDE_PATH"] = str(EVALPLUS_DATASETS["mbpp"])

    workers = []
    for shard_index, gpu in enumerate(gpus):
        shard_root = run_root / f"shard_{shard_index:02d}"
        log_path = run_root / f"shard_{shard_index:02d}.log"
        command = [
            sys.executable,
            "-m",
            "evalplus.codegen",
            "--model",
            args.model,
            "--dataset",
            args.dataset,
            "--root",
            str(shard_root),
            "--backend",
            "vllm",
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--max-num-seqs",
            str(args.max_num_seqs),
            "--tp",
            "1",
            "--n-samples",
            str(args.n_samples),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(len(gpus)),
            "--trust_remote_code",
        ]
        env = base_env.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_handle = log_path.open("w", encoding="utf-8")
        started_epoch = time.time()
        started_at = now_iso()
        print(
            f"[evalplus-dp] shard {shard_index + 1}/{len(gpus)} "
            f"started on GPU {gpu}; log: {log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        workers.append(
            {
                "shard_index": shard_index,
                "gpu": gpu,
                "root": shard_root,
                "log": log_path,
                "log_handle": log_handle,
                "process": process,
                "started_epoch": started_epoch,
                "started_at": started_at,
            }
        )

    failed = False
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        future_to_worker = {
            executor.submit(worker["process"].wait): worker for worker in workers
        }
        for future in as_completed(future_to_worker):
            worker = future_to_worker[future]
            return_code = future.result()
            worker["log_handle"].close()
            finished_epoch = time.time()
            duration = round(finished_epoch - worker["started_epoch"], 3)
            worker_timing = {
                "shard_index": worker["shard_index"],
                "gpu": worker["gpu"],
                "started_at": worker["started_at"],
                "finished_at": now_iso(),
                "duration_seconds": duration,
                "return_code": return_code,
                "log": str(worker["log"]),
            }
            timing["workers"].append(worker_timing)
            print(
                f"[evalplus-dp] shard {worker['shard_index'] + 1}/{len(gpus)} "
                f"on GPU {worker['gpu']} finished in {duration:.1f}s "
                f"(exit {return_code})",
                flush=True,
            )
            failed = failed or return_code != 0
            write_json(timing_path, timing)
    timing["workers"].sort(key=lambda worker: worker["shard_index"])

    if failed:
        timing["status"] = "failed"
        timing["finished_at"] = now_iso()
        write_json(timing_path, timing)
        print("[evalplus-dp] one or more generation shards failed", file=sys.stderr)
        return 1

    task_ids = load_task_ids(args.dataset)
    task_order = {task_id: position for position, task_id in enumerate(task_ids)}
    sanitized_inputs = []
    raw_inputs = []
    all_inputs = []
    counts: dict[str, int] = {}
    for worker in workers:
        shard_dir = worker["root"] / args.dataset
        sanitized = shard_dir / f"{identifier}.jsonl"
        raw = shard_dir / f"{identifier}.raw.jsonl"
        all_solutions = shard_dir / f"{identifier}.all_solutions.jsonl"
        for required in (sanitized, raw, all_solutions):
            if not required.is_file():
                raise FileNotFoundError(f"missing shard output: {required}")
        sanitized_inputs.append(sanitized)
        raw_inputs.append(raw)
        all_inputs.append(all_solutions)
        with sanitized.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    task_id = json.loads(line)["task_id"]
                    counts[task_id] = counts.get(task_id, 0) + 1

    invalid = {
        task_id: counts.get(task_id, 0)
        for task_id in task_ids
        if counts.get(task_id, 0) != args.n_samples
    }
    unexpected = sorted(set(counts) - set(task_ids))
    if invalid or unexpected:
        raise RuntimeError(
            "incomplete EvalPlus shards: "
            f"{len(invalid)} tasks have the wrong sample count; "
            f"{len(unexpected)} unexpected task IDs"
        )

    merge_jsonl(sanitized_inputs, final_samples, task_order)
    merge_jsonl(raw_inputs, final_raw, task_order)
    merge_jsonl(all_inputs, final_all, task_order)
    print(
        f"[evalplus-dp] merged {len(task_ids) * args.n_samples} samples "
        f"into {final_samples}",
        flush=True,
    )

    evaluation_started = time.time()
    evaluation_started_at = now_iso()
    temporary_result_path = run_root / "eval_results.json"
    evaluation_command = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        args.dataset,
        "--samples",
        str(final_samples),
        "--output_file",
        str(temporary_result_path),
        "--min-time-limit",
        "10.0",
        "--gt-time-limit-factor",
        "8.0",
    ]
    return_code = subprocess.run(
        evaluation_command,
        cwd=ROOT,
        env=base_env,
        check=False,
    ).returncode
    timing["evaluation"] = {
        "started_at": evaluation_started_at,
        "finished_at": now_iso(),
        "duration_seconds": round(time.time() - evaluation_started, 3),
        "return_code": return_code,
    }
    timing["status"] = "completed" if return_code == 0 else "failed"
    timing["finished_at"] = now_iso()
    if return_code == 0:
        temporary_result_path.replace(result_path)
    write_json(timing_path, timing)
    if return_code:
        return return_code
    print(f"[evalplus-dp] evaluation saved to {result_path}", flush=True)
    print(f"[evalplus-dp] timing saved to {timing_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
