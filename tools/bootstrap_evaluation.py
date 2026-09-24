#!/usr/bin/env python3
"""Paired bootstrap analysis for existing MOPD-Router evaluation runs.

This program only reads local filesystem paths and does not contact remote
storage APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


MATH = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")
CODE = ("humaneval_plus", "mbpp_plus", "livecodebench")
INSTRUCTION = ("ifeval", "ifbench")
BENCHMARKS = MATH + CODE + INSTRUCTION
METRICS = BENCHMARKS + ("math", "code", "if", "overall")
METHODS = ("standard_mopd", "mean", "expert_delta")


class InputError(RuntimeError):
    pass


def json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read {path}: {exc}") from exc


def jsonl_file(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read {path}: {exc}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise InputError(f"Expected non-empty JSON objects in {path}")
    return rows


def local_root(value: str) -> Path:
    if "://" in value:
        raise InputError(
            f"Only local filesystem paths are accepted, not {value!r}. "
            "Mount or copy the evaluation run to a local path first."
        )
    # Do not call Path.resolve() here. Some mounted filesystems expose logical
    # directory aliases/symlinks, and resolving them can rewrite a valid path.
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.absolute()
    if not root.is_dir():
        raise InputError(
            f"Evaluation run directory does not exist: {root} "
            f"(configured path: {value!r})"
        )
    return root


def add(scores: dict[str, float], key: Any, value: float, path: Path) -> None:
    sample_id = str(key)
    if key is None or sample_id in scores:
        raise InputError(f"Missing or duplicate sample ID {sample_id!r} in {path}")
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InputError(f"Invalid score {value!r} for {sample_id!r} in {path}")
    scores[sample_id] = float(value)


def math_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in jsonl_file(path):
        values = row.get("acc_list")
        if not isinstance(values, list) or not values:
            raise InputError(f"Missing acc_list in {path}")
        add(scores, row.get("id"), float(np.mean(values)), path)
    return scores


def instruction_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in jsonl_file(path):
        prompt = row.get("prompt")
        followed = row.get("follow_all_instructions")
        if not isinstance(prompt, str) or not isinstance(followed, bool):
            raise InputError(f"Malformed instruction result in {path}")
        add(scores, prompt, float(followed), path)
    return scores


def evalplus_scores(path: Path, dataset: str) -> dict[str, float]:
    raw = json_file(path)
    try:
        benchmark = raw["benchmarks"][dataset]
        evaluations = benchmark["metrics"]["eval"]
    except (KeyError, TypeError) as exc:
        raise InputError(f"Missing EvalPlus details for {dataset} in {path}") from exc
    if benchmark.get("status") != "completed":
        raise InputError(f"EvalPlus {dataset} is not completed in {path}")
    scores: dict[str, float] = {}
    for task_id, solutions in evaluations.items():
        if not isinstance(solutions, list) or not solutions:
            raise InputError(f"Empty solutions for {task_id} in {path}")
        add(
            scores,
            task_id,
            float(np.mean([solution.get("plus_status") == "pass" for solution in solutions])),
            path,
        )
    return scores


def lcb_scores(path: Path) -> dict[str, float]:
    raw = json_file(path)
    try:
        benchmark = raw["benchmarks"]["livecodebench"]
        details = benchmark["metrics"]["detail"]["pass@1"]
    except (KeyError, TypeError) as exc:
        raise InputError(f"Missing LiveCodeBench per-task details in {path}") from exc
    if benchmark.get("status") != "completed":
        raise InputError(f"LiveCodeBench is not completed in {path}")
    scores: dict[str, float] = {}
    for task_id, value in details.items():
        add(scores, task_id, float(value), path)
    return scores


def load_run(root: Path) -> dict[str, dict[str, float]]:
    scores = {
        name: math_scores(root / f"outputs/math/{name}/results.jsonl") for name in MATH
    }
    scores["ifeval"] = instruction_scores(
        root / "outputs/instruction/ifeval/eval_results_strict.jsonl"
    )
    scores["ifbench"] = instruction_scores(
        root / "outputs/instruction/ifbench/eval_results_strict.jsonl"
    )
    evalplus = root / "metrics/evalplus.json"
    scores["humaneval_plus"] = evalplus_scores(evalplus, "humaneval")
    scores["mbpp_plus"] = evalplus_scores(evalplus, "mbpp")
    scores["livecodebench"] = lcb_scores(root / "metrics/livecodebench.json")
    return scores


def aggregates(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(values)
    result["math"] = np.mean([values[name] for name in MATH], axis=0)
    result["code"] = np.mean([values[name] for name in CODE], axis=0)
    result["if"] = np.mean([values[name] for name in INSTRUCTION], axis=0)
    result["overall"] = np.mean([values[name] for name in BENCHMARKS], axis=0)
    return result


def derived_seed(seed: int, setting: str, benchmark: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{setting}\0{benchmark}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return 100.0 * float(low), 100.0 * float(high)


def holm(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = previous
    return adjusted


def parse_config(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    raw = json_file(path)
    settings = raw.get("settings") if isinstance(raw, dict) else None
    if not isinstance(settings, dict) or not settings:
        raise InputError("Config requires a non-empty 'settings' object")
    for setting, methods in settings.items():
        if not isinstance(methods, dict):
            raise InputError(f"{setting!r} must contain method entries")
        missing = [method for method in METHODS if method not in methods]
        if missing:
            raise InputError(f"{setting!r} is missing methods: {missing}")
        for method in METHODS:
            value = methods[method]
            if isinstance(value, str):
                methods[method] = {"path": value}
            elif not isinstance(value, dict) or not isinstance(value.get("path"), str):
                raise InputError(f"{setting}.{method} requires a path")
    return settings


def analyze_setting(
    setting: str,
    specs: dict[str, dict[str, Any]],
    n_resamples: int,
    seed: int,
    confidence: float,
    chunk_size: int,
) -> dict[str, Any]:
    runs = {method: load_run(local_root(specs[method]["path"])) for method in METHODS}
    arrays: dict[str, dict[str, np.ndarray]] = {method: {} for method in METHODS}
    sample_counts: dict[str, int] = {}

    for benchmark in BENCHMARKS:
        reference = set(runs[METHODS[0]][benchmark])
        for method in METHODS[1:]:
            candidate = set(runs[method][benchmark])
            if candidate != reference:
                raise InputError(
                    f"{setting}/{benchmark}: sample IDs differ for {method}; "
                    f"missing={sorted(reference - candidate)[:5]}, "
                    f"extra={sorted(candidate - reference)[:5]}"
                )
        ids = sorted(reference)
        sample_counts[benchmark] = len(ids)
        for method in METHODS:
            arrays[method][benchmark] = np.asarray(
                [runs[method][benchmark][sample_id] for sample_id in ids], dtype=float
            )

    observed = {
        method: aggregates(
            {benchmark: float(arrays[method][benchmark].mean()) for benchmark in BENCHMARKS}
        )
        for method in METHODS
    }
    reps: dict[str, dict[str, np.ndarray]] = {
        method: {name: np.empty(n_resamples) for name in BENCHMARKS} for method in METHODS
    }
    for benchmark in BENCHMARKS:
        stacked = np.stack([arrays[method][benchmark] for method in METHODS])
        count = stacked.shape[1]
        rng = np.random.default_rng(derived_seed(seed, setting, benchmark))
        for start in range(0, n_resamples, chunk_size):
            stop = min(start + chunk_size, n_resamples)
            indices = rng.integers(0, count, size=(stop - start, count))
            means = stacked[:, indices].mean(axis=2)
            for method_index, method in enumerate(METHODS):
                reps[method][benchmark][start:stop] = means[method_index]
    for method in METHODS:
        reps[method] = aggregates(reps[method])

    result: dict[str, Any] = {"sample_counts": sample_counts, "methods": {}, "comparisons": {}}
    for method in METHODS:
        result["methods"][method] = {}
        for metric in METRICS:
            low, high = interval(reps[method][metric], confidence)
            result["methods"][method][metric] = {
                "estimate": 100.0 * float(observed[method][metric]),
                "ci_low": low,
                "ci_high": high,
            }
        expected = specs[method].get("expected_overall")
        if expected is not None:
            actual = result["methods"][method]["overall"]["estimate"]
            if abs(actual - float(expected)) > 0.015:
                raise InputError(
                    f"{setting}/{method}: recomputed Overall {actual:.4f} does not "
                    f"match expected {float(expected):.4f}; check the run path"
                )

    for baseline in ("standard_mopd", "mean"):
        name = f"expert_delta-minus-{baseline}"
        result["comparisons"][name] = {}
        for metric in METRICS:
            draws = reps["expert_delta"][metric] - reps[baseline][metric]
            delta = float(observed["expert_delta"][metric] - observed[baseline][metric])
            low, high = interval(draws, confidence)
            centered = draws - delta
            extreme = int(np.count_nonzero(np.abs(centered) >= abs(delta)))
            result["comparisons"][name][metric] = {
                "delta": 100.0 * delta,
                "ci_low": low,
                "ci_high": high,
                "p_value": (extreme + 1.0) / (n_resamples + 1.0),
                "probability_superior": (
                    int(np.count_nonzero(draws > 0.0)) + 1.0
                ) / (n_resamples + 2.0),
            }
    return result


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bootstrap_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "method_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("setting", "method", "metric", "estimate", "ci_low", "ci_high"))
        writer.writeheader()
        for setting, data in result["settings"].items():
            for method in METHODS:
                for metric in METRICS:
                    writer.writerow({"setting": setting, "method": method, "metric": metric, **data["methods"][method][metric]})
    comparison_fields = ("setting", "comparison", "metric", "delta", "ci_low", "ci_high", "p_value", "p_value_holm", "probability_superior")
    with (output_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for setting, data in result["settings"].items():
            for comparison, metrics in data["comparisons"].items():
                for metric in METRICS:
                    writer.writerow({"setting": setting, "comparison": comparison, "metric": metric, **metrics[metric], "p_value_holm": metrics[metric].get("p_value_holm", "")})
    lines = ["# Bootstrap evaluation report", ""]
    for setting, data in result["settings"].items():
        lines += [f"## {setting}", "", "| Method | Overall [95% CI] |", "|---|---:|"]
        for method in METHODS:
            row = data["methods"][method]["overall"]
            lines.append(f"| {method} | {row['estimate']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}] |")
        lines += ["", "| Comparison | Delta [95% CI] | p | Holm p |", "|---|---:|---:|---:|"]
        for name, metrics in data["comparisons"].items():
            row = metrics["overall"]
            lines.append(f"| {name} | {row['delta']:.2f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}] | {row['p_value']:.4g} | {row['p_value_holm']:.4g} |")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_910)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--chunk-size", type=int, default=1_000)
    args = parser.parse_args()
    try:
        if args.n_resamples < 100 or args.chunk_size < 1 or not 0.0 < args.confidence < 1.0:
            raise InputError("Invalid resampling parameters")
        settings = parse_config(args.config.resolve())
        result = {
            "analysis": {
                "n_resamples": args.n_resamples,
                "seed": args.seed,
                "confidence": args.confidence,
                "bootstrap_unit": "benchmark item; math generations clustered by problem",
                "overall": "unweighted macro-average of nine benchmarks",
            },
            "settings": {
                setting: analyze_setting(setting, specs, args.n_resamples, args.seed, args.confidence, args.chunk_size)
                for setting, specs in settings.items()
            },
        }
        primary = [
            metrics["overall"]
            for data in result["settings"].values()
            for metrics in data["comparisons"].values()
        ]
        for row, adjusted in zip(primary, holm([row["p_value"] for row in primary])):
            row["p_value_holm"] = adjusted
        write_outputs(result, args.output_dir.resolve())
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote bootstrap analysis to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
