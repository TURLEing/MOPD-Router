#!/usr/bin/env python3
"""Extract the headline metrics from a full-evaluation metrics directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


MATH_BENCHMARKS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")
EVALPLUS_BENCHMARKS = ("humaneval", "mbpp")
INSTRUCTION_BENCHMARKS = ("ifeval", "ifbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print headline scores from evaluate_checkpoint.py metrics JSON files."
    )
    parser.add_argument("metrics_dir", type=Path, help="Directory containing metrics JSON files")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the extracted result as machine-readable JSON",
    )
    return parser.parse_args()


def load_benchmarks(metrics_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    benchmarks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for path in sorted(metrics_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        file_benchmarks = payload.get("benchmarks")
        if not isinstance(file_benchmarks, dict):
            errors.append(f"{path.name}: missing object key 'benchmarks'")
            continue

        for name, result in file_benchmarks.items():
            if isinstance(result, dict):
                benchmarks[name] = result
            else:
                errors.append(f"{path.name}: benchmark {name!r} is not an object")

    return benchmarks, errors


def benchmark_status(result: dict[str, Any] | None) -> tuple[str, str | None]:
    if result is None:
        return "missing", None
    status = str(result.get("status", "unknown"))
    error = result.get("error")
    return status, str(error) if error else None


def metric_object(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    metrics = result.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def select(
    benchmarks: dict[str, dict[str, Any]],
    name: str,
    extractor: Any,
) -> dict[str, Any]:
    result = benchmarks.get(name)
    status, error = benchmark_status(result)
    selected: dict[str, Any] = {"status": status}
    if error:
        selected["error"] = error
    selected.update(extractor(metric_object(result)))
    return selected


def extract_summary(
    metrics_dir: Path,
    benchmarks: dict[str, dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "metrics_dir": str(metrics_dir),
        "math": {},
        "evalplus": {},
        "livecodebench": {},
        "instruction": {},
    }

    for name in MATH_BENCHMARKS:
        summary["math"][name] = select(
            benchmarks,
            name,
            lambda metrics: {
                "accuracy": metrics.get("accuracy"),
                "pass@8": metrics.get("pass@8"),
            },
        )

    for name in EVALPLUS_BENCHMARKS:
        summary["evalplus"][name] = select(
            benchmarks,
            name,
            lambda metrics: {"pass_at_k": metrics.get("pass_at_k")},
        )

    summary["livecodebench"] = select(
        benchmarks,
        "livecodebench",
        lambda metrics: {"pass@1": metrics.get("pass@1")},
    )

    for name in INSTRUCTION_BENCHMARKS:
        summary["instruction"][name] = select(
            benchmarks,
            name,
            lambda metrics: {
                "strict": {
                    "prompt_accuracy": (
                        metrics.get("strict", {}).get("prompt_accuracy")
                        if isinstance(metrics.get("strict"), dict)
                        else None
                    ),
                    "instruction_accuracy": (
                        metrics.get("strict", {}).get("instruction_accuracy")
                        if isinstance(metrics.get("strict"), dict)
                        else None
                    ),
                }
            },
        )

    if errors:
        summary["read_errors"] = errors
    return summary


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def status_suffix(result: dict[str, Any]) -> str:
    status = result["status"]
    if status == "completed":
        return ""
    suffix = f" [status={status}"
    if result.get("error"):
        suffix += f"; error={result['error']}"
    return suffix + "]"


def print_human(summary: dict[str, Any]) -> None:
    print("=== Evaluation metrics summary ===")
    print(f"metrics_dir: {summary['metrics_dir']}")

    print("\nMath")
    for name, result in summary["math"].items():
        print(
            f"  {name:<12} "
            f"accuracy={format_value(result['accuracy'])}  "
            f"pass@8={format_value(result['pass@8'])}"
            f"{status_suffix(result)}"
        )

    print("\nEvalPlus")
    for name, result in summary["evalplus"].items():
        print(
            f"  {name:<12} pass_at_k={format_value(result['pass_at_k'])}"
            f"{status_suffix(result)}"
        )

    result = summary["livecodebench"]
    print("\nLiveCodeBench")
    print(
        f"  livecodebench pass@1={format_value(result['pass@1'])}"
        f"{status_suffix(result)}"
    )

    print("\nInstruction following (strict)")
    for name, result in summary["instruction"].items():
        strict = result["strict"]
        print(
            f"  {name:<12} "
            f"prompt_accuracy={format_value(strict['prompt_accuracy'])}  "
            f"instruction_accuracy={format_value(strict['instruction_accuracy'])}"
            f"{status_suffix(result)}"
        )

    if summary.get("read_errors"):
        print("\nMetric file read warnings:", file=sys.stderr)
        for error in summary["read_errors"]:
            print(f"  - {error}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    metrics_dir = args.metrics_dir.expanduser().resolve()
    if not metrics_dir.is_dir():
        print(f"Error: metrics directory does not exist: {metrics_dir}", file=sys.stderr)
        return 2

    benchmarks, errors = load_benchmarks(metrics_dir)
    summary = extract_summary(metrics_dir, benchmarks, errors)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
