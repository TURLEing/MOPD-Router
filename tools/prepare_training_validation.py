#!/usr/bin/env python3
"""Build verl-compatible LiveCodeBench v6 and IFBench validation parquet files."""

from __future__ import annotations

import argparse
import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LCB_INPUT = ROOT / "benchmarks/code_eval/coding/LiveCodeBench/code_generation_lite/test6.jsonl"
DEFAULT_IFBENCH_INPUT = ROOT / "benchmarks/IFBench/data/IFBench_test.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"validation source file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _decode_lcb_tests(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        decoded = json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))
    if not isinstance(decoded, list):
        raise ValueError("LiveCodeBench test cases must decode to a list")
    return decoded


def _lcb_prompt(example: dict[str, Any]) -> str:
    prompt = (
        "You will be given a question (problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests.\n\n"
        f"Question: {example['question_content']}\n\n"
    )
    starter_code = example.get("starter_code", "")
    if starter_code:
        return prompt + (
            "Use the following starter code to write the solution. Return the complete solution inside a "
            f"Python code block.\n```python\n{starter_code}\n```"
        )
    return prompt + (
        "Read input from stdin and write the answer to stdout. Do not hard-code the sample inputs. "
        "Return the complete solution inside a Python code block."
    )


def convert_livecodebench(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        tests = _decode_lcb_tests(example["public_test_cases"]) + _decode_lcb_tests(example["private_test_cases"])
        metadata = example.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        ground_truth = {
            "inputs": [test["input"] for test in tests],
            "outputs": [test["output"] for test in tests],
            "fn_name": metadata.get("func_name"),
        }
        rows.append(
            {
                "data_source": "livecodebench/code_generation_lite",
                "prompt": [{"role": "user", "content": _lcb_prompt(example)}],
                "ability": "code",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth)},
                "extra_info": {
                    "split": "test6",
                    "index": index,
                    "question_id": str(example.get("question_id", index)),
                    "platform": str(example.get("platform", "unknown")),
                },
            }
        )
    return rows


def convert_ifbench(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, example in enumerate(examples):
        verifier_spec = {
            "key": example["key"],
            "prompt": example["prompt"],
            "instruction_id_list": example["instruction_id_list"],
            "kwargs": example["kwargs"],
        }
        rows.append(
            {
                "data_source": "ifbench",
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "ability": "instruction_following",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(verifier_spec)},
                "extra_info": {"split": "test", "index": index, "key": str(example["key"])},
            }
        )
    return rows


def _write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output))
    print(f"Wrote {len(rows)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--livecodebench-input", type=Path, default=DEFAULT_LCB_INPUT)
    parser.add_argument("--ifbench-input", type=Path, default=DEFAULT_IFBENCH_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lcb_rows = convert_livecodebench(_read_jsonl(args.livecodebench_input))
    ifbench_rows = convert_ifbench(_read_jsonl(args.ifbench_input))
    _write_parquet(lcb_rows, args.output_dir / "livecodebench_v6.parquet")
    _write_parquet(ifbench_rows, args.output_dir / "ifbench.parquet")


if __name__ == "__main__":
    main()
