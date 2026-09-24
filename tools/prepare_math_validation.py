#!/usr/bin/env python3
"""Convert bundled math JSONL files to verl-compatible validation Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")


def convert(source: Path, benchmark: str) -> list[dict]:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    "data_source": benchmark,
                    "prompt": [{"role": "user", "content": item["problem"]}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": str(item["answer"])},
                    "extra_info": {
                        "split": "test",
                        "index": index,
                        "id": str(item.get("id", index)),
                    },
                }
            )
    if not rows:
        raise ValueError(f"no records found in {source}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks/math_eval/data/training_validation",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for benchmark in BENCHMARKS:
        source = ROOT / f"benchmarks/math_eval/data/{benchmark}/test.jsonl"
        output = output_dir / f"{benchmark}.parquet"
        records = convert(source, benchmark)
        Dataset.from_list(records).to_parquet(str(output))
        print(f"Wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
