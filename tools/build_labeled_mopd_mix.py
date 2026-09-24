#!/usr/bin/env python3
"""Build a deterministic domain-labeled MOPD mixture from three Parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_dataset


SOURCES = ("math", "code", "instruction_following")


def load_rows(path: Path, label: str, count: int, seed: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = load_dataset("parquet", data_files={"train": str(path)}, split="train")
    if len(dataset) < count:
        raise ValueError(f"{label}: requested {count:,} rows, but {path} has {len(dataset):,}")
    dataset = dataset.shuffle(seed=seed).select(range(count))
    rows = []
    for item in dataset:
        record = dict(item)
        extra_info = dict(record.get("extra_info") or {})
        extra_info["opd_teacher"] = label
        record["extra_info"] = extra_info
        rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in SOURCES:
        parser.add_argument(f"--{label.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--math-count", type=int, default=25_000)
    parser.add_argument("--code-count", type=int, default=25_000)
    parser.add_argument("--instruction-following-count", type=int, default=16_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    counts = {
        "math": args.math_count,
        "code": args.code_count,
        "instruction_following": args.instruction_following_count,
    }
    records = []
    for offset, label in enumerate(SOURCES):
        records.extend(
            load_rows(getattr(args, label), label, counts[label], args.seed + offset)
        )
    merged = Dataset.from_list(records).shuffle(seed=args.seed)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(str(output))
    manifest = {
        "output": str(output),
        "total_rows": len(merged),
        "counts": counts,
        "seed": args.seed,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
