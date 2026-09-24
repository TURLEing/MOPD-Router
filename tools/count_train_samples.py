#!/usr/bin/env python3
"""Count samples in the configured parquet training dataset.

The script reads parquet metadata only, so it does not load the dataset into
memory. A directory is treated as a sharded dataset and searched recursively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = ROOT / "benchmarks/math_eval/data/mopd_router_unlabeled_60k/train.parquet"


def parquet_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".parquet", ".pq"}:
            raise ValueError(f"not a parquet file: {dataset_path}")
        return [dataset_path]
    if dataset_path.is_dir():
        files = sorted(
            path
            for path in dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".parquet", ".pq"}
        )
        if not files:
            raise ValueError(f"no parquet files found under {dataset_path}")
        return files
    raise ValueError(f"path does not exist: {dataset_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count parquet training samples using file metadata only."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help=f"parquet file or sharded parquet directory (default: {DEFAULT_TRAIN_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_paths = args.paths or [str(DEFAULT_TRAIN_PATH)]

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "error: pyarrow is required; install the repository dependencies or run "
            "`python3 -m pip install pyarrow`",
            file=sys.stderr,
        )
        return 2

    grand_total = 0
    total_files = 0
    try:
        for raw_path in raw_paths:
            dataset_path = Path(raw_path).expanduser()
            files = parquet_files(dataset_path)
            sample_count = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
            print(f"{dataset_path}: {sample_count:,} samples ({len(files)} parquet file(s))")
            grand_total += sample_count
            total_files += len(files)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if len(raw_paths) > 1:
        print(f"TOTAL: {grand_total:,} samples ({total_files} parquet file(s))")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
