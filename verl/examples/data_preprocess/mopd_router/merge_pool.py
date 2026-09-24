# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Merge per-source parquets into a single MOPD-Router training pool.

Each input parquet already carries its own ``extra_info.pool`` and
``extra_info.skill_tags``. This script concatenates them, reindexes, and
optionally caps per-source counts for the 2x2 experiment pool composition
(e.g. equalize composite vs single-domain). It also writes a manifest with
per-source / per-pool row counts so the experiment config is reproducible.

No ``opd_teacher`` field is added: the token-level router handles teacher
selection at training time (prompt-level pre-select labels are intentionally
out of scope for this pipeline).

Implementation note: the four upstream sources have *heterogeneous*
``extra_info`` subfields (``code_fence_count`` for NuminaMath-TIR,
``constraint_type``/``constraint`` for RLVR, ``math_source_subset`` for the
synthetic composite). ``datasets.concatenate_datasets`` rejects mismatched
struct schemas, so we materialize rows to plain dicts and rebuild via
``Dataset.from_list``, which infers a union schema (missing keys -> null).
"""

import argparse
import collections
import json
import os

import datasets


def _read(path: str) -> "datasets.Dataset":
    # Use the ``datasets`` parquet loader so nested struct/list columns
    # (extra_info, reward_model, prompt) are materialized as Python objects
    # instead of raising ArrowNotImplementedError like pd.read_parquet would.
    return datasets.load_dataset("parquet", data_files={"train": path})["train"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input parquet files to concatenate (in priority order).",
    )
    parser.add_argument(
        "--cap_per_source",
        type=int,
        default=-1,
        help="Cap rows per data_source (-1 = no cap). Useful for balanced pools.",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/mopd_router/pool", help="Output dir for merged train.parquet."
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the merged pool (seed=42).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = []
    counts_by_source = collections.Counter()
    counts_by_pool = collections.Counter()
    for p in args.inputs:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        d = _read(p)
        if args.cap_per_source > 0 and len(d) > args.cap_per_source:
            d = d.select(range(args.cap_per_source))
        # Materialize rows as plain dicts. Extra_info subfields differ across
        # sources (code_fence_count vs constraint_type vs math_source_subset);
        # going through ``from_list`` below yields a union schema with nulls,
        # avoiding the strict-schema cast error concatenate_datasets raises.
        for ex in d.to_list():
            counts_by_source[ex["data_source"]] += 1
            pool = (ex.get("extra_info") or {}).get("pool", "unknown")
            counts_by_pool[pool] += 1
            records.append(ex)

    # Reindex sequentially across the merged pool.
    for i, ex in enumerate(records):
        extra = ex.get("extra_info") or {}
        extra["index"] = i
        ex["extra_info"] = extra

    merged = datasets.Dataset.from_list(records) if records else datasets.Dataset.from_dict({})
    if args.shuffle:
        merged = merged.shuffle(seed=args.seed)

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    merged.to_parquet(os.path.join(local_dir, "train.parquet"))

    manifest = {
        "total_rows": len(merged),
        "by_data_source": dict(counts_by_source),
        "by_pool": dict(counts_by_pool),
        "inputs": args.inputs,
        "cap_per_source": args.cap_per_source,
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    with open(os.path.join(local_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
