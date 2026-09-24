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
"""Split ``allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`` into two single-domain
pools for the MOPD-Router 2x2 experiment (token-router vs prompt-router on
composite vs single-domain pools).

The dataset is 3 disjoint subsets, *not* per-prompt composite:
  - ``gsm8k`` / ``MATH`` rows (constraint_type is None): plain math, single
    skill.  -> ``data_source = "rlvr_math"``      (reward: math answer eq.)
  - ``ifeval`` rows (constraint_type present): a general question plus one
    verifiable IFEval constraint, single skill.
    -> ``data_source = "rlvr_ifeval"``  (reward: IF constraint check)

These subsets form the *single-domain* pool where a prompt-level router is
expected to be sufficient -- the contrast set for the NuminaMath-TIR and
synthesized math x IF composite pools.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

HF_ID = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"


def _user_text(messages):
    # The dataset always carries a single user message.
    if messages and len(messages) > 0:
        return messages[0].get("content", "") or ""
    return ""


def make_map_fn(split: str):
    def process_fn(example, idx):
        messages = example.get("messages") or []
        user_text = _user_text(messages)
        ds = example.get("dataset") or ""
        constraint_type = example.get("constraint_type")
        constraint = example.get("constraint")
        gt = example.get("ground_truth") or ""

        if ds in ("gsm8k", "MATH"):
            data_source = "rlvr_math"
            ability = "math"
            pool = "single_domain"
            skill_tags = ["math"]
        elif ds == "ifeval":
            data_source = "rlvr_ifeval"
            ability = "instruction_following"
            pool = "single_domain"
            skill_tags = ["instruction_following"]
        else:
            # Unknown subset; mark unusable.
            return {"__bad__": True}

        if not user_text or not gt:
            return {"__bad__": True}

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": user_text}],
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split,
                "index": idx,
                "source_dataset": "RLVR-GSM-MATH-IF",
                "source_subset": ds,
                "pool": pool,
                "skill_tags": skill_tags,
                "constraint_type": constraint_type,
                "constraint": constraint,
            },
        }
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument(
        "--local_save_dir", default="~/data/mopd_router/rlvr", help="Output root for the two subset parquets."
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_prompt_chars", type=int, default=6000)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading {HF_ID} from huggingface...", flush=True)
    if args.local_dataset_path is not None:
        ds = datasets.load_dataset(args.local_dataset_path)
    else:
        ds = datasets.load_dataset(HF_ID)
    train_dataset = ds["train"]

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=args.num_proc)
    train_dataset = train_dataset.filter(function=lambda ex: not ex.get("__bad__", False), num_proc=args.num_proc)
    train_dataset = train_dataset.filter(
        function=lambda ex: len(ex["prompt"][-1]["content"]) <= args.max_prompt_chars, num_proc=args.num_proc
    )

    math_subset = train_dataset.filter(
        function=lambda ex: ex["data_source"] == "rlvr_math", num_proc=args.num_proc
    )
    ifeval_subset = train_dataset.filter(
        function=lambda ex: ex["data_source"] == "rlvr_ifeval", num_proc=args.num_proc
    )

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    math_subset.to_parquet(os.path.join(local_dir, "math.parquet"))
    ifeval_subset.to_parquet(os.path.join(local_dir, "ifeval.parquet"))

    import json

    if len(math_subset) > 0:
        with open(os.path.join(local_dir, "math_example.json"), "w") as f:
            json.dump(math_subset[0], f, indent=2, ensure_ascii=False)
    if len(ifeval_subset) > 0:
        with open(os.path.join(local_dir, "ifeval_example.json"), "w") as f:
            json.dump(ifeval_subset[0], f, indent=2, ensure_ascii=False)

    print(
        f"Saved math={len(math_subset)} ifeval={len(ifeval_subset)} rows to {local_dir}",
        flush=True,
    )

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
