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
"""Preprocess ``AI-MO/NuminaMath-TIR`` into the MOPD-Router *composite* pool.

NuminaMath-TIR is a natural **math x code** composite: each trajectory mixes
natural-language reasoning with Python code blocks (tool-integrated reasoning).
For on-policy distillation we only keep the *prompt* (the student rolls out the
response itself); the solution is used solely to (a) extract the boxed final
answer as the reward ground-truth, and (b) count code fences as a free
skill-boundary signal for later CUSUM / switch-point alignment analysis.

Output parquet schema matches ``RLHFDataset`` expectations:
    data_source : "numina_tir"
    prompt      : [{"role":"system",...}, {"role":"user",...}]
    ability     : "math"
    reward_model: {"style":"rule","ground_truth": <boxed answer str>}
    extra_info  : {split, index, source_dataset, pool, skill_tags,
                   code_fence_count}
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

DATA_SOURCE = "numina_tir"
HF_ID = "AI-MO/NuminaMath-TIR"

# System prompt that induces the composite math x code trajectory. Asking the
# student to interleave reasoning and executable Python (and to verify its own
# intermediate results) is what makes a *single* response demand both skills,
# which is the whole point of the token-level router value proposition.
TIR_SYSTEM_PROMPT = (
    "You are a mathematics assistant that solves problems with tool-integrated "
    "reasoning. Think step by step in natural language, and whenever useful "
    "write Python code inside fenced blocks (```python ... ```) to compute or "
    "verify intermediate quantities, then continue reasoning from the result. "
    "Always place your final answer inside \\boxed{}."
)


def extract_solution(solution_str: str):
    return remove_boxed(last_boxed_only_string(solution_str))


def count_code_fences(solution_str: str) -> int:
    # Count code *blocks* (one open + one close fence each). The regex matches
    # both opening fences (```python) and bare closing fences (```), so integer
    # division by two yields the number of blocks = number of math->code
    # transitions, the free skill-boundary ground truth for CUSUM alignment.
    n_fence_lines = len(re.findall(r"(?m)^\s*```[a-zA-Z0-9_+-]*\s*$", solution_str))
    return n_fence_lines // 2


def make_map_fn(split: str):
    def process_fn(example, idx):
        problem = example.get("problem") or ""
        solution = example.get("solution") or ""

        answer = extract_solution(solution)
        # Skip rows we cannot verify (no boxed answer). OPD reward is
        # answer-equivalence, so ground-truth-less rows are unusable.
        if not answer:
            return {"__bad__": True}

        data = {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": TIR_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "source_dataset": "NuminaMath-TIR",
                # composite pool: single prompt requires >=2 skills.
                "pool": "composite",
                "skill_tags": ["math", "code"],
                # free skill-boundary ground truth for CUSUM alignment eval.
                "code_fence_count": count_code_fences(solution),
                "answer": answer,
            },
        }
        return data

    return process_fn


def _filter_bad(batch):
    # ``datasets.map`` returns the raw dict including the sentinel key when
    # extract failed; drop those rows before writing parquet.
    import datasets as _ds

    kept = [i for i, v in enumerate(batch["__bad__"]) if not v]
    out = {k: [batch[k][i] for i in kept] for k in batch if k != "__bad__"}
    return _ds.Dataset.from_dict(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Optional local path to the raw dataset.")
    parser.add_argument(
        "--local_save_dir", default="~/data/mopd_router/numina_tir", help="Output directory for parquet."
    )
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--max_problem_chars", type=int, default=6000, help="Skip overly long problems.")
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading {HF_ID} from huggingface...", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(HF_ID)

    train_dataset = dataset["train"]
    # Optional test split if present (NuminaMath-TIR ships a 99-row test split).
    test_dataset = dataset.get("test")

    train_dataset = train_dataset.map(
        function=make_map_fn("train"), with_indices=True, num_proc=args.num_proc
    )
    # Remove rows flagged unusable.
    train_dataset = train_dataset.filter(
        function=lambda ex: not ex.get("__bad__", False), num_proc=args.num_proc
    )
    # Drop overly long problems to respect max_prompt_length downstream.
    train_dataset = train_dataset.filter(
        function=lambda ex: len(ex["prompt"][-1]["content"]) <= args.max_problem_chars,
        num_proc=args.num_proc,
    )

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=args.num_proc)
        test_dataset = test_dataset.filter(function=lambda ex: not ex.get("__bad__", False), num_proc=args.num_proc)
        test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Example artifact for quick inspection.
    import json

    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_dataset)} train rows to {local_dir}", flush=True)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
