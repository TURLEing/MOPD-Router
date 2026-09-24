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
"""Synthesize a controlled *math x instruction-following* composite pool.

The ``allenai/RLVR-GSM-MATH-IF`` dataset is three disjoint single-domain
subsets (math, and ifeval) -- it does *not* contain per-prompt composite. To
obtain prompts whose single response trajectory requires BOTH mathematical
reasoning AND instruction following (the regime where a token-level router is
provably valuable), we wrap each math question with a sampled, math-compatible
IFEval constraint. The constraint and its programmatic verifier (func_name +
args) are borrowed verbatim from the dataset's ifeval subset, so the reward is
fully verifiable: math answer equivalence (math_verify) AND constraint check
(ifeval), scored densely as 0.5 * math + 0.5 * IF.

Constraints that are structurally incompatible with a \\boxed{} math final
answer (All Lowercase / All Uppercase / No Commas / JSON Format / Quotation /
Two Responses / Response Language) are filtered out by default so the
synthetic composite remains satisfiable.
"""

import argparse
import copy
import json
import os
import random

import datasets

from verl.utils.reward_score.ifeval import MATH_INCOMPATIBLE_CONSTRAINTS

DATA_SOURCE = "math_if_composite"


def _read_parquet(path: str) -> "datasets.Dataset":
    # Use the ``datasets`` parquet loader so nested struct/list columns
    # (extra_info, reward_model, prompt) are materialized as Python objects
    # instead of raising ArrowNotImplementedError like pd.read_parquet would.
    return datasets.load_dataset("parquet", data_files={"train": path})["train"]


def _build_constraint_pool(ifeval_ds, allow_incompatible: bool):
    pool = []
    for i in range(len(ifeval_ds)):
        ex = ifeval_ds[i]
        ct = ex.get("extra_info", {}).get("constraint_type")
        constraint_text = ex.get("extra_info", {}).get("constraint")
        gt = ex.get("reward_model", {}).get("ground_truth")
        if not ct or not constraint_text or not gt:
            continue
        if not allow_incompatible and ct in MATH_INCOMPATIBLE_CONSTRAINTS:
            continue
        try:
            spec = json.loads(gt) if isinstance(gt, str) else dict(gt)
        except (ValueError, TypeError):
            continue
        if not isinstance(spec, dict) or "func_name" not in spec:
            continue
        pool.append({"constraint_type": ct, "constraint": constraint_text, "spec": spec, "src_index": i})
    return pool


def _wrap(math_text: str, constraint_text: str) -> str:
    return (
        f"{math_text.strip()}\n\n{constraint_text.strip()}\n\n"
        "Please reason step by step, satisfy the instruction constraint above, "
        "and put your final answer within \\boxed{}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math_parquet", required=True, help="rlvr_math subset parquet from rlvr_math_if.py.")
    parser.add_argument("--ifeval_parquet", required=True, help="rlvr_ifeval subset parquet from rlvr_math_if.py.")
    parser.add_argument(
        "--local_save_dir", default="~/data/mopd_router/math_if_composite", help="Output dir for composite parquet."
    )
    parser.add_argument("--max_composite", type=int, default=-1, help="Cap composite rows (-1 = all math rows).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_incompatible", action="store_true", help="Keep math-incompatible constraints too.")
    parser.add_argument("--max_prompt_chars", type=int, default=6000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    math_ds = _read_parquet(args.math_parquet)
    ifeval_ds = _read_parquet(args.ifeval_parquet)
    pool = _build_constraint_pool(ifeval_ds, args.allow_incompatible)
    if not pool:
        raise RuntimeError("No usable IF constraints found in the ifeval subset.")

    records = []
    n_math = len(math_ds)
    cap = n_math if args.max_composite < 0 else min(args.max_composite, n_math)
    used = 0
    for j in range(n_math):
        if used >= cap:
            break
        m = math_ds[j]
        math_text = m["prompt"][-1]["content"]
        answer = m["reward_model"]["ground_truth"]
        m_extra = m.get("extra_info", {}) or {}
        if len(math_text) > args.max_prompt_chars:
            continue

        chosen = rng.choice(pool)
        spec = copy.deepcopy(chosen["spec"])
        # validate_repeat_prompt needs the verbatim request to be repeated;
        # point it at the wrapped user message so the IF semantics carry over.
        wrapped = _wrap(math_text, chosen["constraint"])
        if spec.get("func_name") == "validate_repeat_prompt":
            spec["original_prompt"] = wrapped

        composite_gt = json.dumps({"answer": answer, "if_constraint": spec})

        records.append(
            {
                "data_source": DATA_SOURCE,
                "prompt": [{"role": "user", "content": wrapped}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": composite_gt},
                "extra_info": {
                    "split": "train",
                    "index": used,
                    "source_dataset": "synthetic-math-IF",
                    "pool": "composite",
                    "skill_tags": ["math", "instruction_following"],
                    "constraint_type": chosen["constraint_type"],
                    "constraint": chosen["constraint"],
                    "math_source_subset": m_extra.get("source_subset"),
                    "math_base_index": m_extra.get("index"),
                    "ifeval_src_index": chosen["src_index"],
                    "answer": answer,
                },
            }
        )
        used += 1

    local_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_dir, exist_ok=True)
    out = datasets.Dataset.from_list(records)
    out.to_parquet(os.path.join(local_dir, "train.parquet"))
    if len(out) > 0:
        with open(os.path.join(local_dir, "train_example.json"), "w") as f:
            json.dump(out[0], f, indent=2, ensure_ascii=False)
    print(f"Saved {len(out)} composite rows to {local_dir} (constraint pool={len(pool)})", flush=True)


if __name__ == "__main__":
    main()
