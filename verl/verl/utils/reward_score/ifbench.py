"""Adapter from verl's per-sample reward API to the vendored IFBench verifier."""

from __future__ import annotations

import copy
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _evaluation_lib():
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from benchmarks.IFBench import evaluation_lib
    except ImportError as exc:
        raise RuntimeError(
            "IFBench verifier dependencies are missing. Install benchmarks/IFBench/requirements.txt."
        ) from exc
    return evaluation_lib


def _run_check(spec: dict[str, Any], response: str, *, loose: bool):
    evaluation_lib = _evaluation_lib()
    kwargs = [
        {name: value for name, value in item.items() if value is not None}
        for item in copy.deepcopy(spec["kwargs"])
    ]
    example = evaluation_lib.InputExample(
        key=spec["key"],
        instruction_id_list=list(spec["instruction_id_list"]),
        prompt=spec["prompt"],
        kwargs=kwargs,
    )
    check = (
        evaluation_lib.test_instruction_following_loose
        if loose
        else evaluation_lib.test_instruction_following_strict
    )
    return check(example, {example.prompt: response})


def compute_score(solution_str: str, ground_truth: str | dict[str, Any]) -> dict[str, float]:
    """Use IFBench's reported loose prompt accuracy as reward and retain strict metrics."""
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    strict = _run_check(spec, solution_str, loose=False)
    loose = _run_check(spec, solution_str, loose=True)
    return {
        "score": float(loose.follow_all_instructions),
        "strict_prompt": float(strict.follow_all_instructions),
        "loose_prompt": float(loose.follow_all_instructions),
        "strict_instruction": sum(strict.follow_instruction_list) / len(strict.follow_instruction_list),
    }
