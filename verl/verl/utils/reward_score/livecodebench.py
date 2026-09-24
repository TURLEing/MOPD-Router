"""Rule-based LiveCodeBench code-generation verifier."""

from __future__ import annotations

import base64
import json
import multiprocessing
import pickle
import zlib
from typing import Any

from .prime_code.testing_util import run_test


def _decode_test_cases(test_cases: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(test_cases, dict):
        return test_cases
    try:
        return json.loads(test_cases)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))


def _run_tests(test_cases, solution, result, timeout):
    try:
        test_results, _ = run_test(test_cases, test=solution, debug=False, timeout=timeout)
        result.append(test_results)
    except Exception:
        result.append([-1] * len(test_cases.get("inputs", [])))


def compute_score(completion: str, test_cases: str | dict[str, Any], timeout: int = 6) -> bool:
    """Return true only when the generated Python solution passes every test."""
    solution = completion.split("```python")[-1].split("```")[0]
    decoded = _decode_test_cases(test_cases)
    if not decoded.get("inputs"):
        return False

    manager = multiprocessing.Manager()
    result = manager.list()
    process = multiprocessing.Process(target=_run_tests, args=(decoded, solution, result, timeout))
    process.start()
    process.join(timeout=(timeout + 1) * len(decoded["inputs"]) + 5)
    if process.is_alive():
        process.kill()
        process.join()

    passed = bool(result) and all(item is True for item in result[0])
    manager.shutdown()
    return passed
