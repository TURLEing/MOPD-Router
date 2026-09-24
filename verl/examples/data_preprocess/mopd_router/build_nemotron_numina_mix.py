#!/usr/bin/env python3
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
"""Build the prompt-only 60K MOPD-Router training mixture.

Default composition:

* 20K nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 instruction_following
* 20K nvidia/Nemotron-SFT-Instruction-Following-Chat-v3 chat
* 20K AI-MO/NuminaMath-TIR train

Only the context before the final assistant turn is used as ``prompt``. Source
answers/solutions are retained in ``reward_model.ground_truth`` for provenance,
and Nemotron target reasoning is retained in ``extra_info.source_reasoning``;
the OPD training path does not consume either field.

Remote Nemotron data is read from dispersed auto-converted parquet shards. The
script adds shards adaptively until the quality-filtered candidate pool reaches
the requested safety margin, then applies deterministic bottom-k sampling. It
therefore avoids both a full 20GB scan and the ordering bias of taking one
contiguous prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import string
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import datasets
import fsspec
import pyarrow.parquet as pq


NEMOTRON_ID = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
NUMINA_ID = "AI-MO/NuminaMath-TIR"
HF_PARQUET_API = "https://datasets-server.huggingface.co/parquet"

TIR_SYSTEM_PROMPT = (
    "You are a mathematics assistant that solves problems with tool-integrated "
    "reasoning. Think step by step in natural language, and whenever useful "
    "write Python code inside fenced blocks (```python ... ```) to compute or "
    "verify intermediate quantities, then continue reasoning from the result."
)

INSTRUCTION_MARKERS = re.compile(
    r"\b(exactly|at least|at most|format|include|exclude|avoid|must|ensure|"
    r"do not|don't|use|write|create|provide|list|summarize|explain|describe|"
    r"respond|answer|translate|classify|extract|compare|rewrite)\b",
    re.IGNORECASE,
)
MISSING_CONTEXT = re.compile(
    r"\b(attached (?:file|image|document)|image (?:above|below)|figure (?:above|below)|"
    r"as shown (?:above|below|in the image)|listen to (?:this|the) audio)\b",
    re.IGNORECASE,
)
REPEAT_PROMPT = re.compile(r"\brepeat (?:the|this|my) (?:entire )?prompt\b", re.IGNORECASE)
CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n", re.IGNORECASE)
SPACE = re.compile(r"\s+")
LONG_REPEAT = re.compile(r"(.)\1{9,}", re.DOTALL)


def _stable_int(seed: int, *parts: Any) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _normalized_text(text: str) -> str:
    return SPACE.sub(" ", text).strip().casefold()


def _english_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    ascii_letters = sum(char in string.ascii_letters for char in letters)
    return ascii_letters / len(letters)


def _meaningful_user_text(text: str, *, min_chars: int, max_chars: int, english_only: bool) -> bool:
    text = text.strip()
    if not min_chars <= len(text) <= max_chars:
        return False
    if len(re.findall(r"[A-Za-z0-9]", text)) < 8:
        return False
    if LONG_REPEAT.search(text) or MISSING_CONTEXT.search(text) or REPEAT_PROMPT.search(text):
        return False
    if english_only and _english_ratio(text) < 0.85:
        return False
    return True


def _clean_message(message: dict[str, Any]) -> dict[str, str] | None:
    role = message.get("role")
    content = message.get("content")
    if role not in {"system", "user", "assistant"} or not isinstance(content, str):
        return None
    content = content.strip()
    if not content:
        return None
    return {"role": role, "content": content}


def _prompt_before_final_assistant(
    messages: Any,
) -> tuple[list[dict[str, str]], str, str] | None:
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    parsed_messages = []
    for message in messages:
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                return None
        if not isinstance(message, dict):
            return None
        parsed_messages.append(message)
    messages = parsed_messages
    final_idx = next(
        (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx].get("role") == "assistant"),
        None,
    )
    if final_idx is None or final_idx != len(messages) - 1:
        return None

    final_message = messages[final_idx]
    final_content = final_message.get("content")
    if not isinstance(final_content, str) or len(final_content.strip()) < 20:
        return None
    final_reasoning = final_message.get("reasoning_content")
    if not isinstance(final_reasoning, str):
        final_reasoning = ""

    prompt = []
    for raw_message in messages[:final_idx]:
        cleaned = _clean_message(raw_message)
        # Empty system prompts are harmless and can simply be omitted. Missing
        # user/assistant content means the conversation cannot be reconstructed.
        if cleaned is None:
            if raw_message.get("role") == "system" and not raw_message.get("content"):
                continue
            return None
        prompt.append(cleaned)

    if not prompt or prompt[-1]["role"] != "user":
        return None
    return prompt, final_content.strip(), final_reasoning.strip()


def _load_nemotron(split: str, local_dir: str | None) -> Iterable[dict[str, Any]]:
    if local_dir:
        path = Path(local_dir).expanduser() / "data" / f"{split}.jsonl"
        if not path.exists():
            path = Path(local_dir).expanduser() / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Nemotron {split} JSONL not found under {local_dir}")
        return datasets.load_dataset(
            "json", data_files={split: str(path)}, split=split, streaming=True
        )
    return datasets.load_dataset(NEMOTRON_ID, split=split, streaming=True)


def _load_numina(local_path: str | None) -> Iterable[dict[str, Any]]:
    if local_path:
        path = Path(local_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return datasets.load_dataset(
                "parquet", data_files={"train": str(path)}, split="train", streaming=True
            )
        return datasets.load_dataset(
            "json", data_files={"train": str(path)}, split="train", streaming=True
        )
    return datasets.load_dataset(NUMINA_ID, split="train", streaming=True)


def _bottom_k(
    records: Iterable[tuple[str, dict[str, Any]]], target: int, seed: int, source_name: str
) -> tuple[list[dict[str, Any]], int]:
    """Keep a deterministic uniform sample without materializing the source."""
    heap: list[tuple[int, str, dict[str, Any]]] = []
    eligible = 0
    for unique_id, record in records:
        eligible += 1
        priority = _stable_int(seed, source_name, unique_id)
        item = (-priority, unique_id, record)
        if len(heap) < target:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)

    if len(heap) < target:
        raise RuntimeError(
            f"{source_name}: requested {target:,} rows but only {eligible:,} passed quality filters"
        )
    selected = [item[2] for item in heap]
    selected.sort(key=lambda row: _stable_int(seed, source_name, row["extra_info"]["source_id"]))
    return selected, eligible


def _nemotron_records(
    rows: Iterable[dict[str, Any]],
    *,
    subset: str,
    max_prompt_chars: int,
    max_user_chars: int,
    max_chat_turns: int,
    seed: int,
    source_shard: str | None = None,
    dedup_state: dict[str, Any] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    if dedup_state is None:
        dedup_state = {}
    seen_prompts = dedup_state.setdefault("prompts", set())
    seen_chat_seed_depths = dedup_state.setdefault("chat_seed_depths", set())

    for row_idx, row in enumerate(rows):
        if row_idx > 0 and row_idx % 100_000 == 0:
            print(f"  {subset}: scanned {row_idx:,} source rows", flush=True)
        extracted = _prompt_before_final_assistant(row.get("messages"))
        if extracted is None:
            continue
        prompt, source_answer, source_reasoning = extracted
        user_text = prompt[-1]["content"]
        if not _meaningful_user_text(
            user_text, min_chars=20, max_chars=max_user_chars, english_only=True
        ):
            continue
        if sum(len(message["content"]) for message in prompt) > max_prompt_chars:
            continue
        if subset == "instruction_following" and not INSTRUCTION_MARKERS.search(user_text):
            continue

        metadata = row.get("metadata") or {}
        train_turns = metadata.get("train_turns") or []
        if train_turns and (
            len(train_turns) != len(row.get("messages") or [])
            or train_turns[-1] is not True
            or any(train_turns[:-1])
        ):
            continue
        source_uuid = str(row.get("uuid") or f"row-{row_idx}")
        seed_prompt_sha256 = str(metadata.get("seed_prompt_sha256") or "")
        source_seed = seed_prompt_sha256 or source_uuid

        if subset == "chat":
            user_turns = sum(message["role"] == "user" for message in prompt)
            if user_turns > max_chat_turns:
                continue
            # Keep at most one conversation at each depth for an original seed.
            # Different depths are genuinely different multi-turn prompts; rows
            # with the same seed and depth are redundant response variants.
            seed_depth = f"{source_seed}:{user_turns}"
            if seed_depth in seen_chat_seed_depths:
                continue
        else:
            user_turns = sum(message["role"] == "user" for message in prompt)

        prompt_key = hashlib.sha256(
            "\n".join(
                f"{message['role']}:{_normalized_text(message['content'])}" for message in prompt
            ).encode("utf-8")
        ).hexdigest()
        if prompt_key in seen_prompts:
            continue
        seen_prompts.add(prompt_key)
        if subset == "chat":
            seen_chat_seed_depths.add(seed_depth)

        data_source = f"nemotron_{subset}"
        record = {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "instruction_following" if subset == "instruction_following" else "chat",
            # Preserved for provenance and possible offline analysis. The OPD
            # advantage path does not consume this reference response.
            "reward_model": {"style": "reference", "ground_truth": source_answer},
            "extra_info": {
                "split": "train",
                "source_dataset": NEMOTRON_ID,
                "source_subset": subset,
                "source_id": source_uuid,
                "source_row_index": row_idx,
                "source_shard": source_shard,
                "seed_dataset": metadata.get("seed_dataset"),
                "seed_prompt_sha256": seed_prompt_sha256 or None,
                "source_model": metadata.get("model"),
                "source_train_turns": train_turns,
                "source_reasoning": source_reasoning,
                "source_output_type": "assistant_response",
                "used_in": row.get("used_in") or [],
                "user_turns": user_turns,
                "pool": "prompt_only",
                "skill_tags": ["instruction_following"]
                if subset == "instruction_following"
                else ["instruction_following", "chat"],
            },
        }
        yield source_uuid, record


def _spread_indices(size: int) -> list[int]:
    """Order shard indices by repeatedly choosing the least-covered position."""
    if size <= 0:
        return []
    selected = [0]
    remaining = set(range(1, size))
    while remaining:
        next_idx = max(
            remaining,
            key=lambda idx: (min(abs(idx - chosen) for chosen in selected), idx),
        )
        selected.append(next_idx)
        remaining.remove(next_idx)
    return selected


def _remote_nemotron_shards(split: str, hf_endpoint: str | None) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset": NEMOTRON_ID})
    with urllib.request.urlopen(f"{HF_PARQUET_API}?{query}") as response:
        payload = json.load(response)
    shards = [item for item in payload.get("parquet_files", []) if item.get("split") == split]
    if not shards:
        raise RuntimeError(f"No converted parquet shards found for Nemotron split {split!r}")

    endpoint = (hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    if endpoint != "https://huggingface.co":
        for shard in shards:
            parsed = urllib.parse.urlparse(shard["url"])
            shard["url"] = endpoint + parsed.path
    return shards


def _adaptive_remote_nemotron_candidates(
    *,
    subset: str,
    target: int,
    candidate_multiplier: float,
    max_prompt_chars: int,
    max_user_chars: int,
    max_chat_turns: int,
    seed: int,
    hf_endpoint: str | None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[dict[str, Any]]]:
    required = math.ceil(target * candidate_multiplier)
    shards = _remote_nemotron_shards(subset, hf_endpoint)
    order = _spread_indices(len(shards))
    candidates: list[tuple[str, dict[str, Any]]] = []
    dedup_state: dict[str, Any] = {}
    used_shards = []

    for shard_idx in order:
        shard = shards[shard_idx]
        print(
            f"  {subset}: reading shard {shard_idx + 1}/{len(shards)} "
            f"({shard.get('size', 0) / 1e6:.1f} MB)",
            flush=True,
        )
        rows = _iter_remote_parquet_rows(shard["url"])
        candidates.extend(
            _nemotron_records(
                rows,
                subset=subset,
                max_prompt_chars=max_prompt_chars,
                max_user_chars=max_user_chars,
                max_chat_turns=max_chat_turns,
                seed=seed,
                source_shard=shard.get("filename"),
                dedup_state=dedup_state,
            )
        )
        used_shards.append(shard)
        print(
            f"  {subset}: {len(candidates):,}/{required:,} quality candidates",
            flush=True,
        )
        if len(candidates) >= required:
            break

    if len(candidates) < target:
        raise RuntimeError(
            f"{subset}: only {len(candidates):,} quality candidates after reading "
            f"all {len(used_shards)} available converted shards; need {target:,}"
        )
    return candidates, used_shards


def _iter_remote_parquet_rows(url: str, batch_size: int = 512) -> Iterator[dict[str, Any]]:
    """Read parquet rows without interpreting Hugging Face extension metadata.

    Some released Nemotron shards use the newer ``Json`` feature extension,
    which older ``datasets`` versions reject before yielding any rows. PyArrow
    can still read the underlying parquet values directly.
    """
    with fsspec.open(url, "rb") as source:
        parquet_file = pq.ParquetFile(source)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            yield from batch.to_pylist()


def _numina_records(
    rows: Iterable[dict[str, Any]],
    *,
    max_prompt_chars: int,
    require_code: bool,
) -> Iterator[tuple[str, dict[str, Any]]]:
    seen_problems: set[str] = set()
    for row_idx, row in enumerate(rows):
        if row_idx > 0 and row_idx % 25_000 == 0:
            print(f"  numina_math_tir: scanned {row_idx:,} source rows", flush=True)
        problem = row.get("problem")
        solution = row.get("solution")
        if not isinstance(problem, str) or not isinstance(solution, str):
            continue
        problem = problem.strip()
        if not _meaningful_user_text(
            problem, min_chars=20, max_chars=max_prompt_chars, english_only=False
        ):
            continue
        # The solution is inspected to guarantee that the selected source row
        # really contains a nontrivial TIR/code trajectory. It is retained as a
        # reference field, but is not part of the prompt consumed by OPD.
        if len(solution.strip()) < 200 or (require_code and not CODE_FENCE.search(solution)):
            continue
        solution = solution.strip()

        problem_key = hashlib.sha256(_normalized_text(problem).encode("utf-8")).hexdigest()
        if problem_key in seen_problems:
            continue
        seen_problems.add(problem_key)
        source_id = f"train-{row_idx}"
        record = {
            "data_source": "numina_math_tir",
            "prompt": [
                {"role": "system", "content": TIR_SYSTEM_PROMPT},
                {"role": "user", "content": problem},
            ],
            "ability": "math_code",
            "reward_model": {"style": "reference", "ground_truth": solution},
            "extra_info": {
                "split": "train",
                "source_dataset": NUMINA_ID,
                "source_subset": "train",
                "source_id": source_id,
                "source_row_index": row_idx,
                "source_shard": str(NUMINA_ID),
                "seed_dataset": None,
                "seed_prompt_sha256": None,
                "source_model": None,
                "source_train_turns": [],
                "source_reasoning": "",
                "source_output_type": "tir_solution",
                "used_in": [],
                "user_turns": 1,
                "pool": "prompt_only",
                "skill_tags": ["math", "code"],
            },
        }
        yield source_id, record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction_count", type=int, default=20_000)
    parser.add_argument("--chat_count", type=int, default=20_000)
    parser.add_argument("--numina_count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_chars", type=int, default=6000)
    parser.add_argument("--max_user_chars", type=int, default=5000)
    parser.add_argument("--max_chat_turns", type=int, default=4)
    parser.add_argument(
        "--candidate_multiplier",
        type=float,
        default=1.0,
        help="Stop adding remote shards after this many filtered candidates per selected row.",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=None,
        help="Optional Hugging Face endpoint for parquet downloads, e.g. https://hf-mirror.com.",
    )
    parser.add_argument(
        "--numina_require_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a Python/code fence in the retained source solution.",
    )
    parser.add_argument(
        "--nemotron_local_dir",
        default=None,
        help="Optional downloaded Nemotron repo containing data/*.jsonl.",
    )
    parser.add_argument(
        "--numina_local_path",
        default=None,
        help="Optional local Numina train parquet or JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        default="./data/nemotron_numina_prompt_60k",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("instruction_count", "chat_count", "numina_count"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.max_chat_turns <= 0:
        raise ValueError("--max_chat_turns must be positive")
    if args.candidate_multiplier < 1:
        raise ValueError("--candidate_multiplier must be at least 1")
    numina_max_problem_chars = min(
        args.max_user_chars, args.max_prompt_chars - len(TIR_SYSTEM_PROMPT)
    )
    if numina_max_problem_chars < 20:
        raise ValueError("--max_prompt_chars leaves no room for the Numina TIR system prompt")

    used_remote_shards: dict[str, list[dict[str, Any]]] = {}

    def build_nemotron_subset(subset: str, target: int):
        print(f"Streaming and filtering Nemotron {subset}...", flush=True)
        if args.nemotron_local_dir:
            candidates = _nemotron_records(
                _load_nemotron(subset, args.nemotron_local_dir),
                subset=subset,
                max_prompt_chars=args.max_prompt_chars,
                max_user_chars=args.max_user_chars,
                max_chat_turns=args.max_chat_turns,
                seed=args.seed,
                source_shard=f"data/{subset}.jsonl",
            )
        else:
            candidates, used_shards = _adaptive_remote_nemotron_candidates(
                subset=subset,
                target=target,
                candidate_multiplier=args.candidate_multiplier,
                max_prompt_chars=args.max_prompt_chars,
                max_user_chars=args.max_user_chars,
                max_chat_turns=args.max_chat_turns,
                seed=args.seed,
                hf_endpoint=args.hf_endpoint,
            )
            used_remote_shards[subset] = used_shards
        return _bottom_k(candidates, target, args.seed, f"nemotron_{subset}")

    instruction, instruction_eligible = build_nemotron_subset(
        "instruction_following", args.instruction_count
    )
    chat, chat_eligible = build_nemotron_subset("chat", args.chat_count)

    print("Streaming and filtering NuminaMath-TIR...", flush=True)
    numina, numina_eligible = _bottom_k(
        _numina_records(
            _load_numina(args.numina_local_path),
            max_prompt_chars=numina_max_problem_chars,
            require_code=args.numina_require_code,
        ),
        args.numina_count,
        args.seed,
        "numina_math_tir",
    )

    records = instruction + chat + numina
    records.sort(
        key=lambda row: _stable_int(
            args.seed, "final-mix", row["data_source"], row["extra_info"]["source_id"]
        )
    )
    for index, record in enumerate(records):
        record["extra_info"]["index"] = index

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.parquet"
    datasets.Dataset.from_list(records).to_parquet(str(output_path))

    manifest = {
        "total_rows": len(records),
        "counts": {
            "nemotron_instruction_following": len(instruction),
            "nemotron_chat": len(chat),
            "numina_math_tir": len(numina),
        },
        "eligible_after_filters": {
            "nemotron_instruction_following": instruction_eligible,
            "nemotron_chat": chat_eligible,
            "numina_math_tir": numina_eligible,
        },
        "source_datasets": [NEMOTRON_ID, NUMINA_ID],
        "prompt_only": True,
        "source_answers_or_ground_truth_saved": True,
        "numina_require_code": args.numina_require_code,
        "seed": args.seed,
        "max_prompt_chars": args.max_prompt_chars,
        "max_user_chars": args.max_user_chars,
        "max_chat_turns": args.max_chat_turns,
        "numina_max_problem_chars": numina_max_problem_chars,
        "candidate_multiplier": args.candidate_multiplier,
        "hf_endpoint": args.hf_endpoint or os.environ.get("HF_ENDPOINT"),
        "remote_nemotron_shards": used_remote_shards,
        "remote_nemotron_bytes": sum(
            shard.get("size", 0)
            for shards in used_remote_shards.values()
            for shard in shards
        ),
        "output": str(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "sample.json").write_text(
        json.dumps(records[:3], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
