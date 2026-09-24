#!/usr/bin/env python3
"""Merge one verl actor checkpoint into a clearly named Hugging Face directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")


def has_hf_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        next(path.glob(pattern), None) is not None for pattern in WEIGHT_GLOBS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a verl FSDP/Megatron checkpoint to Hugging Face format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="verl global_step_N or actor directory")
    parser.add_argument("--name", required=True, help="experiment name used in the output path")
    parser.add_argument("--backend", choices=("fsdp", "megatron"), default="fsdp")
    parser.add_argument("--output-root", default=str(ROOT / "eval_outputs"))
    parser.add_argument("--target-dir", help="override the complete merged-model output directory")
    parser.add_argument("--path-file", help="write the resolved HF model path here for another script")
    parser.add_argument("--force", action="store_true", help="merge again even when complete HF weights exist")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        print(f"error: checkpoint directory does not exist: {checkpoint}", file=sys.stderr)
        return 2

    actor_dir = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    if not (actor_dir / "huggingface/config.json").is_file():
        print(
            f"error: expected verl actor checkpoint with {actor_dir / 'huggingface/config.json'}",
            file=sys.stderr,
        )
        return 2

    target_dir = (
        Path(args.target_dir).expanduser().resolve()
        if args.target_dir
        else Path(args.output_root).expanduser().resolve() / args.name / "merged_hf_model"
    )
    path_file = Path(args.path_file).expanduser().resolve() if args.path_file else None

    if has_hf_weights(target_dir) and not args.force:
        metadata_path = target_dir.parent / "merge_metadata.json"
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as handle:
                previous = json.load(handle)
            previous_source_value = previous.get("source_checkpoint")
            previous_source = Path(previous_source_value).expanduser() if previous_source_value else None
            if previous_source is not None and previous_source.resolve() != checkpoint:
                print(
                    f"error: name {args.name!r} already points to a different checkpoint: "
                    f"{previous_source}. Use another --name or pass --force.",
                    file=sys.stderr,
                )
                return 2
        print(f"Using existing merged model: {target_dir}")
    else:
        command = [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            args.backend,
            "--local_dir",
            str(actor_dir),
            "--target_dir",
            str(target_dir),
        ]
        print("+", " ".join(command), flush=True)
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(command, cwd=ROOT / "verl", check=True)
            except subprocess.CalledProcessError as exc:
                print(f"error: model merger exited with status {exc.returncode}", file=sys.stderr)
                return exc.returncode

    if path_file and not args.dry_run:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        path_file.write_text(str(target_dir) + "\n", encoding="utf-8")

    metadata = {
        "name": args.name,
        "source_checkpoint": str(checkpoint),
        "actor_checkpoint": str(actor_dir),
        "backend": args.backend,
        "merged_model": str(target_dir),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    if not args.dry_run:
        with (target_dir.parent / "merge_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Merged model path: {target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
