#!/usr/bin/env python3
"""Run the repository's math/code benchmarks against one checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATH_DATASETS = {
    "aime24": ROOT / "benchmarks/math_eval/data/aime24/test.jsonl",
    "aime25": ROOT / "benchmarks/math_eval/data/aime25/test.jsonl",
    "hmmt25_feb": ROOT / "benchmarks/math_eval/data/hmmt25_feb/test.jsonl",
    "hmmt25_nov": ROOT / "benchmarks/math_eval/data/hmmt25_nov/test.jsonl",
}
BENCHMARK_ALIASES = {
    "math": tuple(MATH_DATASETS),
    "evalplus": ("humaneval", "mbpp"),
    "code": ("humaneval", "mbpp", "livecodebench"),
    "instruction": ("ifeval", "ifbench"),
    "all": (*MATH_DATASETS, "humaneval", "mbpp", "livecodebench", "ifeval", "ifbench"),
}
VALID_BENCHMARKS = set(BENCHMARK_ALIASES["all"])
WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin")
INSTRUCTION_DEPENDENCIES = {
    "ifeval": ("absl", "immutabledict", "langdetect", "nltk"),
    "ifbench": ("absl", "emoji", "immutabledict", "langdetect", "nltk", "spacy", "syllapy", "unicodedata2", "pkg_resources"),
}
# LiveCodeBench only accepts model names registered in lcb_runner/lm_styles.py;
# the --model value only selects the prompt style, weights come from --local_model_path.
LCB_STYLE_MODEL = "Qwen3-4B-NonThinking"
MERGER_DEPENDENCIES = ("accelerate", "tensordict", "torch", "transformers")


def has_hf_weights(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        next(path.glob(pattern), None) is not None for pattern in WEIGHT_GLOBS
    )


def find_actor_dir(path: Path) -> Path | None:
    candidates = (path / "actor", path)
    return next((candidate for candidate in candidates if (candidate / "huggingface/config.json").is_file()), None)


def expand_benchmarks(value: str) -> list[str]:
    result: list[str] = []
    for item in value.split(","):
        item = item.strip().lower()
        expanded = BENCHMARK_ALIASES.get(item, (item,))
        for benchmark in expanded:
            if benchmark not in VALID_BENCHMARKS:
                raise ValueError(f"unknown benchmark {benchmark!r}")
            if benchmark not in result:
                result.append(benchmark)
    return result


def run(command: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def publish_evalplus_timing(
    local_path: Path, published_path: Path
) -> tuple[dict | None, bool]:
    """Read timing locally, then publish it without affecting benchmark status."""
    try:
        payload = read_json(local_path)
    except (OSError, ValueError) as exc:
        print(
            f"warning: unable to read local EvalPlus timing {local_path}: {exc}",
            file=sys.stderr,
        )
        return None, False

    try:
        write_json(published_path, payload)
    except OSError as exc:
        print(
            f"warning: unable to publish EvalPlus timing to {published_path}: {exc}",
            file=sys.stderr,
        )
        return payload, False
    return payload, True


def resolve_instruction_python(args: argparse.Namespace) -> str:
    """Python used for IFEval/IFBench scoring (verifier deps live in the `eval` conda env)."""
    candidate = args.instruction_python or os.environ.get("INSTRUCTION_PYTHON")
    if candidate:
        return candidate
    detected = Path.home() / ".conda/envs/eval/bin/python"
    if detected.is_file():
        return str(detected)
    return sys.executable


def missing_instruction_deps(python: str, packages: tuple[str, ...]) -> list[str]:
    missing = []
    for package in packages:
        result = subprocess.run([python, "-c", f"import {package}"], capture_output=True)
        if result.returncode != 0:
            missing.append(package)
    return missing


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def instruction_metrics(path: Path) -> dict:
    prompt_total = prompt_correct = instruction_total = instruction_correct = 0
    tier0_total = defaultdict(int)
    tier0_correct = defaultdict(int)
    tier1_total = defaultdict(int)
    tier1_correct = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            followed = item["follow_instruction_list"]
            instruction_ids = item["instruction_id_list"]
            prompt_total += 1
            prompt_correct += int(all(followed))
            instruction_total += len(followed)
            instruction_correct += sum(followed)
            for instruction_id, correct in zip(instruction_ids, followed):
                tier0 = instruction_id.split(":")[0]
                tier0_total[tier0] += 1
                tier0_correct[tier0] += int(correct)
                tier1_total[instruction_id] += 1
                tier1_correct[instruction_id] += int(correct)
    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
        "tier0_accuracy": {
            key: tier0_correct[key] / total for key, total in sorted(tier0_total.items())
        },
        "tier1_accuracy": {
            key: tier1_correct[key] / total for key, total in sorted(tier1_total.items())
        },
    }


def resolve_model(args: argparse.Namespace, output_dir: Path, env: dict[str, str]) -> Path:
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    if has_hf_weights(checkpoint):
        return checkpoint

    actor_dir = find_actor_dir(checkpoint)
    if actor_dir is None:
        raise ValueError(
            "checkpoint is neither a Hugging Face model nor a verl actor checkpoint "
            "(expected actor/huggingface/config.json)"
        )
    hf_dir = actor_dir / "huggingface"
    if has_hf_weights(hf_dir):
        return hf_dir
    if args.merge_backend is None:
        raise ValueError(
            f"{actor_dir} contains sharded training weights. Pass --merge-backend fsdp "
            "or --merge-backend megatron to merge them before evaluation."
        )

    if not args.dry_run:
        missing = [package for package in MERGER_DEPENDENCIES if importlib.util.find_spec(package) is None]
        if missing:
            raise ValueError(
                f"missing verl model-merger dependencies: {', '.join(missing)}; "
                f"install the local package with `{sys.executable} -m pip install -e {ROOT / 'verl'}`"
            )

    merged_dir = output_dir / "merged_hf_model"
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        args.merge_backend,
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(merged_dir),
    ]
    run(command, cwd=ROOT / "verl", env=env, dry_run=args.dry_run)
    return merged_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face or verl checkpoint on the repository benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", help="HF model directory, global_step_N directory, or actor directory")
    parser.add_argument("--name", help="experiment name used for output directories and model labels")
    parser.add_argument(
        "--benchmarks",
        default="math",
        help="comma-separated names or aliases: math, evalplus, code, instruction, all",
    )
    parser.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES value; all listed GPUs are used by vLLM")
    parser.add_argument("--output-dir", help="result directory (default: eval_outputs/<checkpoint-name>)")
    parser.add_argument("--metrics-file", help="consolidated metrics JSON (default: a timestamped file in output-dir)")
    parser.add_argument("--merge-backend", choices=("fsdp", "megatron"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--math-n", type=int, default=8, help="samples per mathematics problem (Avg@8)")
    parser.add_argument("--code-n", type=int, default=1, help="samples per code problem (Pass@1)")
    parser.add_argument("--if-n", type=int, default=5, help="independent completions per instruction prompt")
    parser.add_argument(
        "--math-data-parallel",
        action="store_true",
        help="for math, run one TP=1 vLLM replica per listed GPU and merge problem shards",
    )
    parser.add_argument(
        "--evalplus-data-parallel",
        action="store_true",
        help="for EvalPlus, run one TP=1 vLLM replica per listed GPU and merge task shards",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stop-token-ids",
        help="comma-separated generation stop IDs; math defaults to the model generation config",
    )
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--if-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--if-temperature",
        type=float,
        default=0.7,
        help="sampling temperature for IFEval/IFBench",
    )
    parser.add_argument(
        "--if-top-p",
        type=float,
        default=0.8,
        help="top-p for IFEval/IFBench (independent of --top-p)",
    )
    parser.add_argument("--if-top-k", type=int, default=20, help="top-k for IFEval/IFBench")
    parser.add_argument(
        "--instruction-python",
        help="python executable for IFEval/IFBench scoring "
        "(default: $INSTRUCTION_PYTHON, else ~/.conda/envs/eval/bin/python if present)",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate and print commands without running them")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="for instruction benchmarks: skip response generation, only run scoring "
        "(useful when responses were pre-generated elsewhere; no GPU required)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = None
    summary = None
    try:
        benchmarks = expand_benchmarks(args.benchmarks)
        if min(args.math_n, args.code_n, args.if_n) < 1:
            raise ValueError("--math-n, --code-n, and --if-n must all be at least 1")
        checkpoint_name = args.name or Path(args.checkpoint.rstrip("/")).name
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "eval_outputs" / checkpoint_name
        output_dir.mkdir(parents=True, exist_ok=True)
        run_timestamp = datetime.now().astimezone()
        metrics_path = (
            Path(args.metrics_file).expanduser().resolve()
            if args.metrics_file
            else output_dir / f"evaluation_metrics_{run_timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
        env["PYTHONUNBUFFERED"] = "1"
        summary = {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "started_at": run_timestamp.isoformat(),
            "finished_at": None,
            "status": "running",
            "output_dir": str(output_dir),
            "metrics_file": str(metrics_path),
            "generation": {
                "math_n": args.math_n,
                "code_n": args.code_n,
                "if_n": args.if_n,
                "max_tokens": args.max_tokens,
                "if_max_tokens": args.if_max_tokens,
                "max_num_seqs": args.max_num_seqs,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "if_temperature": args.if_temperature,
                "if_top_p": args.if_top_p,
                "if_top_k": args.if_top_k,
                "seed": args.seed,
                "stop_token_ids": args.stop_token_ids,
                "enable_thinking": args.enable_thinking,
                "gpus": args.gpus,
                "math_data_parallel": args.math_data_parallel,
                "evalplus_data_parallel": args.evalplus_data_parallel,
            },
            "benchmarks": {},
        }
        write_json(metrics_path, summary)
        model_path = resolve_model(args, output_dir, env)
        summary["checkpoint"] = str(model_path)
        write_json(metrics_path, summary)
        instruction_python = resolve_instruction_python(args)

        print(f"checkpoint: {model_path}")
        print(f"benchmarks: {', '.join(benchmarks)}")
        print(f"output: {output_dir}")
        print(f"metrics: {metrics_path}")
        print(f"instruction scoring python: {instruction_python}")

        for benchmark in benchmarks:
            summary["benchmarks"][benchmark] = {"status": "running"}
            write_json(metrics_path, summary)
            if benchmark in MATH_DATASETS:
                dataset = MATH_DATASETS[benchmark]
                if not dataset.is_file():
                    raise FileNotFoundError(f"missing benchmark data: {dataset}")
                benchmark_dir = output_dir / benchmark
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                if args.math_data_parallel:
                    command = [
                        sys.executable,
                        str(ROOT / "tools/run_math_dp.py"),
                        "--input-file", str(dataset),
                        "--model-path", str(model_path),
                        "--output-file", str(benchmark_dir / "results.jsonl"),
                        "--metrics-file", str(benchmark_dir / "metrics.json"),
                        "--gpus", args.gpus,
                        "--max-tokens", str(args.max_tokens),
                        "--temperature", str(args.temperature),
                        "--top-p", str(args.top_p),
                        "--top-k", str(args.top_k),
                        "--max-num-seqs", str(args.max_num_seqs),
                        "--n", str(args.math_n),
                        "--seed", str(args.seed),
                    ]
                else:
                    command = [
                        sys.executable,
                        str(ROOT / "benchmarks/math_eval/eval_math.py"),
                        "--input_file", str(dataset),
                        "--model_path", str(model_path),
                        "--output_file", str(benchmark_dir / "results.jsonl"),
                        "--metrics_file", str(benchmark_dir / "metrics.json"),
                        "--max_tokens", str(args.max_tokens),
                        "--temperature", str(args.temperature),
                        "--top_p", str(args.top_p),
                        "--top_k", str(args.top_k),
                        "--max_num_seqs", str(args.max_num_seqs),
                        "--n", str(args.math_n),
                        "--seed", str(args.seed),
                    ]
                if args.enable_thinking:
                    command.append("--enable-thinking" if args.math_data_parallel else "--enable_thinking")
                if args.stop_token_ids:
                    command.extend([
                        "--stop-token-ids" if args.math_data_parallel else "--stop_token_ids",
                        args.stop_token_ids,
                    ])
                run(command, cwd=ROOT, env=env, dry_run=args.dry_run)
            elif benchmark in {"humaneval", "mbpp"}:
                evalplus_env = env.copy()
                evalplus_gpus = [
                    gpu for gpu in args.gpus.split(",") if gpu.strip()
                ]
                benchmark_dir = output_dir / benchmark
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                if args.evalplus_data_parallel:
                    if not evalplus_gpus:
                        raise ValueError(
                            "--evalplus-data-parallel requires at least one GPU in --gpus"
                        )
                    local_dp_timing_path = (
                        ROOT
                        / "evalplus_results/.timing"
                        / (
                            f"{benchmark}_{model_path.name}_"
                            f"{run_timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
                        )
                    )
                    command = [
                        sys.executable,
                        str(ROOT / "tools/run_evalplus_dp.py"),
                        "--dataset", benchmark,
                        "--model", str(model_path),
                        "--gpus", args.gpus,
                        "--n-samples", str(args.code_n),
                        "--temperature", str(args.temperature),
                        "--top-p", str(args.top_p),
                        "--top-k", str(args.top_k),
                        "--max-new-tokens", str(args.max_tokens),
                        "--max-num-seqs", str(args.max_num_seqs),
                        "--output-root", str(ROOT / "evalplus_results"),
                        "--timing-file", str(local_dp_timing_path),
                    ]
                else:
                    evalplus_tp = len(evalplus_gpus)
                    command = [
                        "bash", str(ROOT / "benchmarks/code_eval/scripts/run_evalplus.sh"),
                        benchmark, str(model_path), "0", str(args.temperature), str(args.top_p), str(args.code_n),
                        str(args.top_k), str(args.max_tokens), str(args.max_num_seqs), str(evalplus_tp),
                    ]
                run(command, cwd=ROOT, env=evalplus_env, dry_run=args.dry_run)
            elif benchmark == "livecodebench":
                # run_lcb_gen.sh hardcodes --release_version v6, which needs test6.jsonl
                lcb_data_file = (
                    ROOT / "benchmarks/code_eval/coding/LiveCodeBench/code_generation_lite/test6.jsonl"
                )
                if not args.dry_run and not lcb_data_file.is_file():
                    raise FileNotFoundError(
                        f"missing LiveCodeBench data: {lcb_data_file}; download it with "
                        "`huggingface-cli download livecodebench/code_generation_lite "
                        "--repo-type dataset --include test6.jsonl --local-dir "
                        f"{lcb_data_file.parent}`"
                    )
                command = [
                    "bash", str(ROOT / "benchmarks/code_eval/scripts/run_lcb_gen.sh"),
                    "--model", LCB_STYLE_MODEL,
                    "--local_model_path", str(model_path),
                    "--save_name", checkpoint_name,
                    "--gpu", args.gpus,
                    "--n", str(args.code_n),
                    "--temperature", str(args.temperature),
                    "--top_p", str(args.top_p),
                    "--top_k", str(args.top_k),
                    "--max_tokens", str(args.max_tokens),
                    "--batch_size", str(args.max_num_seqs),
                ]
                run(command, cwd=ROOT, env=env, dry_run=args.dry_run)
            else:
                benchmark_root = ROOT / "benchmarks"
                if benchmark == "ifeval":
                    source_dir = benchmark_root / "instruction_following_eval"
                    input_file = source_dir / "data/input_data.jsonl"
                else:
                    source_dir = benchmark_root / "IFBench"
                    input_file = source_dir / "data/IFBench_test.jsonl"
                if not input_file.is_file():
                    raise FileNotFoundError(
                        f"missing vendored {benchmark} data: {input_file}"
                    )
                if not args.dry_run:
                    missing = missing_instruction_deps(
                        instruction_python, INSTRUCTION_DEPENDENCIES[benchmark]
                    )
                    if missing:
                        raise ValueError(
                            f"missing {benchmark} verifier dependencies in {instruction_python}: "
                            f"{', '.join(missing)}; install {source_dir / 'requirements.txt'} "
                            "into that environment"
                        )

                benchmark_dir = output_dir / benchmark
                benchmark_dir.mkdir(parents=True, exist_ok=True)
                responses_file = benchmark_dir / "responses.jsonl"
                if args.eval_only:
                    if not responses_file.is_file():
                        raise FileNotFoundError(
                            f"--eval-only requires pre-generated responses, but "
                            f"{responses_file} does not exist"
                        )
                    print(f"[{benchmark}] skipping generation, using existing {responses_file}")
                    completion_runs = [(benchmark_dir, responses_file, args.seed)]
                else:
                    completion_runs = []
                    for completion_idx in range(args.if_n):
                        completion_dir = benchmark_dir / f"completion_{completion_idx + 1:02d}"
                        completion_dir.mkdir(parents=True, exist_ok=True)
                        completion_runs.append(
                            (completion_dir, completion_dir / "responses.jsonl", args.seed + completion_idx)
                        )

                for completion_dir, completion_responses, completion_seed in completion_runs:
                    if not args.eval_only:
                        generation_command = [
                            sys.executable,
                            str(ROOT / "tools/generate_instruction_responses.py"),
                            "--input-file", str(input_file),
                            "--output-file", str(completion_responses),
                            "--model-path", str(model_path),
                            "--max-tokens", str(args.if_max_tokens),
                            "--max-num-seqs", str(args.max_num_seqs),
                            "--temperature", str(args.if_temperature),
                            "--top-p", str(args.if_top_p),
                            "--top-k", str(args.if_top_k),
                            "--seed", str(completion_seed),
                        ]
                        if args.enable_thinking:
                            generation_command.append("--enable-thinking")
                        run(generation_command, cwd=ROOT, env=env, dry_run=args.dry_run)

                    if benchmark == "ifeval":
                        evaluation_command = [
                            instruction_python,
                            "-m", "instruction_following_eval.evaluation_main",
                            f"--input_data={input_file}",
                            f"--input_response_data={completion_responses}",
                            f"--output_dir={completion_dir}",
                        ]
                        evaluation_cwd = benchmark_root
                    else:
                        evaluation_command = [
                            instruction_python,
                            "run_eval.py",
                            f"--input_data={input_file}",
                            f"--input_response_data={completion_responses}",
                            f"--output_dir={completion_dir}",
                        ]
                        evaluation_cwd = source_dir
                    run(evaluation_command, cwd=evaluation_cwd, env=env, dry_run=args.dry_run)

                if not args.dry_run and not args.eval_only:
                    for filename in (
                        "responses.jsonl",
                        "eval_results_strict.jsonl",
                        "eval_results_loose.jsonl",
                    ):
                        combined_path = benchmark_dir / filename
                        with combined_path.open("w", encoding="utf-8") as combined:
                            for completion_dir, _, _ in completion_runs:
                                combined.write((completion_dir / filename).read_text(encoding="utf-8"))
            if args.dry_run:
                summary["benchmarks"][benchmark] = {"status": "planned"}
            elif benchmark in MATH_DATASETS:
                benchmark_metrics_path = output_dir / benchmark / "metrics.json"
                summary["benchmarks"][benchmark] = {
                    "status": "completed",
                    "metrics": read_json(benchmark_metrics_path),
                    "files": {
                        "metrics": str(benchmark_metrics_path),
                        "generations": str(output_dir / benchmark / "results.jsonl"),
                    },
                }
            elif benchmark in {"humaneval", "mbpp"}:
                result_path = ROOT / "evalplus_results" / benchmark / f"{model_path.name}_eval_results.json"
                files = {"evaluation": str(result_path)}
                dp_timing_path = output_dir / benchmark / "dp_timing.json"
                benchmark_summary = {
                    "status": "completed",
                    "metrics": read_json(result_path),
                    "files": files,
                }
                if args.evalplus_data_parallel:
                    timing_payload, timing_published = publish_evalplus_timing(
                        local_dp_timing_path, dp_timing_path
                    )
                    if timing_payload is not None:
                        benchmark_summary["data_parallel_execution"] = timing_payload
                    if timing_published:
                        files["data_parallel_timing"] = str(dp_timing_path)
                summary["benchmarks"][benchmark] = benchmark_summary
            elif benchmark == "livecodebench":
                # lcb_runner writes outputs under lcb_outputs/<save_name> (we pass
                # --save_name checkpoint_name so different checkpoints don't collide);
                # file names are prefixed by str(Scenario) i.e. "Scenario.codegeneration"
                lcb_dir = ROOT / "benchmarks/code_eval/coding/LiveCodeBench/lcb_outputs" / checkpoint_name
                result_candidates = sorted(
                    lcb_dir.glob(f"*codegeneration*_{args.code_n}_{args.temperature}_eval.json")
                )
                if not result_candidates:
                    raise FileNotFoundError(
                        f"LiveCodeBench evaluation result not found in {lcb_dir} "
                        f"(pattern *codegeneration*_{args.code_n}_{args.temperature}_eval.json)"
                    )
                result_path = result_candidates[0]
                payload = read_json(result_path)
                generations_candidates = sorted(
                    lcb_dir.glob(f"*codegeneration*_{args.code_n}_{args.temperature}.json")
                )
                eval_all_candidates = sorted(
                    lcb_dir.glob(f"*codegeneration*_{args.code_n}_{args.temperature}_eval_all.json")
                )
                summary["benchmarks"][benchmark] = {
                    "status": "completed",
                    "metrics": payload[0] if isinstance(payload, list) and payload else payload,
                    "files": {
                        "evaluation": str(result_path),
                        "evaluation_details": str(eval_all_candidates[0]) if eval_all_candidates else None,
                        "generations": str(generations_candidates[0]) if generations_candidates else None,
                    },
                }
            else:
                benchmark_dir = output_dir / benchmark
                modes = {}
                files = {"responses": str(benchmark_dir / "responses.jsonl")}
                for mode in ("strict", "loose"):
                    result_path = benchmark_dir / f"eval_results_{mode}.jsonl"
                    modes[mode] = instruction_metrics(result_path)
                    files[mode] = str(result_path)
                summary["benchmarks"][benchmark] = {
                    "status": "completed",
                    "metrics": modes,
                    "files": files,
                }
            write_json(metrics_path, summary)
        summary["status"] = "planned" if args.dry_run else "completed"
        summary["finished_at"] = datetime.now().astimezone().isoformat()
        write_json(metrics_path, summary)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        if summary is not None and metrics_path is not None:
            summary["status"] = "failed"
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            summary["error"] = str(exc)
            for result in summary["benchmarks"].values():
                if result.get("status") == "running":
                    result["status"] = "failed"
                    result["error"] = str(exc)
            write_json(metrics_path, summary)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("Evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
