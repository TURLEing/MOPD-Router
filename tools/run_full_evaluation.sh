#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}

MATH_GPUS=${MATH_GPUS:-0,1}
MATH_N=${MATH_N:-8}
MATH_MAX_NUM_SEQS=${MATH_MAX_NUM_SEQS:-128}

EVALPLUS_GPUS=${EVALPLUS_GPUS:-2,3}
EVALPLUS_N=${EVALPLUS_N:-1}
EVALPLUS_MAX_NUM_SEQS=${EVALPLUS_MAX_NUM_SEQS:-64}

LCB_GPUS=${LCB_GPUS:-4,5}
LCB_N=${LCB_N:-1}
LCB_MAX_NUM_SEQS=${LCB_MAX_NUM_SEQS:-64}

INSTRUCTION_GPUS=${INSTRUCTION_GPUS:-6,7}
INSTRUCTION_N=${INSTRUCTION_N:-5}
INSTRUCTION_MAX_NUM_SEQS=${INSTRUCTION_MAX_NUM_SEQS:-64}

usage() {
    echo "Usage: bash tools/run_full_evaluation.sh CHECKPOINT_PATH [options]"
    echo
    echo "Options:"
    echo "  --name NAME                 Experiment/model label"
    echo "  --merge-backend BACKEND     Override detected shard backend: fsdp or megatron"
    echo "  --run-dir PATH              Run logs, metrics, and outputs directory"
    echo
    echo "Optional environment overrides:"
    echo "  PYTHON_BIN"
    echo "  MATH_GPUS MATH_N MATH_MAX_NUM_SEQS"
    echo "  EVALPLUS_GPUS EVALPLUS_N EVALPLUS_MAX_NUM_SEQS"
    echo "  LCB_GPUS LCB_N LCB_MAX_NUM_SEQS"
    echo "  INSTRUCTION_GPUS INSTRUCTION_N INSTRUCTION_MAX_NUM_SEQS"
    echo "  INSTRUCTION_PYTHON          Python for IFEval/IFBench scoring"
    echo "                              (default: ~/.conda/envs/eval/bin/python if present)"
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

CHECKPOINT=$1
shift
EVALUATION_NAME=""
MERGE_BACKEND=""
RUN_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            EVALUATION_NAME=$2
            shift 2
            ;;
        --merge-backend)
            MERGE_BACKEND=$2
            shift 2
            ;;
        --run-dir)
            RUN_DIR=$2
            shift 2
            ;;
        *)
            echo "Error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$CHECKPOINT" ]]; then
    echo "Error: checkpoint directory does not exist: $CHECKPOINT" >&2
    exit 2
fi
CHECKPOINT=$(cd -- "$CHECKPOINT" && pwd)

RUN_TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
CHECKPOINT_NAME=$(basename "$CHECKPOINT")
if [[ -z "$EVALUATION_NAME" ]]; then
    EVALUATION_NAME=$CHECKPOINT_NAME
fi
RUN_DIR=${RUN_DIR:-"$REPO_ROOT/evaluation_runs/${EVALUATION_NAME}_${RUN_TIMESTAMP}"}
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/metrics" "$RUN_DIR/outputs" "$RUN_DIR/status"

format_duration() {
    local total=$1
    printf "%02d:%02d:%02d" "$((total / 3600))" "$(((total % 3600) / 60))" "$((total % 60))"
}

is_hf_model() {
    local model_dir=$1
    [[ -f "$model_dir/config.json" ]] && (
        compgen -G "$model_dir/*.safetensors" >/dev/null ||
        compgen -G "$model_dir/pytorch_model*.bin" >/dev/null
    )
}

run_task() {
    local name=$1
    local benchmarks=$2
    local gpus=$3
    local n=$4
    local max_num_seqs=$5
    shift 5
    local -a extra_args=("$@")
    local log_file="$RUN_DIR/logs/${name}.log"
    local metrics_file="$RUN_DIR/metrics/${name}.json"
    local output_dir="$RUN_DIR/outputs/${name}"
    local status_file="$RUN_DIR/status/${name}.tsv"
    local start_epoch end_epoch duration exit_code start_time end_time
    local -a command

    command=(
        "$PYTHON_BIN" "$REPO_ROOT/tools/evaluate_checkpoint.py" "$CHECKPOINT"
        --name "$EVALUATION_NAME"
        --benchmarks "$benchmarks"
        --gpus "$gpus"
        --max-num-seqs "$max_num_seqs"
        --output-dir "$output_dir"
        --metrics-file "$metrics_file"
    )
    if [[ -n "$n" ]]; then
        case "$name" in
            math) command+=(--math-n "$n") ;;
            evalplus|livecodebench) command+=(--code-n "$n") ;;
            instruction) command+=(--if-n "$n") ;;
        esac
    fi
    if [[ ${#extra_args[@]} -gt 0 ]]; then
        command+=("${extra_args[@]}")
    fi

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%dT%H:%M:%S%z")
    {
        echo "task: $name"
        echo "started_at: $start_time"
        echo "gpus: $gpus"
        echo "command: $(printf '%q ' "${command[@]}")"
        echo
    } >"$log_file"

    echo "[$name] started on GPU(s) $gpus; log: $log_file"
    (
        cd "$REPO_ROOT"
        "${command[@]}"
    ) >>"$log_file" 2>&1
    exit_code=$?

    end_epoch=$(date +%s)
    end_time=$(date "+%Y-%m-%dT%H:%M:%S%z")
    duration=$((end_epoch - start_epoch))
    {
        echo
        echo "finished_at: $end_time"
        echo "duration_seconds: $duration"
        echo "duration: $(format_duration "$duration")"
        echo "exit_code: $exit_code"
    } >>"$log_file"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" "$benchmarks" "$gpus" "$start_time" "$end_time" \
        "$duration" "$(format_duration "$duration")" "$exit_code" "$log_file" >"$status_file"

    if [[ $exit_code -eq 0 ]]; then
        echo "[$name] completed in $(format_duration "$duration")"
    else
        echo "[$name] failed after $(format_duration "$duration"); see $log_file" >&2
    fi
    return "$exit_code"
}

echo "Evaluation run directory: $RUN_DIR"
echo "Source checkpoint: $CHECKPOINT"
echo "Evaluation name: $EVALUATION_NAME"
echo

if is_hf_model "$CHECKPOINT"; then
    echo "[merge] skipped: input is already a complete Hugging Face model"
    echo
else
    ACTOR_DIR="$CHECKPOINT"
    if [[ -d "$CHECKPOINT/actor" ]]; then
        ACTOR_DIR="$CHECKPOINT/actor"
    fi
    if [[ -z "$MERGE_BACKEND" && -f "$ACTOR_DIR/fsdp_config.json" ]]; then
        MERGE_BACKEND=fsdp
        echo "[merge] detected FSDP actor checkpoint"
    fi
    if [[ -z "$MERGE_BACKEND" ]]; then
        echo "Error: input is not a complete Hugging Face model and its shard backend could not be detected." >&2
        echo "Pass --merge-backend fsdp or --merge-backend megatron." >&2
        exit 2
    fi
    if [[ "$MERGE_BACKEND" != "fsdp" && "$MERGE_BACKEND" != "megatron" ]]; then
        echo "Error: --merge-backend must be fsdp or megatron" >&2
        exit 2
    fi
    MERGE_LOG="$RUN_DIR/logs/merge.log"
    MERGED_PATH_FILE="$RUN_DIR/merged_model_path.txt"
    MERGE_START_EPOCH=$(date +%s)
    MERGE_START_TIME=$(date "+%Y-%m-%dT%H:%M:%S%z")
    echo "[merge] started; log: $MERGE_LOG"
    (
        cd "$REPO_ROOT"
        "$PYTHON_BIN" "$REPO_ROOT/tools/merge_checkpoint.py" "$CHECKPOINT" \
            --name "$EVALUATION_NAME" \
            --backend "$MERGE_BACKEND" \
            --path-file "$MERGED_PATH_FILE"
    ) >"$MERGE_LOG" 2>&1
    MERGE_EXIT_CODE=$?
    MERGE_END_EPOCH=$(date +%s)
    MERGE_END_TIME=$(date "+%Y-%m-%dT%H:%M:%S%z")
    MERGE_DURATION=$((MERGE_END_EPOCH - MERGE_START_EPOCH))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "merge" "$MERGE_BACKEND" "cpu" "$MERGE_START_TIME" "$MERGE_END_TIME" \
        "$MERGE_DURATION" "$(format_duration "$MERGE_DURATION")" "$MERGE_EXIT_CODE" "$MERGE_LOG" \
        >"$RUN_DIR/status/merge.tsv"
    if [[ $MERGE_EXIT_CODE -ne 0 ]]; then
        echo "[merge] failed after $(format_duration "$MERGE_DURATION"); see $MERGE_LOG" >&2
        TIMING_SUMMARY="$RUN_DIR/timing_summary.tsv"
        printf "task\tbenchmarks\tgpus\tstarted_at\tfinished_at\tduration_seconds\tduration\texit_code\tlog\n" \
            >"$TIMING_SUMMARY"
        sed -n '1p' "$RUN_DIR/status/merge.tsv" >>"$TIMING_SUMMARY"
        echo "Timing summary: $TIMING_SUMMARY" >&2
        exit "$MERGE_EXIT_CODE"
    fi
    CHECKPOINT=$(sed -n '1p' "$MERGED_PATH_FILE")
    CHECKPOINT_NAME=$(basename "$CHECKPOINT")
    echo "[merge] completed in $(format_duration "$MERGE_DURATION")"
    echo "Merged checkpoint: $CHECKPOINT"
    echo
fi

# =========================================================================
# Phase 1: Generate IFEval and IFBench responses concurrently, with one
# independent TP=1 vLLM replica on each instruction GPU.
# =========================================================================
echo
echo "=== Phase 1: instruction-following response generation ==="
INSTRUCTION_OUTPUT_DIR="$RUN_DIR/outputs/instruction"

IFEVAL_INPUT="$REPO_ROOT/benchmarks/instruction_following_eval/data/input_data.jsonl"
IFBENCH_INPUT="$REPO_ROOT/benchmarks/IFBench/data/IFBench_test.jsonl"
IFEVAL_RESPONSES="$INSTRUCTION_OUTPUT_DIR/ifeval/responses.jsonl"
IFBENCH_RESPONSES="$INSTRUCTION_OUTPUT_DIR/ifbench/responses.jsonl"
mkdir -p "$INSTRUCTION_OUTPUT_DIR/ifeval" "$INSTRUCTION_OUTPUT_DIR/ifbench"

IFS=',' read -r -a INSTRUCTION_GPU_LIST <<<"$INSTRUCTION_GPUS"
if [[ ${#INSTRUCTION_GPU_LIST[@]} -lt 2 ]]; then
    echo "Error: Phase 1 needs two INSTRUCTION_GPUS (default: 6,7)." >&2
    echo "IFEval and IFBench use one TP=1 replica each." >&2
    exit 2
fi
IFEVAL_GPU=${INSTRUCTION_GPU_LIST[0]//[[:space:]]/}
IFBENCH_GPU=${INSTRUCTION_GPU_LIST[1]//[[:space:]]/}

run_instruction_generation() {
    local benchmark=$1
    local input_file=$2
    local responses_file=$3
    local gpu=$4
    local name="instruction-gen-${benchmark}"
    local log_file="$RUN_DIR/logs/${name}.log"
    local status_file="$RUN_DIR/status/${name}.tsv"
    local start_epoch end_epoch duration exit_code start_time end_time
    local -a command

    command=(
        env CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON_BIN" "$REPO_ROOT/tools/generate_instruction_responses.py"
        --input-file "$input_file"
        --output-file "$responses_file"
        --model-path "$CHECKPOINT"
        --max-tokens "${IF_MAX_TOKENS:-4096}"
        --max-num-seqs "$INSTRUCTION_MAX_NUM_SEQS"
        --temperature "${IF_TEMPERATURE:-0.7}"
        --top-p "${IF_TOP_P:-0.8}"
        --top-k "${IF_TOP_K:-20}"
        --seed "${IF_SEED:-42}"
    )

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%dT%H:%M:%S%z")
    {
        echo "task: $name"
        echo "started_at: $start_time"
        echo "gpu: $gpu"
        echo "parallelism: DP replica, TP=1"
        echo "command: $(printf '%q ' "${command[@]}")"
        echo
    } >"$log_file"
    echo "[$name] started on GPU $gpu; log: $log_file"
    (
        cd "$REPO_ROOT"
        "${command[@]}"
    ) >>"$log_file" 2>&1
    exit_code=$?

    end_epoch=$(date +%s)
    end_time=$(date "+%Y-%m-%dT%H:%M:%S%z")
    duration=$((end_epoch - start_epoch))
    {
        echo
        echo "finished_at: $end_time"
        echo "duration_seconds: $duration"
        echo "duration: $(format_duration "$duration")"
        echo "exit_code: $exit_code"
    } >>"$log_file"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$name" "$benchmark" "$gpu" "$start_time" "$end_time" \
        "$duration" "$(format_duration "$duration")" "$exit_code" "$log_file" \
        >"$status_file"
    if [[ $exit_code -eq 0 ]]; then
        echo "[$name] completed in $(format_duration "$duration")"
    else
        echo "[$name] failed after $(format_duration "$duration"); see $log_file" >&2
    fi
    return "$exit_code"
}

echo "Instruction generation is handled by the evaluator with $INSTRUCTION_N completions per prompt."

# =========================================================================
# Phase 2: Run all 4 evaluations in parallel.
# EvalPlus uses GPUs 2,3,6,7 as four DP replicas, each with TP=1.
# Instruction evaluation generates and scores five independent completions.
# =========================================================================
echo
echo "=== Phase 2: parallel evaluation ==="

EVALPLUS_GPUS_PHASE2="${EVALPLUS_GPUS},${INSTRUCTION_GPUS}"

run_task math math "$MATH_GPUS" "$MATH_N" "$MATH_MAX_NUM_SEQS" &
PID_MATH=$!
run_task evalplus evalplus "$EVALPLUS_GPUS_PHASE2" "$EVALPLUS_N" "$EVALPLUS_MAX_NUM_SEQS" \
    --evalplus-data-parallel &
PID_EVALPLUS=$!
run_task livecodebench livecodebench "$LCB_GPUS" "$LCB_N" "$LCB_MAX_NUM_SEQS" &
PID_LCB=$!
run_task instruction instruction "$INSTRUCTION_GPUS" "$INSTRUCTION_N" "$INSTRUCTION_MAX_NUM_SEQS" &
PID_INSTRUCTION=$!

OVERALL_STATUS=0
for pid in "$PID_MATH" "$PID_EVALPLUS" "$PID_LCB" "$PID_INSTRUCTION"; do
    if ! wait "$pid"; then
        OVERALL_STATUS=1
    fi
done

TIMING_SUMMARY="$RUN_DIR/timing_summary.tsv"
printf "task\tbenchmarks\tgpus\tstarted_at\tfinished_at\tduration_seconds\tduration\texit_code\tlog\n" \
    >"$TIMING_SUMMARY"
for name in merge instruction-gen-ifeval instruction-gen-ifbench math evalplus livecodebench instruction; do
    if [[ -f "$RUN_DIR/status/${name}.tsv" ]]; then
        sed -n '1p' "$RUN_DIR/status/${name}.tsv" >>"$TIMING_SUMMARY"
    fi
done

echo
echo "Timing summary: $TIMING_SUMMARY"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "$TIMING_SUMMARY"
else
    cat "$TIMING_SUMMARY"
fi
echo
echo "Logs:    $RUN_DIR/logs/"
echo "Metrics: $RUN_DIR/metrics/"
echo "Outputs: $RUN_DIR/outputs/"

exit "$OVERALL_STATUS"
