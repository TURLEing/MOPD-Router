#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
CODE_EVAL_ROOT="$REPO_ROOT/benchmarks/code_eval"
export PYTHONPATH="$CODE_EVAL_ROOT/coding/evalplus${PYTHONPATH:+:$PYTHONPATH}"
export HUMANEVAL_OVERRIDE_PATH="$CODE_EVAL_ROOT/data/HumanEvalPlus.jsonl"
export MBPP_OVERRIDE_PATH="$CODE_EVAL_ROOT/data/MbppPlus.jsonl"

# Set defaults if not specified - fix argument assignments
DATASET=${1:-humaneval}
MODEL=${2:-"Qwen/Qwen3-4B"}
GREEDY=${3:-0}
TEMP=${4:-0.7}
TOP_P=${5:-0.8}
N_SAMPLES=${6:-1}
TOP_K=${7:-20}
MAX_TOKENS=${8:-16384}
MAX_NUM_SEQS=${9:-256}
TP_SIZE=${10:-1}

# If greedy mode, force n_samples to 1
if [ "$GREEDY" -eq 1 ]; then
    N_SAMPLES=1
fi

echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Greedy: $GREEDY (1=yes, 0=no)"
echo "Temperature: $TEMP"
echo "Top-P: $TOP_P"
echo "Top-K: $TOP_K"
echo "Number of samples: $N_SAMPLES"
echo "Max tokens: $MAX_TOKENS"
echo "Max number of sequences: $MAX_NUM_SEQS"
echo "Tensor parallel size: $TP_SIZE"

# Extract model identifier for output file
MODEL_BASE=$(basename "$MODEL")
echo "Model base: $MODEL_BASE"

# Execute command directly without quoting the arguments
if [ "$GREEDY" -eq 1 ]; then
    python3 -m evalplus.codegen --model "$MODEL" \
                    --dataset $DATASET \
                    --backend vllm \
                    --max-new-tokens "$MAX_TOKENS" \
                    --max-num-seqs "$MAX_NUM_SEQS" \
                    --tp "$TP_SIZE" \
                    --trust_remote_code \
                    --greedy
    TEMP_VAL="0.0"
else
    echo "Running non-greedy mode"
    python3 -m evalplus.codegen --model "$MODEL" \
                    --dataset $DATASET \
                    --backend vllm \
                    --temperature $TEMP \
                    --top-p $TOP_P \
                    --top-k $TOP_K \
                    --max-new-tokens "$MAX_TOKENS" \
                    --max-num-seqs "$MAX_NUM_SEQS" \
                    --tp "$TP_SIZE" \
                    --trust_remote_code \
                    --n-samples $N_SAMPLES
    TEMP_VAL="$TEMP"
fi

# The actual output file - use a glob pattern to find the file
echo "Waiting for output file to be generated..."
sleep 2  # Give some time for the file to be created

# Use find to locate the file with a more flexible pattern that matches actual filename format
OUTPUT_FILE=$(find "evalplus_results/${DATASET}" -name "*${MODEL_BASE}_vllm_temp_${TEMP_VAL}.jsonl" ! -name "*.raw.jsonl" -type f -print -quit)
if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Error: EvalPlus generation produced no output for $DATASET" >&2
    exit 1
fi

# Run evaluation with found file
python3 -m evalplus.evaluate --dataset "$DATASET" \
    --samples "$OUTPUT_FILE" \
    --output_file "evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json" \
    --min-time-limit 10.0 \
    --gt-time-limit-factor 8.0

echo "Evaluation complete. Results saved to evalplus_results/${DATASET}/${MODEL_BASE}_eval_results.json"
