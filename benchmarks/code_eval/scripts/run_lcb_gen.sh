#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)

# Default values
MODEL_PATH="Qwen/Qwen3-4"
LOCAL_MODEL_PATH=""
SAVE_NAME=""
CUDA_GPU_ID="7"
NUM_GPUS=1
BATCH_SIZE=128
N=4
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20
MAX_TOKENS=16384

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -l|--local_model_path)
      LOCAL_MODEL_PATH="$2"
      shift 2
      ;;
    --save_name)
      SAVE_NAME="$2"
      shift 2
      ;;
    -g|--gpu)
      CUDA_GPU_ID="$2"
      shift 2
      ;;
    -n|--n)
      N="$2"
      shift 2
      ;;
    -t|--temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    -p|--top_p)
      TOP_P="$2"
      shift 2
      ;;
    -b|--batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    -k|--max_tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    *)
      # Unknown option
      shift
      ;;
  esac
done

cd "$REPO_ROOT/benchmarks/code_eval/coding/LiveCodeBench"

# Fall back to the local model directory name if no explicit save name was given.
# A unique save name keeps different checkpoints from sharing an output/cache dir
# (every merged checkpoint has the same basename, e.g. "merged_hf_model").
if [ -z "$SAVE_NAME" ]; then
  SAVE_NAME=$(basename "$LOCAL_MODEL_PATH")
fi

# Pass local model path for tokenizer loading (avoids HF download in offline envs)
export LCB_TOKENIZER_PATH="$LOCAL_MODEL_PATH"

# Run LiveCodeBench with the AZR template and a local model
CUDA_VISIBLE_DEVICES=$CUDA_GPU_ID python -m lcb_runner.runner.main \
  --model $MODEL_PATH \
  --local_model_path $LOCAL_MODEL_PATH \
  --trust_remote_code \
  --scenario codegeneration \
  --release_version v6 \
  --tensor_parallel_size $NUM_GPUS \
  --use_cache \
  --n $N \
  --temperature $TEMPERATURE \
  --max_tokens $MAX_TOKENS \
  --custom_output_save_name "$SAVE_NAME" \
  --top_p $TOP_P \
  --top_k $TOP_K \
  --timeout 60 \
  --evaluate --continue_existing --continue_existing_with_eval
