#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL="Qwen3-4B"
MODEL_PATH="Qwen/Qwen3-4B"
MODEL_NAME=$MODEL

echo $MODEL_PATH
echo $MODEL_NAME

# Create output directories if they don't exist
mkdir -p "$SCRIPT_DIR/eval_outputs/aime24"
mkdir -p "$SCRIPT_DIR/eval_outputs/aime25"
mkdir -p "$SCRIPT_DIR/eval_outputs/hmmt25_feb"
mkdir -p "$SCRIPT_DIR/eval_outputs/hmmt25_nov"

# aime24
CUDA_VISIBLE_DEVICES=0,1 python3 "$SCRIPT_DIR/eval_math.py" \
    --input_file "$SCRIPT_DIR/data/aime24/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "$SCRIPT_DIR/eval_outputs/aime24/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# aime25
CUDA_VISIBLE_DEVICES=2,3 python3 "$SCRIPT_DIR/eval_math.py" \
    --input_file "$SCRIPT_DIR/data/aime25/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "$SCRIPT_DIR/eval_outputs/aime25/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Feb
CUDA_VISIBLE_DEVICES=4,5 python3 "$SCRIPT_DIR/eval_math.py" \
    --input_file "$SCRIPT_DIR/data/hmmt25_feb/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "$SCRIPT_DIR/eval_outputs/hmmt25_feb/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Nov
CUDA_VISIBLE_DEVICES=6,7 python3 "$SCRIPT_DIR/eval_math.py" \
    --input_file "$SCRIPT_DIR/data/hmmt25_nov/test.jsonl" \
    --model_path $MODEL_PATH  \
    --output_file "$SCRIPT_DIR/eval_outputs/hmmt25_nov/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 0.7 \
    --top_p 0.8 \
    --top_k 20 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &

wait
echo "Model $MODEL_NAME done!"
