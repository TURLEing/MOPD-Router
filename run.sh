#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Public defaults. Teacher checkpoints are intentionally explicit because the
# three domain-specialized checkpoints are not bundled with this repository.
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_BASE_MODEL="${TEACHER_BASE_MODEL:-Qwen/Qwen3-4B}"
MATH_TEACHER_MODEL="${MATH_TEACHER_MODEL:-}"
CODE_TEACHER_MODEL="${CODE_TEACHER_MODEL:-}"
INSTRUCT_FOLLOWING_TEACHER_MODEL="${INSTRUCT_FOLLOWING_TEACHER_MODEL:-}"
DATA_ROOT="$REPO_ROOT/benchmarks/math_eval/data"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$DATA_ROOT/mopd_router_unlabeled_60k/train.parquet}"

for required_name in MATH_TEACHER_MODEL CODE_TEACHER_MODEL INSTRUCT_FOLLOWING_TEACHER_MODEL; do
    if [[ -z "${!required_name}" ]]; then
        echo "Error: $required_name must point to a local directory or Hugging Face model ID." >&2
        exit 2
    fi
done
if [[ ! -f "$TRAIN_DATA_PATH" ]]; then
    echo "Error: training parquet not found: $TRAIN_DATA_PATH" >&2
    echo "Ensure the released data is present under $DATA_ROOT." >&2
    exit 2
fi

# verl creates a validation dataloader even when periodic validation is off.
# The release includes the corresponding validation files under DATA_ROOT.
AIME24_VAL_PATH="${AIME24_VAL_PATH:-$DATA_ROOT/training_validation/aime24.parquet}"
AIME25_VAL_PATH="${AIME25_VAL_PATH:-$DATA_ROOT/training_validation/aime25.parquet}"
for validation_file in "$AIME24_VAL_PATH" "$AIME25_VAL_PATH"; do
    if [[ ! -f "$validation_file" ]]; then
        echo "Error: validation parquet not found: $validation_file" >&2
        echo "Ensure the released validation data is present under $DATA_ROOT/training_validation." >&2
        exit 2
    fi
done
VAL_FILES="['$AIME24_VAL_PATH', '$AIME25_VAL_PATH']"

VALIDATION="${VALIDATION:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-}"
TEST_FREQ="${TEST_FREQ:-}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-}"
if [[ "$VALIDATION" =~ ^([Tt][Rr][Uu][Ee]|1|[Yy][Ee][Ss]|[Oo][Nn])$ ]]; then
    LCB_VAL_PATH="${LCB_VAL_PATH:-$DATA_ROOT/training_validation/livecodebench_v6.parquet}"
    IFBENCH_VAL_PATH="${IFBENCH_VAL_PATH:-$DATA_ROOT/training_validation/ifbench.parquet}"
    for validation_file in "$LCB_VAL_PATH" "$IFBENCH_VAL_PATH"; do
        if [[ ! -f "$validation_file" ]]; then
            echo "Error: optional validation parquet not found: $validation_file" >&2
            echo "Ensure the released validation data is present under $DATA_ROOT/training_validation." >&2
            exit 2
        fi
    done
    VAL_FILES="['$AIME24_VAL_PATH', '$AIME25_VAL_PATH', '$LCB_VAL_PATH', '$IFBENCH_VAL_PATH']"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
    TEST_FREQ="${TEST_FREQ:-20}"
    LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-10}"
else
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    TEST_FREQ="${TEST_FREQ:--1}"
    LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-0}"
fi

NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-mopd-router-expertdelta-qwen3-1.7b}"
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/outputs/$EXPERIMENT_NAME}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
SAVE_FREQ="${SAVE_FREQ:-50}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1024}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRADIENT_CLIPPING="${GRADIENT_CLIPPING:-1.0}"
OBJECTIVE_CLIP_EPS="${OBJECTIVE_CLIP_EPS:-0.2}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
VAL_MAX_PROMPT_LENGTH="${VAL_MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((VAL_MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.7}"
EVAL_TOP_P="${EVAL_TOP_P:-0.8}"
EVAL_TOP_K="${EVAL_TOP_K:-20}"
DATA_SEED="${DATA_SEED:-42}"

# Router presets:
#   delta   + cosine = ExpertDelta
#   entropy                = predictive-entropy routing
#   accessible_novelty     = shared-support novelty routing
#   mean                   = equal-weight multi-teacher baseline
# Set ENTROPY_AWARE_ROUTER=false with labeled data for Standard MOPD.
ENTROPY_AWARE_ROUTER="${ENTROPY_AWARE_ROUTER:-true}"
TEACHER_SIGNAL_AGGREGATION="${TEACHER_SIGNAL_AGGREGATION:-delta}"
ENTROPY_ROUTER_TEMPERATURE="${ENTROPY_ROUTER_TEMPERATURE:-0.1}"
ENTROPY_CALIBRATION_ENABLED="${ENTROPY_CALIBRATION_ENABLED:-false}"
ACCESSIBLE_NOVELTY_TOP_K="${ACCESSIBLE_NOVELTY_TOP_K:-${ACCESSIBLE_NOVELTY_TOPK:-16}}"
ACCESSIBLE_NOVELTY_JS_TEMPERATURE="${ACCESSIBLE_NOVELTY_JS_TEMPERATURE:-0.1}"
ACCESSIBLE_NOVELTY_UTILITY_MODE="${ACCESSIBLE_NOVELTY_UTILITY_MODE:-default}"
DELTA_ROUTER_TOPK="${DELTA_ROUTER_TOPK:-16}"
DELTA_ROUTER_WEIGHTING="${DELTA_ROUTER_WEIGHTING:-cosine}"
DELTA_ROUTER_ALIGNMENT_EPS="${DELTA_ROUTER_ALIGNMENT_EPS:-0.000001}"

ENTROPY_ROUTER_LOG_SAMPLE="${ENTROPY_ROUTER_LOG_SAMPLE:-false}"
ENTROPY_ROUTER_LOG_INTERVAL="${ENTROPY_ROUTER_LOG_INTERVAL:-10}"
ENTROPY_ROUTER_LOG_MAX_TOKENS="${ENTROPY_ROUTER_LOG_MAX_TOKENS:-128}"
ENTROPY_ROUTER_LOG_SAMPLE_INDEX="${ENTROPY_ROUTER_LOG_SAMPLE_INDEX:--1}"

WANDB_MODE="${WANDB_MODE:-disabled}"
if [[ "$WANDB_MODE" == "disabled" || "$WANDB_MODE" == "offline" ]]; then
    TRAINER_LOGGER='["console"]'
else
    TRAINER_LOGGER='["console","wandb"]'
fi
export WANDB_MODE
export USED_MODEL="${USED_MODEL:-no_api}"

mkdir -p "$CKPT_DIR"
cd "$REPO_ROOT"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.val_max_prompt_length="$VAL_MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.shuffle=true \
    data.seed="$DATA_SEED" \
    data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="$STUDENT_MODEL" \
    +actor_rollout_ref.ref.model.path="$MATH_TEACHER_MODEL" \
    +actor_rollout_ref.ref.model.base_model_path="$CODE_TEACHER_MODEL" \
    +actor_rollout_ref.ref.model.teacher_base_model_path="$TEACHER_BASE_MODEL" \
    +actor_rollout_ref.ref.model.teacher_name=math \
    +actor_rollout_ref.ref.model.base_model_teacher_name=code \
    +actor_rollout_ref.ref.model.additional_teacher_model_paths.instruction_following="$INSTRUCT_FOLLOWING_TEACHER_MODEL" \
    actor_rollout_ref.actor.optim.lr="$LEARNING_RATE" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.clip_grad="$GRADIENT_CLIPPING" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=true \
    actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=true \
    actor_rollout_ref.actor.policy_loss.entropy_aware_router="$ENTROPY_AWARE_ROUTER" \
    actor_rollout_ref.actor.policy_loss.teacher_signal_aggregation="$TEACHER_SIGNAL_AGGREGATION" \
    actor_rollout_ref.actor.policy_loss.entropy_router_temperature="$ENTROPY_ROUTER_TEMPERATURE" \
    actor_rollout_ref.actor.policy_loss.entropy_calibration_enabled="$ENTROPY_CALIBRATION_ENABLED" \
    actor_rollout_ref.actor.policy_loss.accessible_novelty_top_k="$ACCESSIBLE_NOVELTY_TOP_K" \
    actor_rollout_ref.actor.policy_loss.accessible_novelty_js_temperature="$ACCESSIBLE_NOVELTY_JS_TEMPERATURE" \
    actor_rollout_ref.actor.policy_loss.accessible_novelty_utility_mode="$ACCESSIBLE_NOVELTY_UTILITY_MODE" \
    actor_rollout_ref.actor.policy_loss.delta_router_topk="$DELTA_ROUTER_TOPK" \
    actor_rollout_ref.actor.policy_loss.delta_router_weighting="$DELTA_ROUTER_WEIGHTING" \
    actor_rollout_ref.actor.policy_loss.delta_router_alignment_epsilon="$DELTA_ROUTER_ALIGNMENT_EPS" \
    actor_rollout_ref.actor.policy_loss.entropy_router_log_sample="$ENTROPY_ROUTER_LOG_SAMPLE" \
    actor_rollout_ref.actor.policy_loss.entropy_router_log_interval="$ENTROPY_ROUTER_LOG_INTERVAL" \
    actor_rollout_ref.actor.policy_loss.entropy_router_log_max_tokens="$ENTROPY_ROUTER_LOG_MAX_TOKENS" \
    actor_rollout_ref.actor.policy_loss.entropy_router_log_sample_index="$ENTROPY_ROUTER_LOG_SAMPLE_INDEX" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.clip_ratio="$OBJECTIVE_CLIP_EPS" \
    actor_rollout_ref.actor.clip_ratio_low="$OBJECTIVE_CLIP_EPS" \
    actor_rollout_ref.actor.clip_ratio_high="$OBJECTIVE_CLIP_EPS" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU" \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP_SIZE" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.temperature="$ROLLOUT_TEMPERATURE" \
    actor_rollout_ref.rollout.top_p="$ROLLOUT_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature="$EVAL_TEMPERATURE" \
    actor_rollout_ref.rollout.val_kwargs.top_p="$EVAL_TOP_P" \
    actor_rollout_ref.rollout.val_kwargs.top_k="$EVAL_TOP_K" \
    actor_rollout_ref.rollout.val_kwargs.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.use_kl_in_reward=false \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.project_name=mopd-router \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$NGPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "$@"
