#! /bin/bash

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"

if [ -d "/workspace/RoboTwin" ]; then
  DEFAULT_ROBOTWIN_ROOT="/workspace/RoboTwin"
elif [ -d "${REPO_PATH}/../RoboTwin" ]; then
  DEFAULT_ROBOTWIN_ROOT="$(cd "${REPO_PATH}/../RoboTwin" && pwd)"
else
  DEFAULT_ROBOTWIN_ROOT="/workspace/RoboTwin"
fi

export EMBODIED_PATH
export REPO_PATH
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-$DEFAULT_ROBOTWIN_ROOT}"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RLINF_DP_EXPERT_BACKEND="${RLINF_DP_EXPERT_BACKEND:-remote}"

RDT_PORT="${RDT_PORT:-8769}"
ENV_PORT="${ENV_PORT:-8765}"
RDT_GPU="${RDT_GPU:-0}"
ENV_GPU="${ENV_GPU:-1}"
TRAIN_GPU="${TRAIN_GPU:-2}"

export ROBOTWIN_SERVER_ADDR="${ROBOTWIN_SERVER_ADDR:-127.0.0.1:${ENV_PORT}}"
export ROBOTWIN_RDT_SERVER_ADDR="${ROBOTWIN_RDT_SERVER_ADDR:-127.0.0.1:${RDT_PORT}}"
export ROBOTWIN_RDT_CKPT="${ROBOTWIN_RDT_CKPT:-${ROBOTWIN_ROOT}/policy/RDT/checkpoints/place_empty_cup_only_170m/checkpoint-20000}"
export ROBOTWIN_RDT_CONFIG="${ROBOTWIN_RDT_CONFIG:-${ROBOTWIN_ROOT}/policy/RDT/configs/base_170m.yaml}"
export ROBOTWIN_RDT_VISION_ENCODER="${ROBOTWIN_RDT_VISION_ENCODER:-${ROBOTWIN_ROOT}/policy/weights/RDT/siglip-so400m-patch14-384}"
export ROBOTWIN_RDT_TEXT_ENCODER="${ROBOTWIN_RDT_TEXT_ENCODER:-${ROBOTWIN_ROOT}/policy/weights/RDT/t5-v1_1-xxl}"
export ROBOTWIN_RDT_LORA_ADAPTER="${ROBOTWIN_RDT_LORA_ADAPTER:-}"
export ROBOTWIN_RDT_MERGE_LORA="${ROBOTWIN_RDT_MERGE_LORA:-1}"

ENV_CONFIG="${ENV_CONFIG:-${REPO_PATH}/examples/embodiment/config/env/robotwin_place_empty_cup.yaml}"
TASK_NAME="${TASK_NAME:-place_empty_cup}"
STEP_LIM="${STEP_LIM:-500}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
RDT_ACTION_STEPS="${RDT_ACTION_STEPS:-64}"
RDT_SUCCESS_DATASET_DIR="${RDT_SUCCESS_DATASET_DIR:-${ROBOTWIN_ROOT}/policy/RDT/training_data/place_empty_cup_rdt_success}"
RDT_SUCCESS_TARGET="${RDT_SUCCESS_TARGET:-50}"
RDT_SUCCESS_MAX_ATTEMPTS="${RDT_SUCCESS_MAX_ATTEMPTS:-200}"
RDT_LORA_CONFIG_NAME="${RDT_LORA_CONFIG_NAME:-place_empty_cup_success_lora_170m}"
RDT_LORA_ADAPTER_DEFAULT="${ROBOTWIN_ROOT}/policy/RDT/checkpoints/${RDT_LORA_CONFIG_NAME}/lora_adapter"

usage() {
  cat <<'USAGE'
Usage:
  run_robotwin_rdt_post_training_gate.sh <command> [extra args]

Commands:
  start-rdt-server     Start RoboTwin/script/robotwin_rdt_server.py with official 64-step RDT settings.
  start-env-server     Start RoboTwin/script/robotwin_env_server.py for place_empty_cup.
  smoke-eval           Run remote RDT policy smoke eval before any RLinf training.
  collect-success-dataset
                       Collect successful remote RDT episodes as processed RDT HDF5 data.
  prepare-success-dataset
                       Generate per-episode language embeddings for the collected HDF5 data.
  train-lora           Train a PEFT LoRA adapter on the collected success dataset.
  smoke-lora           Run remote smoke eval against a server started with the trained LoRA adapter.
  zero-noise-smoke     Run RLinf rollout/SAC smoke with actor.model.dsrl_noise_scale=0.0.
  dsrl-smoke           Run RLinf DSRL smoke with actor.model.dsrl_noise_scale=0.05.

Important env overrides:
  ROBOTWIN_ROOT, ROBOTWIN_RDT_CKPT, ROBOTWIN_RDT_CONFIG
  ROBOTWIN_RDT_VISION_ENCODER, ROBOTWIN_RDT_TEXT_ENCODER
  ROBOTWIN_RDT_LORA_ADAPTER, ROBOTWIN_RDT_MERGE_LORA
  ROBOTWIN_SERVER_ADDR, ROBOTWIN_RDT_SERVER_ADDR
  RDT_SUCCESS_DATASET_DIR, RDT_SUCCESS_TARGET, RDT_SUCCESS_MAX_ATTEMPTS, RDT_LORA_CONFIG_NAME
  RDT_GPU, ENV_GPU, TRAIN_GPU, EPISODES, SEED, STEP_LIM
USAGE
}

require_file_or_dir() {
  local path="$1"
  local name="$2"
  if [ ! -e "$path" ]; then
    echo "Missing ${name}: ${path}" >&2
    exit 2
  fi
}

print_context() {
  cat <<EOF
[rdt-gate] REPO_PATH=${REPO_PATH}
[rdt-gate] ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
[rdt-gate] ROBOTWIN_SERVER_ADDR=${ROBOTWIN_SERVER_ADDR}
[rdt-gate] ROBOTWIN_RDT_SERVER_ADDR=${ROBOTWIN_RDT_SERVER_ADDR}
[rdt-gate] ROBOTWIN_RDT_CKPT=${ROBOTWIN_RDT_CKPT}
[rdt-gate] ROBOTWIN_RDT_CONFIG=${ROBOTWIN_RDT_CONFIG}
[rdt-gate] ROBOTWIN_RDT_LORA_ADAPTER=${ROBOTWIN_RDT_LORA_ADAPTER:-<none>}
[rdt-gate] ENV_CONFIG=${ENV_CONFIG}
[rdt-gate] RDT_ACTION_STEPS=${RDT_ACTION_STEPS}
[rdt-gate] RDT_SUCCESS_DATASET_DIR=${RDT_SUCCESS_DATASET_DIR}
[rdt-gate] RDT_LORA_CONFIG_NAME=${RDT_LORA_CONFIG_NAME}
EOF
}

start_rdt_server() {
  require_file_or_dir "${ROBOTWIN_ROOT}/script/robotwin_rdt_server.py" "RDT server"
  require_file_or_dir "${ROBOTWIN_RDT_CKPT}" "RDT checkpoint"
  require_file_or_dir "${ROBOTWIN_RDT_CONFIG}" "RDT config"
  require_file_or_dir "${ROBOTWIN_RDT_VISION_ENCODER}" "RDT vision encoder"
  require_file_or_dir "${ROBOTWIN_RDT_TEXT_ENCODER}" "RDT text encoder"
  if [ -n "${ROBOTWIN_RDT_LORA_ADAPTER}" ]; then
    require_file_or_dir "${ROBOTWIN_RDT_LORA_ADAPTER}" "RDT LoRA adapter"
  fi
  print_context
  cd "${ROBOTWIN_ROOT}"
  local lora_args=()
  if [ -n "${ROBOTWIN_RDT_LORA_ADAPTER}" ]; then
    lora_args+=(--lora-adapter-path "${ROBOTWIN_RDT_LORA_ADAPTER}")
    if [ "${ROBOTWIN_RDT_MERGE_LORA}" = "1" ] || [ "${ROBOTWIN_RDT_MERGE_LORA}" = "true" ]; then
      lora_args+=(--merge-lora)
    fi
  fi
  exec env CUDA_VISIBLE_DEVICES="${RDT_GPU}" python script/robotwin_rdt_server.py \
    --host 0.0.0.0 \
    --port "${RDT_PORT}" \
    --ckpt "${ROBOTWIN_RDT_CKPT}" \
    --config "${ROBOTWIN_RDT_CONFIG}" \
    --vision-encoder-path "${ROBOTWIN_RDT_VISION_ENCODER}" \
    --text-encoder-path "${ROBOTWIN_RDT_TEXT_ENCODER}" \
    --device cuda:0 \
    --instruction "Place the empty cup to the target area." \
    --n-action-steps "${RDT_ACTION_STEPS}" \
    "${lora_args[@]}" \
    "$@"
}

start_env_server() {
  require_file_or_dir "${ROBOTWIN_ROOT}/script/robotwin_env_server.py" "RoboTwin env server"
  require_file_or_dir "${ENV_CONFIG}" "RoboTwin env config"
  print_context
  cd "${ROBOTWIN_ROOT}"
  exec env CUDA_VISIBLE_DEVICES="${ENV_GPU}" python script/robotwin_env_server.py \
    --host 0.0.0.0 \
    --port "${ENV_PORT}" \
    --config "${ENV_CONFIG}" \
    --assets-path "${ROBOTWIN_ROOT}/" \
    --debug-level 2 \
    --debug-every 1 \
    "$@"
}

smoke_eval() {
  require_file_or_dir "${REPO_PATH}/toolkits/eval_scripts_robotwin/eval_remote_rdt_smoke.py" "remote RDT smoke eval"
  require_file_or_dir "${ENV_CONFIG}" "RoboTwin env config"
  print_context
  cd "${REPO_PATH}"
  python toolkits/eval_scripts_robotwin/eval_remote_rdt_smoke.py \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --config "${ENV_CONFIG}" \
    --task-name "${TASK_NAME}" \
    --step-lim "${STEP_LIM}" \
    "$@"
}

collect_success_dataset() {
  require_file_or_dir "${REPO_PATH}/toolkits/eval_scripts_robotwin/eval_remote_rdt_smoke.py" "remote RDT smoke eval"
  require_file_or_dir "${ENV_CONFIG}" "RoboTwin env config"
  mkdir -p "${RDT_SUCCESS_DATASET_DIR}"
  print_context
  cd "${REPO_PATH}"
  python toolkits/eval_scripts_robotwin/eval_remote_rdt_smoke.py \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --config "${ENV_CONFIG}" \
    --task-name "${TASK_NAME}" \
    --step-lim "${STEP_LIM}" \
    --collect-success-dataset-dir "${RDT_SUCCESS_DATASET_DIR}" \
    --target-successes "${RDT_SUCCESS_TARGET}" \
    --max-attempts "${RDT_SUCCESS_MAX_ATTEMPTS}" \
    "$@"
}

prepare_success_dataset() {
  require_file_or_dir "${ROBOTWIN_ROOT}/policy/RDT/scripts/prepare_success_dataset.py" "success dataset prepare script"
  require_file_or_dir "${RDT_SUCCESS_DATASET_DIR}" "RDT success dataset"
  print_context
  cd "${ROBOTWIN_ROOT}/policy/RDT"
  env CUDA_VISIBLE_DEVICES="${RDT_GPU}" python scripts/prepare_success_dataset.py \
    --dataset-dir "${RDT_SUCCESS_DATASET_DIR}" \
    --gpu 0 \
    "$@"
}

train_lora() {
  require_file_or_dir "${ROBOTWIN_ROOT}/policy/RDT/finetune.sh" "RDT finetune script"
  require_file_or_dir "${ROBOTWIN_ROOT}/policy/RDT/model_config/${RDT_LORA_CONFIG_NAME}.yml" "RDT LoRA config"
  require_file_or_dir "${RDT_SUCCESS_DATASET_DIR}" "RDT success dataset"
  print_context
  cd "${ROBOTWIN_ROOT}/policy/RDT"
  bash finetune.sh "${RDT_LORA_CONFIG_NAME}" "$@"
}

smoke_lora() {
  local adapter="${ROBOTWIN_RDT_LORA_ADAPTER:-${RDT_LORA_ADAPTER_DEFAULT}}"
  require_file_or_dir "${adapter}" "trained RDT LoRA adapter"
  if [ "${ROBOTWIN_RDT_LORA_ADAPTER:-}" = "" ]; then
    export ROBOTWIN_RDT_LORA_ADAPTER="${adapter}"
  fi
  echo "[rdt-gate] smoke-lora expects start-rdt-server to be running with ROBOTWIN_RDT_LORA_ADAPTER=${ROBOTWIN_RDT_LORA_ADAPTER}" >&2
  smoke_eval "$@"
}

train_common() {
  local noise_scale="$1"
  local log_path="$2"
  shift 2
  require_file_or_dir "${EMBODIED_PATH}/train_embodied_agent.py" "RLinf train entry"
  print_context
  cd "${EMBODIED_PATH}"
  env CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python train_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name robotwin_place_empty_cup_dsrl_rdt \
    runner.logger.log_path="${log_path}" \
    runner.max_epochs=10 \
    runner.save_interval=1 \
    +runner.keep_last_checkpoints=3 \
    env.train.total_num_envs=1 \
    env.eval.total_num_envs=1 \
    env.train.video_cfg.save_video=false \
    env.eval.video_cfg.save_video=false \
    actor.micro_batch_size=8 \
    actor.global_batch_size=8 \
    rollout.enable_offload=false \
    actor.enable_offload=false \
    actor.fsdp_config.gradient_checkpointing=true \
    rollout.collect_prev_infos=false \
    actor.model.dsrl_noise_scale="${noise_scale}" \
    actor.model.num_action_chunks="${RDT_ACTION_STEPS}" \
    env.train.max_steps_per_rollout_epoch=512 \
    env.eval.max_steps_per_rollout_epoch=512 \
    "$@"
}

zero_noise_smoke() {
  train_common "0.0" "${REPO_PATH}/logs/dsrl-rdt-place-empty-cup-zero-noise-smoke" "$@"
}

dsrl_smoke() {
  train_common "0.05" "${REPO_PATH}/logs/dsrl-rdt-place-empty-cup-dsrl-smoke" \
    algorithm.update_epoch=5 \
    algorithm.train_actor_steps=1 \
    algorithm.entropy_tuning.target_entropy=auto_half_action_dim \
    "$@"
}

cmd="${1:-}"
if [ -z "$cmd" ]; then
  usage
  exit 1
fi
shift

case "$cmd" in
  start-rdt-server)
    start_rdt_server "$@"
    ;;
  start-env-server)
    start_env_server "$@"
    ;;
  smoke-eval)
    smoke_eval "$@"
    ;;
  collect-success-dataset)
    collect_success_dataset "$@"
    ;;
  prepare-success-dataset)
    prepare_success_dataset "$@"
    ;;
  train-lora)
    train_lora "$@"
    ;;
  smoke-lora)
    smoke_lora "$@"
    ;;
  zero-noise-smoke)
    zero_noise_smoke "$@"
    ;;
  dsrl-smoke)
    dsrl_smoke "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac
