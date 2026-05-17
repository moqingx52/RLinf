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
export ROBOTWIN_PLANNER_BACKEND="${ROBOTWIN_PLANNER_BACKEND:-curobo}"

# Use the RoboTwin task config that matches the official RDT eval distribution.
# The generic RLinf env yaml is piper/no-wrist by default; RDT was trained/evaled
# with aloha-agilex and head/right/left wrist cameras.
ENV_CONFIG="${ENV_CONFIG:-${ROBOTWIN_ROOT}/task_config/demo_randomized.yml}"
TASK_NAME="${TASK_NAME:-place_empty_cup}"
STEP_LIM="${STEP_LIM:-500}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"
RDT_ACTION_STEPS="${RDT_ACTION_STEPS:-64}"
RDT_SUCCESS_DATASET_DIR="${RDT_SUCCESS_DATASET_DIR:-${ROBOTWIN_ROOT}/policy/RDT/training_data/place_empty_cup_rdt_success}"
RDT_SUCCESS_TARGET="${RDT_SUCCESS_TARGET:-50}"
RDT_SUCCESS_MAX_ATTEMPTS="${RDT_SUCCESS_MAX_ATTEMPTS:-200}"
RDT_SUCCESS_SEED_CHECK_MODE="${RDT_SUCCESS_SEED_CHECK_MODE:-setup}"
RDT_LORA_CONFIG_NAME="${RDT_LORA_CONFIG_NAME:-place_empty_cup_success_lora_170m}"
RDT_LORA_ADAPTER_DEFAULT="${ROBOTWIN_ROOT}/policy/RDT/checkpoints/${RDT_LORA_CONFIG_NAME}/lora_adapter"
RDT_EXPERT_TASK_CONFIG="${RDT_EXPERT_TASK_CONFIG:-$(basename "${ENV_CONFIG%.*}")}"
RDT_EXPERT_DATA_NUM="${RDT_EXPERT_DATA_NUM:-${RDT_SUCCESS_TARGET}}"
RDT_EXPERT_EPISODE_NUM="${RDT_EXPERT_EPISODE_NUM:-${RDT_EXPERT_DATA_NUM}}"
RDT_EXPERT_GPU="${RDT_EXPERT_GPU:-${ENV_GPU}}"
RDT_EXPERT_PROCESS_GPU="${RDT_EXPERT_PROCESS_GPU:-${RDT_GPU}}"

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
  collect-expert-dataset
                       Collect official RoboTwin expert demos, convert to RDT format, and add them to the LoRA dataset.
  prepare-success-dataset
                       Generate per-episode language embeddings for the collected HDF5 data.
  train-lora           Train a PEFT LoRA adapter on the collected success dataset.
  smoke-lora           Run remote smoke eval against a server started with the trained LoRA adapter.
  debug-curobo         Check whether RoboTwin can import and instantiate the cuRobo planner module.
  zero-noise-smoke     Run RLinf rollout/SAC smoke with actor.model.dsrl_noise_scale=0.0.
  dsrl-smoke           Run RLinf DSRL smoke with actor.model.dsrl_noise_scale=0.05.

Important env overrides:
  ROBOTWIN_ROOT, ROBOTWIN_RDT_CKPT, ROBOTWIN_RDT_CONFIG
  ROBOTWIN_RDT_VISION_ENCODER, ROBOTWIN_RDT_TEXT_ENCODER
  ROBOTWIN_RDT_LORA_ADAPTER, ROBOTWIN_RDT_MERGE_LORA
  ROBOTWIN_SERVER_ADDR, ROBOTWIN_RDT_SERVER_ADDR, ROBOTWIN_PLANNER_BACKEND
  RDT_SUCCESS_DATASET_DIR, RDT_SUCCESS_TARGET, RDT_SUCCESS_MAX_ATTEMPTS, RDT_SUCCESS_SEED_CHECK_MODE, RDT_LORA_CONFIG_NAME
  RDT_EXPERT_TASK_CONFIG, RDT_EXPERT_DATA_NUM, RDT_EXPERT_EPISODE_NUM, RDT_EXPERT_GPU, RDT_EXPERT_PROCESS_GPU
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
[rdt-gate] ROBOTWIN_PLANNER_BACKEND=${ROBOTWIN_PLANNER_BACKEND}
[rdt-gate] ENV_CONFIG=${ENV_CONFIG}
[rdt-gate] RDT_ACTION_STEPS=${RDT_ACTION_STEPS}
[rdt-gate] RDT_SUCCESS_DATASET_DIR=${RDT_SUCCESS_DATASET_DIR}
[rdt-gate] RDT_SUCCESS_SEED_CHECK_MODE=${RDT_SUCCESS_SEED_CHECK_MODE}
[rdt-gate] RDT_LORA_CONFIG_NAME=${RDT_LORA_CONFIG_NAME}
[rdt-gate] RDT_EXPERT_TASK_CONFIG=${RDT_EXPERT_TASK_CONFIG}
[rdt-gate] RDT_EXPERT_DATA_NUM=${RDT_EXPERT_DATA_NUM}
[rdt-gate] RDT_EXPERT_EPISODE_NUM=${RDT_EXPERT_EPISODE_NUM}
[rdt-gate] RDT_EXPERT_GPU=${RDT_EXPERT_GPU}
[rdt-gate] RDT_EXPERT_PROCESS_GPU=${RDT_EXPERT_PROCESS_GPU}
EOF
}

yaml_episode_num() {
  local config_path="$1"
  python - "$config_path" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}
print(int(payload.get("episode_num", 0)))
PY
}

ensure_expert_task_config() {
  local source_name="$1"
  local episode_num="$2"
  local source_path="${ROBOTWIN_ROOT}/task_config/${source_name}.yml"
  require_file_or_dir "${source_path}" "RoboTwin expert task config"

  local source_episode_num
  source_episode_num="$(yaml_episode_num "${source_path}")"
  if [ "${episode_num}" -le "${source_episode_num}" ]; then
    echo "${source_name}"
    return
  fi

  local generated_name="${source_name}_expert_${episode_num}"
  local generated_path="${ROBOTWIN_ROOT}/task_config/${generated_name}.yml"
  python - "$source_path" "$generated_path" "$episode_num" <<'PY'
import sys
import yaml

source_path, generated_path, episode_num = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(source_path, "r", encoding="utf-8") as f:
    payload = yaml.safe_load(f) or {}
payload["episode_num"] = episode_num
with open(generated_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
PY
  echo "${generated_name}"
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
    --planner-backend "${ROBOTWIN_PLANNER_BACKEND}" \
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
    --planner-backend "${ROBOTWIN_PLANNER_BACKEND}" \
    --seed-check-mode "${RDT_SUCCESS_SEED_CHECK_MODE}" \
    --collect-success-dataset-dir "${RDT_SUCCESS_DATASET_DIR}" \
    --target-successes "${RDT_SUCCESS_TARGET}" \
    --max-attempts "${RDT_SUCCESS_MAX_ATTEMPTS}" \
    "$@"
}

collect_expert_dataset() {
  require_file_or_dir "${ROBOTWIN_ROOT}/collect_data.sh" "RoboTwin expert collector"
  require_file_or_dir "${ROBOTWIN_ROOT}/policy/RDT/process_data_rdt.sh" "RDT data processor"
  require_file_or_dir "${ROBOTWIN_ROOT}/policy/RDT/scripts/copy_processed_to_training_data.sh" "RDT training data copy script"

  local task_config
  task_config="$(ensure_expert_task_config "${RDT_EXPERT_TASK_CONFIG}" "${RDT_EXPERT_EPISODE_NUM}")"
  local raw_data_dir="${ROBOTWIN_ROOT}/data/${TASK_NAME}/${task_config}/data"
  local processed_dir="processed_data/${TASK_NAME}-${task_config}-${RDT_EXPERT_DATA_NUM}"
  local processed_abs="${ROBOTWIN_ROOT}/policy/RDT/${processed_dir}"
  local training_root="${ROBOTWIN_ROOT}/policy/RDT/training_data"
  local dest_base
  dest_base="$(basename "${processed_dir}")"

  print_context
  echo "[rdt-gate] Collecting official expert data: task=${TASK_NAME} task_config=${task_config} episodes=${RDT_EXPERT_EPISODE_NUM}" >&2
  cd "${ROBOTWIN_ROOT}"
  bash collect_data.sh "${TASK_NAME}" "${task_config}" "${RDT_EXPERT_GPU}" "$@"

  require_file_or_dir "${raw_data_dir}" "collected RoboTwin expert data"
  echo "[rdt-gate] Converting official expert data to RDT format: num=${RDT_EXPERT_DATA_NUM}" >&2
  cd "${ROBOTWIN_ROOT}/policy/RDT"
  bash process_data_rdt.sh "${TASK_NAME}" "${task_config}" "${RDT_EXPERT_DATA_NUM}" "${RDT_EXPERT_PROCESS_GPU}"
  require_file_or_dir "${processed_abs}" "processed RDT expert dataset"

  mkdir -p "${RDT_SUCCESS_DATASET_DIR}"
  if [[ "${RDT_SUCCESS_DATASET_DIR}" == "${training_root}/"* ]]; then
    local run_name="${RDT_SUCCESS_DATASET_DIR#${training_root}/}"
    local default_dest="${training_root}/${run_name}/${dest_base}"
    if [ -d "${default_dest}" ]; then
      cp -a "${processed_abs}/." "${default_dest}/"
      echo "Merged ${processed_abs} -> ${default_dest}"
    else
      bash scripts/copy_processed_to_training_data.sh "${run_name}" "${processed_dir}"
    fi
  else
    local dest_dir="${RDT_SUCCESS_DATASET_DIR}/${dest_base}"
    mkdir -p "${dest_dir}"
    cp -a "${processed_abs}/." "${dest_dir}/"
    echo "Copied ${processed_abs} -> ${dest_dir}"
  fi

  echo "[rdt-gate] Expert RDT dataset is available under ${RDT_SUCCESS_DATASET_DIR}/${dest_base}" >&2
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

debug_curobo() {
  print_context
  cd "${ROBOTWIN_ROOT}"
  env CUDA_VISIBLE_DEVICES="${ENV_GPU}" python - <<'PY'
import inspect
from envs.robot import planner

print("[rdt-gate] CuroboPlanner defined:", hasattr(planner, "CuroboPlanner"))
if hasattr(planner, "CuroboPlanner"):
    print("[rdt-gate] CuroboPlanner:", planner.CuroboPlanner)
    print("[rdt-gate] CuroboPlanner.plan_path:", inspect.signature(planner.CuroboPlanner.plan_path))
try:
    import curobo
    print("[rdt-gate] curobo module:", getattr(curobo, "__file__", curobo))
except Exception as exc:
    print("[rdt-gate] curobo import failed:", repr(exc))
    raise
PY
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
  collect-expert-dataset)
    collect_expert_dataset "$@"
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
  debug-curobo)
    debug_curobo "$@"
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
