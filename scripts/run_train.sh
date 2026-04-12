#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

PRESET="${1:-smoke}"
PROFILE="${PROFILE:-local}"

shift $(( $# > 0 ? 1 : 0 ))
if [ $# -gt 0 ] && [ -f "${REPO_ROOT}/env/${1}.env" ]; then
    PROFILE="${1}"
    shift
fi
EXTRA_ARGS=("$@")

ENV_FILE="${REPO_ROOT}/env/${PROFILE}.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "Unknown profile '${PROFILE}'. Expected ${REPO_ROOT}/env/${PROFILE}.env" >&2
    exit 1
fi

source "${ENV_FILE}"

if [ ! -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    echo "Missing virtualenv at ${REPO_ROOT}/.venv. Run a bootstrap script first." >&2
    exit 1
fi

mkdir -p "${DATA_PATH}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

source "${REPO_ROOT}/.venv/bin/activate"

case "${PRESET}" in
    smoke)
        : "${POP_SIZE:=512}"
        : "${RANK:=2}"
        : "${SIGMA:=0.02}"
        : "${LR:=0.005}"
        : "${EPOCHS:=30}"
        : "${BATCH_SIZE:=128}"
        : "${CHUNK_SIZE:=256}"
        : "${TIMESTEPS:=25}"
        ;;
    tune)
        : "${POP_SIZE:=2048}"
        : "${RANK:=2}"
        : "${SIGMA:=0.01}"
        : "${LR:=0.005}"
        : "${EPOCHS:=100}"
        : "${BATCH_SIZE:=256}"
        : "${CHUNK_SIZE:=512}"
        : "${TIMESTEPS:=25}"
        ;;
    full)
        : "${POP_SIZE:=10000}"
        : "${RANK:=3}"
        : "${SIGMA:=0.007}"
        : "${LR:=0.001}"
        : "${EPOCHS:=4000}"
        : "${BATCH_SIZE:=256}"
        : "${CHUNK_SIZE:=1024}"
        : "${TIMESTEPS:=25}"
        ;;
    *)
        echo "Unknown preset '${PRESET}'. Use smoke, tune, or full." >&2
        exit 1
        ;;
esac

STAMP="$(date +%Y%m%d-%H%M%S)"
: "${RUN_NAME:=${PRESET}-${PROFILE}-${STAMP}}"
: "${LOG_INTERVAL:=10}"
: "${TEST_INTERVAL:=100}"
: "${CHECKPOINT_INTERVAL:=100}"
: "${DATASET:=mnist}"
: "${MODEL_NAME:=mlp_snn}"
: "${UPDATES_PER_EPOCH:=1}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.stdout.log"

if [ "${DATASET}" = "cifar10" ] && [ "${MODEL_NAME}" = "spiking_resnet18" ]; then
    : "${TIMESTEPS:=16}"
    : "${BATCH_SIZE:=32}"
    : "${CHUNK_SIZE:=96}"
    : "${UPDATES_PER_EPOCH:=64}"
    : "${FITNESS_SHAPING:=centered_rank}"
    : "${USE_BATCHED_UPDATE:=true}"
    : "${DTYPE:=bfloat16}"
    : "${SIGMA_MAX:=0.012}"
    : "${NUM_TEST_EVAL_SAMPLES:=1024}"
fi

echo "Profile: ${PROFILE}"
echo "Preset: ${PRESET}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_NAME}"
echo "Data path: ${DATA_PATH}"
echo "Log file: ${LOG_FILE}"
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR}"
echo "Run name: ${RUN_NAME}"

CMD=(
  python -m spikyeggroll.train
  --dataset "${DATASET}" \
  --model_name "${MODEL_NAME}" \
  --pop_size "${POP_SIZE}" \
  --rank "${RANK}" \
  --sigma "${SIGMA}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --chunk_size "${CHUNK_SIZE}" \
  --updates_per_epoch "${UPDATES_PER_EPOCH}" \
  --timesteps "${TIMESTEPS}" \
  --data_path "${DATA_PATH}" \
  --run_name "${RUN_NAME}" \
  --log_dir "${LOG_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --log_interval "${LOG_INTERVAL}" \
  --test_interval "${TEST_INTERVAL}" \
  --checkpoint_interval "${CHECKPOINT_INTERVAL}"
)

if [ -n "${FITNESS_SHAPING:-}" ]; then
    CMD+=(--fitness_shaping "${FITNESS_SHAPING}")
fi

if [ -n "${DTYPE:-}" ]; then
    CMD+=(--dtype "${DTYPE}")
fi

if [ -n "${SIGMA_MAX:-}" ]; then
    CMD+=(--sigma_max "${SIGMA_MAX}")
fi

if [ -n "${NUM_TEST_EVAL_SAMPLES:-}" ]; then
    CMD+=(--num_test_eval_samples "${NUM_TEST_EVAL_SAMPLES}")
fi

if [ "${USE_BATCHED_UPDATE:-}" = "true" ]; then
    CMD+=(--use_batched_update)
elif [ "${USE_BATCHED_UPDATE:-}" = "false" ]; then
    CMD+=(--no-use_batched_update)
fi

if [ -n "${RESUME_FROM:-}" ]; then
    CMD+=(--resume_from "${RESUME_FROM}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}" | tee "${LOG_FILE}"
