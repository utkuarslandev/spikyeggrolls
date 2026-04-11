#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

source "${REPO_ROOT}/env/runpod.env"

if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y git curl
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "${WORKSPACE_ROOT}" "${DATA_PATH}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

cd "${REPO_ROOT}"

if [ ! -d ".venv" ]; then
    uv venv
fi

source .venv/bin/activate

uv pip install -e "${SPIKYEGGROLL_INSTALL_TARGET}"

bash "${REPO_ROOT}/scripts/doctor.sh" runpod

echo
echo "Runpod bootstrap complete."
echo "Next:"
echo "  make doctor-runpod"
echo "  make smoke-runpod"
