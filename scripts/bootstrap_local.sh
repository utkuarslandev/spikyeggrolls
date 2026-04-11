#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

source "${REPO_ROOT}/env/local.env"

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

bash "${REPO_ROOT}/scripts/doctor.sh" local

echo
echo "Local bootstrap complete."
echo "Next:"
echo "  make doctor-local"
echo "  make smoke-local"
