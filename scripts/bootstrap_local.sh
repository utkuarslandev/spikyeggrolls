#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

source "${REPO_ROOT}/env/local.env"

PARENT_DIR="$(cd "${REPO_ROOT}/.." && pwd)"
HYPERSCALEES_DIR="${HYPERSCALEES_DIR:-${PARENT_DIR}/HyperscaleES}"
HYPERSCALEES_REPO="${HYPERSCALEES_REPO:-https://github.com/ESHyperscale/HyperscaleES.git}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

mkdir -p "${WORKSPACE_ROOT}" "${DATA_PATH}" "${LOG_DIR}" "${CHECKPOINT_DIR}" "${JAX_COMPILATION_CACHE_DIR}"

if [ ! -d "${HYPERSCALEES_DIR}" ]; then
    git clone "${HYPERSCALEES_REPO}" "${HYPERSCALEES_DIR}"
fi

cd "${REPO_ROOT}"

if [ ! -d ".venv" ]; then
    uv venv
fi

source .venv/bin/activate

uv pip install -e "${SPIKYEGGROLL_INSTALL_TARGET}"
uv pip install -e "${HYPERSCALEES_DIR}"

bash "${REPO_ROOT}/scripts/doctor.sh" local

echo
echo "Local bootstrap complete."
echo "Next:"
echo "  make doctor-local"
echo "  make smoke-local"
