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

resolve_install_target() {
    if [ -n "${SPIKYEGGROLL_INSTALL_TARGET:-}" ] && [ "${SPIKYEGGROLL_INSTALL_TARGET}" != "auto" ]; then
        printf '%s\n' "${SPIKYEGGROLL_INSTALL_TARGET}"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        printf '%s\n' '.[cpu]'
        return 0
    fi

    printf '%s\n' 'auto-gpu'
}

install_project() {
    local install_target="$1"

    if [ "${install_target}" != "auto-gpu" ]; then
        echo "Installing with target ${install_target}"
        uv pip install -e "${install_target}"
        return 0
    fi

    local targets=('.[cuda13]' '.[cuda12]')
    local target
    for target in "${targets[@]}"; do
        echo "Attempting GPU install target ${target}"
        if uv pip install -e "${target}"; then
            export SPIKYEGGROLL_INSTALL_TARGET="${target}"
            echo "Selected GPU install target ${target}"
            return 0
        fi
        echo "Install target ${target} failed; trying next option."
    done

    echo "No compatible GPU JAX wheel found; falling back to CPU install."
    uv pip install -e '.[cpu]'
    export SPIKYEGGROLL_INSTALL_TARGET='.[cpu]'
}

install_project "$(resolve_install_target)"

bash "${REPO_ROOT}/scripts/doctor.sh" runpod

echo
echo "Runpod bootstrap complete."
echo "Next:"
echo "  make doctor-runpod"
echo "  make smoke-runpod"
