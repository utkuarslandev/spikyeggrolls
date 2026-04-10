#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional local-only file for SSH endpoint overrides.
if [ -f "${REPO_ROOT}/env/runpod.ssh.env" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/env/runpod.ssh.env"
fi

RUNPOD_SSH_USER="${RUNPOD_SSH_USER:-root}"
RUNPOD_SSH_HOST="${RUNPOD_SSH_HOST:-}"
RUNPOD_SSH_PORT="${RUNPOD_SSH_PORT:-}"
RUNPOD_SSH_KEY="${RUNPOD_SSH_KEY:-${HOME}/.ssh/id_ed25519}"
RUNPOD_REMOTE_REPO="${RUNPOD_REMOTE_REPO:-/workspace/spikyeggrolls}"

if [ -z "${RUNPOD_SSH_HOST}" ] || [ -z "${RUNPOD_SSH_PORT}" ]; then
    echo "Set RUNPOD_SSH_HOST and RUNPOD_SSH_PORT (env vars or env/runpod.ssh.env)." >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "Usage: $0 <remote command>" >&2
    echo "Example: $0 make doctor-runpod" >&2
    exit 1
fi

printf -v remote_repo_quoted "%q" "${RUNPOD_REMOTE_REPO}"
remote_cmd="cd ${remote_repo_quoted} &&"
for arg in "$@"; do
    printf -v arg_quoted "%q" "${arg}"
    remote_cmd+=" ${arg_quoted}"
done

ssh -i "${RUNPOD_SSH_KEY}" \
    -p "${RUNPOD_SSH_PORT}" \
    -o StrictHostKeyChecking=accept-new \
    "${RUNPOD_SSH_USER}@${RUNPOD_SSH_HOST}" \
    "${remote_cmd}"
