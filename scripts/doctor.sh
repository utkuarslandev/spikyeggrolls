#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export REPO_ROOT

PROFILE="${1:-${PROFILE:-local}}"
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

source "${REPO_ROOT}/.venv/bin/activate"

python - <<'PY'
import importlib
import os
import sys

modules = ["jax", "torch", "torchvision", "spikyeggroll", "hyperscalees"]
failed = False

print(f"profile={os.environ.get('PLATFORM_NAME')}")
print(f"repo_root={os.environ.get('REPO_ROOT')}")
print(f"workspace_root={os.environ.get('WORKSPACE_ROOT')}")
print(f"data_path={os.environ.get('DATA_PATH')}")
print(f"log_dir={os.environ.get('LOG_DIR')}")
print(f"checkpoint_dir={os.environ.get('CHECKPOINT_DIR')}")
print(f"jax_cache={os.environ.get('JAX_COMPILATION_CACHE_DIR')}")
print(f"python={sys.executable}")

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}=ok version={version}")
    except Exception as exc:
        failed = True
        print(f"{name}=error {exc}")

try:
    import jax
    devices = jax.devices()
    print(f"jax_devices={devices}")
    require_accelerator = os.environ.get("REQUIRE_ACCELERATOR", "0") == "1"
    if require_accelerator and not any(device.platform != "cpu" for device in devices):
        failed = True
        print("accelerator=error expected non-CPU JAX device, found CPU-only runtime")
except Exception as exc:
    failed = True
    print(f"jax_devices=error {exc}")

if failed:
    sys.exit(1)
PY
