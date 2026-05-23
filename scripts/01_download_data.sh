#!/usr/bin/env bash
# Fetch SDDB, MIT-BIH, and INCART recordings from PhysioNet into data/raw/.
set -euo pipefail

# Run from the repository root regardless of where the user invoked the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -d .venv ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

python -m src.data.download --config config/default.yaml "$@"
