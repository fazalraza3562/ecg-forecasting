#!/usr/bin/env bash
# Run filtering, normalization, windowing, and the patient-disjoint split.
set -euo pipefail

# Run from the repository root regardless of where the user invoked the script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

if [[ -d .venv ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

python -m src.data.preprocess --config config/default.yaml "$@"
