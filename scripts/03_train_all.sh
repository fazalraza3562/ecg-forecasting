#!/usr/bin/env bash
# Train all six models in sequence. Each model gets its own --model
# invocation against the model-agnostic trainer in src/training/train.py.
# A failure in one model does not stop the others (we want the full grid
# of checkpoints even if one architecture blows up mid-run); the script
# tallies per-model status at the end so failures are visible.
#
# Usage:
#   ./scripts/03_train_all.sh [--device cpu|cuda|auto] [--seed N]
#
# Defaults to --device auto and the seed in config/default.yaml.

set -u  # error on undefined variables; intentionally NOT set -e

# -------- CLI parsing --------
DEVICE="auto"
SEED_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --seed)   SEED_ARG="--seed $2"; shift 2 ;;
        *)        echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

MODELS=(
    baseline_lstm
    cnn_lstm_attention
    transformer
    resnet1d
    inception1d
    tcn
)

# -------- run training --------
declare -A STATUS
for model in "${MODELS[@]}"; do
    echo
    echo "============================================================"
    echo "  training: ${model}"
    echo "============================================================"
    # shellcheck disable=SC2086  # we want word-splitting on $SEED_ARG
    if python -m src.training.train --model "${model}" --device "${DEVICE}" ${SEED_ARG}; then
        STATUS["${model}"]="OK"
    else
        STATUS["${model}"]="FAILED (exit $?)"
    fi
done

# -------- summary --------
echo
echo "============================================================"
echo "  summary"
echo "============================================================"
for model in "${MODELS[@]}"; do
    printf "  %-22s %s\n" "${model}" "${STATUS[${model}]}"
done

# Exit non-zero if any run failed so CI / wrapping scripts notice.
for s in "${STATUS[@]}"; do
    [[ "$s" == "OK" ]] || exit 1
done
