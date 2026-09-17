#!/usr/bin/env bash
set -euo pipefail

# The authoritative outer-loop state machine lives in ~/Karpathy. This target
# wrapper only selects this checkout; the shared orchestrator owns cadence.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KARPATHY_TARGET_DIR="${KARPATHY_TARGET_DIR:-$ROOT_DIR}"
export KARPATHY_STORE_DIR="${KARPATHY_STORE_DIR:-$HOME/Karpathy/experiments-q36-opt-27b}"
exec python3 "$HOME/Karpathy/orchestrator.py" "$@"
