#!/usr/bin/env bash
set -euo pipefail

# The authoritative inner-loop implementation lives in ~/Karpathy. This
# target wrapper only selects this checkout and its local experiment store.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KARPATHY_TARGET_DIR="${KARPATHY_TARGET_DIR:-$ROOT_DIR}"
export KARPATHY_STORE_DIR="${KARPATHY_STORE_DIR:-$HOME/Karpathy/experiments-q36-opt-27b}"
export KARPATHY_PROVIDER="${KARPATHY_PROVIDER:-deepseek}"
if [[ "${1:-}" == "--verbose" ]]; then
    export KARPATHY_VERBOSE=1
    shift
fi
exec bash "$HOME/Karpathy/run_inner.sh" "$@"
