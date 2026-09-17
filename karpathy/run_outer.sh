#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export KARPATHY_TARGET_DIR="${KARPATHY_TARGET_DIR:-$ROOT_DIR}"
export KARPATHY_STORE_DIR="${KARPATHY_STORE_DIR:-$HOME/Karpathy/experiments-q36-opt}"
export KARPATHY_STATE_FILE="${KARPATHY_STATE_FILE:-$HOME/Karpathy/state-q36-opt.json}"
export KARPATHY_PROVIDER="${KARPATHY_PROVIDER:-deepseek}"
export KARPATHY_TARGET_NAME="${KARPATHY_TARGET_NAME:-Qwen3.6-35B-A3B MoE}"
export KARPATHY_PRIMARY_MODEL="${KARPATHY_PRIMARY_MODEL:-/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf}"
export KARPATHY_COMPAT_MODEL="${KARPATHY_COMPAT_MODEL:-$KARPATHY_PRIMARY_MODEL}"
exec python3 "$HOME/Karpathy/orchestrator.py" "$@"
