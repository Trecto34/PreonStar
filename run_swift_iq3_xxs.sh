#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
MODEL=${P36_SWIFT_MODEL:-"$ROOT/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf"}

P36_SWIFT_MODEL="$MODEL" exec "$ROOT/run_swift.sh" "$@"
