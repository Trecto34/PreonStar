#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${KARPATHY_QWEN36_MODEL:-/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf}"
EVIDENCE_DIR="${KARPATHY_EVIDENCE_DIR:-${ROOT_DIR}/karpathy/evidence}"
LOG_FILE="${KARPATHY_COMPAT_LOG:-${EVIDENCE_DIR}/compatibility-gate.log}"
TIMEOUT_SECONDS="${KARPATHY_GATE_TIMEOUT:-180}"
CASE="${KARPATHY_GATE_CASE:-short_reasoning_plain}"
ACTIVE_PID=0

cleanup() {
    local status=$?
    trap - INT TERM EXIT
    if [[ "$ACTIVE_PID" -gt 0 ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    exit "$status"
}
trap 'exit 130' INT
trap 'cleanup' TERM EXIT

mkdir -p -- "$(dirname -- "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() { echo "COMPATIBILITY_GATE: FAIL: $*"; exit 1; }
[[ -x "$ROOT_DIR/q36_test" ]] || fail "missing executable: $ROOT_DIR/q36_test"
[[ -s "$MODEL" ]] || fail "missing Qwen3.6 model: $MODEL"

for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        fail "GPU process already running: $process_name"
    fi
done

echo "Karpathy Qwen3.6 compatibility gate"
echo "root=$ROOT_DIR"
echo "qwen36_model=$(basename -- "$MODEL")"
echo "case=$CASE timeout=${TIMEOUT_SECONDS}s"
echo "[1/1] Qwen3.6 CPU/Vulkan parity"

set +e
setsid env Q36_TEST_THREADS="${Q36_TEST_THREADS:-4}" \
    timeout --kill-after=5s "${TIMEOUT_SECONDS}s" \
    "$ROOT_DIR/q36_test" --gpu-cpu-parity --model "$MODEL" --case "$CASE" &
ACTIVE_PID=$!
wait "$ACTIVE_PID"
status=$?
ACTIVE_PID=0
set -e
[[ "$status" -eq 0 ]] || fail "Qwen3.6 parity exited with $status"

for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        fail "GPU process remained after parity: $process_name"
    fi
done

echo "COMPATIBILITY_GATE: PASS"
