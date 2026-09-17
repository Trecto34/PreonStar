#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_MODEL="${KARPATHY_SWIFT_MODEL:-/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf}"
QWEN35_MODEL="${KARPATHY_QWEN35_MODEL:-/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf}"
EVIDENCE_DIR="${KARPATHY_EVIDENCE_DIR:-${ROOT_DIR}/karpathy/evidence}"
LOG_FILE="${KARPATHY_COMPAT_LOG:-${EVIDENCE_DIR}/compatibility-gate.log}"
TIMEOUT_SECONDS="${KARPATHY_GATE_TIMEOUT:-180}"
CTX="${KARPATHY_GATE_CTX:-1024}"
GEN_TOKENS="${KARPATHY_GATE_GEN_TOKENS:-4}"

mkdir -p -- "$(dirname -- "$LOG_FILE")"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

fail() { echo "COMPATIBILITY_GATE: FAIL: $*"; exit 1; }
for command_name in q36 q36-bench q36_test; do
    [[ -x "$ROOT_DIR/$command_name" ]] || fail "missing executable: $ROOT_DIR/$command_name"
done
[[ -s "$SWIFT_MODEL" ]] || fail "missing Swift model: $SWIFT_MODEL"
[[ -s "$QWEN35_MODEL" ]] || fail "missing Qwen3.5/Qwen3.6 model: $QWEN35_MODEL"

# pgrep -x does not accept shell alternation; check each process explicitly.
for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        fail "GPU process already running: $process_name"
    fi
done

echo "Karpathy compatibility gate"
echo "root=$ROOT_DIR"
echo "swift_model=$(basename -- "$SWIFT_MODEL")"
echo "qwen35_model=$(basename -- "$QWEN35_MODEL")"
echo "context=$CTX generation=$GEN_TOKENS timeout=${TIMEOUT_SECONDS}s"

echo "[1/2] Qwen3.5/Qwen3.6 MoE CPU/Vulkan parity"
set +e
Q36_TEST_THREADS="${Q36_TEST_THREADS:-4}" \
    timeout --kill-after=5s "${TIMEOUT_SECONDS}s" \
    "$ROOT_DIR/q36_test" --gpu-cpu-parity --model "$QWEN35_MODEL" \
    --case short_reasoning_plain
qwen_status=$?
set -e
[[ "$qwen_status" -eq 0 ]] || fail "Qwen3.5/Qwen3.6 parity exited with $qwen_status"

for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        fail "GPU process remained after parity: $process_name"
    fi
done

echo "[2/2] Swift IQ3_XXS Vulkan smoke benchmark"
set +e
env \
    Q36_SWIFT_MODEL="$SWIFT_MODEL" \
    Q36_SWIFT_CTX="$CTX" \
    Q36_SWIFT_MTP_DRAFT=1 \
    timeout --kill-after=5s "${TIMEOUT_SECONDS}s" \
    "$ROOT_DIR/q36-bench" --model "$SWIFT_MODEL" --vulkan \
    --prompt-file "$ROOT_DIR/tests/long_context_story_prompt.txt" \
    --prefill-chunk 512 --ctx-start "$CTX" --ctx-max "$CTX" \
    --ctx-alloc "$((CTX + GEN_TOKENS + 1))" --gen-tokens "$GEN_TOKENS" \
    -ctk q8_0 -ctv q4_0 --mtp-draft 1
swift_status=$?
set -e
[[ "$swift_status" -eq 0 ]] || fail "Swift smoke exited with $swift_status"

for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        fail "GPU process remained after Swift smoke: $process_name"
    fi
done

echo "COMPATIBILITY_GATE: PASS"
