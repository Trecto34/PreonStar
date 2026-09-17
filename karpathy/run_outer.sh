#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MAX_ITERATIONS="${KARPATHY_MAX_ITERATIONS:-1}"
META_EVERY="${KARPATHY_META_EVERY:-5}"
GATE_TIMEOUT="${KARPATHY_GATE_TIMEOUT:-180}"

if ! [[ "$MAX_ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "KARPATHY_MAX_ITERATIONS must be a positive integer" >&2
    exit 2
fi

run_gate() {
    echo "[OUTER] compatibility pre/postflight"
    KARPATHY_GATE_TIMEOUT="$GATE_TIMEOUT" "$ROOT_DIR/karpathy/compat_gate.sh"
}

run_meta() {
    local prompt
    prompt="$(cat <<EOF
You are the outer-loop process architect for $ROOT_DIR.
Read karpathy/Task.md, karpathy/AlreadyTried.md, changelog.txt, and the latest
records under .karpathy/experiments. Do not edit Vulkan code. Diagnose whether
the inner loop is making valid progress, then update only the local Karpathy
task/ledger guidance with the next untried Swift IQ3_XXS hypothesis. Preserve
the hard compatibility requirement: every candidate must pass
karpathy/compat_gate.sh for both Swift IQ3_XXS and Qwen3.5/Qwen3.6 qwen35moe.
Do not touch /home/server/Karpathy or another checkout.
EOF
  )"
    if [[ -n "${KARPATHY_OUTER_COMMAND:-}" ]]; then
        timeout --kill-after=20s "${KARPATHY_AGENT_TIMEOUT:-900}s" \
            env KARPATHY_PROMPT="$prompt" bash -lc "${KARPATHY_OUTER_COMMAND} \"\$KARPATHY_PROMPT\""
    elif command -v claude >/dev/null 2>&1; then
        timeout --kill-after=20s "${KARPATHY_AGENT_TIMEOUT:-900}s" \
            claude --model "${KARPATHY_OUTER_MODEL:-opus}" --dangerously-skip-permissions -p "$prompt"
    else
        echo "[OUTER] no outer agent configured; skipping process audit" >&2
    fi
}

cd "$ROOT_DIR"
for ((iteration = 1; iteration <= MAX_ITERATIONS; iteration++)); do
    echo "[OUTER] pass $iteration/$MAX_ITERATIONS"
    run_gate
    if ! "$ROOT_DIR/karpathy/run_inner.sh"; then
        echo "[OUTER] inner pass failed; stopping before another GPU launch" >&2
        exit 1
    fi
    run_gate
    if (( iteration % META_EVERY == 0 )); then
        run_meta
    fi
done

echo "[OUTER] completed $MAX_ITERATIONS pass(es) with compatibility postflight"
