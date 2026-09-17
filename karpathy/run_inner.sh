#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KARPATHY_CORE="${KARPATHY_CORE_DIR:-/home/server/Karpathy}"
STORE_DIR="${KARPATHY_STORE_DIR:-${ROOT_DIR}/.karpathy/experiments}"

if [[ "${1:-}" != "--worker" ]]; then
    exec python3 "$KARPATHY_CORE/experiments.py" --store "$STORE_DIR" run \
        --target "$ROOT_DIR" -- \
        bash "$ROOT_DIR/karpathy/run_inner.sh" --worker
fi

: "${EXPERIMENT_WORKTREE:?Run through experiments.py}"
: "${EXPERIMENT_RESULT:?Missing experiment result path}"

for process_name in q36 q36-bench q36_test; do
    if pgrep -x "$process_name" >/dev/null 2>&1; then
        echo "[INNER] GPU is occupied by $process_name; refusing to overlap it." >&2
        exit 1
    fi
done

WORKTREE="$EXPERIMENT_WORKTREE"
mkdir -p -- "$WORKTREE/karpathy/evidence"

PROMPT=$(cat <<EOF
You are the inner optimization worker for the Swift Qwen3.8-27B IQ3_XXS target.
Work only in: $WORKTREE

Read these files first:
- $WORKTREE/karpathy/Task.md
- $WORKTREE/karpathy/AlreadyTried.md
- $WORKTREE/karpathy/Program.md
- $WORKTREE/karpathy/ExperimentProtocol.md

Perform exactly one hypothesis-driven iteration. Edit source in the worktree,
build with 'make vulkan-generic -j2', and use a focused benchmark when useful.
Do not commit, revert, or edit shared files. Before deciding, run:
  ./karpathy/compat_gate.sh
It independently checks the Swift IQ3_XXS model and the local
Qwen3.5/Qwen3.6 qwen35moe model in serialized GPU processes. A failure is a
hard rejection. Write the JSON report required by ExperimentProtocol.md to
$EXPERIMENT_RESULT. An accepted report must contain measured positive
throughput, all required protocol gates set true, and evidence paths relative
to the worktree. Print DECISION: ACCEPTED or DECISION: REJECTED.
EOF
)

set +e
if [[ -n "${KARPATHY_INNER_COMMAND:-}" ]]; then
    timeout --kill-after=20s "${KARPATHY_AGENT_TIMEOUT:-1500}s" \
        env KARPATHY_PROMPT="$PROMPT" bash -lc "${KARPATHY_INNER_COMMAND} \"\$KARPATHY_PROMPT\""
elif command -v opencode >/dev/null 2>&1; then
    timeout --kill-after=20s "${KARPATHY_AGENT_TIMEOUT:-1500}s" \
        opencode run --model "${KARPATHY_INNER_MODEL:-opencode/union-alpha}" --auto "$PROMPT"
elif command -v claude >/dev/null 2>&1; then
    timeout --kill-after=20s "${KARPATHY_AGENT_TIMEOUT:-1500}s" \
        claude --model "${KARPATHY_INNER_MODEL:-sonnet}" --dangerously-skip-permissions -p "$PROMPT"
else
    echo "[INNER] no agent CLI found (set KARPATHY_INNER_COMMAND)" >&2
    exit 1
fi
agent_status=$?
set -e

# The independent gate is mandatory even if the agent forgot to invoke it.
set +e
KARPATHY_EVIDENCE_DIR="$WORKTREE/karpathy/evidence" \
    "$WORKTREE/karpathy/compat_gate.sh"
gate_status=$?
set -e

python3 "$WORKTREE/karpathy/finalize_report.py" "$EXPERIMENT_RESULT" \
    "$gate_status" "karpathy/evidence/compatibility-gate.log"

if [[ "$agent_status" -ne 0 ]]; then
    echo "[INNER] worker exited with $agent_status; report retained for diagnosis." >&2
    exit "$agent_status"
fi
if [[ "$gate_status" -ne 0 ]]; then
    echo "[INNER] compatibility gate failed; candidate cannot be accepted." >&2
    exit 0
fi
