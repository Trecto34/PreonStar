#!/usr/bin/env bash
# Initial prompt size (system prompt + tool schemas + one-word user turn) as the
# agent tokenizes it.  Usage: MODEL=gguf/x.gguf scripts/agent_prompt_tokens.sh [agent-binary]
set -e
BIN=${1:-./q36-agent}; T=$(mktemp); D=$(mktemp -d)
timeout 300 "$BIN" --vulkan -m "$MODEL" --ctx 4096 --nothink -n 1 --chdir "$D" \
    --non-interactive --trace "$T" -p "hi" >/dev/null 2>&1 || true
grep -m1 "prefill tool_round=0" "$T" | sed 's/.*transcript=/initial_prompt_tokens=/'
rm -rf "$T" "$D"
