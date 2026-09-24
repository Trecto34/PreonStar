#!/usr/bin/env bash
# R1: the vulkan-kernels n_tok-invariance gate went red at 9d54dc7 (FA prefill
# for GQA 8).  Bisect pinned it; the FA batch rows now differ from the per-row
# decode step by ~1e-8.  This sweep collects the e2e evidence that the drift is
# inert: greedy token identity and frontier-logits agreement with FA on vs off,
# on both FA shapes (guard ratio 8, Swift ratio 6), then the compat gate.
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
PROMPT=tests/long_context_story_prompt.txt
OUT=/tmp/r1/out
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

cli () { # cli NAME MODEL ENVP
    local name="$1" model="$2" envp="$3"
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$model" \
          -p "Write a short story about a lazy duck who avoids work." \
          -n 64 -c 1400 --seed 42 --temp 0 < /dev/null ) > "$OUT/$name.txt" 2>&1
    echo "--- cli $name rc=$?"
}

dump () { # dump NAME MODEL ENVP
    local name="$1" model="$2" envp="$3"
    mkdir -p "$OUT/$name"
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$model" --prompt-file "$PROMPT" \
          --ctx-start 512 --ctx-max 520 --ctx-alloc 1100 --prefill-chunk 256 \
          --gen-tokens 6 --dump-frontier-logits-dir "$OUT/$name" ) > "$OUT/$name.log" 2>&1
    echo "--- dump $name rc=$?"
}

echo "### greedy token parity, FA on vs off (production kv q8_0/q4_0)"
cli guard-fa1 "$GUARD" ""
cli guard-fa0 "$GUARD" "Q36_VK_ATTN_FA=0"
cli swift-fa1 "$SWIFT" ""
cli swift-fa0 "$SWIFT" "Q36_VK_ATTN_FA=0"

echo "### frontier logits, FA on vs off"
dump guard-fa1d "$GUARD" ""
dump guard-fa0d "$GUARD" "Q36_VK_ATTN_FA=0"
dump swift-fa1d "$SWIFT" ""
dump swift-fa0d "$SWIFT" "Q36_VK_ATTN_FA=0"

echo "### compat gate"
./karpathy/compat_gate.sh
echo "### done"
