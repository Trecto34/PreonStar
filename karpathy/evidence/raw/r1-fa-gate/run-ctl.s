#!/usr/bin/env bash
# R1 controls: is the FA-on/FA-off greedy token difference caused by FA, or is
# the 64-token greedy text simply not reproducible run to run?  Also: does an
# already-accepted configuration change (prefill chunk 256 -> 128) change the
# tokens too?  Three arms each run twice, same binary, same seed, temp 0.
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
OUT=/tmp/r1/ctl
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

run () { # run NAME MODEL ENVP extra...
    local name="$1" model="$2" envp="$3"; shift 3
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$model" \
          -p "Write a short story about a lazy duck who avoids work." \
          -n 64 -c 1400 --seed 42 --temp 0 "$@" < /dev/null ) > "$OUT/$name.txt" 2>&1
    echo "--- $name rc=$?"
}

for i in 1 2; do
    run swift-fa1-r$i "$SWIFT" ""
    run swift-fa0-r$i "$SWIFT" "Q36_VK_ATTN_FA=0"
    run swift-c128-r$i "$SWIFT" "" --prefill-chunk 128
    run guard-fa1-r$i "$GUARD" ""
    run guard-fa0-r$i "$GUARD" "Q36_VK_ATTN_FA=0"
done
echo "### done"
