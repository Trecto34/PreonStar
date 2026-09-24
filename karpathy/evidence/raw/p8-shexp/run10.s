#!/usr/bin/env bash
# Sweep 10: P8 premise probe -- how much of IQ2_M decode is the shared-expert
# path?  The shexp gate/up/down are IQ2_S (unlike the guard's Q5_K/Q6_K), so
# they miss the Q8_0 fast path and run as separate plain matvecs.  Per-shape
# profile names the shapes, so the 2048x512 (gate/up) and 512x2048 (down)
# entries can be summed directly.
set -uo pipefail
cd /home/server/q36
MOE=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out10
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () {
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$MOE" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

# decode-only profile, per shape, 128 generated tokens
PROF="Q36_VK_PROF=1 Q36_VK_PROF_SHAPE=1 Q36_VK_PROF_KERNEL=1"
bench p8-shape "$PROF" --ctx-start 512 --ctx-max 512 --ctx-alloc 1100 \
      --prefill-chunk 256 --gen-tokens 128
echo "### done"
grep -h "^q36:   op " "$OUT/p8-shape.log" | sort -t= -k5 -rn | head -30
