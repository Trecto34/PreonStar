#!/usr/bin/env bash
set -uo pipefail
cd /home/server/q36
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
OUT=/tmp/p36/out11
mkdir -p "$OUT"
( flock -w 7200 /tmp/q36-gpu.lock env Q36_VK_PROF=1 Q36_VK_PROF_KERNEL=1 \
    ./q36-bench --vulkan -m "$GUARD" --prompt-file tests/long_context_story_prompt.txt \
    --ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 32 ) \
    > "$OUT/guard-prof.log" 2>&1
echo "--- guard rc=$?"
echo "guard nx dispatches: $(grep -c 'dense_iq3_xxs_decode_nx' "$OUT/guard-prof.log")"
grep -h "^512" "$OUT/guard-prof.log"
