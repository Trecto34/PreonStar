#!/usr/bin/env bash
# R2: kernel-level profile + interleaved e2e A/B, old vs new IQ2_S sum-decode blob.
# Decode-only kernel (identity = n_tok == 1), so the e2e gate is the decode median.
set -uo pipefail
cd /home/server/q36
MOE=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
OUT=/tmp/r2/ab
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

prof () { # prof NAME BLOB
    cp -f "$2" vulkan/moe_down_q2k_sum_decode_iq2s.spv
    Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1 \
    flock -w 7200 "$L" ./q36-bench --vulkan -m "$MOE" \
        --prompt-file tests/long_context_story_prompt.txt \
        --ctx-start 512 --ctx-max 512 --ctx-alloc 900 --prefill-chunk 256 \
        --gen-tokens 128 > "$OUT/prof-$1.log" 2>&1
    echo "--- prof $1 rc=$?"
}

prof old /tmp/r2/old.spv
prof new /tmp/r2/new.spv

Q36_AB_REPS=7 Q36_AB_CTX=1024 Q36_AB_GEN=128 Q36_AB_CHUNK=256 \
    tests/bench_ab.sh /tmp/r2/arm-old.sh /tmp/r2/arm-new.sh "$MOE" \
    > "$OUT/ab-iq2m.csv" 2> "$OUT/ab-iq2m.summary"
echo "--- ab rc=$?"
cp -f /tmp/r2/new.spv vulkan/moe_down_q2k_sum_decode_iq2s.spv
echo "### done"
