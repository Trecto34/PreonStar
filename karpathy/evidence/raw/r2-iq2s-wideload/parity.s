#!/usr/bin/env bash
# R2 parity: the IQ2_S down sum-decode rewrite must be bit-identical to the
# shipped blob.  Same binary, same model, same prompt; only
# vulkan/moe_down_q2k_sum_decode_iq2s.spv is swapped between the two arms.
set -uo pipefail
cd /home/server/q36
MOE=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
SPV=vulkan/moe_down_q2k_sum_decode_iq2s.spv
OUT=/tmp/r2/parity
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

arm () { # arm NAME MODEL BLOB
    local name="$1" model="$2" blob="$3"
    cp -f "$blob" "$SPV"
    mkdir -p "$OUT/$name"
    flock -w 7200 "$L" ./q36-bench --vulkan -m "$model" \
        --prompt-file tests/long_context_story_prompt.txt \
        --ctx-start 512 --ctx-max 520 --ctx-alloc 1100 --prefill-chunk 256 \
        --gen-tokens 6 --dump-frontier-logits-dir "$OUT/$name" \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$? blob=$(md5sum "$SPV" | cut -c1-8)"
}

for i in 1 2; do
    arm moe-old-r$i "$MOE" /tmp/r2/old.spv
    arm moe-new-r$i "$MOE" /tmp/r2/new.spv
    arm guard-old-r$i "$GUARD" /tmp/r2/old.spv
    arm guard-new-r$i "$GUARD" /tmp/r2/new.spv
done
cp -f /tmp/r2/new.spv "$SPV"
echo "### done"
