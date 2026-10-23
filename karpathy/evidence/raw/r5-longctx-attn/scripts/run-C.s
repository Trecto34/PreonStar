#!/usr/bin/env bash
# R5 stage C: close the span curve from the narrow side (256, 128) at ctx 16384.
# The stock span comment claims a 128-key span buys occupancy but pays in
# rescaled combine drift and combine cost, measured only at ctx 2048.
set -uo pipefail
cd /home/server/p36
SWIFT=/home/server/p36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
GUARD=/home/server/p36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-P36-IQ2XXS.gguf
OUT=/tmp/r5/outB
L=/tmp/p36-gpu.lock
pgrep -x p36-bench >/dev/null && { echo "p36-bench running, abort"; exit 1; }
bench () {
    local name="$1"; shift; local envp="$1"; shift; local model="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./p36-bench --vulkan -m "$model" --prompt-file tests/long_context_story_prompt.txt "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}
for span in 256 128; do
    for m in swift guard; do
        if [ "$m" = swift ]; then MOD="$SWIFT"; else MOD="$GUARD"; fi
        bench $m-span$span "P36_VK_PROF_OP=1 P36_VK_PROF_KERNEL=1 P36_VK_ATTN_SPAN=$span" "$MOD" \
            --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16896 \
            --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
    done
done
echo "### stage C done"
