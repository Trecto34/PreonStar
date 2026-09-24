#!/usr/bin/env bash
# R5: decode attention at long context.  One fresh process per (model, ctx),
# decode-only attribution by dispatch count (attn_decode_split runs once per
# full-attention layer per decode token, so its per-token gpu_ms is exact even
# though the profile also contains prefill rows).
#
#   stage A: Swift and guard at ctx 8K/16K/32K
#   stage B: span sweep (Q36_VK_ATTN_SPAN) at ctx 16K
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/r5/out
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
STAGE="${1:-A}"
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () { # bench NAME ENVP MODEL -- args...
    local name="$1"; shift
    local envp="$1"; shift
    local model="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$model" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

PROF="Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1"
RUN () { # RUN CTX GEN
    echo "$1" "$2"
}

if [ "$STAGE" = A ]; then
    for rep in 1 2; do
        for ctx in 8192 16384; do
            bench swift-$ctx-r$rep "$PROF" "$SWIFT" \
                --ctx-start $ctx --ctx-max $ctx --ctx-alloc $((ctx + 512)) \
                --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
            bench guard-$ctx-r$rep "$PROF" "$GUARD" \
                --ctx-start $ctx --ctx-max $ctx --ctx-alloc $((ctx + 512)) \
                --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
        done
    done
    for rep in 1 2 3; do
        bench swift-32768-r$rep "$PROF" "$SWIFT" \
            --ctx-start 32768 --ctx-max 32768 --ctx-alloc 33280 \
            --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
        bench guard-32768-r$rep "$PROF" "$GUARD" \
            --ctx-start 32768 --ctx-max 32768 --ctx-alloc 33280 \
            --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
    done
else
    for span in 512 1024 2048 4096; do
        for rep in 1 2; do
            bench swift-span$span-r$rep "$PROF Q36_VK_ATTN_SPAN=$span" "$SWIFT" \
                --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16896 \
                --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
            bench guard-span$span-r$rep "$PROF Q36_VK_ATTN_SPAN=$span" "$GUARD" \
                --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16896 \
                --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0
        done
    done
fi
echo "### stage $STAGE done"
grep -h "^ctx_tokens\|attn_decode_split\|attn_combine" "$OUT"/*.log | tail -40
