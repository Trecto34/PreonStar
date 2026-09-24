#!/usr/bin/env bash
# Sweep 4 (fixed arg order; run2.s passed a stray positional and --temperature).
#   A. P3  prefill f16-accumulation: chunk 256 vs decode path (chunk 1) frontier dumps
#   B. P3b DeltaNet recurrent state f16 vs f32, ctx 2048..8192 and 16384..32768
#   C. P2b exact MTP acceptance per depth (Q36_MTP_STATS)
#   D. P6 per-kernel profile of a 1-, 2- and 3-token step
#   E. P8 IQ2_M per-kernel decode profile
#   F. P9 sampler wall-vs-GPU gap (greedy vs default min-p sampler)
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
MOE=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out3
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

t0=$(date +%s)
bench () { # bench NAME "ENV ..." [@ MODEL] extra...
    local name="$1"; shift
    local envp="$1"; shift
    local model="$SWIFT"
    if [ "${1:-}" = "@" ]; then shift; model="$1"; shift; fi
    local st=$(date +%s)
    ( export Q36_BENCH_SPEC_TRACE=1; [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$model" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$? $(( $(date +%s) - st ))s"
}

cli () { # cli NAME "ENV ..." extra...
    local name="$1"; shift
    local envp="$1"; shift
    local st=$(date +%s)
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" < /dev/null ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$? $(( $(date +%s) - st ))s"
}

echo "### tokens for frontier_nll.py"
( flock -w 7200 "$L" ./q36 --vulkan -m "$SWIFT" --prompt-file "$STORY" --dump-tokens ) \
    > "$OUT/tokens.json" 2>"$OUT/tokens.err"
echo "--- tokens rc=$? lines=$(wc -l < "$OUT/tokens.json")"

echo "### A. P3 f16 prefill accumulation: chunk 256 vs chunk 1 (decode path), ctx 512..2048"
rm -rf /tmp/p36/p3h-a /tmp/p36/p3h-b; mkdir -p /tmp/p36/p3h-a /tmp/p36/p3h-b
PA="--ctx-start 512 --ctx-max 2048 --step-incr 64 --ctx-alloc 2176 --gen-tokens 0"
bench p3h-chunk256 "" $PA --prefill-chunk 256 --dump-frontier-logits-dir /tmp/p36/p3h-a
bench p3h-chunk1   "" $PA --prefill-chunk 1   --dump-frontier-logits-dir /tmp/p36/p3h-b

echo "### B. P3b recurrent state f16 vs f32, ctx 2048..8192"
rm -rf /tmp/p36/p3h-c /tmp/p36/p3h-d; mkdir -p /tmp/p36/p3h-c /tmp/p36/p3h-d
S3="--ctx-start 2048 --ctx-max 8192 --step-incr 2048 --ctx-alloc 8320 --prefill-chunk 256 --gen-tokens 0"
bench p3h-state-f16 ""                             $S3 --dump-frontier-logits-dir /tmp/p36/p3h-c
bench p3h-state-f32 "Q36_VK_RECURRENT_STATE_F16=0" $S3 --dump-frontier-logits-dir /tmp/p36/p3h-d

echo "### B2. P3b at ctx 16384..32768"
rm -rf /tmp/p36/p3h-e /tmp/p36/p3h-f; mkdir -p /tmp/p36/p3h-e /tmp/p36/p3h-f
S4="--ctx-start 16384 --ctx-max 32768 --step-incr 16384 --ctx-alloc 33024 --prefill-chunk 256 --gen-tokens 0"
bench p3h-state32-f16 ""                             $S4 --dump-frontier-logits-dir /tmp/p36/p3h-e
bench p3h-state32-f32 "Q36_VK_RECURRENT_STATE_F16=0" $S4 --dump-frontier-logits-dir /tmp/p36/p3h-f

echo "### C. P2b exact acceptance per depth (margin 0 = confidence gate off)"
P2="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 256 --mtp-margin 0"
bench p2b-stats-d2 "Q36_MTP_STATS=1" $P2 --mtp-draft 2
bench p2b-stats-d3 "Q36_MTP_STATS=1" $P2 --mtp-draft 3
bench p2b-stats-d4 "Q36_MTP_STATS=1" $P2 --mtp-draft 4

echo "### D. P6 per-kernel profile of a 1-, 2- and 3-token step"
PROF="Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1"
P6="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 16 --mtp-margin 0"
bench p6-prof-off "$PROF Q36_MTP_SPEC_DISABLE=1" $P6 --mtp-draft 2
bench p6-prof-d2  "$PROF"                       $P6 --mtp-draft 2
bench p6-prof-d3  "$PROF"                       $P6 --mtp-draft 3

echo "### E. P8 IQ2_M per-kernel decode profile"
bench p8-iq2m-prof "$PROF Q36_MTP_SPEC_DISABLE=1" @ "$MOE" \
    --ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 16

echo "### F. P9 sampler wall-vs-GPU gap"
cli p9-greedy  "Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1" -n 96 -c 700 --temp 0
cli p9-sampled "Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1" -n 96 -c 700 --temp 1

echo "### DONE total $(( $(date +%s) - t0 ))s"
