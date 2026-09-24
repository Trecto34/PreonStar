#!/usr/bin/env bash
# Sweep 5: P3 follow-ups.  Sweep 4 (run4.s) showed mean|dlogit| of 1.5-2.0 nats
# between the state f16/f32 arms and up to 16 nats between chunk 256/chunk 1, so
# before reading any NLL out of those, three things have to be separated:
#   1. determinism control   - chunk 256 twice, identical settings (dir g)
#   2. the audit's >=2048-token window for the f16 accumulator (dirs i/j)
#   3. many paired samples for the recurrent state at long ctx   (dirs k/l)
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out3
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () {
    local name="$1"; shift
    local envp="$1"; shift
    local model="$SWIFT"
    if [ "${1:-}" = "@" ]; then shift; model="$1"; shift; fi
    local st=$(date +%s)
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$model" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$? $(( $(date +%s) - st ))s"
}

echo "### 1. determinism control: chunk 256 twice, same settings"
rm -rf /tmp/p36/p3h-g; mkdir -p /tmp/p36/p3h-g
PA="--ctx-start 512 --ctx-max 2048 --step-incr 64 --ctx-alloc 2176 --gen-tokens 0"
bench p3h-chunk256-bis "" $PA --prefill-chunk 256 --dump-frontier-logits-dir /tmp/p36/p3h-g

echo "### 2. P3 f16 accumulator, ctx 2048..4096 every 64 tokens"
rm -rf /tmp/p36/p3h-i /tmp/p36/p3h-j; mkdir -p /tmp/p36/p3h-i /tmp/p36/p3h-j
PB="--ctx-start 2048 --ctx-max 4096 --step-incr 64 --ctx-alloc 4224 --gen-tokens 0"
bench p3h-chunk256-b "" $PB --prefill-chunk 256 --dump-frontier-logits-dir /tmp/p36/p3h-i
bench p3h-chunk1-b   "" $PB --prefill-chunk 1   --dump-frontier-logits-dir /tmp/p36/p3h-j

echo "### 3. P3b recurrent state f16 vs f32, ctx 8192..16384 every 256 tokens"
rm -rf /tmp/p36/p3h-k /tmp/p36/p3h-l; mkdir -p /tmp/p36/p3h-k /tmp/p36/p3h-l
PC="--ctx-start 8192 --ctx-max 16384 --step-incr 256 --ctx-alloc 16512 --prefill-chunk 256 --gen-tokens 0"
bench p3h-state2-f16 ""                             $PC --dump-frontier-logits-dir /tmp/p36/p3h-k
bench p3h-state2-f32 "Q36_VK_RECURRENT_STATE_F16=0" $PC --dump-frontier-logits-dir /tmp/p36/p3h-l

echo "### 4. P9 sampler gap: same model, greedy vs the default min-p sampler, short prompt"
# The story file is ~34k tokens and cannot fit --ctx 700, which is why the run4
# p9 arms exited 1 with dispatches=0 before generating.
cli () { # cli NAME "ENV ..." extra...
    local name="$1"; shift
    local envp="$1"; shift
    local st=$(date +%s)
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$SWIFT" -p "Write a long detailed story about a lazy duck who avoids work." "$@" < /dev/null ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$? $(( $(date +%s) - st ))s"
}
P9="Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1 -n 128 -c 1400 --seed 42"
cli p9b-greedy  "$P9" --temp 0
cli p9b-sampled "$P9" --temp 1

echo "### DONE"
