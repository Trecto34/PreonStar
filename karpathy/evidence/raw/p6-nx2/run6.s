#!/usr/bin/env bash
# Sweep 6: P6 small-batch (n_tok == 2) dense IQ3_XXS decode kernel.
#   - the gate: a 2-token step costs <= 1.3x a 1-token step
#   - arms: plain decode | MTP d2 (verify row is 1 token, nx inert)
#           MTP d3 nx OFF (the shipped path: dense_iq3_xxs_mmq tile)
#           MTP d3 nx ON
#   - 3 interleaved reps per arm, gen 64, ctx 512
#   - profile arm to confirm the nx label and per-kernel cost
#   - greedy parity: MTP off vs d3 nx ON must be byte-identical
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out6
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () { # bench NAME ENV -- args...
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

COMMON="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 64 --mtp-margin 0"
PLAIN="Q36_MTP_SPEC_DISABLE=1"
NXOFF="Q36_VK_DENSE_IQ3_NX=0"

echo "### P6 A/B: 3 interleaved reps"
for rep in 1 2 3; do
    bench r$rep-plain "$PLAIN"            $COMMON --mtp-draft 2
    bench r$rep-d2    ""                  $COMMON --mtp-draft 2
    bench r$rep-d3off "$NXOFF"            $COMMON --mtp-draft 3
    bench r$rep-d3on  ""                  $COMMON --mtp-draft 3
done

echo "### P6 profile: d3 nx OFF vs ON (gen 16)"
PROF="Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1"
P16="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 16 --mtp-margin 0"
bench prof-d3off "$PROF $NXOFF" $P16 --mtp-draft 3
bench prof-d3on  "$PROF"        $P16 --mtp-draft 3

echo "### P6 greedy parity: MTP off vs d2 vs d3 nx ON, 64 greedy tokens"
cli () { # cli NAME ENV extra...
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$SWIFT" \
          -p "Write a short story about a lazy duck who avoids work." \
          -n 64 -c 1400 --seed 42 --temp 0 "$@" < /dev/null ) \
        > "$OUT/$name.txt" 2>&1
    echo "--- $name rc=$?"
}
cli par-mtp-off "$PLAIN"
cli par-d2      ""
cli par-d3nxoff "$NXOFF" --mtp-draft 3
cli par-d3nxon  ""       --mtp-draft 3

echo "### DONE"
