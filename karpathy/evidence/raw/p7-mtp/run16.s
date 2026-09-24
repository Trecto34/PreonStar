#!/usr/bin/env bash
# Sweep 16: P7.  The nx small-batch IQ3_XXS kernel widened from n_tok == 2 to
# 2..8, so the --mtp-draft 3 verify (3 rows) stops taking the 128-row MMQ tile.
#   - inertness: par-d2 rerun must reproduce P6's par-d2 byte for byte (the
#     n_tok == 2 path is untouched: the tail guard's `tn` is 2 for every chunk)
#   - attribution: d3 kernel profile, nx ON vs OFF, gen 16
#   - the gate: 7 interleaved reps, d3 nx OFF vs ON, gen 256, ctx 512
#   - parity: MTP off vs d2 vs d3 nx ON/OFF, 64 greedy tokens, seed 42
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out16
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () { # bench NAME ENVP -- args...
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

cli () { # cli NAME ENVP extra...
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$SWIFT" \
          -p "Write a short story about a lazy duck who avoids work." \
          -n 64 -c 1400 --seed 42 --temp 0 "$@" < /dev/null ) \
        > "$OUT/$name.txt" 2>&1
    echo "--- $name rc=$?"
}

COMMON="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 256 --mtp-margin 0"
STATS="Q36_MTP_STATS=1"
NXOFF="Q36_VK_DENSE_IQ3_NX=0 $STATS"
NXON="$STATS"
PROF="Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1"
P16="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 16 --mtp-margin 0"

echo "### P7 attribution: d3 profile, nx OFF vs ON, gen 16"
bench prof-d3off "$PROF Q36_VK_DENSE_IQ3_NX=0" $P16 --mtp-draft 3
bench prof-d3on  "$PROF"                       $P16 --mtp-draft 3

echo "### P7 parity: MTP off vs d2 vs d3 nx ON/OFF, 64 greedy tokens"
cli par-mtp-off "Q36_MTP_SPEC_DISABLE=1"
cli par-d2      ""
cli par-d3nxoff "Q36_VK_DENSE_IQ3_NX=0" --mtp-draft 3
cli par-d3nxon  ""                      --mtp-draft 3

echo "### P7 gate: 7 interleaved reps, d3 nx OFF vs ON, gen 256"
for rep in 1 2 3 4 5 6 7; do
    bench r$rep-d3off "$NXOFF" $COMMON --mtp-draft 3
    bench r$rep-d3on  "$NXON"  $COMMON --mtp-draft 3
done
echo "### done"
grep -h "gen_tps\|MTP stats" "$OUT"/r*.log 2>/dev/null | tail -40
