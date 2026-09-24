#!/usr/bin/env bash
# Sweep 15: P9a.  First the bit-identity gate for the GPU min-p prefilter --
# the same seed and prompt sampled with the pack ON and OFF must produce the
# same bytes -- then a profiled smoke so the pack dispatches are visible, then
# the decode A/B (pack ON vs OFF) on both models, 7 interleaved reps, ctx 1400
# Swift / 1024 MoE, --temp 0.8 --min-p 0.05 so the sampler is actually used.
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
MOE=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
OUT=/tmp/p36/out15
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36 >/dev/null && { echo "q36 running, abort"; exit 1; }
PROMPT="Explain in detail how a rope bridge is built, step by step."

run () { # run NAME ENVP MODEL N C extra...
    local name="$1"; shift
    local envp="$1"; shift
    local model="$1"; shift
    local n="$1"; shift
    local c="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36 --vulkan -m "$model" -p "$PROMPT" \
        -n "$n" -c "$c" --seed 42 "$@" < /dev/null ) > "$OUT/$name.txt" 2>&1
    echo "--- $name rc=$?"
}

echo "### parity: pack ON vs OFF, sampled, seed 42"
run par-swift-on  ""                       "$SWIFT" 64 1400 --temp 0.8 --min-p 0.05
run par-swift-off "Q36_VK_MINP_PACK=0"     "$SWIFT" 64 1400 --temp 0.8 --min-p 0.05
run par-moe-on   ""                        "$MOE"   64 1024 --temp 0.8 --min-p 0.05
run par-moe-off  "Q36_VK_MINP_PACK=0"      "$MOE"   64 1024 --temp 0.8 --min-p 0.05
run par-moe-on-t1  ""                      "$MOE"   64 1024 --temp 1.0 --min-p 0.05
run par-moe-off-t1 "Q36_VK_MINP_PACK=0"    "$MOE"   64 1024 --temp 1.0 --min-p 0.05
run par-swift-on-mp2 ""                    "$SWIFT" 64 1400 --temp 0.8 --min-p 0.2
run par-swift-off-mp2 "Q36_VK_MINP_PACK=0" "$SWIFT" 64 1400 --temp 0.8 --min-p 0.2

echo "### profiled smoke"
run prof-on "Q36_VK_PROF=1" "$SWIFT" 16 512 --temp 0.8 --min-p 0.05

echo "### A/B: pack ON vs OFF, sampled, 7 interleaved reps"
for rep in 1 2 3 4 5 6 7; do
    run r$rep-swift-off "Q36_VK_MINP_PACK=0" "$SWIFT" 256 1400 --temp 0.8 --min-p 0.05
    run r$rep-swift-on  ""                   "$SWIFT" 256 1400 --temp 0.8 --min-p 0.05
    run r$rep-moe-off   "Q36_VK_MINP_PACK=0" "$MOE"   128 1024 --temp 0.8 --min-p 0.05
    run r$rep-moe-on    ""                   "$MOE"   128 1024 --temp 0.8 --min-p 0.05
done
echo "### done"
grep -h "generation:" "$OUT"/r*.txt
