#!/usr/bin/env bash
# Sweep 9: P6 gate run.  7 interleaved reps of MTP draft=3 with the nx
# two-row kernel OFF (shipped MMQ tile) vs ON, gen 256, ctx 512.
# Q36_MTP_STATS=1 so every rep also yields (calls, drafted, accepted, accept%),
# which is what the accept-count drift between the two arms has to be judged on.
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/p36/out9
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && { echo "q36-bench already running, abort"; exit 1; }

bench () { # bench NAME ENV
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

COMMON="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 256 --mtp-margin 0 --mtp-draft 3"
NXOFF="Q36_VK_DENSE_IQ3_NX=0 Q36_MTP_STATS=1"
NXON="Q36_MTP_STATS=1"

echo "### P6 gate: 7 reps, d3 nx OFF vs ON"
for rep in 1 2 3 4 5 6 7; do
    bench r$rep-d3off "$NXOFF" $COMMON
    bench r$rep-d3on  "$NXON"  $COMMON
done
echo "### done"
grep -h "gen_tps\|MTP stats" "$OUT"/*.log 2>/dev/null | tail -40
