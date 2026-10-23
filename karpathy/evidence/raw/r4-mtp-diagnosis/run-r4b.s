#!/usr/bin/env bash
# R4 attribution: which kernels the 2-row verify pass actually takes, decode-only
# (P36_VK_PROF_DECODE=1 resets the profiler after prefill), MTP d3 nx ON vs OFF,
# plus a plain-decode control on the same binary.
set -uo pipefail
cd /home/server/p36
SWIFT=/home/server/p36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/r4/out
mkdir -p "$OUT"
L=/tmp/p36-gpu.lock
pgrep -x p36-bench >/dev/null && { echo "p36-bench running, abort"; exit 1; }

bench () {
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./p36-bench --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" ) \
        > "$OUT/$name.log" 2>&1
    echo "--- $name rc=$?"
}

P16="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 16 --mtp-margin 0"
PROF="P36_VK_PROF_OP=1 P36_VK_PROF_KERNEL=1 P36_VK_PROF_DECODE=1"

bench prof-mtp3      "P36_MTP_TIMING=1 $PROF"                $P16 --mtp-draft 3
bench prof-mtp3-nxoff "P36_MTP_TIMING=1 $PROF P36_VK_DENSE_IQ3_NX=0" $P16 --mtp-draft 3
bench prof-mtp2      "P36_MTP_TIMING=1 $PROF"                $P16 --mtp-draft 2
bench prof-plain     "$PROF P36_MTP_SPEC_DISABLE=1"          $P16
bench mtp3-32-nxoff  "P36_MTP_TIMING=1 P36_MTP_STATS=1 P36_VK_DENSE_IQ3_NX=0" \
      --ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 32 --mtp-margin 0 --mtp-draft 3
echo "### done"
grep -h "^512," "$OUT"/prof-*.log "$OUT"/mtp3-32-nxoff.log
