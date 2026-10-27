#!/bin/bash
# Task 1: interleaved timestamp ON/OFF A/B + prefill-only split run.
# ctx 2048, 192 decode tokens, 3 interleaved pairs.
set -u
cd /home/server/q36
M=${1:?model path}
TAG=${2:?tag}
OUT=evidence/raw/w-gpu-ms
mkdir -p "$OUT"

run() { # $1=label $2=extra env $3...=bench args
  local label=$1; shift
  local envs=$1; shift
  echo "=== START $label $(date +%H:%M:%S) ==="
  env $envs ./p36-bench -m "$M" --vulkan \
      --prompt-file "$OUT/prompt.txt" --ctx-start 2048 --ctx-max 2048 \
      -n 192 --csv "$OUT/${TAG}_${label}.csv" "$@" \
      > "$OUT/${TAG}_${label}.log" 2>&1
  echo "=== END $label rc=$? $(date +%H:%M:%S) ==="
  tail -1 "$OUT/${TAG}_${label}.csv"
}

for rep in 1 2 3; do
  run "off_${rep}" "" 
  run "on_${rep}"  "P36_VK_PROF_KERNEL=1 P36_VK_PROF_WG_HIST=1"
done
# prefill-only reference: no decode, so op rows are pure prefill and the
# 192-token rows minus these give decode GPU ms.
run "prefill" "P36_VK_PROF_KERNEL=1 P36_VK_PROF_WG_HIST=1" -n 0
echo DONE
