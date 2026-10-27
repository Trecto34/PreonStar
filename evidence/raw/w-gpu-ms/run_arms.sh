#!/bin/bash
# Contamination attribution: which part of the instrumentation costs t/s?
#   off   - no prof env at all
#   host  - P36_VK_PROF=1 (host-side clocks only, no timestamp queries)
#   wg    - P36_VK_PROF_WG_HIST=1 (host-side bucket counters only)
#   ts    - P36_VK_PROF_KERNEL=1 (timestamp queries, per-kernel-type brackets)
#   skip  - timestamps, first 2600 dispatches (the prefill burst) untimed,
#           so the profile describes a pure decode window
set -u
cd /home/server/q36
M=${1:?model}; TAG=${2:?tag}; OUT=evidence/raw/w-gpu-ms

run() {
  local label=$1 envs=$2; shift 2
  echo "=== START $label $(date +%H:%M:%S) ==="
  env $envs ./p36-bench -m "$M" --vulkan \
      --prompt-file "$OUT/prompt.txt" --ctx-start 2048 --ctx-max 2048 \
      -n 192 --csv "$OUT/${TAG}_${label}.csv" "$@" \
      > "$OUT/${TAG}_${label}.log" 2>&1
  echo "=== END $label rc=$? ==="
  tail -1 "$OUT/${TAG}_${label}.csv"
}

for rep in 1 2; do
  run "a_off_${rep}"  ""
  run "b_host_${rep}" "P36_VK_PROF=1"
  run "c_wg_${rep}"   "P36_VK_PROF_WG_HIST=1"
  run "d_ts_${rep}"   "P36_VK_PROF_KERNEL=1"
done
run "e_skipdec" "P36_VK_PROF_KERNEL=1 P36_VK_PROF_WG_HIST=1 P36_VK_PROF_SKIP=2600"
echo DONE
