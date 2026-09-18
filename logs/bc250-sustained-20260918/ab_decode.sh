#!/usr/bin/env bash
# Interleaved, soak-gated A/B for a single Vulkan shader. Swaps only the .spv,
# so both arms run the same host binary and any difference is the kernel.
#
#   logs/bc250-sustained-20260918/ab_decode.sh PAIRS OUT.csv A.spv B.spv [TARGET_SPV]
#
# Defaults to the Q2_0 decode kernel. Env: SOAK_TEMP (default 55), MODEL.
# Run from the repo root, with the GPU idle. Records entry/exit edge temp, live
# SCLK, MCLK and package power alongside each measurement, because this box
# throttles hard enough that an unpaired comparison is meaningless.
set -uo pipefail

PAIRS="${1:?usage: ab_decode.sh PAIRS OUT.csv A.spv B.spv [TARGET_SPV]}"
OUT="${2:?missing OUT.csv}"
A_SPV="${3:?missing A.spv}"
B_SPV="${4:?missing B.spv}"
SPV="${5:-vulkan/dense_extra_decode_q2_0.spv}"
MODEL="${MODEL:-gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf}"
TGT="${SOAK_TEMP:-55}"

HW=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
[ -z "$HW" ] && { echo "no hwmon temp node" >&2; exit 2; }
HWD=$(dirname "$HW")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

soak () { while :; do t=$(( $(cat "$HW") / 1000 )); [ "$t" -le "$TGT" ] && break; sleep 2; done; }
sclk () { echo $(( $(cat "$HWD/freq1_input") / 1000000 )); }          # live, in MHz
mclk () { awk '/\*/{gsub(/[^0-9]/,"",$2); print $2; exit}' \
              /sys/class/drm/card0/device/pp_dpm_mclk; }              # DPM display only

echo "pair,arm,entry_temp_c,exit_temp_c,sclk_mhz,mclk_mhz,power_w,pp512_tps,decode_tps,decode_ms_per_tok" > "$OUT"

run_arm () {  # $1 pair  $2 arm  $3 spv
    cp "$3" "$SPV"
    soak
    local t0 sc mc pw t1 pp dec BP out
    t0=$(( $(cat "$HW") / 1000 ))
    flock -w 1800 /tmp/q36-gpu.lock ./q36-bench --vulkan -m "$MODEL" \
        --prompt-file tests/long_context_story_prompt.txt \
        --ctx-start 512 --ctx-max 512 --ctx-alloc 641 \
        --prefill-chunk 256 --gen-tokens 128 > "$TMP/run" 2>&1 &
    BP=$!
    sleep 12                      # sample clocks mid-run, under load
    sc=$(sclk); mc=$(mclk); pw=$(( $(cat "$HWD/power1_average") / 1000000 ))
    wait $BP
    out=$(cat "$TMP/run")
    t1=$(( $(cat "$HW") / 1000 ))
    pp=$(printf '%s' "$out"  | awk -F, '/^512,/{print $3; exit}')
    dec=$(printf '%s' "$out" | awk -F, '/^512,/{print $5; exit}')
    [ -z "$dec" ] && { echo "pair $1 arm $2: NO RESULT" >&2; printf '%s\n' "$out" | tail -3 >&2; return 1; }
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$1" "$2" "$t0" "$t1" "$sc" "$mc" "$pw" "$pp" "$dec" \
        "$(awk -v d="$dec" 'BEGIN{printf "%.4f", 1000/d}')" >> "$OUT"
    echo "pair $1 arm $2: pp=$pp dec=$dec entry=${t0}C sclk=${sc}"
}

for i in $(seq 1 "$PAIRS"); do
    run_arm "$i" A "$A_SPV"
    run_arm "$i" B "$B_SPV"
done
cp "$A_SPV" "$SPV"    # leave the tree on the A (baseline) shader
echo "DONE -> $OUT"
