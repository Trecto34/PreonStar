#!/usr/bin/env bash
# Interleaved, soak-gated A/B between two MODEL FILES. Nothing is swapped: the
# kernel is selected from the weight type at runtime, so both arms run the same
# binary and the same shader set. Use this for a quantization-format change
# (Q2_0-g64 vs PQ2_0); use ab_decode.sh when comparing two .spv on one model.
#
#   logs/bc250-sustained-20260918/ab_model.sh PAIRS OUT.csv A.gguf B.gguf
#
# Env: SOAK_TEMP (default 55).
# Run from the repo root, with the GPU idle. Records entry/exit edge temp, live
# SCLK, MCLK and package power alongside each measurement, because this box
# throttles hard enough that an unpaired comparison is meaningless.
set -uo pipefail

PAIRS="${1:?usage: ab_decode.sh PAIRS OUT.csv A.spv B.spv [TARGET_SPV]}"
OUT="${2:?missing OUT.csv}"
A_MODEL="${3:?missing A.gguf}"
B_MODEL="${4:?missing B.gguf}"
TGT="${SOAK_TEMP:-55}"

HW=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
[ -z "$HW" ] && { echo "no hwmon temp node" >&2; exit 2; }
HWD=$(dirname "$HW")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

soak () { while :; do t=$(( $(cat "$HW") / 1000 )); [ "$t" -le "$TGT" ] && break; sleep 2; done; }
sclk () { echo $(( $(cat "$HWD/freq1_input") / 1000000 )); }          # live, in MHz
mclk () { awk '/\*/{gsub(/[^0-9]/,"",$2); print $2; exit}' \
              /sys/class/drm/card0/device/pp_dpm_mclk; }              # DPM display only

echo "pair,arm,order,entry_temp_c,exit_temp_c,sclk_mhz,mclk_mhz,power_w,pp512_tps,decode_tps,decode_ms_per_tok" > "$OUT"

run_arm () {  # $1 pair  $2 arm  $3 model  $4 position-within-pair
    local MODEL="$3" ORDER="$4"
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
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$1" "$2" "$ORDER" "$t0" "$t1" "$sc" "$mc" "$pw" "$pp" "$dec" \
        "$(awk -v d="$dec" 'BEGIN{printf "%.4f", 1000/d}')" >> "$OUT"
    echo "pair $1 arm $2 (order $ORDER): pp=$pp dec=$dec entry=${t0}C exit=${t1}C sclk=${sc} pw=${pw}W"
}

# Alternate which arm runs first. Soak-gating already equalizes entry temp, but
# a fixed A-then-B order still gives B a systematically warmer die and a warmer
# page cache; alternating cancels whatever the gate does not.
for i in $(seq 1 "$PAIRS"); do
    if [ $(( i % 2 )) -eq 1 ]; then
        run_arm "$i" A "$A_MODEL" 1
        run_arm "$i" B "$B_MODEL" 2
    else
        run_arm "$i" B "$B_MODEL" 1
        run_arm "$i" A "$A_MODEL" 2
    fi
done
echo "DONE -> $OUT"
