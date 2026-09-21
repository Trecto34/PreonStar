#!/usr/bin/env bash
# Interleaved, soak-gated A/B of the IQ2_S fused MoE gate/up path against the
# generic matvec fallback on one model file.  Both arms run the same binary;
# the arm is selected by Q36_VK_MOE_GATE_UP=0 (fallback) vs unset (fused).
#
#   logs/iq2s-gateup-20260920/ab_iq2s_gate_up.sh PAIRS OUT.csv MODEL.gguf
#
# Env: SOAK_TEMP (default 55), CTX (default 512), GEN (default 128).
set -uo pipefail

PAIRS="${1:?usage: ab_iq2s_gate_up.sh PAIRS OUT.csv MODEL.gguf}"
OUT="${2:?missing OUT.csv}"
MODEL="${3:?missing MODEL.gguf}"
TGT="${SOAK_TEMP:-55}"
CTX="${CTX:-512}"
GEN="${GEN:-128}"
ALLOC=$((CTX + GEN + 1))

HW=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
[ -z "$HW" ] && { echo "no hwmon temp node" >&2; exit 2; }
HWD=$(dirname "$HW")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

soak () { while :; do t=$(( $(cat "$HW") / 1000 )); [ "$t" -le "$TGT" ] && break; sleep 2; done; }
sclk () { echo $(( $(cat "$HWD/freq1_input") / 1000000 )); }

echo "pair,arm,order,entry_temp_c,exit_temp_c,sclk_mhz,power_w,pp_tps,decode_tps" > "$OUT"

run_arm () {  # $1 pair $2 arm $3 order
    local PAIR="$1" ARM="$2" ORDER="$3"
    soak
    local t0 sc pw t1 out pp dec BP
    t0=$(( $(cat "$HW") / 1000 ))
    if [ "$ARM" = ref ]; then
        flock -w 1800 /tmp/q36-gpu.lock env Q36_VK_MOE_DOWN_SUM_DECODE=0 ./q36-bench --vulkan -m "$MODEL" \
            --prompt-file tests/long_context_story_prompt.txt \
            --ctx-start "$CTX" --ctx-max "$CTX" --ctx-alloc "$ALLOC" \
            --prefill-chunk 256 --gen-tokens "$GEN" > "$TMP/run" 2>&1 &
    else
        flock -w 1800 /tmp/q36-gpu.lock ./q36-bench --vulkan -m "$MODEL" \
            --prompt-file tests/long_context_story_prompt.txt \
            --ctx-start "$CTX" --ctx-max "$CTX" --ctx-alloc "$ALLOC" \
            --prefill-chunk 256 --gen-tokens "$GEN" > "$TMP/run" 2>&1 &
    fi
    BP=$!
    sleep 8
    sc=$(sclk); pw=$(( $(cat "$HWD/power1_average") / 1000000 ))
    wait $BP
    out=$(cat "$TMP/run")
    t1=$(( $(cat "$HW") / 1000 ))
    pp=$(printf '%s' "$out"  | awk -F, -v c="$CTX" '$1==c{print $3; exit}')
    dec=$(printf '%s' "$out" | awk -F, -v c="$CTX" '$1==c{print $5; exit}')
    [ -z "$dec" ] && { echo "pair $PAIR arm $ARM: NO RESULT" >&2; printf '%s\n' "$out" | tail -3 >&2; return 1; }
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$PAIR" "$ARM" "$ORDER" "$t0" "$t1" "$sc" "$pw" "$pp" "$dec" >> "$OUT"
    echo "pair $PAIR arm $ARM (order $ORDER): pp=$pp dec=$dec entry=${t0}C exit=${t1}C sclk=${sc} pw=${pw}W"
}

for i in $(seq 1 "$PAIRS"); do
    if [ $(( i % 2 )) -eq 1 ]; then
        run_arm "$i" ref 1
        run_arm "$i" new 2
    else
        run_arm "$i" new 1
        run_arm "$i" ref 2
    fi
done
echo "DONE -> $OUT"
