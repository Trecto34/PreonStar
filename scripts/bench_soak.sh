#!/usr/bin/env bash
# Soak-gated benchmark wrapper. Waits until the GPU edge temp is <= TARGET
# before starting, runs the command, logs temps/wall/result to a CSV.
# Read-only sysfs, so it works unprivileged; the PWM/DPM override lives in
# scripts/setup_bc250_env.sh (root).
#
#   scripts/bench_soak.sh LABEL -- ./q36-bench --vulkan -m ... --gen-tokens 128
#
# Env: SOAK_TEMP (default 55), SOAK_LOG (default logs/thermal_run_log.csv)
set -uo pipefail

TARGET="${SOAK_TEMP:-55}"
LABEL="${1:?usage: bench_soak.sh LABEL -- command...}"
shift
[ "${1:-}" = "--" ] && shift
LOG="${SOAK_LOG:-logs/thermal_run_log.csv}"

NODE=$(ls /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input 2>/dev/null | head -1)
if [ -z "$NODE" ]; then
    echo "bench_soak: no hwmon temp node found" >&2
    exit 2
fi

if [ ! -f "$LOG" ]; then
    echo "timestamp,label,start_temp_c,end_temp_c,wall_s,result" > "$LOG"
fi

# ponytail: busy-wait on temp; if the box cannot reach the target it waits
# forever, which is the intended gate. Ctrl-C to bail.
while :; do
    t=$(( $(cat "$NODE") / 1000 ))
    [ "$t" -le "$TARGET" ] && break
    sleep 0.5
done
start=$t
s=$(date +%s)
out=$("$@" 2>&1)
rc=$?
e=$(date +%s)
end=$(( $(cat "$NODE") / 1000 ))
last=$(printf '%s' "$out" | grep -v '^q36:' | tail -1 | tr ',' ';')
printf '%s,%s,%s,%s,%s,"%s"\n' "$(date -Is)" "$LABEL" "$start" "$end" "$((e-s))" "$last" >> "$LOG"
printf '%s\n' "$out" | tail -2
exit $rc
