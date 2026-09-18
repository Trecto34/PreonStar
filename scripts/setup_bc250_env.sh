#!/usr/bin/env bash
# BC-250 platform override: NCT6687 fan PWM to max + DPM manual clocks.
# MUST run as root (writes /sys). Corrected from the hand-off spec:
#  - nct6687 hwmon is found by reading each hwmon's name, not by grep -l on a glob.
#  - pp_dpm_* takes the DPM level INDEX; the spec's `awk '{print $1}'` yields "2:".
#    Strip the trailing colon.
#  - Idempotent, exits 0 even if a node is missing (so it is safe in harnesses).
set -uo pipefail

MODE="${1:-apply}"

if [ "$(id -u)" != 0 ]; then
    echo "setup_bc250_env: must run as root (try: sudo $0)" >&2
    exit 1
fi

if [ "$MODE" = "restore" ]; then
    echo "=== restoring DPM auto and fan auto ==="
    for dev in /sys/class/drm/card*/device; do
        [ -e "$dev/power_dpm_force_performance_level" ] &&
            echo auto > "$dev/power_dpm_force_performance_level" 2>/dev/null || true
    done
    for h in /sys/class/hwmon/hwmon*; do
        case "$(cat "$h/name" 2>/dev/null)" in
            nct6686|nct6687) ;;
            *) continue ;;
        esac
        for pwm in "$h"/pwm[1-8]; do
            [ -e "${pwm}_enable" ] && { echo 2 > "${pwm}_enable" 2>/dev/null ||
                                        echo 0 > "${pwm}_enable" 2>/dev/null || true; }
        done
    done
    echo "restored (DPM auto, PWM auto). Verify with: cat /sys/class/drm/card*/device/power_dpm_force_performance_level"
    exit 0
fi

echo "=== [1/3] NCT668x PWM -> 255 ==="
HWMON_DIR=""
for h in /sys/class/hwmon/hwmon*; do
    case "$(cat "$h/name" 2>/dev/null)" in
        nct6686|nct6687) HWMON_DIR="$h"; break ;;
    esac
done
if [ -n "$HWMON_DIR" ]; then
    for pwm in "$HWMON_DIR"/pwm[1-8]; do
        [ -e "$pwm" ] || continue
        [ -e "${pwm}_enable" ] && echo 1 > "${pwm}_enable" 2>/dev/null || true
        echo 255 > "$pwm" 2>/dev/null || true
    done
    echo "nct668x at $HWMON_DIR: PWM set to 255"
else
    echo "WARNING: nct6687 hwmon not found; fan override skipped"
fi

echo "=== [2/3] DPM clocks ==="
for dev in /sys/class/drm/card*/device; do
    [ -e "$dev/power_dpm_force_performance_level" ] || continue
    # Some kernels reject "manual"; fall back to "high" (highest perf level).
    echo manual > "$dev/power_dpm_force_performance_level" 2>/tmp/.dpm_err || true
    err=$(cat /tmp/.dpm_err 2>/dev/null); rm -f /tmp/.dpm_err
    lvl=$(cat "$dev/power_dpm_force_performance_level" 2>/dev/null)
    if [ "$lvl" != "manual" ]; then
        echo high > "$dev/power_dpm_force_performance_level" 2>/tmp/.dpm_err || true
        [ -s /tmp/.dpm_err ] && echo "  manual err: $err" && echo "  high err: $(cat /tmp/.dpm_err)"
        rm -f /tmp/.dpm_err
        lvl=$(cat "$dev/power_dpm_force_performance_level" 2>/dev/null)
    fi
    # manual mode needs an explicit level index (highest state = last line).
    if [ "$lvl" = "manual" ]; then
        for clk in pp_dpm_sclk pp_dpm_mclk; do
            [ -e "$dev/$clk" ] || continue
            idx=$(grep -oE '^[0-9]+' "$dev/$clk" | tail -1)
            [ -n "$idx" ] && echo "$idx" > "$dev/$clk" 2>/dev/null || true
        done
    fi
    echo "$dev: DPM level=$lvl"
done
echo "sclk states: $(tr '\n' ' ' < /sys/class/drm/card0/device/pp_dpm_sclk)"

echo "=== [3/3] Environment ready ==="
