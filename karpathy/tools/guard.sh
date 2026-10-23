#!/bin/bash
# guard.sh - run a model command under a hard memory ceiling so an oversized
# model, a leak, or a runaway mlock kills this process instead of taking the dev
# box down. Changes no engine behaviour; it only bounds it.
#
#   ./.orchestration/guard.sh MODEL.gguf -- ./p36-bench -m MODEL.gguf ...
#
# Two layers, both cheap:
#   1. refuse to start when the model cannot plausibly fit in RAM + swap;
#   2. watch RSS and kill the child if it exceeds the ceiling.
#
# Deliberately NOT used: systemd-run scopes (silently unenforced on this host -
# a 512 MiB allocation survived a 64M scope) and ulimit -v (Vulkan reserves
# enormous address space, so an address-space cap breaks the GPU path).
set -uo pipefail

usage() { echo "usage: $0 MODEL.gguf -- command [args...]" >&2; exit 2; }
[ $# -ge 3 ] || usage
MODEL="$1"; shift
[ "$1" = "--" ] || usage; shift
[ -f "$MODEL" ] || { echo "guard: no such model: $MODEL" >&2; exit 2; }

model_mib=$(( $(stat -c %s "$MODEL") / 1048576 ))
avail_mib=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
swap_mib=$(awk '/SwapFree/{print int($2/1024)}' /proc/meminfo)
total_mib=$(awk '/MemTotal/{print int($2/1024)}' /proc/meminfo)
margin_mib=3072

echo "guard: model ${model_mib} MiB, available ${avail_mib} MiB, swap free ${swap_mib} MiB, ram ${total_mib} MiB"

if [ $(( model_mib + margin_mib )) -gt $(( avail_mib + swap_mib )) ]; then
    echo "guard: REFUSING - model ${model_mib} MiB + ${margin_mib} MiB margin exceeds" >&2
    echo "guard: ${avail_mib} MiB available + ${swap_mib} MiB swap free." >&2
    echo "guard: free memory or pick a smaller quant." >&2
    exit 3
fi

# Ceiling: model plus a fixed working set, capped at 90% of RAM so the kernel
# keeps room to reclaim rather than invoking the global OOM killer.
cap_mib=$(( model_mib + margin_mib ))
[ "${P36_GUARD_CAP_MIB:-0}" -gt 0 ] && cap_mib=$P36_GUARD_CAP_MIB
[ "$cap_mib" -gt $(( total_mib * 9 / 10 )) ] && cap_mib=$(( total_mib * 9 / 10 ))
echo "guard: RSS ceiling ${cap_mib} MiB"

"$@" &
child=$!
killed=0
while kill -0 "$child" 2>/dev/null; do
    rss_kb=$(awk '/VmRSS/{print $2}' /proc/$child/status 2>/dev/null)
    if [ -n "${rss_kb:-}" ] && [ "$rss_kb" -gt $(( cap_mib * 1024 )) ]; then
        echo "guard: KILL - pid $child reached $(( rss_kb / 1024 )) MiB > ${cap_mib} MiB ceiling" >&2
        kill -TERM "$child" 2>/dev/null
        sleep 1
        kill -KILL "$child" 2>/dev/null
        killed=1
        break
    fi
    sleep 0.1
done
wait "$child"; rc=$?
[ "$killed" = 1 ] && exit 137
exit $rc
