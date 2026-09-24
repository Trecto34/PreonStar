#!/usr/bin/env bash
# R3: does the P6 nx small-batch kernel buy anything for --batched-session?
#   N = 1, 2, 4, 8 concurrent greedy streaming clients, Q36_VK_DENSE_IQ3_NX=1 vs 0,
#   arms interleaved per rep, 2 reps, no profiler.
#   Then one profiled run per (N in {1,4}) x (NX in {0,1}) for the op/kernel tables.
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
OUT=/tmp/r3/out
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock

srv_kill () {
    pkill -TERM -x q36-server 2>/dev/null
    for _ in $(seq 1 40); do
        pgrep -x q36-server >/dev/null || return 0
        sleep 0.5
    done
    pkill -KILL -x q36-server 2>/dev/null
    sleep 1
}

run () { # run TAG N NX TOKENS PROF
    local tag="$1" n="$2" nx="$3" toks="$4" prof="$5"
    local port=$((22000 + RANDOM % 2000))
    local log="$OUT/srv-$tag.log"
    local envs=(Q36_VK_DENSE_IQ3_NX=$nx Q36_SERVER_BATCH_LOG=1)
    [ -n "$prof" ] && envs+=(Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_SHAPE=1)
    srv_kill
    pgrep -x q36-bench >/dev/null && { echo "q36-bench running, abort $tag"; return 1; }
    ( exec flock -w 7200 "$L" env "${envs[@]}" ./q36-server -m "$SWIFT" --vulkan \
          --ctx 1024 --prefill-chunk 256 --tokens 128 --batched-session "$n" \
          --host 127.0.0.1 --port "$port" ) > "$log" 2>&1 &
    local ready=0
    for _ in $(seq 1 300); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            ready=1; break
        fi
        pgrep -x q36-server >/dev/null || { echo "server died: $tag"; tail -5 "$log"; return 1; }
        sleep 1
    done
    if [ "$ready" != 1 ]; then echo "server timeout: $tag"; srv_kill; return 1; fi
    python3 /tmp/r3/batchbench.py --url "http://127.0.0.1:$port" \
        --streams "$n" --max-tokens "$toks" --label "$tag" >> "$OUT/results.jsonl"
    srv_kill
    echo "--- $tag done"
}

: > "$OUT/results.jsonl"
for rep in 1 2; do
    for n in 1 2 4 8; do
        if (( rep % 2 == 1 )); then run "n${n}-nx0-r${rep}" "$n" 0 64 ""; run "n${n}-nx1-r${rep}" "$n" 1 64 ""
        else                        run "n${n}-nx1-r${rep}" "$n" 1 64 ""; run "n${n}-nx0-r${rep}" "$n" 0 64 ""; fi
    done
done

for n in 1 4; do
    for nx in 0 1; do
        run "prof-n${n}-nx${nx}" "$n" "$nx" 32 1
    done
done
srv_kill
echo "### done"
