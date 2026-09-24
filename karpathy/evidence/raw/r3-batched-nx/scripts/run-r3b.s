#!/usr/bin/env bash
# R3 follow-up probe: the batched n_tok 2..8 step still sends K-quant trunk
# tensors to the 128-row MMQ tile (Q36_VK_DENSE_KQUANT_MMQ).  There is no
# switch for the IQ3_XXS pair tile or the IQ4_XS tile, so the existing K-quant
# switch is the one cheap way to test "is the 128-row tile the problem?".
#   N=4, nx ON in both arms, kquant MMQ tile on vs off, 2 interleaved reps,
#   profiled so the logs show which kernel replaced which.
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

run () { # run TAG KMQ
    local tag="$1" kmq="$2"
    local port=$((22000 + RANDOM % 2000))
    local log="$OUT/srv-$tag.log"
    srv_kill
    pgrep -x q36-bench >/dev/null && { echo "q36-bench running, abort $tag"; return 1; }
    ( exec flock -w 7200 "$L" env Q36_VK_DENSE_IQ3_NX=1 Q36_VK_DENSE_KQUANT_MMQ=$kmq \
        Q36_SERVER_BATCH_LOG=1 Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 \
        ./q36-server -m "$SWIFT" --vulkan --ctx 1024 --prefill-chunk 256 --tokens 128 \
        --batched-session 4 --host 127.0.0.1 --port "$port" ) > "$log" 2>&1 &
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
        --streams 4 --max-tokens 64 --label "$tag" >> "$OUT/results-kmq.jsonl"
    srv_kill
    echo "--- $tag done"
}

: > "$OUT/results-kmq.jsonl"
for rep in 1 2; do
    run "kmq-on-r$rep"  1
    run "kmq-off-r$rep" 0
done
srv_kill
echo "### done"
cat "$OUT/results-kmq.jsonl"
