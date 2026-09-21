#!/usr/bin/env bash
# Wait for GPU quiet, then run the item-2 A/B under the shared GPU lock.
set -uo pipefail
ROOT=/home/server/q36-opt-27b
A=/tmp/q36-base-44143d8/q36-bench
B=$ROOT/q36-bench
M=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
cd "$ROOT"

deadline=$(( $(date +%s) + 2700 ))
while pgrep -x q36-bench >/dev/null 2>&1 || pgrep -x q36_test >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then echo "waited too long for GPU"; exit 3; fi
    echo "waiting for GPU ($(date +%H:%M:%S)): $(pgrep -a q36-bench | head -1)"
    sleep 20
done
sleep 5

for chunk in 16 256; do
    reps=7; [ "$chunk" = 256 ] && reps=5
    echo "=== A/B chunk=$chunk reps=$reps ($(date +%H:%M:%S)) ==="
    flock -w 3600 /tmp/q36-gpu.lock env \
        Q36_AB_CTX=512 Q36_AB_GEN=128 Q36_AB_CHUNK=$chunk Q36_AB_MIN_GAIN=3.4 \
        tests/bench_ab.sh "$A" "$B" "$M" $reps \
        > "/tmp/ab-iq2s-smallbatch-chunk$chunk.csv" \
        2> "/tmp/ab-iq2s-smallbatch-chunk$chunk.summary"
    echo "A/B chunk=$chunk done exit=$?"
    cat "/tmp/ab-iq2s-smallbatch-chunk$chunk.summary"
done
echo "=== all done ($(date +%H:%M:%S)) ==="
