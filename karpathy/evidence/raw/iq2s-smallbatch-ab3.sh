#!/usr/bin/env bash
# Wait for GPU quiet, then run the item-2 A/B under the shared GPU lock.
set -uo pipefail
ROOT=/home/server/p36-opt-27b
A=/tmp/p36-base-44143d8/p36-bench
B=$ROOT/p36-bench
M=/home/server/p36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
cd "$ROOT"

deadline=$(( $(date +%s) + 2700 ))
while pgrep -x p36-bench >/dev/null 2>&1 || pgrep -x p36_test >/dev/null 2>&1; do
    if (( $(date +%s) > deadline )); then echo "waited too long for GPU"; exit 3; fi
    echo "waiting for GPU ($(date +%H:%M:%S)): $(pgrep -a p36-bench | head -1)"
    sleep 20
done
sleep 5

for chunk in 16 256; do
    reps=7; [ "$chunk" = 256 ] && reps=5
    echo "=== A/B chunk=$chunk reps=$reps ($(date +%H:%M:%S)) ==="
    flock -w 3600 /tmp/p36-gpu.lock env \
        P36_AB_CTX=512 P36_AB_GEN=128 P36_AB_CHUNK=$chunk P36_AB_MIN_GAIN=3.4 \
        tests/bench_ab.sh "$A" "$B" "$M" $reps \
        > "/tmp/ab-iq2s-smallbatch-chunk$chunk.csv" \
        2> "/tmp/ab-iq2s-smallbatch-chunk$chunk.summary"
    echo "A/B chunk=$chunk done exit=$?"
    cat "/tmp/ab-iq2s-smallbatch-chunk$chunk.summary"
done
echo "=== all done ($(date +%H:%M:%S)) ==="
