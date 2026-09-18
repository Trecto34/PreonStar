#!/usr/bin/env bash
# D2 regression test: build attn_parity, run span 512 and 256, assert
#   1) each GPU output matches its own fp64 CPU reference (< 5e-3), and
#   2) the span-vs-span difference stays at reassociation magnitude (< 1e-4).
# Run from the repo root after `make`.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
HERE=logs/bc250-sustained-20260918
BIN=$HERE/attn_parity
cc -O2 -I. -o "$BIN" "$HERE/attn_parity.c" \
    q36_gpu_core.o q36_vulkan.o q36_ssd.o q36_prompt_prefix.o q36_image.o \
    -lm -pthread -ldl -lvulkan || exit 2
run_span() { flock -w 900 /tmp/q36-gpu.lock -c "$ROOT/$BIN 24 4 256 $1 /tmp/attn-$1.bin" 2>/dev/null; }
a=$(run_span 512); b=$(run_span 256)
printf '%s\n%s\n' "$a" "$b" > "$HERE/runs/attn-parity-ref.txt"
worst=$(printf '%s\n%s\n' "$a" "$b" | grep -o 'maxabs_gpu_ref=[0-9.eE+-]*' | cut -d= -f2 | sort -g | tail -1)
echo "worst |gpu-ref| = $worst (bound 5e-3)"
python3 "$HERE/attn_parity_cmp.py" /tmp/attn-512.bin /tmp/attn-256.bin
cmp_rc=$?
ref_rc=1
[ -n "$worst" ] && python3 -c "import sys; sys.exit(0 if float('$worst') < 5e-3 else 1)" && ref_rc=0
if [ $ref_rc -eq 0 ] && [ $cmp_rc -eq 0 ]; then echo "ATTN PARITY PASS"; exit 0; fi
echo "ATTN PARITY FAIL"; exit 1
