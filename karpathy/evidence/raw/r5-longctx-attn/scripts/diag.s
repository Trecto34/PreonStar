#!/usr/bin/env bash
# R5 diagnostics that only need a Vulkan device (no model process).
# Run AFTER the bench loop has finished, never during.
set -uo pipefail
cd /home/server/q36
mkdir -p /tmp/r5/diag
pgrep -x q36-bench >/dev/null && { echo "q36-bench running, abort"; exit 1; }
for s in attn_decode_split attn_combine attn_prefill_fa; do
    [ -f "vulkan/$s.spv" ] || continue
    /tmp/mmq_info "vulkan/$s.spv" > "/tmp/r5/diag/mmq-$s.txt" 2>&1
    echo "--- $s rc=$? $(grep -c . /tmp/r5/diag/mmq-$s.txt) lines"
done
echo "### diag done"
