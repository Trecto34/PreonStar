#!/usr/bin/env bash
# Sweep 11: P6 gates.  compat_gate (Swift + guard), q36_test --vulkan-kernels,
# and a guard-side profile to show the nx kernel never fires for an IQ2XXS model
# (it is gated on Q36_VK_TENSOR_IQ3_XXS, so the guard is inert by type).
set -uo pipefail
cd /home/server/q36
GUARD=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
OUT=/tmp/p36/out11
mkdir -p "$OUT"
L=/tmp/q36-gpu.lock
pgrep -x q36-bench >/dev/null && pgrep -x q36_test >/dev/null && { echo busy; exit 1; }

echo "### compat gate"
( flock -w 7200 "$L" ./karpathy/compat_gate.sh ) > "$OUT/compat.log" 2>&1
echo "--- compat rc=$?"
tail -5 "$OUT/compat.log"

echo "### vulkan kernel self-test"
( flock -w 7200 "$L" ./q36_test --vulkan-kernels ) > "$OUT/vkern.log" 2>&1
echo "--- vkern rc=$?"
tail -5 "$OUT/vkern.log"

echo "### guard decode profile: nx dispatch count must be 0"
( flock -w 7200 "$L" env Q36_VK_PROF=1 Q36_VK_PROF_KERNEL=1 \
    ./q36-bench --vulkan -m "$GUARD" -p "Count to five." \
    --ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 32 ) \
    > "$OUT/guard-prof.log" 2>&1
echo "--- guard rc=$?"
echo "guard nx dispatches: $(grep -c 'dense_iq3_xxs_decode_nx' "$OUT/guard-prof.log")"
echo "### done"
