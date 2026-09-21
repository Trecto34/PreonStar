#!/usr/bin/env bash
# Item 2 parity: IQ2_S small-batch f32 pair vs q8 route and vs accepted GEMM pair.
set -uo pipefail
ROOT=/home/server/q36-opt-27b
M=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
G=/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf
cd "$ROOT"
run() { d="$1" m="$2"; shift 2; mkdir -p "$d"
  ( env "$@" flock -w 1800 /tmp/q36-gpu.lock "$ROOT/q36-bench" --vulkan -m "$m" \
      --prompt-file "$ROOT/tests/long_context_story_prompt.txt" \
      --ctx-start 512 --ctx-max 512 --ctx-alloc 900 --prefill-chunk 16 \
      --dump-frontier-logits-dir "$d" --gen-tokens 0 >"$d.log" 2>&1 ) || echo "$d exit=$?"
}
rm -rf /tmp/p-f32b /tmp/p-q8 /tmp/p-gemm16 /tmp/pg-f32b /tmp/pg-q8
run /tmp/p-f32b   "$M"
run /tmp/p-q8     "$M" Q36_VK_MOE_F32B=0
run /tmp/p-gemm16 "$M" Q36_VK_MOE_GEMM_MIN=16
run /tmp/pg-f32b  "$G"
run /tmp/pg-q8    "$G" Q36_VK_MOE_F32B=0

echo "--- IQ2_M chunk16: f32b(new) vs q8"
python3 /tmp/cmp_logits.py /tmp/p-q8     /tmp/p-f32b
echo "--- IQ2_M chunk16: gemm16(accepted) vs q8"
python3 /tmp/cmp_logits.py /tmp/p-q8     /tmp/p-gemm16
echo "--- IQ2_M chunk16: f32b(new) vs gemm16(accepted)"
python3 /tmp/cmp_logits.py /tmp/p-gemm16 /tmp/p-f32b
echo "--- GUARD chunk16: f32b vs q8"
python3 /tmp/cmp_logits.py /tmp/pg-q8    /tmp/pg-f32b
echo "=== parity done ($(date +%H:%M:%S)) ==="
