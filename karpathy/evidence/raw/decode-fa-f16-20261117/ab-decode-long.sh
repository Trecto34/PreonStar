#!/usr/bin/env bash
# Whole-model long-context decode profile + interleaved A/B for P36_VK_ATTN_DECODE_FA_F16.
# Arm A: P36_VK_ATTN_DECODE_FA_F16=0 (f32 LDS staging in attn_decode_fa)
# Arm B: P36_VK_ATTN_DECODE_FA_F16=1 (f16 LDS staging in attn_decode_fa)
# Prefill uses the same attn_prefill_fa_f16 kernel in both arms.
set -euo pipefail
cd /home/server/q36
EV="karpathy/evidence/raw/decode-fa-f16-20261117"
STORY="tests/long_context_story_prompt.txt"
MOE="gguf/Axion-1.0-35B.APEX-I-NanoPlus-V2.1.gguf"
DENSE="gguf/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.VAL-APEX-I-NanoPlus-V3.gguf"

pgrep -x p36-bench >/dev/null && { echo "p36-bench running, abort"; exit 1; }

echo "=== [1/3] MoE 32k decode kernel GPU timestamp profile ==="
for arm in A:0 B:1; do
  lab="${arm%%:*}"; v="${arm#*:}"
  P36_VK_ATTN_DECODE_FA_F16="$v" P36_VK_PROF_KERNEL=1 P36_VK_PROF_DECODE=1 \
    ./p36-bench --vulkan -m "$MOE" --prompt-file "$STORY" \
    --ctx-start 32768 --ctx-max 32768 --ctx-alloc 32864 --prefill-chunk 256 --gen-tokens 32 \
    > "$EV/prof32k_moe_${lab}.csv" 2> "$EV/prof32k_moe_${lab}.txt"
  echo "moe prof $lab: $(tail -n 1 "$EV/prof32k_moe_${lab}.csv")"
  grep -E "attn_decode_fa|attn_combine|total_ms=" "$EV/prof32k_moe_${lab}.txt" || true
done

echo "=== [2/3] Dense 16k decode kernel GPU timestamp profile ==="
for arm in A:0 B:1; do
  lab="${arm%%:*}"; v="${arm#*:}"
  P36_VK_ATTN_DECODE_FA_F16="$v" P36_VK_PROF_KERNEL=1 P36_VK_PROF_DECODE=1 \
    ./p36-bench --vulkan -m "$DENSE" --prompt-file "$STORY" \
    --ctx-start 16384 --ctx-max 16384 --ctx-alloc 16480 --prefill-chunk 256 --gen-tokens 32 \
    > "$EV/prof16k_dense_${lab}.csv" 2> "$EV/prof16k_dense_${lab}.txt"
  echo "dense prof $lab: $(tail -n 1 "$EV/prof16k_dense_${lab}.csv")"
  grep -E "attn_decode_fa|attn_combine|total_ms=" "$EV/prof16k_dense_${lab}.txt" || true
done

echo "=== [3/3] MoE 8k/16k/32k interleaved A/B (3 reps, gen=32) ==="
CSV="$EV/moe_long_ab.csv"
echo "label,rep,ctx,prefill_tps,decode_tps" > "$CSV"
run_moe() {
  local lab="$1" v="$2" rep="$3" tmp
  tmp="$(mktemp)"
  P36_VK_ATTN_DECODE_FA_F16="$v" ./p36-bench --vulkan -m "$MOE" --prompt-file "$STORY" \
    --ctx-start 8192 --ctx-max 32768 --step-mul 2 --ctx-alloc 32864 --prefill-chunk 256 --gen-tokens 32 \
    > "$tmp" 2>/dev/null
  awk -F, -v l="$lab" -v r="$rep" '$1 ~ /^[0-9]+$/ { printf "%s,%s,%s,%s,%s\n", l, r, $1, $3, $5 }' "$tmp" | tee -a "$CSV"
  rm -f "$tmp"
}
for rep in 1 2 3; do
  if (( rep % 2 == 1 )); then
    run_moe A 0 "$rep"
    run_moe B 1 "$rep"
  else
    run_moe B 1 "$rep"
    run_moe A 0 "$rep"
  fi
done

python3 - "$CSV" <<'PY' | tee "$EV/moe_long_ab.summary"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for ctx in sorted({int(r["ctx"]) for r in rows}):
    a_d = [float(r["decode_tps"]) for r in rows if r["label"] == "A" and int(r["ctx"]) == ctx]
    b_d = [float(r["decode_tps"]) for r in rows if r["label"] == "B" and int(r["ctx"]) == ctx]
    a_p = [float(r["prefill_tps"]) for r in rows if r["label"] == "A" and int(r["ctx"]) == ctx]
    b_p = [float(r["prefill_tps"]) for r in rows if r["label"] == "B" and int(r["ctx"]) == ctx]
    ma, mb = statistics.median(a_d), statistics.median(b_d)
    mada = statistics.median([abs(x - ma) for x in a_d])
    madb = statistics.median([abs(x - mb) for x in b_d])
    mp_a, mp_b = statistics.median(a_p), statistics.median(b_p)
    sep = "clean" if min(b_d) > max(a_d) else "overlap"
    print(f"ctx {ctx:5d}: decode A {ma:6.2f} (MAD {mada:.2f}) -> B {mb:6.2f} (MAD {madb:.2f})  {(mb-ma)/ma*100:+.2f}% [{sep}: minB {min(b_d):.2f} vs maxA {max(a_d):.2f}] | prefill med A {mp_a:.2f} B {mp_b:.2f}")
PY
