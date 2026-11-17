#!/usr/bin/env bash
# Whole-model MoE prefill at 16k and 16k..32k: f32 FA tile (P36_VK_ATTN_FA_F16=0)
# vs shipped f16 K/V + packed-f16 PV tile (P36_VK_ATTN_FA_F16=1).
set -euo pipefail
cd /home/server/q36
EV="karpathy/evidence/raw/moe-prefill-16k-20261117"
mkdir -p "$EV"
STORY="tests/long_context_story_prompt.txt"
MOE="gguf/Axion-1.0-35B.APEX-I-NanoPlus-V2.1.gguf"

pgrep -x p36-bench >/dev/null && { echo "p36-bench running, abort"; exit 1; }

echo "=== [1/2] Profiled 0..16k and 0..32k prefill GPU timestamps ==="
for ctx in 16384 32768; do
  for arm in A:0 B:1; do
    lab="${arm%%:*}"; v="${arm#*:}"
    P36_VK_ATTN_FA_F16="$v" P36_VK_PROF_KERNEL=1 \
      ./p36-bench --vulkan -m "$MOE" --prompt-file "$STORY" \
      --ctx-start "$ctx" --ctx-max "$ctx" --ctx-alloc "$((ctx + 128))" \
      --prefill-chunk 256 --gen-tokens 0 \
      > "$EV/prof${ctx}_${lab}.csv" 2> "$EV/prof${ctx}_${lab}.txt"
    echo "prof ctx=$ctx $lab: $(tail -n 1 "$EV/prof${ctx}_${lab}.csv")"
    grep -E "attn_prefill_fa|attn_combine|total_ms=" "$EV/prof${ctx}_${lab}.txt" || true
  done
done

echo "=== [2/2] 3 interleaved reps: 0..16k and 16k..32k prefill ==="
CSV="$EV/moe_16k_32k.csv"
echo "label,rep,ctx,prefill_tokens,prefill_tps" > "$CSV"
run_one() {
  local lab="$1" v="$2" rep="$3" tmp
  tmp="$(mktemp)"
  P36_VK_ATTN_FA_F16="$v" ./p36-bench --vulkan -m "$MOE" --prompt-file "$STORY" \
    --ctx-start 16384 --ctx-max 32768 --step-mul 2 --ctx-alloc 32896 \
    --prefill-chunk 256 --gen-tokens 0 > "$tmp" 2>/dev/null
  awk -F, -v l="$lab" -v r="$rep" '$1 ~ /^[0-9]+$/ { printf "%s,%s,%s,%s,%s\n", l, r, $1, $2, $3 }' "$tmp" | tee -a "$CSV"
  rm -f "$tmp"
}
for rep in 1 2 3; do
  if (( rep % 2 == 1 )); then
    run_one A 0 "$rep"
    run_one B 1 "$rep"
  else
    run_one B 1 "$rep"
    run_one A 0 "$rep"
  fi
done

python3 - "$CSV" <<'PY' | tee "$EV/moe_16k_32k.summary"
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
for ctx in sorted({int(r["ctx"]) for r in rows}):
    a = [float(r["prefill_tps"]) for r in rows if r["label"] == "A" and int(r["ctx"]) == ctx]
    b = [float(r["prefill_tps"]) for r in rows if r["label"] == "B" and int(r["ctx"]) == ctx]
    ma, mb = statistics.median(a), statistics.median(b)
    mada = statistics.median([abs(x - ma) for x in a])
    madb = statistics.median([abs(x - mb) for x in b])
    sep = "clean" if min(b) > max(a) else "overlap"
    span = "0..16k" if ctx == 16384 else "16k..32k"
    print(f"ctx {ctx:5d} ({span:8s}): prefill A {ma:6.2f} (MAD {mada:.2f}) -> B {mb:6.2f} (MAD {madb:.2f})  {(mb-ma)/ma*100:+.2f}% [{sep}: minB {min(b):.2f} vs maxA {max(a):.2f}]")
PY
