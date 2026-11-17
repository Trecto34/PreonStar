#!/usr/bin/env bash
# Refresh the README long-context table on the f16 prefill + decode FA build
# (2026-11-17): 3 reps of Axion 1k..32k (--prefill-chunk 512, --gen-tokens 64)
# and 1 sweep of Huihui dense 27B 1k..32k (--prefill-chunk 256, --gen-tokens 32).
set -euo pipefail
cd /home/server/q36
EV="karpathy/evidence/raw/readme-longctx-20261117"
mkdir -p "$EV"
STORY="tests/long_context_story_prompt.txt"
MOE="gguf/Axion-1.0-35B.APEX-I-NanoPlus-V2.1.gguf"
DENSE="gguf/Huihui-Qwen3.8-27B-Coder390-EfficientThink-abliterated.VAL-APEX-I-NanoPlus-V3.gguf"

pgrep -x p36-bench >/dev/null && { echo "p36-bench running, abort"; exit 1; }

for rep in 1 2 3; do
  ./p36-bench --vulkan -m "$MOE" --prompt-file "$STORY" \
    --ctx-start 1024 --ctx-max 32768 --step-mul 2 --ctx-alloc 32896 \
    --prefill-chunk 512 --gen-tokens 64 > "$EV/axion_rep${rep}.csv" 2> "$EV/axion_rep${rep}.err"
  echo "=== axion rep $rep ==="
  cat "$EV/axion_rep${rep}.csv"
done

python3 - "$EV" <<'PY' | tee "$EV/axion_medians.txt"
import csv, glob, os, statistics, sys
ev = sys.argv[1]
by_ctx = {}
for p in sorted(glob.glob(os.path.join(ev, "axion_rep*.csv"))):
    for r in csv.DictReader(open(p)):
        c = int(r["ctx_tokens"])
        by_ctx.setdefault(c, ([], []) )[0].append(float(r["prefill_tps"]))
        by_ctx[c][1].append(float(r["gen_tps"]))
for c in sorted(by_ctx):
    ps, ds = by_ctx[c]
    print(f"ctx {c:5d}: prefill med {statistics.median(ps):6.1f} (runs {ps}) | decode med {statistics.median(ds):5.1f} (runs {ds})")
PY

./p36-bench --vulkan -m "$DENSE" --prompt-file "$STORY" \
  --ctx-start 1024 --ctx-max 32768 --step-mul 2 --ctx-alloc 32896 \
  --prefill-chunk 256 --gen-tokens 32 > "$EV/dense_1k_32k.csv" 2> "$EV/dense_1k_32k.err"
echo "=== dense 1k..32k ==="
cat "$EV/dense_1k_32k.csv"
