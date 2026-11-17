#!/usr/bin/env bash
# Frontier-NLL check for P36_VK_ATTN_FA_F16 (P3 protocol, karpathy/evidence/raw/p3-f16-nll).
set -uo pipefail
cd /home/server/q36
OUT="$1"; shift
STORY=tests/long_context_story_prompt.txt
pgrep -x p36-bench >/dev/null && { echo "p36-bench already running, abort"; exit 1; }
SW="--ctx-start 8192 --ctx-max 16384 --step-incr 256 --ctx-alloc 16512 --prefill-chunk 256 --gen-tokens 0"
for spec in "$@"; do
  tag="${spec%%=*}"; model="${spec#*=}"
  mkdir -p "$OUT/$tag"
  ./p36 --vulkan -m "$model" --prompt-file "$STORY" --dump-tokens > "$OUT/$tag/tokens.json" 2>"$OUT/$tag/tokens.err"
  echo "--- $tag tokens rc=$? $(head -c 200 "$OUT/$tag/tokens.json" | tr -cd , | wc -c)+ toks"
  for arm in ref:"P36_VK_ATTN_FA_F16=0" f16:"P36_VK_ATTN_FA_F16=1" ctl:"P36_VK_ATTN_FA=0"; do
    a="${arm%%:*}"; e="${arm#*:}"; d="$OUT/$tag/$a"; rm -rf "$d"; mkdir -p "$d"
    st=$(date +%s)
    env $e ./p36-bench --vulkan -m "$model" --prompt-file "$STORY" $SW --dump-frontier-logits-dir "$d" > "$OUT/$tag/$a.log" 2>&1
    echo "--- $tag $a rc=$? $(( $(date +%s) - st ))s frontiers=$(ls "$d" | wc -l)"
  done
  python3 karpathy/evidence/raw/p3-f16-nll/p3stats.py "$OUT/$tag/tokens.json" "$OUT/$tag/ref" "$OUT/$tag/f16" "$OUT/$tag/ctl" > "$OUT/$tag/stats.txt" 2>&1
  python3 karpathy/evidence/raw/p3-f16-nll/nonfinite.py "$OUT/$tag/f16" > "$OUT/$tag/nonfinite.txt" 2>&1
  cat "$OUT/$tag/stats.txt"
done
