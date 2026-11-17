#!/usr/bin/env bash
# Frontier-NLL check for step B (packed-f16 PV), P3 protocol. Arms: ref = f32-staged
# (P36_VK_ATTN_FA_F16=0), h16 = shipped step A, pv = step A + FA_F16_PV.
# A/B dirs hold the same p36-bench; B differs only in attn_prefill_fa_f16_gqa{6,8}.spv.
set -uo pipefail
cd /home/server/q36
OUT="$1"; AB="$2"; shift 2
STORY=tests/long_context_story_prompt.txt
pgrep -x p36-bench >/dev/null && { echo "p36-bench already running, abort"; exit 1; }
SW="--ctx-start 8192 --ctx-max 16384 --step-incr 256 --ctx-alloc 16512 --prefill-chunk 256 --gen-tokens 0"
for spec in "$@"; do
  tag="${spec%%=*}"; model="${spec#*=}"
  mkdir -p "$OUT/$tag"
  ./p36 --vulkan -m "$model" --prompt-file "$STORY" --dump-tokens > "$OUT/$tag/tokens.json" 2>"$OUT/$tag/tokens.err"
  for arm in ref:A:"P36_VK_ATTN_FA_F16=0" h16:A:"P36_VK_ATTN_FA_F16=1" pv:B:"P36_VK_ATTN_FA_F16=1"; do
    a="${arm%%:*}"; r="${arm#*:}"; bin="$AB/${r%%:*}/p36-bench"; e="${r#*:}"
    d="$OUT/$tag/$a"; rm -rf "$d"; mkdir -p "$d"; st=$(date +%s)
    env -u P36_SHADER_DIR $e "$bin" --vulkan -m "$model" --prompt-file "$STORY" $SW --dump-frontier-logits-dir "$d" > "$OUT/$tag/$a.log" 2>&1
    echo "--- $tag $a rc=$? $(( $(date +%s) - st ))s frontiers=$(ls "$d" | wc -l)"
  done
  python3 karpathy/evidence/raw/p3-f16-nll/p3stats.py "$OUT/$tag/tokens.json" "$OUT/$tag/ref" "$OUT/$tag/h16" "$OUT/$tag/pv" > "$OUT/$tag/stats.txt" 2>&1
  python3 karpathy/evidence/raw/p3-f16-nll/nonfinite.py "$OUT/$tag/pv" > "$OUT/$tag/nonfinite.txt" 2>&1
  cat "$OUT/$tag/stats.txt" "$OUT/$tag/nonfinite.txt"
done
