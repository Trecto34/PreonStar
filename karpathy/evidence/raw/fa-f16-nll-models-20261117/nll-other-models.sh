#!/usr/bin/env bash
# Frontier-NLL check across the remaining local FA-path models (P3 protocol:
# ctx 8192..16384 step 256, n=33 frontiers; ref=f32 FA, f16=shipped f16 K/V+PV,
# ctl=qtile2 f32 reorder control).
set -uo pipefail
cd /home/server/q36
SCR="$1"; shift
EV="karpathy/evidence/raw/fa-f16-nll-models-20261117"
mkdir -p "$EV" "$SCR"
STORY="tests/long_context_story_prompt.txt"
pgrep -x p36-bench >/dev/null && { echo "p36-bench already running, abort"; exit 1; }
SW="--ctx-start 8192 --ctx-max 16384 --step-incr 256 --ctx-alloc 16512 --prefill-chunk 256 --gen-tokens 0"

for spec in "$@"; do
  tag="${spec%%=*}"; model="${spec#*=}"
  mkdir -p "$SCR/$tag"
  ./p36 --vulkan -m "$model" --prompt-file "$STORY" --dump-tokens > "$SCR/$tag/tokens.json" 2>"$SCR/$tag/tokens.err"
  echo "=== $tag ($(basename "$model")) tokens rc=$? ==="
  for arm in ref:"P36_VK_ATTN_FA_F16=0" f16:"P36_VK_ATTN_FA_F16=1" ctl:"P36_VK_ATTN_FA=0"; do
    a="${arm%%:*}"; e="${arm#*:}"; d="$SCR/$tag/$a"; rm -rf "$d"; mkdir -p "$d"
    st=$(date +%s)
    env $e ./p36-bench --vulkan -m "$model" --prompt-file "$STORY" $SW --dump-frontier-logits-dir "$d" > "$SCR/$tag/$a.log" 2>&1
    echo "--- $tag $a rc=$? $(( $(date +%s) - st ))s frontiers=$(ls "$d" | wc -l)"
  done
  python3 karpathy/evidence/raw/p3-f16-nll/p3stats.py "$SCR/$tag/tokens.json" "$SCR/$tag/ref" "$SCR/$tag/f16" "$SCR/$tag/ctl" > "$EV/nll-${tag}-stats.txt" 2>&1
  python3 karpathy/evidence/raw/p3-f16-nll/nonfinite.py "$SCR/$tag/f16" > "$EV/nll-${tag}-nonfinite.txt" 2>&1
  cat "$EV/nll-${tag}-stats.txt" "$EV/nll-${tag}-nonfinite.txt"
  rm -rf "$SCR/$tag"
done
