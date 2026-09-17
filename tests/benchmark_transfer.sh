#!/usr/bin/env bash
# Reproducible, sequential BC-250 throughput check for both q36 target models.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
swift_model="${Q36_TRANSFER_SWIFT_MODEL:-/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf}"
qwen_model="${Q36_TRANSFER_QWEN_MODEL:-/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf}"
reps="${Q36_TRANSFER_REPS:-3}"
ctx="${Q36_TRANSFER_CTX:-1024}"
gen="${Q36_TRANSFER_GEN:-16}"

[[ "$reps" =~ ^[1-9][0-9]*$ ]] || { echo "invalid Q36_TRANSFER_REPS" >&2; exit 2; }
[[ "$ctx" =~ ^[1-9][0-9]*$ ]] || { echo "invalid Q36_TRANSFER_CTX" >&2; exit 2; }
[[ "$gen" =~ ^[1-9][0-9]*$ ]] || { echo "invalid Q36_TRANSFER_GEN" >&2; exit 2; }
[[ -x "$root_dir/q36-bench" ]] || { echo "build q36-bench first" >&2; exit 2; }
[[ -s "$swift_model" && -s "$qwen_model" ]] || { echo "both model files are required" >&2; exit 2; }
if pgrep -x q36-bench >/dev/null 2>&1 || pgrep -x q36_test >/dev/null 2>&1; then
    echo "another q36 benchmark/test is running" >&2
    exit 2
fi

active_pid=0
log_file=""
cleanup() {
    local status=$?
    trap - INT TERM EXIT
    if (( active_pid > 0 )) && kill -0 "$active_pid" 2>/dev/null; then
        kill -TERM -- "-$active_pid" 2>/dev/null || kill -TERM "$active_pid" 2>/dev/null || true
        wait "$active_pid" 2>/dev/null || true
    fi
    if [[ -n "$log_file" ]]; then rm -f -- "$log_file"; fi
    exit "$status"
}
trap 'exit 130' INT
trap cleanup TERM EXIT

echo "model,rep,ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes"
for model in swift qwen36; do
    if [[ "$model" == swift ]]; then
        path="$swift_model"
        chunk=256 # GFX1013 GQA-6 watchdog-safe maximum
    else
        path="$qwen_model"
        chunk=1024 # GFX1013 GQA-8 watchdog-safe maximum
    fi
    for (( rep=1; rep<=reps; rep++ )); do
        log_file="$(mktemp)"
        setsid timeout --kill-after=5s 180s "$root_dir/q36-bench" --vulkan \
            -m "$path" --prompt-file "$root_dir/tests/long_context_story_prompt.txt" \
            --ctx-start "$ctx" --ctx-max "$ctx" --ctx-alloc "$((ctx + gen + 256))" \
            --prefill-chunk "$chunk" --gen-tokens "$gen" >"$log_file" 2>&1 &
        active_pid=$!
        if ! wait "$active_pid"; then
            sed -n '1,80p' "$log_file" >&2
            rm -f -- "$log_file"
            active_pid=0
            exit 1
        fi
        active_pid=0
        row="$(awk -F, '$1 ~ /^[0-9]+$/ && NF == 6 { print; exit }' "$log_file")"
        if [[ -z "$row" ]]; then
            sed -n '1,80p' "$log_file" >&2
            rm -f -- "$log_file"
            exit 1
        fi
        printf '%s,%s,%s\n' "$model" "$rep" "$row"
        rm -f -- "$log_file"
        log_file=""
    done
done
