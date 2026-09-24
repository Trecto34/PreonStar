#!/usr/bin/env bash
# Stage R3c + R4 in one GPU session.
#   R3c: prefill-only-plus-one-decode-step profiled server arms (4 streams,
#        max_tokens 1) at nx ON and OFF, so the 32-token arms' kernel gpu_ms
#        can be split into prefill and per-decode-step parts.
#   R4:  the MTP cycle diagnosis (plain control, phase timers, op profile).
set -uo pipefail
cd /home/server/q36
SWIFT=/home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf
STORY=tests/long_context_story_prompt.txt
OUT=/tmp/r3/out
L=/tmp/q36-gpu.lock

srv_kill () {
    pkill -TERM -x q36-server 2>/dev/null
    for _ in $(seq 1 40); do
        pgrep -x q36-server >/dev/null || return 0
        sleep 0.5
    done
    pkill -KILL -x q36-server 2>/dev/null
    sleep 1
}

srv_run () { # srv_run TAG NX TOKENS
    local tag="$1" nx="$2" toks="$3"
    local port=$((22000 + RANDOM % 2000))
    local log="$OUT/srv-$tag.log"
    srv_kill
    pgrep -x q36-bench >/dev/null && { echo "q36-bench running, abort $tag"; return 1; }
    ( exec flock -w 7200 "$L" env Q36_VK_DENSE_IQ3_NX=$nx Q36_SERVER_BATCH_LOG=1 \
        Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 \
        ./q36-server -m "$SWIFT" --vulkan --ctx 1024 --prefill-chunk 256 --tokens 128 \
        --batched-session 4 --host 127.0.0.1 --port "$port" ) > "$log" 2>&1 &
    for _ in $(seq 1 300); do
        if curl -fsS --max-time 1 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
        pgrep -x q36-server >/dev/null || { echo "server died: $tag"; tail -5 "$log"; return 1; }
        sleep 1
    done
    python3 /tmp/r3/batchbench.py --url "http://127.0.0.1:$port" \
        --streams 4 --max-tokens "$toks" --label "$tag" >> "$OUT/results-prefillonly.jsonl"
    srv_kill
    echo "--- $tag done"
}

bench () {
    local name="$1"; shift
    local envp="$1"; shift
    ( [ -n "$envp" ] && export $envp
      flock -w 7200 "$L" ./q36-bench --vulkan -m "$SWIFT" --prompt-file "$STORY" "$@" ) \
        > "/tmp/r4/out/$name.log" 2>&1
    echo "--- $name rc=$?"
}

mkdir -p /tmp/r4/out
: > "$OUT/results-prefillonly.jsonl"

echo "### R3c: prefill+1-step profiled arms"
srv_run prefill-nx1 1 1
srv_run prefill-nx0 0 1

echo "### R4: MTP cycle diagnosis"
P32="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 32 --mtp-margin 0"
P256="--ctx-start 512 --ctx-max 512 --ctx-alloc 1100 --prefill-chunk 256 --gen-tokens 256 --mtp-margin 0"
TIMING="Q36_MTP_TIMING=1 Q36_MTP_STATS=1"
PROF="Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1"
pgrep -x q36-server >/dev/null && srv_kill

bench plain32  "Q36_MTP_SPEC_DISABLE=1 Q36_MTP_STATS=1" $P32
bench mtp3-32  "$TIMING"                               $P32 --mtp-draft 3
bench mtp2-32  "$TIMING"                               $P32 --mtp-draft 2
bench plain256 "Q36_MTP_SPEC_DISABLE=1 Q36_MTP_STATS=1" $P256
bench mtp3-256 "$TIMING"                               $P256 --mtp-draft 3
echo "### done"
grep -h "gen_tps" /tmp/r4/out/*.log | tail -10
