#!/usr/bin/env bash
# Interleaved A/B throughput comparison for q36 target models on BC-250.
#
# Why this exists: tests/benchmark_transfer.sh runs A fully, then B fully. Anything
# that drifts during a session (GPU state, page cache, background load) lands on the
# second binary and looks like a regression. This harness alternates A and B rep by
# rep, so drift hits both equally, and reports medians instead of means.
#
# Measured on 2026-09-17 (Swift 27B IQ3_XXS, ctx 1024, 3 reps):
#   prefill spread within a session: 0.6-0.7%   -> usable signal
#   decode  spread within a session: up to 9.4% -> needs many reps to say anything
#   start die temp 49C vs 63C: 171.96 vs 171.91 tok/s prefill, i.e. no thermal effect.
#     So NO cool-down is performed; it would double wall time for nothing.
#
# Usage: tests/bench_ab.sh <A_BIN> <B_BIN> <MODEL> [REPS]
#   A_BIN = current/known-good binary, B_BIN = candidate binary.
# Output: raw CSV rows on stdout, then a summary block on stderr.
# Exit status: 0 always if the comparison ran; the verdict is in the summary.
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
usage="usage: bench_ab.sh <A_BIN> <B_BIN> <MODEL> [REPS]"
a_bin="${1:?$usage}"
b_bin="${2:?$usage}"
model="${3:?$usage}"
reps="${4:-${Q36_AB_REPS:-3}}"
ctx="${Q36_AB_CTX:-1024}"
gen="${Q36_AB_GEN:-16}"
chunk="${Q36_AB_CHUNK:-256}"
min_gain="${Q36_AB_MIN_GAIN:-1.5}"

for pair in "reps=$reps" "ctx=$ctx" "gen=$gen" "chunk=$chunk"; do
    name="${pair%%=*}"
    val="${pair#*=}"
    [[ "$val" =~ ^[1-9][0-9]*$ ]] || { echo "invalid $name: $val" >&2; exit 2; }
done
[[ -x "$a_bin" ]] || { echo "A binary not executable: $a_bin" >&2; exit 2; }
[[ -x "$b_bin" ]] || { echo "B binary not executable: $b_bin" >&2; exit 2; }
[[ -s "$model" ]] || { echo "model not found: $model" >&2; exit 2; }
if pgrep -x q36-bench >/dev/null 2>&1 || pgrep -x q36_test >/dev/null 2>&1; then
    echo "another q36 benchmark/test is running" >&2
    exit 2
fi

hw_dir="$(ls -d /sys/class/drm/card0/device/hwmon/hwmon* 2>/dev/null | head -1 || true)"
gpu_temp() {
    local raw
    raw="$(cat "$hw_dir/temp1_input" 2>/dev/null || echo 0)"
    echo $(( raw / 1000 ))
}

out_file="$(mktemp)"
cleanup() { rm -f -- "$out_file"; }
trap cleanup EXIT

run_one() { # $1=label $2=binary $3=rep
    local label="$1" bin="$2" rep="$3" tag st et row
    tag="$(mktemp)"
    st="$(gpu_temp)"
    if ! timeout --kill-after=5s 300s "$bin" --vulkan -m "$model" \
        --prompt-file "$root_dir/tests/long_context_story_prompt.txt" \
        --ctx-start "$ctx" --ctx-max "$ctx" --ctx-alloc "$(( ctx + gen + 256 ))" \
        --prefill-chunk "$chunk" --gen-tokens "$gen" >"$tag" 2>&1; then
        echo "  [$label rep$rep] bench exited non-zero" >&2
        sed -n '1,40p' "$tag" >&2
    fi
    et="$(gpu_temp)"
    row="$(awk -F, '$1 ~ /^[0-9]+$/ && NF == 6 { print; exit }' "$tag")"
    rm -f -- "$tag"
    if [[ -z "$row" ]]; then
        printf '%s,%s,%s,%s,error,error\n' "$label" "$rep" "$st" "$et"
    else
        # row is: ctx_start,ctx_end,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
        printf '%s,%s,%s,%s,%s,%s\n' "$label" "$rep" "$st" "$et" \
            "$(awk -F, '{print $3}' <<<"$row")" "$(awk -F, '{print $5}' <<<"$row")"
    fi
}

echo "label,rep,start_temp_c,end_temp_c,prefill_tps,decode_tps" | tee "$out_file"
for (( rep=1; rep<=reps; rep++ )); do
    # Alternate order every rep so warm-up and drift split evenly between A and B.
    if (( rep % 2 == 1 )); then
        run_one A "$a_bin" "$rep" | tee -a "$out_file"
        run_one B "$b_bin" "$rep" | tee -a "$out_file"
    else
        run_one B "$b_bin" "$rep" | tee -a "$out_file"
        run_one A "$a_bin" "$rep" | tee -a "$out_file"
    fi
done

awk -F, -v min_gain="$min_gain" -v a_name="$a_bin" -v b_name="$b_bin" '
function med(arr, n,    i, j, t) {
    for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++)
        if (arr[j] < arr[i]) { t = arr[i]; arr[i] = arr[j]; arr[j] = t }
    if (n == 0) return -1
    if (n % 2) return arr[(n + 1) / 2]
    return (arr[n / 2] + arr[n / 2 + 1]) / 2
}
function mad_of(arr, n, m,    i, k, d) {
    for (i = 1; i <= n; i++) d[i] = (arr[i] > m ? arr[i] - m : m - arr[i])
    return med(d, n)
}
$1 == "A" || $1 == "B" {
    lab = $1
    if ($5 == "error" || $5 == "") { fails[lab]++; next }
    p[lab, ++np[lab]] = $5 + 0
    d[lab, ++nd[lab]] = $6 + 0
}
END {
    printf "\n%-4s %-5s %-14s %-9s %-14s %-9s %s\n", \
        "bin", "n", "prefill_med", "mad", "decode_med", "mad", "source" > "/dev/stderr"
    for (i = 1; i <= 2; i++) {
        lab = (i == 1 ? "A" : "B")
        # p[]/d[] are SUBSEP-joined multipart arrays (p[lab,k]); med() indexes its
        # argument with plain integers, so handing them over directly reads only
        # uninitialized elements and returns 0 -- which prints a fake "median 0.00"
        # and a fake 0.00% gain. Flatten into the copy arrays first, then med() the
        # flat copy (med() sorts in place, which does not affect the MAD).
        for (k = 1; k <= np[lab]; k++) pcopy[k] = p[lab, k]
        pa = (np[lab] > 0 ? med(pcopy, np[lab]) : -1)
        for (k = 1; k <= nd[lab]; k++) dcopy[k] = d[lab, k]
        pb = (nd[lab] > 0 ? med(dcopy, nd[lab]) : -1)
        printf "%-4s %-5d %-14.2f %-9.3f %-14.2f %-9.3f %s\n", \
            lab, np[lab], pa, mad_of(pcopy, np[lab], pa), pb, mad_of(dcopy, nd[lab], pb), \
            (lab == "A" ? a_name : b_name) > "/dev/stderr"
        P[lab] = pa; D[lab] = pb
    }
    gain_p = (P["A"] > 0 ? (P["B"] - P["A"]) / P["A"] * 100 : 0)
    gain_d = (D["A"] > 0 ? (D["B"] - D["A"]) / D["A"] * 100 : 0)
    if (fails["A"] + fails["B"] > 0)
        printf "\nfailed runs: A=%d B=%d\n", fails["A"] + 0, fails["B"] + 0 > "/dev/stderr"
    printf "\nB vs A prefill: %+.2f%%   decode: %+.2f%%   (gate: %+.2f%%)\n", \
        gain_p, gain_d, min_gain > "/dev/stderr"
    verdict = (gain_p >= min_gain ? "PASS (prefill)" : "FAIL (prefill)")
    printf "verdict: %s\n", verdict > "/dev/stderr"
}
' "$out_file" >&2
