# Reproducing the Q2_0 decode audit

Everything below runs unprivileged on the BC-250. Analysis and verdict live in
`reports/decode_coalescing_audit.md`; this file is just the commands.

Run from the repo root, and **only when the GPU is idle** — `pgrep -x q36-bench`
must come back empty. Never build while a benchmark runs; that has corrupted a
batch on this box before.

```sh
cd /home/server/q36-opt-27b
pgrep -x q36-bench || echo "GPU idle"
make -j16                      # rebuilds the .spv too; they are gitignored
```

## 1. Disassemble a shader (the part the old reports said was impossible)

`shader_isa.c` asks RADV for the compiled pipeline through
`VK_KHR_pipeline_executable_properties`. No `RADV_DEBUG`, no root, and it runs
in its own short-lived process, so a driver fault costs nothing.

```sh
cc -O2 -o /tmp/shader_isa logs/bc250-sustained-20260918/shader_isa.c -lvulkan
/tmp/shader_isa vulkan/dense_extra_decode_q2_0.spv 20 > /tmp/decode.isa
#                                                  ^ push-constant bytes

awk '/IR .Assembly./{f=1} f' /tmp/decode.isa > /tmp/decode.asm   # ACO assembly
grep -oP 'buffer_load_\w+' /tmp/decode.asm | sort | uniq -c      # load widths
grep -n 'buffer_load\|s_waitcnt vmcnt\|^BB' /tmp/decode.asm      # latency structure
```

The second argument is the shader's push-constant size (20 for the decode
kernel, 24 for the mmq one); the descriptor count is read out of the SPIR-V.
Three representations come back: `NIR Shader(s)`, `ACO IR`, `Assembly`.

## 2. Register / LDS / occupancy

```sh
cc -O2 -o /tmp/mmq_info logs/bc250-sustained-20260918/mmq_info.c -lvulkan
/tmp/mmq_info vulkan/dense_extra_decode_q2_0.spv | grep -E 'VGPR|SGPR|Spill|LDS|Code|Subgroups'
```

Baseline is 108 SGPR / 28 VGPR / 0 spills / 1024 B LDS / 3824 B code /
36 subgroups per SIMD.

## 3. Parity — always before benchmarking, it is far cheaper

Bit-exact logits:

```sh
for arm in base cand; do
  cp /path/to/$arm.spv vulkan/dense_extra_decode_q2_0.spv
  flock -w 900 /tmp/q36-gpu.lock ./q36-bench --vulkan \
    -m gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf \
    --prompt-file tests/long_context_story_prompt.txt \
    --ctx-start 512 --ctx-max 512 --ctx-alloc 673 \
    --prefill-chunk 256 --gen-tokens 160 \
    --dump-frontier-logits-dir /tmp/par-$arm
done

python3 -c "
import json
a=json.load(open('/tmp/par-base/frontier_000512.logits.json'))['logits']
b=json.load(open('/tmp/par-cand/frontier_000512.logits.json'))['logits']
print(len(a), 'entries, max_abs_diff =', max(abs(x-y) for x,y in zip(a,b)))"
```

Requirement: `max_abs_diff = 0` over 248,320 entries.

Greedy generation must also be byte-identical. `q36-bench` does not print tokens,
so use the CLI (`--dump-tokens` only dumps the *prompt* tokenization, not the
generation — don't reach for it):

```sh
flock -w 900 /tmp/q36-gpu.lock ./q36 --vulkan \
  -m gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf \
  -p "Explain how a memory-bound GPU kernel differs from a compute-bound one." \
  --temp 0 --seed 1 --nothink > /tmp/out-$arm.txt
cmp /tmp/out-base.txt /tmp/out-cand.txt && echo BYTE-IDENTICAL
```

## 4. Interleaved, soak-gated A/B

Swaps only the `.spv`, so both arms share one host binary. It waits for the edge
temperature to fall to `SOAK_TEMP` (default 55 °C) before *every* arm, and
alternates A/B pair by pair — running all of A then all of B puts thermal drift
on the second binary and fakes a regression.

```sh
logs/bc250-sustained-20260918/ab_decode.sh 10 /tmp/ab.csv \
  /path/to/baseline.spv /path/to/candidate.spv
```

Expect ~3 min per arm, so roughly an hour for 10 pairs — most of it soaking.
Then report **median paired change with MAD**, never means:

```sh
python3 -c "
import csv,statistics as st
r=list(csv.DictReader(open('/tmp/ab.csv')))
A={int(x['pair']):x for x in r if x['arm']=='A'}; B={int(x['pair']):x for x in r if x['arm']=='B'}
ch=[(float(B[p]['decode_tps'])/float(A[p]['decode_tps'])-1)*100 for p in sorted(set(A)&set(B))]
print('median paired %+.2f%%  positive %d/%d'%(st.median(ch),sum(c>0 for c in ch),len(ch)))"
```

Gate: **>= +1.5 %** median paired decode. Judge a decode-only patch on decode;
`tests/bench_ab.sh` declares `FAIL (prefill)` by construction when only a decode
kernel moved.

## 5. Caveats that cost real time here

- The `.spv` files are gitignored and absent from a fresh worktree. `make clean`
  deletes them. Copy them in or rebuild before anything else.
- MCLK has no live counter on this box; `pp_dpm_mclk` shows a DPM table, not the
  active clock. `freq1_input` *is* a live SCLK. Do not parse `pp_dpm_sclk` for a
  current frequency — it reports the selected level, which idles at 57 MHz.
- Thermal throttling is large: SCLK ranged 1220–1690 MHz across the recorded
  run. Only paired, interleaved comparisons mean anything.
- Restore the baseline `.spv` when finished; the harness does this for you on a
  clean exit, but not if you interrupt it.
