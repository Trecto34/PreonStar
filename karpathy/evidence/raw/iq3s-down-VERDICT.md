# IQ3_S down projection in the fused f32 expert path — ACCEPTED

Date: 2026-09-21 · branch `trackB-ptq1_0` · base HEAD `e6c15ba`
Model: `Qwen3.8-35B-A3B-IQ2_M.gguf` · guard `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`

Closes HANDOFF-iq2s-20260920.md §5 item 3 ("explain/patch the q8-route
residue") for this file.

## Root cause — measured, not guessed

`Q36_VK_MOE_ROUTE_DEBUG=1` (env-gated diagnostic added to `q36.c`, prints on
every fused-path miss) at ctx 512, gen 8:

```
q36: moe f32 route miss il=0 n_tok=1   gate=22 up=22 down=21
q36: moe f32 route miss il=0 n_tok=256 gate=22 up=22 down=21
q36: moe f32 route miss il=1 n_tok=1   gate=22 up=22 down=21
q36: moe f32 route miss il=1 n_tok=256 gate=22 up=22 down=21
q36: moe f32 route miss il=2 n_tok=1   gate=22 up=22 down=21
q36: moe f32 route miss il=2 n_tok=256 gate=22 up=22 down=21
```

Types are `q36_vk` tensor ids: 22 = `IQ2_S`, 21 = `IQ3_S`. Exactly three layers
(0, 1, 2) leave the fused path, on both prefill (`n_tok=256`) and decode
(`n_tok=1`) — matching the 15 `moe_iq2s_gate_up` + 15 `moe_matvec` dispatches
the prefill profile showed. `q36_gpu_moe_ffn_f32_tensor()` required gate, up
*and* down to be the same quant; IQ2_M's layers 0-2 mix an **IQ3_S down under
IQ2_S gate/up** (confirmed independently from the GGUF tensor table:
`ffn_down_exps` IQ2_S x37, IQ3_S x3 at il 0-2, Q8_0 x1 at il 40 (MTP)).

## Cost it was paying (post-K-quant-widening profile, ctx 512)

Decode, differential (gen 1 vs gen 65, delta/64), 12.275 ms/tok GPU total:
`moe_iq2s_gate_up` 0.360 (3 disp) + `moe_matvec` 0.349 (3 disp) + `moe_tiles`
0.127 (6 disp) = 0.836 ms/tok = 6.8% still on the q8 route.

Prefill (ctx 512, single gen): `moe_iq2s_gate_up` 227.6 ms (15 disp, 15.2 ms
each) + `moe_matvec` 145.2 ms (15 disp, 9.7 ms each) = **372.8 ms = 19.6%** of
profiled prefill GPU, against 4.25 ms/dispatch for the fused `moe_iq2s_gate_up_gemm`
on the other 37 layers.

## Change

- `vulkan/moe_down_q2k_sum_decode.comp` and `vulkan/moe_down_gemm.comp`: new
  `-DQ36_MOE_IQ3S` builds (`moe_down_q2k_sum_decode_iq3s.spv`,
  `moe_down_gemm_iq3s.spv`). Element slotting is unchanged from the IQ2_S
  variants; the IQ3_S branch decodes the 110-byte block
  `[d f16][64 qs][8 qh][32 signs][4 scales]` with the 9-bit grid index
  (8 qs bits + 1 qh bit per 4 weights) against the IQ3_S grid table already
  present in the shared IQ tables buffer at word 2592 (`Q36_VK_TAB_*` layout:
  XXS 2048 B + ksigns 128 B + IQ2_S 8192 B -> 2592). Scale is `d*(2*nib+1)`,
  the same convention as `dense_iq3_s_decode.comp`.
- `q36_vulkan.c`: `iq2s` split into `gu_iq2s` (gate/up) and
  `down_iq2s`/`down_iq3s`; `down_stride` is 82 or 110 bytes accordingly; the
  down dispatch site picks the IQ3_S kernel; kernel create/destroy/`Q36_VK_KERNEL`
  entries added (7 bindings / 24 B push for the decode kernel, 6 / 24 for the
  GEMM, same as their IQ2_S twins).
- `q36.c`: env-gated `Q36_VK_MOE_ROUTE_DEBUG` miss print (diagnostic only).
- Makefile: the two `.spv` rules + list entries.

The IQ2_XXS/Q2_K and legacy builds stay byte-identical: all new code is inside
`#ifdef Q36_MOE_IQ3S`, and the widened guards only add the IQ3_S define.

## Parity

Frontier dumps (`--dump-frontier-logits-dir`), baseline `e6c15ba` binary vs
candidate, both at the default posture, 248,320 logits:

| arm | frontier | max_abs | mean_abs | p99 | top-1 | top64 |
|---|---|---|---|---|---|---|
| IQ2_M gemm path | 512 | 0.7619 | 0.1121 | 0.371 | match (13) | 62/64 |
| IQ2_M decode+gemm | 513 | 0.8652 | 0.1381 | 0.466 | match (5316) | 61/64 |
| guard | 513 | **0.0** (sha256 identical) | 0 | — | match | 64/64 |

Calibration: W1's own accepted posture measured max_abs 0.907 / top64 62/64
(`dense_iq2s`) and 1.245 / 60/64 against the all-matvec arm
(`HANDOFF-iq2s-20260920.md` §3). These deltas are *smaller* and of the same
shape — spread across ~all logits (max only ~2x p99, 7.7% / 15.0% of logits
above 0.25), not localized. The guard is byte-identical because it has no
IQ3_S expert tensors at all, so this is a pure no-op for it.

## Verdict

7-rep interleaved A/B (`tests/bench_ab.sh /tmp/q36-bench-ref2 ./q36-bench
<IQ2_M> 7`, ctx 512, gen 128, `Q36_AB_MIN_GAIN=3.4`):

| arm | prefill median (MAD) | decode median (MAD) |
|---|---|---|
| A `e6c15ba` | 497.00 (15.150) | 75.26 (0.180) |
| B candidate | **556.21** (12.400) | **78.40** (0.100) |
| delta | **+11.91%** | **+4.17%** |

Gate is MoE prefill >= 3.4% -> **PASS**, cleared ~3.5x. Decode separates
cleanly rep-by-rep (A 75.23/75.26/75.44/75.44/74.93/75.43/74.94 vs B
78.46/78.52/78.40/78.30/78.71/78.23/78.39), so the +4.17% is real, not spread.

Also run: `./q36_test --vulkan-kernels` OK (routed gemm `mv_max_rel=3.9e-06`,
`gemm_max_rel=8.9e-03`); `./karpathy/compat_gate.sh` -> `COMPATIBILITY_GATE:
PASS` (Swift smoke unchanged at 172.67 / 18.56, gpu-cpu-parity OK).

**ACCEPTED, default ON.** No new env var — the IQ3_S kernels are chosen by
tensor type inside the existing fused path; `Q36_VK_MOE_F32B=0` still reverts
the whole fused path.

`reconsider_if`: a future MoE file mixes `IQ2_S`/`IQ3_S` the *other* way
(IQ3_S gate/up under an IQ2_S/IQ3_S down) — that needs the gate/up side ported
too, and the current admission still rejects it (falls back to the q8 route,
unchanged behaviour, no wrong-kernel risk), or an `IQ2_S`-down file whose
`in_dim % 256 != 0`.

Raw evidence: `ab-iq3s-down.csv`, `ab-iq3s-down-summary.txt`,
`iq3s-down-parity.txt`, `iq3s-down-parity-stats.txt`,
`iq2m-prof-postkquant-{decode,prefill}.txt`, `prof-postfix-gen{1,65}.txt`.
