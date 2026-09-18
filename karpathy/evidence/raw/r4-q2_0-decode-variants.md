# Q2_0 decode kernel variants — closed (2026-09-18)

Baseline f1e3dc4: vulkan/dense_extra_decode_q2_0.spv 1848 ms / 64 tok
(28.9 ms/token), whole-model 26.6 t/s, ~250 GB/s effective of 454.4 GB/s.
Mission gate was <1150 ms and tg128 > 40 t/s. Not reached by shader variants.

| variant | change | kernel ms/64 | tg128 t/s | verdict |
|---|---|---|---|---|
| baseline | ROWS=4, 64 lanes, 8 partitions | 1848 | 26.60 | — |
| V1 | ROWS=1, grid=out_dim | — | 25.35 | worse |
| V2 | ROWS=4 + q8 staged in LDS + 2-block unroll | — | 18.06 | much worse (barriers) |
| V3 | per-lane 32B aligned uvec4 loads | — | 22.23 | worse (3.55x traffic) |
| V-opt | consolidate overlapping u32/u16 loads | 1963 | 27.59 | noise, not kept |
| V4 | split-K | — | — | rejected: no fp32 atomics, M 7x oversubscribed |
| V5 | ROWS=8, grid=out_dim/8 | 1962.7 | 26.63 | kernel +6%, no win |
| V6 | ROWS=4 + pure 2x block unroll (no LDS) | 2142.1 | 24.51 | worse (VGPR pressure) |

Conclusion: two independent kernels (this one and dense_iq3_xxs decode on
Swift IQ3_XXS) both sit at ~230-250 GB/s, i.e. ~52-55% of the 454.4 GB/s spec.
Raising this is an engine-level memory-subsystem/latency item (per-layer
dispatch and barrier structure, weight layout), not a variant sweep.
Do not re-run these variants. Raw narration: /tmp/opencode/agy-r4b.log,
/tmp/opencode/agy-r4c.log (ephemeral).
