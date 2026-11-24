# Inkling MoE down IQ4_NL decode: split + step-major layout (harness-validated, not integrated)

Harness: standalone Vulkan, real .spv, 48-dispatch chain with barriers (decode shape), ~2 s governor
warm-up first (without it everything reads ~2x slow). Matches production profile (45.9 vs 46.8 us/disp).

Ceiling: plain stream of the same 11.94 MB in the same chain = 28.0 us/disp (426 GB/s).

| variant (timing on constant-filled weights) | us/disp |
|---|---|
| current kernel, GGUF layout | 45.9 |
| activations staged in LDS | 50.3 (worse) |
| step-major layout [expert][b>>2][row][4 blocks] | 41.8 |
| step-major + split (16 B quants as uvec4, f16 scales after, per expert) | 38.6 |
| same + full 9-step unroll | 61.5 (worse) |

Validation (validate.c, real weights, 64 random cases x 4 layers incl. experts 0 and 127):
bit-identical 64/64 on MiniPlus-R2, APEX-I MiniPlus-V2.1, APEX-I NanoPlus-V2.1, UD-IQ3_XXS.
Old kernel vs CPU double reference: worst 3.6e-8 relative. Real-weight timing: 46.0 -> 38.7 us/disp (-15.8..-15.9%).

Bit-identity trap: selecting the f16 scale half BEFORE unpackHalf2x16 lets ACO fold the convert into a
fused v_fma_mix_f32 for acc = fma(d, s, acc); the GGUF kernel uses unfused v_mac_f32. Convert both
halves then select (d = odd ? dd.y : dd.x) -> 32 v_mac, bits match.

Layout per expert (stride unchanged, out_dim*blocks*18 bytes; needs out_dim*blocks % 8 == 0, blocks % 4 == 0):
quants: lin = ((b>>2)*out_dim + row)*4 + (b&3), 16 B at ebase + lin*16
scales: f16 at ebase + out_dim*blocks*16 + lin*2
Prefill inkling2_moe_down_gemm_iq4nl reads the same tensor and must switch too (not done).

## Integrated (env-gated P36_INK_DOWN_SPLIT=1), GPU-validated 2026-11-23
- Re-layout at weight-cache copy (vk_04 registry keyed by GGUF source pointer, registered by the Inkling engine before prewarm): one copy, no extra memory. Prefill gemm reads the same layout.
- Parity off==on on all 4 Inkling GGUFs (md5 above in ~/q36-ab/PLAN.md). Independent review: no findings.
- Skill bench R2 (ctl/cand/ctl, 6 samples): 78.17 / 81.04 / 78.16 t/s (+3.7%), startup and memory unchanged.

## Follow-up: gate_up step-major (P36_INK_GU_STEPMAJOR, default on)
Whole blocks re-ordered step-major per expert (block i of row R at i*rows+R). Kernel: Q4_K -7.7%, IQ3_XXS -2.6%, IQ3_S none (excluded).
Parity identical on all 4 Inkling GGUFs; independent review clean. Skill bench R2: 80.52 / 81.39 / 80.63 t/s (+1.0%).
