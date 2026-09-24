# P4 Verdict — bit-exact integer IQ3_XXS decode: NOT VIABLE on GFX1013

2026-09-24, audit §P4. Model `gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf`.
Reopens **"Integer sign-mask IQ3_XXS decode — REJECTED"** (2026-09-17: Swift
decode kernel 373.83 -> 407.97 ms over 16 generation tokens) and the §7 closed
line **"`int dot: 0` is a hardware truth on GFX1013"**.  The audit's new evidence
for reopening is real and is reproduced here: the loop is 5.4 VALU per weight
MACC (`evidence/raw/iq3xxs-isa/dense_iq3_xxs_decode_r4.isa.txt`: 690 VALU ops per
128 weight MACCs per lane) and the exposed ALU cost is 6.497 ms/tok
(`evidence/raw/iq3xxs-isa/W2-VERDICT.md`: 26.578 baseline vs 20.081 loads-only
floor, 24.4% of the kernel).  What kills the design is the instruction set, not
the arithmetic.

## Probe: does the target have a packed signed 4x8 dot?

`dot4.comp` is a minimal shader that sums `dotPacked4x8EXT(int, int)` over two
SSBOs.  `./glslc --target-env=vulkan1.1 -O` accepts it (the extension is
supported by the compiler), and `logs/bc250-sustained-20260918/shader_isa.c` on
the resulting SPIR-V shows how ACO compiles it on this device:

```
v_mul_i32_i24_sdwa v5, sext(v4), sext(v3) dst_sel:DWORD src0_sel:BYTE_0 src1_sel:BYTE_0
v_mul_i32_i24_sdwa v6, sext(v4), sext(v3) dst_sel:DWORD src0_sel:BYTE_1 src1_sel:BYTE_1
v_mul_i32_i24_sdwa v7, sext(v4), sext(v3) dst_sel:DWORD src0_sel:BYTE_2 src1_sel:BYTE_2
v_mul_i32_i24_sdwa v4, sext(v4), sext(v3) dst_sel:DWORD src0_sel:BYTE_3 src1_sel:BYTE_3
v_add_nc_u32_e32   v5, v5, v6
v_add3_u32         v5, v5, v7, v4
v_add_nc_u32_e32   v2, v5, v2
```

`v_dot4_i32_i8` appears **zero** times in the disassembly: the packed dot is
emulated as 4 byte-selected signed 24-bit multiplies plus 3 adds + 1 accumulate,
i.e. **8 VALU per 4 weights = 2 VALU/weight for the MAC alone**, with the byte
selection eating SDWA slots.  Raw: `dot4.comp`, `dot4.isa.txt`,
`dot4.stats.txt`.

## Op accounting for the integer inner loop

| layout | ops per 4 weights | VALU/weight | note |
|---|---|---|---|
| shipped f32 path | ~24 | 5.4-6 | 4 bfe + 4 cvt + 4 cmp + 4 sel + 4 mul + 4 fma, per ISA dump |
| int, signs via second packed dot | ~20 | 5.0 | 3 index + 1 LDS + 8 (dot4) + 4 mask build/apply + 8 (dot4 of `q & mask`) |
| int, signs pre-folded into the weight word | ~12 | 3.0 | needs a *signed* grid word |
| int, signs pre-folded + SWAR negation | ~19 | 4.75 | the byte-wise `+1` in `~w + 1` carries across bytes when a magnitude is 0 |

The audit's target (<= 2.5 VALU/weight) needs the *sign-free* row: sign-free
accumulation means the grid word must already be signed, and the only way to get
that without per-4-weight work is a grid LUT indexed by the sign nibble - 16
variants of the current 512-entry table, 32 KB of LDS built per workgroup, which
is both over the LDS budget and a per-dispatch build cost.  The two realistic
integer layouts land at 5.0 and 4.75 VALU/weight, i.e. a 5-15% cut of the
exposed 6.497 ms ALU, ~0.3-1.0 ms/tok, ~1-2% of Swift's 46.7 ms/tok decode.
That is below any gate and well below the audit's 13-15% estimate, which assumed
a packed dot that this hardware does not have.

## Verdict

- **Probe NEGATIVE; not built.** The arithmetic premise (all-integer, bit-exact)
  stands - grid magnitudes <= 62, q8 <= 127, 32-term partial sums < 2^24, so an
  int32 inner product would be bit-identical - but there is no instruction to
  exploit it with.  `dotPacked4x8EXT` costs 2 VALU/weight before signs, and the
  per-4-weight sign handling adds ~2-3 more, so the integer path cannot beat the
  shipped f32 loop by enough to matter.
- This is consistent with, and now explains, the 2026-09-17 rejection
  (373.83 -> 407.97 ms): that variant applied the sign per weight and still paid
  the per-weight convert/fma, exactly as the audit suspected.  Reopening it with
  a packed dot was the right call; the ISA says the dot does not exist here.
- The same restructure for `dense_kquant_decode.comp` Q4_K/Q5_K inherits the same
  ceiling (no packed dot) - do not spend a build on it either.
- `reconsider_if`: hardware with a real `V_DOT4_I32_I8` (RDNA3+/gfx11) where the
  MAC drops to 1 op per 4 weights, or a repacked IQ3_XXS layout that stores
  magnitudes already signed (then the 3.0 VALU/weight row becomes reachable
  without any per-4-weight sign work).
