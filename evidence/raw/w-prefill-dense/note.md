# Dense prefill profile: Ternary-Bonsai-2-27B PTQ1_0 (2026-10-27)

Model `Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf` (5.95 GB, 1.75 bpw,
65-block qwen35 hybrid). Prefill only (`-n 0`), `--prefill-chunk 256`, prompt
`tests/long_context_story_prompt.txt` (24,478 words). Decode and Hadamard
untouched.

## 1. Clean prefill throughput (4 reps each, interleaved 512/2048)

| rep | 512 tok | 2048 tok |
|---|---:|---:|
| 1 | 99.80 | 98.01 |
| 2 | 98.34 | 96.07 |
| 3 | 97.02 | 94.99 |
| 4 | 96.69 | 94.67 |
| **median** | **97.68** | **95.53** |

Matches the earlier ~97 t/s baseline (no regression, no drift). Both columns
decline monotonically across reps = thermal, so any prefill A/B must interleave.

## 2. GPU ms by op (grouped brackets, 2048-token prefill)

Bracket sum **21,314 ms vs 21,389 ms wall = 1.00x** — prefill has no
gap/over-count, unlike decode's 1.04x. These numbers are trustworthy as-is.

| op | disp | WG/disp | µs/disp | ms | ms/tok | % |
|---|---:|---:|---:|---:|---:|---:|
| dense_extra_mmq | 3200 | 1219 | 6316 | 20212.2 | 9.87 | **94.8** |
| attn_prefill_fa_gqa6 | 128 | 128 | 2263 | 289.7 | 0.141 | 1.4 |
| hadamard_prepare | 2049 | 2111 | 82 | 168.2 | 0.082 | 0.8 |
| delta_net_decode | 384 | 192 | 367 | 140.7 | 0.069 | 0.7 |
| dense_f16 | 768 | 12288 | 164 | 126.1 | 0.062 | 0.6 |
| add_rms_norm_q8_k | 1016 | 256 | 90 | 91.7 | 0.045 | 0.4 |
| swiglu_q8_k | 512 | 17408 | 158 | 80.7 | 0.039 | 0.4 |
| recurrent_conv_silu_decode | 384 | **40** | 190 | 72.8 | 0.036 | 0.3 |
| recurrent_norm_gate_q8_k | 384 | 6144 | 182 | 69.7 | 0.034 | 0.3 |
| delta_qkv_l2 | 384 | 12288 | 81 | 31.2 | 0.015 | 0.1 |
| everything else (9 ops) | 203 | — | — | 39.0 | 0.019 | 0.2 |

Dispatches: 10,028 = **4.90 per token** (512-token run identical: 2510, 4.90).
Prefill is not dispatch bound; each `dense_extra_mmq` dispatch runs 6.3 ms.

## 3. Flags

* **Dominant kernel:** `dense_extra_mmq` at 94.8% of prefill. Nothing else is
  above 1.4%.
* **Under-occupied:** none that matters. WG histogram for `dense_extra_mmq` is
  `>40` for **3200 of 3200** dispatches. The only kernel at the 40-CU line is
  `recurrent_conv_silu_decode` at exactly 40 WGs, worth 0.3% of prefill — a
  zero-value fix.

## 4. Is there a >=5% lever?

No, not visible in the numbers.

* Achieved compute: 2 x 27e9 params x 2048 tok / 21.389 s = **5.17 TFLOPS**
  (= 2.58 T-MAC/s). Board fp32 peak is 10.24 TFLOPS, but this kernel's packed
  i16 MAC path tops out near 6.1 TFLOPS, so it already runs at ~85-88% of its
  own roofline. Closing that gap completely is worth ~+4-5% prefill, i.e. right
  at the bar, not clearly above it.
* The one >=5% idea (wider MAC: int8 `dotPacked4x8`, 4 MAC/lane instead of 2)
  was already built and measured in this tree: **85.7 t/s vs 98.0** — gfx1013
  `v_dot4` is not wider than 2x `v_mad_i16` in practice. Deleting the trit
  unpack entirely was also neutral (100.03 vs 98.7-100), so unpacking is not
  the cost.
* Beating this needs a different weight layout (32-byte aligned blocks so one
  sector feeds a block), which is a format change, not a kernel tweak - and the
  decode half of that analysis is off-limits this pass.

Verdict: prefill on this model is a single kernel at ~85% of its instruction
roofline, with no dispatch, occupancy or gap pathology. **Not worth pursuing.**
