# Decode Non-GEMM Overhead Breakdown & Prefill LUT Attempt

Branch `experiment/radiance-transfer-bc250`. Shipped baseline for this task:
`a687cc9` (Q2_0 decode shared-LUT, tg128 ≈ 33.3 t/s, pp512 ≈ 200 t/s).
Hardware soak gate ≤55 °C on every run.

## 1. Method

`Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1`, ctx 512, `--gen-tokens 1` vs
`--gen-tokens 65` (identical 512-token prefill). The per-op delta divided by 64
isolates exactly one decode token's cost per op, free of the prefill
contribution. Wall-step at this point: **30.03 ms/token** (33.30 t/s); the
`dense_extra_decode` GEMM alone is **22.96 ms/token**. Raw dumps:
`logs/profiles/decode_diff_gen{1,65}.txt`, `logs/profiles/decode_overhead_breakdown.txt`.

## 2. Per-token GPU accounting (ctx 512)

| op | gpu_ms/token | dispatches/token | share |
|---|---|---|---|
| `dense_extra_decode` (GEMM) | 22.961 | 401 | 78.5 % |
| `hadamard_prepare` | 1.546 | 257 | 5.3 % |
| `attn_decode_split` | 1.444 | 16 | 4.9 % |
| `add_rms_norm` | 1.032 | 128 | 3.5 % |
| `delta_net_decode` | 0.724 | 48 | 2.5 % |
| `dense_f16` | 0.509 | 96 | 1.7 % |
| `recurrent_norm_gate_q8_k` | 0.202 | 48 | 0.7 % |
| `swiglu_q8_k` | 0.170 | 64 | 0.6 % |
| `norm_rope_kv_quant` | 0.127 | 16 | 0.4 % |
| `recurrent_conv_silu_decode` | 0.112 | 48 | 0.4 % |
| `delta_qkv_l2` | 0.111 | 48 | 0.4 % |
| `q8_k_quant` | 0.106 | 48 | 0.4 % |
| `dense_extra_mmq` | 0.055 | 0 | 0.2 % |
| `norm_rope` / `delta_gates` / `attn_combine` / `norm` / `attn_prefill` / `top2` | 0.160 | 52 | 0.5 % |
| **total GPU** | **29.259** | ~900 | 100 % |

Non-GEMM total = **6.30 ms/token (21.0 % of the 30.03 ms wall)**, matching the
spec's 6.22 ms estimate. The 401 decode dispatches/token are ~6.3 matvecs per
layer × 64 layers.

## 3. Triage answers

1. **GDN recurrent updates** are *not* 48 isolated micro-dispatches causing
   bubbles. They are six small kernels (`delta_net_decode` 0.72,
   `recurrent_norm_gate_q8_k` 0.20, `recurrent_conv_silu_decode` 0.11,
   `delta_qkv_l2` 0.11, `q8_k_quant` 0.11, `delta_gates` 0.04) totalling
   **1.29 ms/token (4.3 %)**. `delta_net_decode` has the worst per-dispatch cost
   (15 µs) and is the largest single recurrent piece.
2. **`hadamard_prepare`** is the largest non-GEMM kernel: **1.55 ms/token across
   257 dispatches** (≈6 µs each). It is not a VRAM round-trip problem — inputs
   and outputs stay in the activation buffers — it is **barrier/occupancy
   bound**: one 256-thread workgroup per 1024-block, only 5 (width 5120) to 17
   (width 17408) workgroups per dispatch, ~19 `barrier()` calls per workgroup
   (10 FWHT stages + 6 shared max-reduction steps + setup). On 80 SIMD32 the
   dispatch never fills the die.
3. **Host sync/sampling is not the leak.** Per token, `record_ms` for the whole
   decode is ~0.33 ms; the recorded host waits (`submit_eager` 21.9 ms,
   `submit_wait_tensor_read` 3.9 ms, `submit_wait_query_pool` 2.4 ms) are
   *waiting for the GPU*, not doing host work, and they sum to the GPU time
   rather than adding to it. Sampling is GPU-side (`top2` 0.01 ms/token).
   GPU total 29.26 ms < wall 30.03 ms, so decode is fully GPU-serialized with
   ~0.8 ms/token of gaps.
4. **Command buffer submission** is not a factor: `record_ms` is ~0.33 ms/token
   and `submit_wait_ms` is 0 in every individual op row. The ring is batched;
   the only waits are the explicit sync points above.

**Conclusion:** the 6.22 ms is legitimate GPU work, distributed across ~10
kernel types. There is no host-side or submission bubble to remove. The two
largest sinks are `hadamard_prepare` (5.3 %) and `attn_decode_split` (4.9 %,
inherent KV scan at ctx 512).

## 4. Phase 2 — mmq LUT port (tested, not warranted)

### 4.1 Ablation

Temporary `Q36_Q2_0_MMQ_ABLATE` in `dense_extra_mmq.comp`, `--gen-tokens 1`,
`Q36_VK_PROF_KERNEL=1`, `gpu_ms` of `op dense_extra_mmq` (512 prefill tokens):

| mode | change | gpu_ms | pp512 t/s |
|---|---|---|---|
| 0 | production per-code extract/sub/cvt/mul | 4930.7 (start 50 °C) | 185.51 |
| 1 | keep load+stores, drop the unpack ALU | 4351.2 (start 55 °C) | 204.74 |

The entire unpack ALU is only **~11 %** of the mmq kernel — even removing it
completely tops out near 205 t/s, below the 235 t/s target.

### 4.2 LUT implementation

The decode-style idea ported to mmq as a 256-entry `shared f16vec4` table
(2 KiB) holding `(code-1)` for each 8-bit code byte, so the hot loop becomes one
LDS read + one f16 multiply + two stores (same f16 values, bit-identical):

| variant | gpu_ms | pp512 t/s |
|---|---|---|
| mode 0 SWAR/unpack | 4930.7 | 185.51 |
| mode 2 f16 LUT | 4874.6 | 190.52 |

**~1.1 % — within noise, and the run was hotter.** Unlike decode (where the
SWAR spread was 10 ops), mmq's per-code sequence is already cheap, and the LDS
table plus `f16vec4` multiply costs about as much as the arithmetic it removes.
The mmq kernel is shared-tile/barrier bound, not unpack bound. **The port was
reverted; `dense_extra_mmq_q2_0.comp` is unchanged from `a687cc9`.**

## 5. Benchmark matrix

| build | tg128 t/s (ctx 512, soak) | pp512 t/s |
|---|---|---|
| `a687cc9` (shipped) | 33.30 | ~200 |
| decode row-load pipeline | 31.03 (−5.2 %) | ~198 | 
| mmq f16 LUT (prefill) | 33.3 (unchanged) | ~190 (neutral) |

No candidate from this task passed its gate, so `a687cc9` remains HEAD.

## 6. Recommendations

1. **Transport/share the 6.3 ms or accept it.** The decode frame is 78.5 % GEMM;
   the GEMM is ~4 % off its own no-ALU memory floor. A >5 % decode win now needs
   GEMM-level work (packed fp16/DP4A scheduling), not overhead cleanup.
2. If `hadamard_prepare` is attacked, the lever is **occupancy, not arithmetic**:
   merge the per-activation prepares into fewer dispatches, or replace the
   shared max-reduction (6 barriers) with subgroup reductions (each 64-lane
   sub-block is exactly one wave64). This risks the bit-exact quantizer, so it
   needs the CPU parity check.
3. Prefill's 200 t/s ceiling is **not** the unpack. Removing the unpack ALU
   entirely only reaches ~205 t/s. The remaining headroom is in the mmq
   shared-tile/barrier structure and would be a separate campaign.

## 7. Artifacts

- `logs/profiles/decode_overhead_breakdown.txt` (full op + kernel profile)
- `logs/profiles/decode_diff_gen1.txt`, `decode_diff_gen65.txt` (differential)
- `logs/profiles/mmq-ablate{0,1}.txt`, `mmq-lut{0,2}.txt`
- `logs/thermal_run_log.csv`
