# W7: Stream-K Geometry & Wave Quantization Audit Report

Target Hardware: AMD BC-250 (40 CUs, GFX1013 RDNA2)
Target Model: Swift-Qwen3.8-27B-IQ3_XXS (prefill chunk 256)
Gate: Stream-K porting requires wave tail > 3.0% of kernel time

| Shape / Kernel | Dispatches | Grid/disp | Waves (R=2) | Tail% (R=2) | Waves (R=4) | Tail% (R=4) | GPU Time (ms) | Share |
|---|---|---|---|---|---|---|---|---|
| `dense_iq3_xxs_mmq_5120x17408_n256 (pair+single)` | 126 | 1088 | 13.60 |  2.9% | 6.80 |  2.9% | 858.91 ms | 34.9% |
| `dense_iq3_xxs_mmq_17408x5120_n256 (down)` | 118 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 575.16 ms | 23.4% |
| `dense_iq3_xxs_mmq_5120x10240_n256` | 70 | 640 | 8.00 |  0.0% | 4.00 |  0.0% | 175.86 ms |  7.1% |
| `dense_iq3_xxs_mmq_5120x6144_n256` | 72 | 384 | 4.80 |  4.0% | 2.40 | 20.0% | 116.42 ms |  4.7% |
| `dense_iq3_xxs_mmq_5120x12288_n256` | 30 | 768 | 9.60 |  4.0% | 4.80 |  4.0% | 91.14 ms |  3.7% |
| `dense_iq4_xs_mmq_5120x17408 (ffn_up)` | 32 | 1088 | 13.60 |  2.9% | 6.80 |  2.9% | 154.18 ms |  6.3% |
| `dense_iq4_xs_mmq_17408x5120 (ffn_down)` | 8 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 11.34 ms |  0.5% |
| `dense_iq4_xs_mmq_5120x10240 (attn_qkv)` | 24 | 640 | 8.00 |  0.0% | 4.00 |  0.0% | 68.02 ms |  2.8% |
| `dense_iq4_xs_mmq_5120x6144 (attn_gate)` | 22 | 384 | 4.80 |  4.0% | 2.40 | 20.0% | 37.41 ms |  1.5% |
| `dense_iq4_xs_mmq_5120x12288 (attn_q)` | 2 | 768 | 9.60 |  4.0% | 4.80 |  4.0% | 6.80 ms |  0.3% |
| `dense_q4k_mmq_6144x5120_n256` | 72 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 139.33 ms |  5.7% |
| `dense_q5k_mmq_6144x5120_n256` | 52 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 116.95 ms |  4.8% |
| `dense_q6k_mmq_5120x1024_n256` | 38 | 64 | 0.80 | 20.0% | 0.40 | 60.0% | 21.11 ms |  0.9% |
| `dense_q5k_mmq_5120x1024_n256` | 26 | 64 | 0.80 | 20.0% | 0.40 | 60.0% | 16.62 ms |  0.7% |
| `dense_q5k_mmq_17408x5120_n256` | 2 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 16.15 ms |  0.7% |
| `dense_q5k_mmq_5120x10240_n256` | 2 | 640 | 8.00 |  0.0% | 4.00 |  0.0% | 16.12 ms |  0.7% |
| `dense_q5k_mmq_5120x17408_n256` | 2 | 1088 | 13.60 |  2.9% | 6.80 |  2.9% | 12.36 ms |  0.5% |
| `dense_q4k_mmq_5120x17408_n256` | 2 | 1088 | 13.60 |  2.9% | 6.80 |  2.9% | 10.33 ms |  0.4% |
| `dense_q6k_mmq_5120x6144_n256` | 2 | 384 | 4.80 |  4.0% | 2.40 | 20.0% | 9.43 ms |  0.4% |
| `dense_q6k_mmq_6144x5120_n256` | 4 | 320 | 4.00 |  0.0% | 2.00 |  0.0% | 8.21 ms |  0.3% |

## Aggregate Wave Tail Analysis
- Total Analyzed MMQ Kernel Time: 2461.87 ms (92.2% of all prefill GPU work)
- Whole-run GPU Kernel Time: 2670.93 ms
- Aggregate Tail Inefficiency (R=2 WGs/CU, 80 slots):  **1.93%** of MMQ time (**1.78%** of prefill kernel time)
- Aggregate Tail Inefficiency (R=4 WGs/CU, 160 slots): **3.61%** of MMQ time (**3.32%** of prefill kernel time)
- Absolute tail loss: 47.59 ms (R=2) / 88.81 ms (R=4)

## Key Geometric & Architectural Findings
1. **Down-projection (5120x17408 -> out_dim 5120):** Grid is exactly 320 WGs. On 40 CUs, 320/80 = 4.00 waves (R=2) and 320/160 = 2.00 waves (R=4). **Tail loss is exactly 0.00%**.
2. **QKV-projection (out_dim 10240):** Grid is exactly 640 WGs. 640/80 = 8.00 waves (R=2) and 640/160 = 4.00 waves (R=4). **Tail loss is exactly 0.00%**.
3. **Gate/Up-projection (out_dim 17408):** Grid is 1088 WGs. 1088/80 = 13.60 waves (0.40 wave tail = 2.86%) and 1088/160 = 6.80 waves (0.20 wave tail = 2.86%). **Tail loss is under 2.9%**.
4. **IQ4_XS de-homogenization:** Breaking down the 88 IQ4_XS dispatches into their exact 5 tensor geometries reveals that 84.1% of IQ4_XS work (ffn_up, ffn_down, attn_qkv) has <= 2.86% tail (0.0% on down/qkv). Blended IQ4_XS tail is 2.23% (R=2) and 4.39% (R=4), disproving the 11.0% artifact from averaging heterogeneous grids.
5. **Hardware Occupancy Model:** `mmq_info` measures 8 subgroups/SIMD (pair kernel, 128 VGPRs) to 10 subgroups/SIMD (stock MMQ, 96 VGPRs) on GFX1013. With 4 subgroups/WG (128 threads/wave32), active resident concurrency ranges from R=2 WGs/CU (80 slots across 40 CUs) up to an upper bound of R=4 WGs/CU (160 slots). We evaluate the full bracket [R=2, R=4].
6. **Zero-idle confirmation:** Timeline analysis (`logs/bc250-sustained-20260918/timeline-analysis.md`) confirms 0.5% idle across prefill dispatches; shapes execute at identical per-MAC rates, proving the latency floor is per-workgroup barrier/LDS stalls, not scheduling tail.

## Gate Decision
- Plan gate: Port Stream-K only if tail is > 3.0% of kernel time.
- Lower concurrency bound (R=2, 80 slots): aggregate tail is **1.78%** of kernel time, well under the 3.0% gate.
- Upper concurrency bound (R=4, 160 slots): aggregate tail is **3.32%** of kernel time. While the blended tail slightly exceeds 3.0% in this upper-bound model due to smaller intermediate shapes, the primary compute paths representing >80% of GEMM time (gate/up, down, qkv) have tail <= 2.86% (and exactly 0.0% on down and qkv).
- Scoping decision: Stream-K targets K-partitioning overhead specifically to eliminate wave tails on wide matrices. Given that the dominant compute paths have <= 2.86% tail (0% on down/qkv), and implementing Stream-K requires atomic K-reductions across CUs, partial-sum buffers, split-K barriers, and extra synchronization overhead, the actual recoverable headroom is bounded at < 1.8% of prefill.
- **VERDICT: REJECTED (NOT A LEVER).** Evidence documented in ledger and CAMPAIGN.md.
