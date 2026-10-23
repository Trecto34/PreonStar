# Decode-window GPU profile — 2026-10-23

First measurement of where decode time actually goes on APEX-I-NanoPlus. Until now
the engine reported `gpu_ms=0.000` for every op, so the campaign ranked work on
end-to-end t/s with a 3-6% noise floor. Now it can rank kernels directly.

## What was broken

The timestamp query pool was capped at 2048 queries (`P36_VK_PROF_QUERY_CAP`), i.e.
**1024 timed dispatches** — the prefill burst only. The report was identical for
`-n 16` and `-n 64` runs, which is how it was noticed.

Fixes, both diagnostic-only and gated behind `P36_VK_PROF_KERNEL`:

- `P36_VK_PROF_QUERY_CAP` 2048 -> 65536 queries (32k dispatches, ~8 decode tokens).
- New `P36_VK_PROF_SKIP=N`: leave the first N dispatches untimed so the window can
  start after prefill.

## Decode window (8 tokens, GPU busy 82.3%)

```
device span 115.4 ms   dispatch time 95.0 ms   busy 82.3%
per token:  span 14.43 ms   gpu 11.87 ms   gap 2.55 ms
```

| GPU ms | share | kernel |
|---:|---:|---|
| 25.00 | 26.3% | `dense_q4k_decode` |
| 20.98 | 22.1% | `moe_down_q2k_sum_decode_iq3xxs` |
| 10.83 | 11.4% | `dense_kquant_decode` |
| 6.11 | 6.4% | `moe_gate_up_decode_iq2s_iq2xxs` |
| 5.60 | 5.9% | `matmul_q8_0_f32b` |
| 5.20 | 5.5% | `matmul_f32_fast` |
| 3.41 | 3.6% | `moe_gate_up_decode_iq2s` |
| 2.81 | 3.0% | `add_rms_norm` |

Note the MoE path uses **model-specific kernels** (`moe_gate_up_decode_iq2s_*`,
`moe_down_q2k_sum_decode_*`), not the generic `moe_matvec`. That is why the
type-specialization work on `moe_matvec` measured neutral: on this model it is not
the decode path at all.

## Where the 17.7% stall lives

Gaps concentrated at the sampling handoff, once per token:

```
top2.spv -> logits_minp_pack.spv : 630 us mean, 8 occurrences, 5.04 ms total
sampling sync = 25% of all gap time
```

Per token that is 2.55 ms of gap, of which 0.63 ms is the sampling round trip.
Even with zero gaps the window implies ~84 t/s ceiling at current GPU work, so
kernel time has to come down as well — `dense_q4k_decode` (3.1 ms/token for
attention projections) is the largest single line item.

## Reproduce

```sh
P36_VK_PROF_KERNEL=1 P36_VK_PROF_SKIP=3000 P36_VK_PROF_TIMELINE=/tmp/tl.txt \
  ./p36 -m MODEL.gguf -p hello -n 64 --ctx 512 --nothink --vulkan
python3 karpathy/tools/timeline_report.py /tmp/tl.txt
```

Tools: `karpathy/tools/timeline_report.py` (ranking, busy%, gaps),
`karpathy/tools/timeline_sync.py` (sampling-sync share).
