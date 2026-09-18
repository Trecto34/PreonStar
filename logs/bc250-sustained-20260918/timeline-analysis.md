# Interval-level dispatch timeline — BC-250 Bonsai (2026-09-18 continuation)

Diagnostic: `Q36_VK_PROF_TIMELINE=<path>` in `q36_vulkan.c` dumps every
dispatch's two GPU query-pool timestamps plus the host record time.  Run:
`ctx 512 / gen 128`, chunk 256, soak-gated.  Raw gzip:
`runs/cont-timeline-decode.txt.gz`; profiler text `runs/cont-timeline-profile.txt`.

Timestamp period 10 ns.  Rows are in submission order; `start` is the
`COMPUTE_SHADER` stage, `end` the `BOTTOM_OF_PIPE` stage.

## Prefill (first 2595 dispatches)

- span 2563.16 ms, union of busy intervals 2550.21 ms -> **0.5 % idle**.
- **0 of 2594 consecutive pairs overlap** (each next `start` >= previous `end`).
  Aggregate GPU time equals union busy time: prefill is strictly serialized,
  not merely "unproven to overlap".
- 88.9 % of that time is `dense_extra_mmq_q2_0` (2278.89 ms / 800 dispatches).
- Per-shape (Q36_VK_PROF_SHAPE): 5120x17408 1040.24 ms / 256 disp;
  17408x5120 554.36 / 128; 5120x10240 235.29 / 96; 6144x5120 195.37 / 128;
  5120x6144 152.13 / 96; 5120x12288 92.59 / 32; 5120x1024 33.14 / 64.
  Every shape runs at roughly the same per-MAC rate, so the inefficiency is
  per-workgroup (barriers/LDS), not a single bad shape.

## Decode (166661 dispatches after the last MMQ dispatch, 128 tokens)

- span 4005.93 ms, sum of dispatch durations 3583.59 ms, gaps 422.35 ms.
- **0 overlapping pairs** out of 166660.  Decode is also strictly serialized.
- Gaps split by flush boundary (256 flushes):
  - within a flush: 293.65 ms, median **1.52 us**, p99 5.96 us
    -> 2.29 ms/token of pure inter-dispatch launch gap (~7 %).
  - between flushes: 128.69 ms, median **630 us** (max 1247 us)
    -> 1.01 ms/token of GPU idle around the per-token host round-trip.
- Total decode-timeline idle ~3.30 ms/token (10.5 %).
- Kernel GPU time (128 tokens): `dense_extra_decode_q2_0` 2855.95 ms
  (22.31 ms/token, 79.7 %), `hadamard_prepare` 178.84, `attn_decode_split`
  167.18, `add_rms_norm` 119.97, `delta_net_decode_reg_f16` 84.69,
  `matmul_bf16` 61.93, remainder < 25 ms each.

## Reading

The two phases answer the earlier open question: *aggregate GPU time does not
prove serialization*, but the timestamps do — neither phase overlaps work.  The
only recoverable decode time not in a kernel is the launch/sync idle above;
shrinking it means fewer dispatches or a device-side sampling path that avoids
the per-token host round-trip, both structural.  The dense decode kernel itself
is already within ~4 % of its no-ALU load floor (r5 ablation: mode 4 3048 ms vs
mode 1 2937 ms per 128 tokens), so a further kernel-local win is not indicated.

## Instrumentation caveat (important)

The decode profile reports 133 flushes / 65 tokens = ~2/token, and they are
`submit_wait_tensor_read` (65, the greedy argmax host read) plus
`submit_wait_query_pool` (67, the 2048-query cap forcing a mid-token flush).
The query-pool flush and the two timestamp writes per dispatch exist only while
`Q36_VK_PROF_KERNEL` is on, so the between-flush 630 us gaps and per-dispatch
1.5 us gaps above are **inflated by the measurement itself**.  The kernel
durations are unaffected, but decode verdicts must come from uninstrumented
runs and this idle must not be read as fully recoverable.  The measured
uninstrumented decode (secondary run) is 34.66 t/s at ctx 512, above the
gate's 34.87 median within noise; the instrumented span is 31.3 ms/token, which
is not a production number.
