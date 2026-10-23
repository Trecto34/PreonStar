# One expert->pair schedule per layer (2026-10-23)

## Finding

Timeline, one layer, before:

    71 moe_tiles        1 workgroup
    72 moe_matvec_fast  512 groups   gate  (98.2 us median)
    73 moe_tiles        1 workgroup  <-- identical schedule, rebuilt
    74 moe_matvec_fast  512 groups   up    (98.2 us median)
    75 swiglu / 76 quantize_q8_k
    77 moe_tiles        1 workgroup  <-- rebuilt a third time
    78 moe_matvec_fast  2048 groups  down  (64.0 us median)

3 builds per layer, 120 per token, all identical whenever
`stream_part == P36_VK_STREAM_PART_NONE` (this model's path).  The schedule
maps pairs to experts and nothing else; gate, up and down all use
`n_slot = n_tok * n_used` over the same `selected`, and the down leg runs on
the same pair indices.

## Change

`p36_vk_moe_matvec` takes `p36_gpu_tensor **tiles_handoff, uint32_t
*tile_count_handoff`: the first of the three builds the schedule and hands
ownership to the caller, the other two borrow it and skip their build
dispatch.  The caller (the routed-FFN decode path) offers the handoff only
when `stream_slots` is off — with streaming enabled each part carries its own
cache-slot map, so the schedule genuinely differs.

Bindings inside the matvec now use `tiles_use` (borrowed or built) while the
existing frees still track `tiles` (owned); a handoff sets `tiles = NULL`, so
every error path keeps freeing what it built and the success path transfers.

## Result (MiniPlus Q3_K, interleaved pairs, 5 runs each, baseline 98b6bb4)

    arm        runs (t/s)                median
    baseline   48.84 49.01 48.82 48.23 48.32   48.82
    reuse      51.49 50.91 50.82 51.06 50.74   50.91   +4.3%

Per token: 20.48 -> 19.64 ms, -0.84 ms.  Predicted from dispatch bubbles and
kernel time was -0.3 ms; the rest is the per-call scratch allocation,
descriptor write and hazard bookkeeping that the two redundant builds also
paid.

Profiler: `moe_tiles` dispatches 1200 -> 400 over the same window, kernel time
2.18 -> 0.70 ms, total dispatches 13170 -> 12370.

md5 (--temp 0 -n 64 "The capital of France is"): `55f555660f6158501394ff16bde68faa`,
unchanged.
