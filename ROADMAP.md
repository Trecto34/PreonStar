# Roadmap

To-dos at a glance. Details, measurements and acceptance criteria live in
`roadmap/`. Performance verdicts still go to `karpathy/AlreadyTried.md`.

## f16 attention — [roadmap/attention-f16.md](roadmap/attention-f16.md)

- [x] Step A: f16 K/V staging in prefill FA LDS (`670febe`, default on)
- [x] Step B: packed-f16 PV math (`96a78f5`; kernel -10%, MoE +1.45% at 8k)
- [ ] Step C: packed-f16 QK math — **delayed**: f16 products push error > 2e-3 for a -7% kernel; needs an f32-accumulating dot2
- [x] Step C.1 (`78bb5d6`): dense gqa6 VGPR relief — 2 rows/lane: 256→128 VGPRs, kernel −11..−14%, bit-identical; whole model +0.4% at 16k
- [x] f16 K/V staging in `attn_decode_fa` (`4328c72`, default on; kernel −19% MoE 32k / −33% dense 16k, MoE 16k decode +2.13%)
- [x] Whole-model prefill at 16k+ (`724f250`; dense 0..16k attention 13.1% of GPU; MoE Axion `0..16k` +6.33%, `16k..32k` +12.87%)
- [x] NLL check on the other FA-path models (`a343f1a`; `Apex`, `Ornith`, `Qwen3.6-35B-A3B`: 0 non-finite, drift in line with f32 reorder control)
- [x] Update README long-context table (`532c53d`; Axion 8k/16k/32k prefill 803/690/537 t/s, decode 82.6/76.6/64.9 t/s)
- [ ] Confirm `v_fma_mix_f32` in the ISA
- [ ] Double buffering (low value; revisit if load time grows)

## Workload-specific MoE expert pruning — [roadmap/moe-expert-pruning.md](roadmap/moe-expert-pruning.md)

- [ ] Complete the routing profiler (resident path, router weight, mergeable)
- [ ] Diagnostic expert mask `P36_MOE_EXPERT_MASK`
- [ ] Offline pruning tool `gguf-tools/prune-experts`
- [ ] Engine audit for `expert_count != 256`

## Decode synchronisation — `karpathy/AlreadyTried.md` (2026-11-17 hazard barrier)

- [x] Hazard barrier carries TRANSFER masks only when a copy is pending (`6c5f966`; +3.98% MoE / +1.35% dense decode)
- [x] Dst-only barrier form (`234b660`; 1.90 vs 2.25 us/barrier; +1.55% MoE decode) — relies on GFX10 L2 coherence, re-check after Mesa upgrades
- [x] Last-workgroup ticket on the MoE down sum — **rejected**, -2.14% decode (per-workgroup atomic costs more than the 80 barriers it removes); router matvec -> top-k ticket untried
- [x] Counter-chained dispatches / persistent per-layer kernel / router ticket — **rejected by probe** (25 / 6.5 / 27.7 us per link vs 3.7-4.3 for dispatch + barrier)

## Decode kernels

- [x] Q4_K decode at 1 row per workgroup (+1.37% MoE decode, bit-identical; dense neutral)
- [x] IQ3_XXS MoE down sum-decode on the gate/up lane map (+4.70% MoE decode, reordered sums, NLL-checked)
- [x] Q3_K down: bytes preloaded + 4 rows per workgroup instead of the lane map (-60% kernel, +0.72% MoE decode, bit-identical)
- [x] IQ3_XXS MoE down sum-decode, 4 rows per workgroup sharing activations (+4.20% MoE decode, bit-identical)
- [x] Activation sharing for the IQ2_S / IQ2_S+IQ2_XXS gate-up decode kernels (+1.76% MoE decode, bit-identical)
- [x] `add_rms_norm_q8_k` fp64 divisions on one wave (-30% kernel, +0.78% MoE decode, bit-identical)
- [x] Decode after a chunk-1024 prefill 23% slow: scratch pool kept prefill-sized buffers, decode allocated host memory every token — fixed (decode 76 -> 99 t/s at chunk 1024, bit-identical)
- [x] Same fp64 hoist in the other small decode kernels — **closed**, no second case (route_reduce is latency, not fp64: 64-thread groups 0%, native fma -7% kernel)

### Next decode items and their gates

Every item: a kernel-level win in a 63-token `P36_VK_PROF_KERNEL` window (2 runs
per arm, same binary or `P36_SHADER_DIR`), then `bench_ab.sh` 3 interleaved reps
(ctx 1024, gen 64) with clean decode separation; stop and reject on a >5% first
pair. **Bit-identical** items need byte-identical `--dump-logprobs` vs HEAD on
Axion + one more NanoPlus file; `5008064` and `582e1ff` were kept on kernel win +
clean separation with the decode median under 1.25% (**owner to confirm** that
as the bit-identical gate). **Reordered-sum** items also need the frontier NLL
check (`3ade95c` protocol) and the 1.25% median. No LFM2 runs; 14 GB MiniPlus
files only with memory headroom.

- [x] Fuse `moe_down_route_reduce` into `ffn_tail` — 40 dispatches/token removed (`moe_down_route_reduce_tail`, −20.7% combined kernel GPU time, +0.65% Axion decode, 6/7 pairs positive, bit-identical on Axion + Qwen3.6)
- [ ] IQ3_XXS gate/up decode, 4 rows per workgroup — 4 layers on every NanoPlus file, 175 us/token at ~148 GB/s, est. −15% kernel. Gate: bit-identical; kernel win only (≈0.2% decode is below A/B resolution).
- [ ] Q3_K gate/up decode, preloaded bytes as `582e1ff` — 20 layers on MiniPlus, 8 on LFM2.5. Gate: bit-identical. **Blocked:** no runnable test model (LFM2 off-limits, MiniPlus has no memory headroom); needs an owner call.
- [ ] IQ2_S / IQ3_S down (Qwen3.8-IQ2_M: 37 / 3 layers) — preloaded bytes first (bit-identical gate); the lane map only if that falls short (NLL gate).
- [x] Q4_K trunk decode — read precomputed `q8_K` `bsums`, `unpackHalf2x16` d/dmin, direct word scale unpack, stream `q8` words for `ROWS=1` (`dense_q4k_decode_r1` −16.1% kernel, Axion decode **103.89 → 108.34 t/s, +4.28%**, clean separation, bit-identical on Axion + Qwen3.6)
