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

- [x] Hazard barrier carries TRANSFER masks only when a copy is pending (+3.98% MoE / +1.35% dense decode)
- [ ] Dst-only barrier form (1.90 vs 2.25 us/barrier in the probe) — spec-loose, relies on GFX10 L2 coherence; owner decision
