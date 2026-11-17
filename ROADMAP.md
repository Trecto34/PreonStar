# Roadmap

To-dos at a glance. Details, measurements and acceptance criteria live in
`roadmap/`. Performance verdicts still go to `karpathy/AlreadyTried.md`.

## f16 attention — [roadmap/attention-f16.md](roadmap/attention-f16.md)

- [x] Step A: f16 K/V staging in prefill FA LDS (`670febe`, default on)
- [x] Step B: packed-f16 PV math (`96a78f5`; kernel -10%, MoE +1.45% at 8k)
- [ ] Step C: packed-f16 QK math — **delayed**: f16 products push error > 2e-3 for a -7% kernel; needs an f32-accumulating dot2
- [x] Step C.1 (`78bb5d6`): dense gqa6 VGPR relief — 2 rows/lane: 256→128 VGPRs, kernel −11..−14%, bit-identical; whole model +0.4% at 16k
- [x] f16 K/V staging in `attn_decode_fa` (default on; kernel −19% MoE 32k / −33% dense 16k, MoE 16k decode +2.13%)
- [ ] Whole-model prefill at 16k+ (dense done 2026-11-17: attention only 13% of GPU time at 0..16k; MoE still open)
- [ ] NLL check on the other FA-path models
- [ ] Update README long-context table
- [ ] Confirm `v_fma_mix_f32` in the ISA
- [ ] Double buffering (low value; revisit if load time grows)

## Workload-specific MoE expert pruning — [roadmap/moe-expert-pruning.md](roadmap/moe-expert-pruning.md)

- [ ] Complete the routing profiler (resident path, router weight, mergeable)
- [ ] Diagnostic expert mask `P36_MOE_EXPERT_MASK`
- [ ] Offline pruning tool `gguf-tools/prune-experts`
- [ ] Engine audit for `expert_count != 256`
