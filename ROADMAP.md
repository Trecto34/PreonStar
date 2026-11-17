# Roadmap

To-dos at a glance. Details, measurements and acceptance criteria live in
`roadmap/`. Performance verdicts still go to `karpathy/AlreadyTried.md`.

## f16 attention — [roadmap/attention-f16.md](roadmap/attention-f16.md)

- [x] Step A: f16 K/V staging in prefill FA LDS (`670febe`, default on)
- [x] Step B: packed-f16 PV math (kernel -10%, MoE +1.45% at 8k)
- [ ] Step C: packed-f16 QK math (riskier; only after B)
- [ ] Dense gqa6 VGPR relief via f16 Q/accumulators
- [ ] f16 K/V staging in `attn_decode_fa` (~15% of a 32k MoE decode token)
- [ ] Whole-model prefill at 16k+ (dense and MoE, one short run)
- [ ] NLL check on the other FA-path models
- [ ] Update README long-context table
- [ ] Confirm `v_fma_mix_f32` in the ISA
- [ ] Double buffering (low value; revisit if load time grows)

## Workload-specific MoE expert pruning — [roadmap/moe-expert-pruning.md](roadmap/moe-expert-pruning.md)

- [ ] Complete the routing profiler (resident path, router weight, mergeable)
- [ ] Diagnostic expert mask `P36_MOE_EXPERT_MASK`
- [ ] Offline pruning tool `gguf-tools/prune-experts`
- [ ] Engine audit for `expert_count != 256`
