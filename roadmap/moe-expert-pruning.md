# Workload-specific MoE expert pruning

**Status:** proposed

## Motivation

The target model routes each token through 8 of 256 experts per layer. A
deployment with a narrow workload (one language, a fixed tool set, a known task
mix) concentrates its routing on a subset of those experts, yet the full expert
set ships in the GGUF and competes for the ~10-14 GB of unified memory left
after the OS and KV cache.

Removing experts the workload rarely selects shrinks the model file and its
resident footprint, with no retraining. Decode bytes per token are essentially
unchanged (still 8 active experts), so this is a memory and model-size
feature, not a decode-throughput one.

## Current state

- `P36_VK_STREAMING_EXPERT_PROFILE` / `P36_VK_STREAMING_EXPERT_HOTLIST_OUT`
  already write a per-`(layer, expert)` selection histogram
  (`src/vulkan/vk_01_prof_reset.inc`). It records only on the SSD-streaming
  path, counts selections only (the `weight` column is always `0`), and is sized
  by the fixed `P36_VK_STREAM_PROFILE_EXPERTS = 256`.
- The expert count is read from GGUF metadata (`P36_N_EXPERT`) and validated
  against tensor shapes in `core_09_weights_bind.inc`; it is uniform across
  layers.

## Scope

1. **Complete the routing profiler** (diagnostic, no semantic change)
   - Record on the resident (non-streaming) MoE path as well.
   - Fill the existing `weight` column with summed post-softmax router weight;
     add the mean expert-output norm needed for REAP saliency.
   - Make profiles mergeable across runs so calibration can combine the eval
     suite with recorded production traffic.

2. **Diagnostic expert mask** (`P36_MOE_EXPERT_MASK=<file>`)
   - Forces listed experts' router logits to `-inf` before top-k, which
     reproduces the routing of a pruned model on the unmodified GGUF.
   - Exists only to validate the pruned release path before weights are cut,
     consistent with the rule against permanent semantic flags. No memory
     saving is expected here: weights are mmap-backed, so a masked expert's
     pages are simply not touched.

3. **Offline pruning tool** (`gguf-tools/prune-experts`)
   - Inputs: source GGUF, one or more profiles, and a policy (fraction to
     remove, or a saliency threshold). Saliency follows REAP
     (router-weighted expert activation) and is ranked per layer.
   - Removes the same number of experts from every layer, so the result keeps
     a single uniform `expert_count` that the engine already validates.
   - Slices `ffn_gate_inp` rows, `ffn_{gate,up,down}_exps` and the optional
     `*_exps_scale` tensors along the expert axis without re-quantization;
     shared-expert tensors are untouched. Records provenance (source hash,
     profiles, policy) in GGUF metadata.

4. **Engine audit for `expert_count != 256`**
   - Replace fixed-256 assumptions (`P36_VK_STREAM_PROFILE_EXPERTS`, the
     streaming hotlist, kernel workgroup tiling over experts) with
     `P36_N_EXPERT`.
   - Non-uniform per-layer counts are out of scope for the first iteration.

## Acceptance criteria

- Profiler disabled: no measurable prefill or decode change.
- An empty mask is bit-exact with the baseline.
- A pruned model and the masked baseline produce identical logits for the same
  prompt (same experts, same router renormalization).
- A pruned model passes `make test` and `p36-eval` on Vulkan.
- Reported for 25% and 40% removal: file size, measured RSS, and quality delta
  per task category on a held-out set not used for calibration.

## Risks

- **Distribution shift:** inputs outside the calibration set (other
  languages, coding, vision) degrade first. Pruned GGUFs should be labelled as
  workload-specific and never replace the general model.
- **Layer sensitivity:** a uniform count per layer leaves quality on the
  table; per-layer counts are the follow-up if measurements justify them.
- **RSS gain may be smaller than file-size gain:** with mmap-backed weights,
  cold experts are often not resident already. Measure RSS and page-cache
  pressure under the real workload before claiming a memory win.

## References

- Lasby et al., *REAP: Router-weighted Expert Activation Pruning* (Cerebras, 2025).
