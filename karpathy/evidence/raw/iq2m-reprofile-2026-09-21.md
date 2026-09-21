# IQ2_M re-profile after 1b (K-quant gate widening) + 1c (IQ3_S down projection)

`karpathy/HANDOFF-iq2s-20260920.md` §4's profile predates both fixes landed
today (`e6c15ba`, K-quant gate widening; `5c53253`, IQ3_S down projection,
Mcode). Per HANDOFF §5 item 4 ("re-profile prefill with the current
binary"): fresh differential profile on current HEAD (`5c53253`), same
method HANDOFF §4 used so the two are directly comparable.

Model `Qwen3.8-35B-A3B-IQ2_M.gguf`. Binary `q36-bench` built at HEAD
`5c53253` (verified current via `make q36-bench` — "up to date").

## Method

- Decode: `Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1 ./q36-bench --vulkan -m
  <model> --prompt-file tests/long_context_story_prompt.txt --ctx-start 512
  --ctx-max 512 --ctx-alloc 640 --prefill-chunk 256 --gen-tokens {1,65}`,
  two separate runs (profiler counters are cumulative from process start),
  per-op `dispatches`/`gpu_ms` diffed and divided by 64 to isolate one decode
  token — identical to HANDOFF §4's method.
- Prefill: same flags, `--gen-tokens 0` (prefill-only, no decode/snapshot),
  single-shot totals at ctx 512 and ctx 1024 — the K-quant fix (1b) touched
  the `n_tok > 1` mmq branch too, so both frontiers are checked.

## Decode: 22.235 ms/tok (HANDOFF) -> 11.902 ms/tok (now), -46.5%

| ms/tok | share | op | disp/tok | note |
| --- | --- | --- | --- | --- |
| 2.498 | 21.0% | `moe_iq2s_down_sum_decode` | 37 | tuned/existing (W1) |
| 2.074 | 17.4% | `dense_extra_decode` | 258 | **new #2 — see root cause below** |
| 1.441 | 12.1% | `moe_iq2s_gate_up_decode` | 40 | tuned/existing (W1) |
| 1.192 | 10.0% | `dense_q4k_decode` | 40 | new from 1b — 40x `attn_qkv` Q4_K |
| 1.057 | 8.9% | `dense_q5k_decode` | 1 | new from 1b — `output.weight` Q5_K |
| 0.781 | 6.6% | `attn_decode_split` | 10 | |
| 0.411 | 3.4% | `dense_f32f_d_256x2048` | 40 | |
| 0.386 | 3.2% | `q8_k_quant` | 191 | |
| 0.378 | 3.2% | `add_rms_norm` | 80 | |
| 0.340 | 2.9% | `router_topk` | 40 | |
| 0.303 | 2.5% | `delta_net_decode` | 30 | recurrent layers |
| 0.197 | 1.7% | `moe_iq3s_down_sum_decode` | 3 | from 1c |
| 0.192 | 1.6% | `dense_iq3_s_decode` | 12 | |
| rest | 3.5% | (24 ops, each <1.1%) | | |

`dense_kquant` / `matmul_kquant` (48.2% in HANDOFF §4, the old #1) **does not
appear at all** — confirms 1b eliminated it, not just shrank it. The two new
kernels it was replaced by (`dense_q4k_decode` disp=40, `dense_q5k_decode`
disp=1) match the K-quant fix's own commit message exactly (40x `attn_qkv`
Q4_K + 1x `output.weight` Q5_K).

## Prefill: 3666 ms (HANDOFF, ctx 512, "before the decode pair") -> 862 ms now, -76.5%

ctx 512:

| ms | share | op | disp | note |
| --- | --- | --- | --- | --- |
| 376.65 | 43.7% | `moe_iq2s_gate_up_gemm` | 80 | tuned/existing (W1) |
| 144.63 | 16.8% | `dense_extra_mmq` | 516 | **new #2 — see root cause below** |
| 119.69 | 13.9% | `moe_iq2s_down_gemm` | 74 | tuned/existing (W1) |
| 68.03 | 7.9% | `dense_kquant_mmq` | 80 | new from 1b, now small |
| 27.03 | 3.1% | `dense_f32f_p_256x2048` | 80 | |
| 24.48 | 2.8% | `moe_iq3s_down_gemm` | 6 | from 1c |
| 21.71 | 2.5% | `attn_prefill_qtile2` | 20 | |
| rest | 9.2% | (8 ops, each <2.1%) | | |

ctx 1024 (same ranking, totals double as expected): `moe_iq2s_gate_up_gemm`
706.50 ms (42.2%), `dense_extra_mmq` 275.48 ms (16.5%), `moe_iq2s_down_gemm`
241.64 ms (14.5%), `dense_kquant_mmq` 127.11 ms (7.6%), total 1672.22 ms.

Total prefill GPU time collapsed 3666 -> 862 ms at ctx 512 (-76.5%) —
combines both 1b and 1c; not a clean single-fix attribution, but consistent
in direction and magnitude with the two landed A/B results (+107.85% and
+11.91% median throughput respectively — GPU-busy-time and wall-clock
throughput are different measures and won't reconcile exactly).

## The next lever: `dense_extra_mmq` / `dense_extra_decode` (~17% of both prefill and decode)

**What it is, confirmed:** `q36_vk_run_unlocked("dense_extra_decode"/"dense_extra_mmq", ...)`
has exactly one call site (`q36_vulkan.c:8366`), inside
`q36_gpu_matmul_iq_quant_q8_scaled_tensor`'s `extra_type` branch —
reachable only for `{IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ4_NL, PQ2_0, PTQ1_0}`.
This GGUF's only tensor of those types is **IQ2_S** (confirmed via
`gguf.GGUFReader`: 375 IQ2_S tensors, no IQ2_XXS/IQ2_XS/IQ1_S/IQ4_NL/PQ2_0/PTQ1_0
at all). So this is a **generic single-tensor IQ2_S matmul** competing with
the tuned, batched `moe_iq2s_*` kernels above it in the table — the same
"quant type falls through to a slow generic path" shape as the K-quant bug
(1b) and the IQ3_S route-miss (1c).

**Ruled out:** the MoE routed-expert route-miss fallback that explained the
old q8-route residue and that 1c fixed for IQ3_S. `Q36_VK_MOE_ROUTE_DEBUG=1`
on a 3-token run at ctx 512 produced **zero** "route miss" lines — the fused
f32 expert path (`q36_gpu_moe_ffn_f32_tensor`) is not missing for any layer
now, so `q36_gpu_moe_ffn_q8_tensor` (the explicit q8 fallback, gated on
`!routed_done`) never runs. Whatever is hitting `dense_extra_mmq/decode`
is not routed-expert traffic.

**Root cause, confirmed for a large fraction:** the **shared-expert**
gate/up/down projections (`ffn_gate_shexp`/`ffn_up_shexp`/`ffn_down_shexp`,
computed once per layer regardless of routing, `q36.c:8283-8307`). The
engine's only fast path for these (`q36_gpu_shared_ffn_decode_tensor`,
`q36.c:~8218`) requires all three tensors to be `Q8_0` — but this GGUF's
shared-expert tensors are IQ2_S on **38-40 of 41 layers** (confirmed via
`gguf.GGUFReader` tensor-type audit; only 1 layer is Q8_0). So for ~40 of 41
layers, all three shared-expert matmuls (`q36_gpu_tensor_matmul_q8_or_float_scaled`
at `q36.c:8283`/`8288`, `q36_gpu_tensor_matmul_scaled` at `q36.c:8305`) fall
through to the generic path — accounting for **~120 of the 258
dispatches/token** (40 layers x 3 tensors). The remaining ~138 dispatches/token
are not yet traced to a specific call site — next step if this is picked up:
instrument `q36_gpu_matmul_iq_quant_q8_scaled_tensor`'s IQ2_S branch with a
one-off caller tag, or check whether some routed-expert bookkeeping also
touches this path despite `routed_done=true`.

**Why this is a real lever, not a re-run of 1c:** structurally identical
opportunity to 1b/1c — a generic per-tensor quant fallback doing work a
specialized/batched kernel family already exists for nearby quant types.
`q36_gpu_shared_ffn_decode_tensor` (or a new IQ2_S-aware variant of it,
following the exact pattern 1c used for IQ3_S in `moe_down_gemm.comp` /
`moe_down_q2k_sum_decode.comp`) is the natural target. Not yet attempted;
no A/B run. `Q36_N_EXPERT=256`, `Q36_N_EXPERT_USED=8` if useful context for
sizing the fix.

## Evidence

Raw profiler dumps: `iq2m-reprof-gen1.txt`, `iq2m-reprof-gen65.txt` (decode
diff inputs), `iq2m-reprof-prefill512.txt`, `iq2m-reprof-prefill1024.txt`
(prefill single-shot), `iq2m-route-debug.txt` (`Q36_VK_MOE_ROUTE_DEBUG=1`,
zero misses) — copied into `karpathy/evidence/raw/`.
