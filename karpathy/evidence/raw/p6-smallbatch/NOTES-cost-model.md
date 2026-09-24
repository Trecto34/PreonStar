# P6/P7 running notes — why MTP cannot pay yet (2026-09-24)

Working notes; final wording lands in `AlreadyTried.md` + `CAMPAIGN.md` §6.8.
Numbers from the P2 arms in `evidence/raw/p2-mtp/`; per-kernel profiles under
`evidence/raw/p6-smallbatch/`.

## Reading the histogram

`q36_session_eval_speculative_argmax` returns `1 + commit_n`
(`q36.c:12483`), so `q36-bench`'s `histogram=k:count` (`q36_bench.c:701`) counts
**tokens committed per spec call**, not rows verified:

- bucket 1 = the call returned before verifying (backoff, gate miss, EOS);
- bucket k>=2 = the call verified and committed k-1 drafts.

Check: 168 + 2x172 = 512 for `p2-m0-d2`; 124 + 2x44 + 3x100 = 512 for
`p2-m0-d3`.

## The cost model at draft_cap = 1 (`--mtp-draft 2`)

Two Swift arms, greedy, ctx 512, gen 512, `--mtp-margin 0`, same prompt:

| arm | calls | bucket 1 | bucket 2 | gen s | t/s |
|---|---|---|---|---|---|
| `p2-m0-d2` | 340 | 168 | 172 | 24.22 | 21.14 |
| `p2-m3-d2` | 452 | 392 |  60 | 24.29 | 21.08 |
| plain `p2-off` | - | - | - | 22.63 | **22.63** |

Solving: **bucket 1 = 47.5 ms per committed token**, **bucket 2 = 94.4 ms per 2
committed tokens** (47.2 ms/token), against **44.19 ms for a plain step**.

That is the whole story at draft depth 1, and it is structural, not a bug: with
`draft_cap = 1` the verify row is a *new position*, so accepting the draft costs
a second full model forward.  Two forwards for two tokens is break-even at best,
and the piggybacked draft head (+3.3 ms) makes it a loss.  Depth-1 MTP can never
pay; it can only be made not to hurt.

## Depth 2+

`--mtp-draft 3` (draft_cap 2) = **4.17 t/s**, `--mtp-draft 4` (draft_cap 3) =
**4.44 t/s**, while committing 1.910 / 1.673 tokens per call and prefill staying
at 187 t/s.  The two bucket-2/bucket-3 equations imply ~1.1 s per call that
commits 3, i.e. ~0.4 s per token, 8-9x a plain step; the *direct* evidence is
the per-kernel decode profile in `p6-prof-d3` (dispatches + `gpu_ms` per op),
which is what this write-up quotes.

Mechanism, from the dispatch code: for IQ3_XXS only `n_tok == 1` is offered a
decode kernel (`q36_vulkan.c:8687-8730`); `n_tok >= 2` takes
`dense_iq3_xxs_mmq` with `ceil(n_tok/128)` workgroups in y, i.e. the 128-token
prefill tile, plus a `predequant_b16` pass.  `q36_vk_micro_batch` exists but is
only wired for Q8_0 (`q36_vulkan.c:5173`, `matmul_q8_0_f32b_nx`), which is the
audit's P6.

## What the P6 kernel is worth

At draft depth 1 nothing.  At depth 2, the measured second-draft acceptance is
100/144 = 69.4%, so a call that drafts both commits 1 + 1 + 0.694 = 2.694 tokens
on average.  If the two verify rows shared one weight pass (the MoE sum-decode
shape, or `dense_extra_small_q2`), a 2-row verify would cost ~1.2 plain steps:
2.694 tokens per ~53 ms = **~50 t/s against 22.63 (2.2x)**, minus whatever the
backoff and the draft head still cost.

## P7 fixes, with the measured target each attacks

1. **No per-draft logits readback.**  `q36_mtp_eval_vulkan` ends with
   `q36_gpu_tensor_read(rt->mtp_logits, 0, s->mtp_logits, Q36_N_VOCAB * 4)`
   (`q36.c:9607-9611`) = **993 KB per draft** (vocab 248320) followed by a CPU
   `sample_argmax_margin` (`q36.c:12419-12421`).  The main path already keeps
   `top2` on the GPU (`s->gpu_top2`, `q36_gpu_top2_tensor`, `q36.c:9377`); the
   draft head needs the same, with the margin, on the device.
2. **Reduced-vocab draft head.**  The draft logits come from the full
   `weights.output` projection (`q36.c:9599-9606`), so every draft pays a whole
   Q5_K lm_head.  Drafting over the top-N lm_head rows only changes the draft
   distribution, so acceptance has to be re-measured afterwards; verification
   stays full-vocab and the emitted tokens are unchanged by construction.
3. **Per-row recurrent-state capture.**  A partial accept rolls the whole
   DeltaNet frontier back (`q36_vulkan_spec_frontier_restore`, `q36.c:9440`) and
   replays `commit_n` tokens as a micro-batch (`q36.c:12456-12462`); keeping one
   saved state per verify row replaces the replay with a select.
4. **P6's kernel for the verify step** — the only item that moves the structural
   cost above.

## Backoff

`mtp_backoff_len` doubles 1 -> 2 -> 4 on a gate miss (`q36.c:12402-12406`), and
each miss costs up to four draftless tokens: at margin 0 only 172 of 340 calls
verifed a draft, so the effective tokens/call (1.506) is far below the 2.694 the
acceptance rates imply.  Measure any of the above with the backoff disabled or it
is measured through a ~50% attenuator.
