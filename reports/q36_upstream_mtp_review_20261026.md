# Upstream q36 MTP: what it does, what p36 does differently

Reviewed upstream `Ninnix/q36` at `fe48cb9` (2026-10-26) against p36 `c57a130`.
Paths below are upstream (`q36.c`, `q36_vulkan.c`, `q36_mtp.h`) unless marked
local. Focus: the speculative (MTP) loop, not the draft model itself.

## How upstream runs one speculative cycle

`q36_session_eval_speculative` (`q36.c:12057-12250`) is a single-forward loop:

1. **`verify[0] = first_token`**, at `start_pos = checkpoint.len`. The target
   forward of the pending token *is* verify row 0. There is no separate
   "decode the pending token first" step.
2. Drafts are appended to `verify[]` by recursive MTP-head calls, using
   `rt->last_h` for the first draft and `rt->mtp_cur` after that.
3. One batched forward over `verify[0..n)` produces per-row logits and
   per-row top-2 on the GPU (`q36_vulkan_verify_suffix_tops`, `q36.c:9091`).
4. Commit the longest agreeing prefix; reuse `row_tops[commit_n-1]` as the
   next token; `mtp_catchup` (`q36.c:9277`) advances the MTP head over the
   committed rows from saved target hiddens, so the next cycle starts warm.
5. Partial commit restores/replays only the committed recurrent rows.

Caps and gates:

- `mtp_cap` is hard-capped at **4 rows on Vulkan** (`q36.c:12104`), buffer
  max 8 (`Q36_MTP_VERIFY_MAX`, `q36.c:75`).
- If the target's top logit beats the first token by **> 1.0**, `mtp_cap` drops
  to 2; an uncertain prefix keeps exactly one probe row (`q36.c:12114-12123`).
- `--mtp-margin` > 0 switches to margin-based gating and disables lookup.
- Prompt lookup (`q36_mtp.h`): reuse an earlier continuation only when the
  suffix match is **>= 12 tokens and unambiguous**, scanning at most 64k of
  history; it can fill the remaining draft rows with zero head cost.
- Greedy only. `excluded_id` is threaded through draft and accept; no
  positive-temperature mode.

## Two upstream-only tricks

**Sparse draft head.** `vulkan/mtp_head_mask.comp` builds one bit per 64-token
vocabulary block from the target's per-row logits: keep a block when its peak
is within 8.0 of that row's top logit, plus a 4-cycle TTL for recent
alternatives. `q36_gpu_mtp_head_tensor` (`q36_vulkan.c:7626`) then runs
format-specific head kernels (`mtp_head_q8`, `_q5k`, `_iq4_xs`) that compute
only the unmasked blocks, with the scale folded in. Requires subgroup 64 and
GPU argmax; disabled by `Q36_MTP_HEAD_MASK=0` / `Q36_VK_ARGMAX=0`.

**Draft sampling without a vocabulary readback.** `q36_mtp_pick`
(`q36.c:9295`) reads two int ids plus two floats from a GPU top-2. The full
993 KB logits copy is the fallback, not the path. A cycle costs one 2-int
read, not several vocabulary drains.

**Compact head.** Since `d5b4e4b` the release GGUF carries an embedded MTP head
(`mtp_embedded`), so `--mtp` works without a second file.

## p36 differences that matter

| | upstream q36 | p36 (`c57a130`) |
|---|---|---|
| Target forwards per cycle | 1 (verify includes pending token) | **2**: `p36_session_eval_with_draft(first_token)` then `verify(drafts)` (`core_18_session_free.inc:727`, `core_19_mtp_timing_report.inc:200`) |
| Pending-token path | folded into verify row 0 | 1-row forward + **993 KB logits read** + host argmax, every cycle |
| Verify rows | <= 4 on Vulkan | up to **16** (`P36_MTP_MAX_DRAFT`); MoE verify above 4 rows drops off the 2-4-row pair kernels to serial per-row, dense keeps nx (tuned to r4) |
| 2-4-row verify kernels | dedicated small-M verify | nx kernels since `c57a130` (pair path gated to `n_tok > 4`), GPU top-2 for all rows, single small read |
| Draft head | block-masked, 3 quantized kernel variants, GPU top-2 | full-vocab generic matvec, host readback per draft |
| Head state | maintained across cycles via `mtp_catchup` | recomputed per cycle from `mtp_norm`/`last_h` |
| Excluded tokens | threaded through draft + accept | not in the spec path |
| Lookup draft | n-gram, >= 12-token unambiguous suffix, 64k scan | `p36_lookup_propose` with `--lookup-order`/`--lookup-k` (newest match, simpler) |
| MTP weights | embedded compact head in release GGUF | separate MTP GGUF; `--strip-nextn` needed for foreign files |
| Stats | calls/drafted/accepted/full + lookup per depth, atexit | `P36_MTP_TIMING` per-cycle phase line |

Both do: strict argmax acceptance, top-2 on GPU, snapshot/restore of recurrent
state on partial accept, greedy-only gating, and a shared commit path for MTP
and prompt-lookup drafts.

## Why upstream wins and p36 loses

p36's last published MTP measurement (`karpathy/AlreadyTried.md:2062-2110`,
2026-09-25): plain 22.68/23.04 t/s, draft 2 **0.92x**, draft 3 **0.30x**;
cycle breakdown `fwd` 51.70 ms, `draft` 4.84, `snap` 0.13, `verify` **391.03**
(max 482), total 462.69 ms; 4850 KB read and 3 queue drains per cycle.

The dominant term was the 2-row verify running MMQ pair kernels
(`dense_iq3_xxs_mmq_pair` +104 ms, `dense_kquant_mmq` +70, `dense_iq4_xs_mmq`
+40, `dense_iq3_xxs_mmq` +30 over the nx baseline). `c57a130` changed that
selection (pair path now `n_tok > 4`, verify asks for logits of all rows and
does top-2 on device), so the 391 ms figure predates the current build and
must be re-measured before any port decision.

What still differs regardless of that fix:

1. **One extra 1-row target forward per cycle** (~52 ms at ctx 512). Upstream
   never pays it. Removing it is a loop restructure, not a kernel change:
   seed the first draft from `last_h` and let verify row 0 be the pending
   token.
2. **Verify cap 16 vs 4.** With `--mtp-draft 8` or more, p36 verifies more
   rows than its 2-4-row kernels serve, then falls back to serial. Upstream
   refuses to draft past 4 on Vulkan.
3. **Draft head cost.** 4.84 ms/cycle in p36, so the masked head buys at most
   ~1% of a cycle. Not the lever.
4. **Readbacks.** 4.85 MB and 3 drains per cycle in the old measurement;
   upstream's cycle reads two ids and two floats per draft. Worth a few ms and removes queue
   stutter, but again not the main term.

## Port order

1. Re-measure MTP on `c57a130` with `P36_MTP_TIMING` first; the 391 ms verify
   number is stale. If 2-row verify is now ~80 ms, the remaining loss is the
   extra forward and the row cap.
2. Cap verify at 4 rows (one-line clamp, mirrors upstream) so `--mtp-draft 8`
   cannot silently become serial verification.
3. Try the single-forward cycle as an experiment branch: `verify[0] =
   first_token`, first draft seeded from the previous `last_h`, catch-up head
   over committed rows. This is the real structural gap.
4. Low priority: exclude-token plumbing, masked head, embedded head.

Upstream's own MTP win (90.63 -> 151.22 tok/s on 35B-A3B, 27.07 -> 36.44 on
27B at 2k) comes from the combination of 1, 2 and its MoE verify kernels, not
from the sparse head.

## Addendum: why the 391 ms number is stale, and why the loop still loses

No GPU device nodes exist in this sandbox (`/dev/dri` absent despite the
BC-250 being the host), so the re-measurement has to run on the box itself.
Static routing says the Sep figure should not be reused:

- The 128-row pair/MMQ tile is now gated to `n_tok > 4`
  (`src/core/core_15_forward_tokens_cpu.inc:368`), so a 2-4-row verify falls
  through to the nx kernels that stream each weight once per batch.
- The `serial_qkv` / `serial_attn` / `serial_recur` paths are diagnostic only,
  behind `P36_VK_SERIAL_*`; they are not taken by default
  (`core_15_forward_tokens_cpu.inc:839,1129,1181`).
- Verify already asks for logits of all rows and does top-2 on device
  (`core_15_forward_tokens_cpu.inc:1875-1890`), so the old
  "five 993 KB readbacks per cycle" is at most one readback now (the pending
  token in `p36_session_eval_with_draft`, `core_18_session_free.inc:748`).

But removing the kernel penalty does not fix the loop. Per cycle p36 pays:

    t_cycle = t_fwd(pending token) + t_verify(drafts) + t_draft_head + t_snap

with `t_fwd` a full one-row target forward. Accepted tokens per cycle are
`1 + commit_n`, typically 1.9-2.4. Even in the best case where a 3-draft
verify costs only ~1.2-1.4 one-row forwards, that is ~2.4 forwards for ~2.2
tokens: break-even at best. The measured 0.92x at draft 2 is exactly this
shape once the multi-row verify penalty is added. Upstream avoids it by making
verify row 0 the pending token, so the same acceptance rate costs one forward.

Consequence: the single-forward restructure is the necessary change, not the
4-row clamp and not the masked head. The clamp only stops `--mtp-draft 8+` from
serializing MoE verify; the head mask saves ~1% of a cycle.

Sketch of the restructure (not applied):

1. `verify[0] = first_token` at `start_pos = checkpoint.len`; do not call
   `p36_session_eval_with_draft` first.
2. Seed the first draft from `last_h` (hidden before `first_token`) instead of
   `rt->norm` after forwarding it; p36's head currently consumes the hidden at
   the same position, upstream's consumes the previous one. Confirm acceptance
   does not regress before trusting the swap.
3. Commit path must not push `first_token` twice: it is already in `verify[]`.
4. Keep `mtp_full` as is; the head rewrites its KV at the same positions each
   cycle, so no new rollback mechanism is needed.

Run on the host:

    ./download_model.sh swift          # embedded MTP head, no --mtp file
    P36_MTP_TIMING=1 P36_SWIFT_MTP_DRAFT=3 P36_SWIFT_MTP_MARGIN=0 \
        P36_SWIFT_CTX=512 ./run_swift_iq3_xxs.sh -n 128 --seed 42

Read `verify` and `fwd` from the per-cycle lines. If `verify` is now ~50-80 ms
rather than ~391 ms, proceed with the restructure; if it is still hundreds of
ms, the 2-row kernel path is still wrong and that comes first.
