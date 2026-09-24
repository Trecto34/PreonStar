# Campaign handoff — q36 inference optimization on BC-250

**Read this file first; it is the map.** If you have five minutes, read
`## Pick up in 5 minutes` and `## The rules that decide a verdict`, then go
straight to `## Open items, ranked`.

Last updated: 2026-09-23 (BC-250 session) · repo `/home/server/q36` + worktree
`/home/server/q36-wt/pq2-persist`, branch `experiment/pq2-smallbatch`
(uncommitted) · the 2026-09-21 notes below are from `/home/server/q36-opt-27b` ·
branch `trackB-ptq1_0`. PLAN items W2–W7 (`karpathy/PLAN-implementation-2026-09-20.md`)
are all closed; see the entries in this section and the matching
`AlreadyTried.md` rows. HANDOFF's ranked item #1
(`karpathy/HANDOFF-iq2s-20260920.md` §5) was **misdiagnosed** — the "IQ2_S
tuned dense decode" it proposed doesn't exist; the actual hotspot was IQ2_M's
Q4_K/Q5_K trunk tensors (`attn_qkv`/`output.weight`) missing a dense-only
kernel-selection gate. Fixed and landed this session (prefill +107.85% /
decode +59.61%, see §6.1 and `AlreadyTried.md`). W8 (long-context KV dequant
redundancy) measured its go/no-go positive but rejected the QT-widening
mechanism on a VGPR-ceiling probe before any build; a KV-scratch-cache
mechanism is real, unstarted work (see 4f). HANDOFF §5 item 3 (the q8-route
residue on IQ2_M) was **closed this session** by Mcode: the fused f32 expert
path now also accepts IQ3_S down projections (prefill +11.91% / decode +4.17%
at ctx 512, 7 interleaved reps — see 1c and `evidence/raw/iq3s-down-VERDICT.md`).
HANDOFF §5 item 2 (small-batch 2..31-token IQ2_S kernels) was **closed this
session** by the same instance that diagnosed it: the fused f32 expert pair now
takes IQ2_S at `n_tok` 2..127, route misses at `--prefill-chunk 16` fell
1280 -> 96, and prefill at that chunk rose 72.92 -> 116.21 t/s (+59.37%, see 1g
and `evidence/raw/iq2s-smallbatch-VERDICT.md`). Remaining open work: the W8
scratch-cache mechanism, plus PLAN's W9.

**2026-09-23 session (this worktree).** Two new things. (1) A new loadable file,
`TERNARY-BONSAI-2-27B-DERISKED-PQ2_0.gguf` — the DERISKED re-release of the
ternary model §6.6 could not open; it runs at 211.08 / 32.97 t/s at ctx 512
(§8). (2) The campaign's **first accepted kernel change on this file**: a
PQ2_0/Q2_0 **small-batch matmul** that is bit-exact against one-token decode and
takes `--prefill-chunk 2` from 2.95 to 54.60 t/s and a prompt tail by +45-72%
(§6.7). Two decode-side attempts were measured and **rejected** and are now
closed lines: PQ2_0 matvec dispatch shapes (§7) and GQA-grouped split-K decode
attention (§7) — plus a decode-variance pitfall that made the first sweeps of the
latter look like wins (§9). The `ssm_alpha`/`ssm_beta` fusion was not attempted:
the IQ2_S precedent for that fusion measured +1.42% on a MoE file, and this dense
file's ceiling for it is under 1%. Everything is uncommitted in this worktree per
the session's instruction; the ledger, this map and `evidence/raw/` are the record.

---

## 1. What this campaign is trying to do

Make the locally available Qwen models — dense **Qwen3.8-27B** and the
**Qwen3.6/3.8 35B-A3B MoE** family — run as fast as possible in *this* engine
(`q36`) on *this* box (BC-250), with no quality loss and no regression on the
MoE "guard" model.

Current standing, measured on both engines, ctx 1024:

| model | q36 prefill / decode | upstream llama.cpp | q36 delta |
|---|---|---|---|
| Swift Qwen3.8-27B IQ3_XXS (dense) | 171.09 / 23.19 | 105.43 / 22.13 | **+58.7% / +8.0%** |
| Qwen3.6-35B-A3B IQ2XXS (MoE guard) | 909.49 / 87.37 | 615.08 / 78.72 | **+47.9% / +11.0%** |
| RavenX-35B-Q36-IQ2XXS (MoE) | 717.36 / 89.47 | 545.46 / 77.96 | **+31.5% / +14.8%** |
| Qwen3.8-35B-A3B-IQ2_M (MoE) | **556.21 / 78.40** at ctx 512 (landed K-quant gate widening + IQ3_S down projection, this session; row above is ctx 1024) | 529.90 / **91.53** at ctx 1024 | **+5% / -14%** at comparable prefill scale — prefill now ahead, was -54%/-48%; decode still the gap, ranked next §6.1b |

The engine leads on every file it can load. It is already ~1.6x upstream on
dense prefill. So the remaining work is not "make it fast" — it is "close the
one file we can't open, and find the next real lever". **No kernel-level
optimization has been accepted this campaign**; the deliverable so far is a
trustworthy measurement base, a list of closed dead ends, and one identified
landed loader fix (`f9d537d`) that closed the one file q36 could not open — and,
in the process, proved that file's real gap is kernel *selection*, not loading.

## 2. The document set — what each file is, and when to touch it

There are three documentation layers. They do not overlap; use the map below
instead of searching.

| file | what it is | read when | write when |
|---|---|---|---|
| **`karpathy/CAMPAIGN.md`** (this file) | the map: state, rules, open items, commands | **first, every session** | whenever state changes |
| **`karpathy/AlreadyTried.md`** | the rejection ledger — the campaign's memory of dead ends. **Authoritative.** | **before proposing any change** | after every verdict (accept or reject) |
| `karpathy/AlreadyTried.active.md` | generated digest of the ledger | rarely | **never by hand** — `python3 karpathy/tools/build_active_ledger.py karpathy/AlreadyTried.md karpathy/AlreadyTried.active.md` |
| `karpathy/evidence/cross-engine-matrix.md` | q36 vs upstream table, every row traced to a raw file, with the asymmetries disclosed | before claiming "we are faster" | when new head-to-head numbers land |
| `karpathy/evidence/ab-hostdrain.md` | a complete worked experiment record — use it as the template for your own | when writing up an experiment | once per experiment |
| **`karpathy/evidence/raw/`** | raw CSVs and logs. **The only source that outranks a summary.** | when re-checking any verdict | immediately after a run, before `/tmp` is cleared |
| `karpathy/evidence/compatibility-gate.log` | gate output for the Swift + MoE load-compat check | before accepting a shader change | automatically (gate writes it) |
| `karpathy/compat_gate.sh` | runs that gate | before accepting a shader change | never |
| `tests/bench_ab.sh` | the interleaved A/B harness that produces every verdict | to measure anything | only to fix a bug in it (see §5) |
| `AGENT.md` | the repo's own rules for agents (layout, quality bar, hardware) | before touching engine source | when repo-wide rules change |
| `karpathy/README.md` | docs for the *separate* `~/Karpathy` autoresearch loop | only if you want the automated loop | rarely |
| `tests/REGRESSIONS.md` | the repo's pre-existing regression list | alongside the ledger | when you add a repo-level regression |

Also, outside the repo: the procedure layer lives in the Hermes skill
`software-development/q36-inference-optimization` (with
`references/bc250-q36-runtime.md`). That is where *how to work on this box*
belongs; the repo holds *what happened on this box*.

**Why this split.** The failure mode this set exists to prevent is a future
agent re-running a dead end because the reason it died was only ever in a chat
log, or quoting a number whose raw data is gone. Ledger = what not to retry.
Evidence = why we believe a number. Campaign map = where we are. Skill = how to
operate. Nothing else.

## 3. Pick up in 5 minutes

```sh
cd /home/server/q36-opt-27b
git log --oneline -6                 # campaign commits
git status --porcelain               # expect: dirty docs, or clean
pgrep -x q36-bench || echo "GPU idle"   # NEVER start a bench or build while one runs

# what the engines can/can't load, and the model set
ls -lL --block-size=M /home/server/q36/gguf/     # NOTE: some entries are symlinks
git worktree list                                # w1-w6; w6-drain is the rejected experiment
```

Then read `## 4` (rules) and `## 6` (open items) and you are current. Do **not**
start by reading `changelog.txt` or `AlreadyTried.md` end to end — the ledger
only needs reading before you propose a specific change.

**There is no committed engine binary.** `q36-bench` / `q36` are gitignored and
must be rebuilt from a pinned commit in their own worktree (§5).

## 4. The rules that decide a verdict

1. **Gate: prefill median gain ≥ 1.5%** (`Q36_AB_MIN_GAIN`), 7 interleaved reps,
   ctx 1024, `--gen-tokens 16`, `--prefill-chunk 256` on Swift.
2. **Noise floors — know them before you claim anything:**
   - dense 27B prefill: **0.7%** within a session
   - **MoE prefill: 3.4%** (identical code produced 713.98 → 738.81 t/s)
   - decode: spread up to **9.4%** — it needs many reps before it says anything
   ⇒ a MoE prefill delta under ~3% is noise, and the guard's real acceptance
   band is 3.4%, not 1.5%.
3. **Interleave A and B rep by rep.** Never run A completely then B: drift lands
   on the second binary and looks like a regression. That is what `bench_ab.sh`
   exists for. Report **median + MAD**, never means.
4. **Serialize GPU work.** One model process at a time (the instance lock is
   intentional), and **never build while a benchmark runs** — a background
   `make -j16` during a matrix run already corrupted one batch this campaign.
5. **A change gated behind a runtime flag must be A/B'd with that flag ON**, or
   not A/B'd for speed at all. Learned the hard way: the drain patch was gated
   on `q36_vk.prof_kernel`, so the two binaries *could not* differ and the
   experiment was null by construction (see `evidence/ab-hostdrain.md`).
6. **Read the gate before designing the experiment.** `grep` the flag/env var
   that guards the code you are about to change, and confirm it is live in the
   configuration you will measure.
7. **Quality is not negotiable.** Preserve MoE routing parity for the guard
   model. Do not keep a faster path with unexplained attention, KV or logits
   drift. Every candidate runs the compatibility gate.

## 5. Exact commands

```sh
# --- A/B a candidate binary against the known-good one (the standard verdict) ---
tests/bench_ab.sh <A_BIN> <B_BIN> /home/server/q36/gguf/<MODEL>.gguf 7 \
  > /tmp/ab-<slug>.csv 2> /tmp/ab-<slug>.summary
cp /tmp/ab-<slug>.csv karpathy/evidence/raw/

# --- one engine run, by hand ---
./q36-bench --vulkan -m /home/server/q36/gguf/<MODEL>.gguf \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 1024 --ctx-max 1024 --ctx-alloc 1296 \
  --prefill-chunk 256 --gen-tokens 16

# --- upstream llama.cpp head-to-head ---
cd /home/server/llama.cpp && git log --oneline -1     # 972d231, build 190
build/bin/llama-bench -m /home/server/q36/gguf/<MODEL>.gguf \
  -ngl 99 -p 1024 -n 16 -r 3 -o csv > /tmp/lb-<slug>.csv
# caveat: llama-bench `tg` is greedy; q36's decode carries the sampling chain.

# --- load/compat gate (Swift + MoE in separate GPU processes) ---
./karpathy/compat_gate.sh

# --- a clean worktree for a candidate (pitfall 6.7 needs the .spv copy) ---
git worktree add -b experiment/<slug> /home/server/q36-wt/<slug> 11bb345
cp -a vulkan/*.spv /home/server/q36-wt/<slug>/vulkan/
cd /home/server/q36-wt/<slug> && make -j16          # only when the GPU is idle
```

## 6. Open items, ranked

1. **Quant-aware dispatch for the `IQ2_S` / K-quant MoE — W1 LANDED & ACCEPTED (2026-09-20).**
   Landed in `6ef9ceb` (fused gate/up + GEMM pair) and `e8cb6de` (f32 identity decode pair).
   Prefill rose **108.21 → 241.27 t/s (2.23×)**, decode rose **29.52 → 47.14 t/s (1.60×)**
   (reproduced 245.55 / 48.68 at ctx 512). Guard is 100% byte-identical (`b46fa81a…`,
   `93a38508…`, `601d56a9…`, `946097d8…`, `3f316c16…`), frontier-513 parity verified.
   **1b. Dense K-quant fast path widened to MoE — LANDED & ACCEPTED (2026-09-21).**
   HANDOFF's own "ranked next step #1" (a new IQ2_S dense decode shader for
   `matmul_kquant`, 48% of decode) was misdiagnosed: parsing the GGUF header
   showed the 41 hot tensors are 40x **Q4_K** `attn_qkv` + 1x **Q5_K**
   `output.weight`, not IQ2_S. The engine already had tuned
   `dense_kquant_decode`/`dense_q4k_decode`/`dense_q5k_decode`/
   `dense_kquant_mmq` kernels for exactly this, gated on `q36_gpu_dense_model`
   (dense-launch-only, never widened for MoE). Dropping that gate (2 lines in
   `q36_vulkan.c`) plus widening the kernel prewarm to match: 7-rep
   interleaved A/B (ctx 512, gen 128) prefill **244.35 → 507.87 t/s
   (+107.85%)**, decode **47.34 → 75.56 t/s (+59.61%)**, gate ≥3.4% cleared by
   ~32x. Guard frontier-513 byte-identical (has no K-quant trunk tensors);
   IQ2_M frontier-513 top-1 preserved in every tested configuration, max_abs
   deltas same order/shape as HANDOFF's own accepted W1 calibration (uniform
   across all 248,320 logits, consistent with `output.weight` reassociation,
   not a localized bug). `./q36_test --vulkan-kernels` CPU-reference oracle
   passes and now actually exercises this path for the first time. Upstream
   gap nearly closed: 507.87/75.56 (ctx 512) vs llama.cpp 529.90/91.53 (ctx
   1024, not directly comparable but close).
   **1c. IQ2_M's q8-route residue — CLOSED & ACCEPTED (2026-09-21, Mcode).**
   HANDOFF §5 item 3's suspects (in-file MTP/`nextn` head, bank-cache-off
   layers) were wrong: `Q36_VK_MOE_ROUTE_DEBUG=1` showed the fused f32 expert
   path was missing on exactly `il=0,1,2`, whose `ffn_down_exps` are
   **IQ3_S×3** while the other 37 are IQ2_S — a mixed-quant file, not an MTP
   effect. Fixed by adding an `#ifdef Q36_MOE_IQ3S` branch (110-byte blocks,
   9-bit grid index) to the *existing* `moe_down_q2k_sum_decode.comp` /
   `moe_down_gemm.comp` rather than writing a new shader, plus the dispatch
   split in `q36_vulkan.c`. 7-rep interleaved A/B (ctx 512, gen 128) prefill
   **497.00 → 556.21 t/s (+11.91%, MAD 15.15/12.40)**, decode **75.26 → 78.40
   t/s (+4.17%, MAD 0.18/0.10)** — gate ≥3.4% cleared ~3.5x on prefill; the
   decode claim rests on non-overlapping reps, not a formal gate. Route-miss
   check now clean; frontier parity at 512/513 preserved (top-1 match, top64
   62-64), guard byte-identical, `q36_test --vulkan-kernels` + compat gate
   PASS. Evidence: `evidence/raw/iq3s-down-VERDICT.md`,
   `evidence/raw/ab-iq3s-down.{csv}` + summary, `evidence/raw/iq3s-down-parity*.txt`.
   **1d. Re-profile after 1b+1c — done (2026-09-21).** HANDOFF §5 item 4.
   Decode collapsed **22.235 -> 11.902 ms/tok (-46.5%)**; prefill (ctx 512)
   **3666 -> 862 ms (-76.5%)**. `dense_kquant`/`matmul_kquant` (old #1, 48.2%
   of decode) is gone from the profile entirely, replaced by `dense_q4k_decode`
   (disp=40) / `dense_q5k_decode` (disp=1) at a fraction of the cost, exactly
   as 1b's commit predicted. New ranking: decode led by
   `moe_iq2s_down_sum_decode` (21.0%, tuned/existing) then **`dense_extra_decode`
   (17.4%, 258 disp/tok)**; prefill led by `moe_iq2s_gate_up_gemm` (43.7%,
   tuned/existing) then **`dense_extra_mmq`** (16.8%). `dense_extra_mmq/decode`
   is IQ2_S-exclusive (single call site, `q36_vulkan.c:8366`, reachable only
   for IQ2_XXS/IQ2_XS/IQ2_S/IQ1_S/IQ4_NL/PQ2_0/PTQ1_0 — this file has only
   IQ2_S among those). `Q36_VK_MOE_ROUTE_DEBUG=1` shows **zero** route misses,
   ruling out the mechanism 1c just fixed. Root cause instead: the
   **shared-expert** gate/up/down (`ffn_*_shexp`) are IQ2_S on 38-40/41
   layers, but the only fast shared-expert path
   (`q36_gpu_shared_ffn_decode_tensor`) requires Q8_0 — so ~40 layers x 3
   tensors (~120 of the 258 disp/tok) fall through to this generic path;
   remainder not yet traced. **Next lever, not yet attempted:** teach the
   shared-expert fast path (or a new decode/mmq variant) to accept IQ2_S,
   same pattern as 1b/1c. Evidence: `evidence/raw/iq2m-reprofile-2026-09-21.md`
   (full ranked tables + method), `evidence/raw/iq2m-reprof-*.txt`,
   `evidence/raw/iq2m-route-debug.txt`.
   **1e. `ssm_alpha`/`ssm_beta` IQ2_S pair-fusion — ACCEPTED, small win
   (2026-09-21).** First of 1d's two-part plan (the small, overhead-bound
   half; the tuned-kernel half for `attn_gate`/`ssm_out`/shared-expert is
   unstarted). New `dense_extra_decode_iq2s_pair.comp` +
   `q36_gpu_matmul_iq2s_pair_scaled_tensor()` fuse `ssm_alpha`+`ssm_beta`
   (2048x32 each, tiny) into one dispatch sharing one Q8_K read instead of
   two, mirroring the existing `q36_gpu_matmul_q8_0_pair_scaled_tensor`.
   Parity bit-exact (`max_abs_diff=0.0`, 248,320 logits, top-1/top-64
   identical) — the dequant math is copied verbatim, just evaluated twice
   per block instead of once per dispatch. 21-rep interleaved A/B (ctx 512,
   gen 128; 7-rep was inconclusive, overlapping ranges): decode **78.15 ->
   79.26 = +1.42%** (MAD 0.190/0.270, right at the formal gate, 19/21 reps
   per arm in disjoint bands), prefill -2.01% (noise — fix is
   `n_tok==1`-gated, never touches prefill; well inside the 3.4% MoE floor).
   Gated on both tensors being exactly IQ2_S, inert by construction
   elsewhere (not separately A/B'd on the guard for that reason). A/B'd in
   isolated worktrees (`q36-wt/iq2s-ssm-pair{,-base}`) built from clean
   `b738610`, since a second instance had unrelated uncommitted work in the
   shared main worktree. Honest framing: real but modest, accepted on
   parity + mechanism + 21-rep separation rather than a clean gate pass.
   Evidence: `evidence/raw/ab-iq2s-ssm-pair{,-7rep}.csv`,
   `ab-iq2s-ssm-pair-summary.txt`, `iq2s-ssm-pair-parity.txt`.
   **1f. Tuned IQ2_S decode/mmq kernel (`attn_gate`/`ssm_out`/shared-expert)
   — ACCEPTED, real but small (2026-09-21).** Second half of 1d/1e's plan.
   Same lever as the existing `Q36_Q2_0_ONLY` compile-time specialization
   (cited "-19..28%" precedent): new `Q36_IQ2S_ONLY` `#ifdef` in
   `dense_extra_decode.comp`/`dense_extra_mmq.comp` (math copied verbatim
   from the existing IQ2_S branch, just without the runtime `pc.type`
   branch chain) -> `dense_extra_decode_iq2s.spv`/`dense_extra_mmq_iq2s.spv`.
   ISA checked before building anything real (`mmq_info`, W8-probe style):
   decode VGPRs 64->56 + occupancy 16->18 subgroups/SIMD (both improved, no
   cliff), mmq VGPRs 88->84 (LDS/occupancy unchanged) — no register-pressure
   risk. Parity bit-exact three ways (combined, prefill-only, decode-only),
   shader hashes of every pre-existing build variant (generic/Q2_0/PQ2_0)
   confirmed byte-identical. A real bug was caught before landing: the first
   draft's specialized mmq had `barrier()` inside the accumulation loop
   instead of after it (16x too many per slice, from a copy-paste slip) —
   found by re-reading the function side-by-side with the generic one, not
   by a failed test, matching the W3-precedent risk this item was flagged
   for going in. Whole-model 21-rep A/B was weak/ambiguous on its own
   (decode +0.52% with heavy overlap, prefill -2.02% inside the 3.4% floor
   but this fix *does* touch prefill unlike 1e). Resolved by kernel-level
   profiling instead of more reps: decode op cost 1.891->1.822 ms/tok
   (-3.7%, ~0.58% of whole-model decode — matches the noisy +0.52% almost
   exactly), and a single-shot prefill profile settles prefill directly:
   `dense_extra_mmq_iq2s` 132.151ms vs `dense_extra_mmq` 138.320ms (-4.46%),
   whole-prefill GPU time 808.025ms vs 825.393ms (-2.10%) — prefill is
   measurably *faster*, confirming the A/B's -2.02% was session noise.
   Why smaller than the ISA numbers suggested: `pc.type` is a push constant
   (uniform across a dispatch), so the removed branches were already cheap
   uniform/predicated branches, not per-thread-divergent ones — the
   remaining cost is the dequant arithmetic itself, left untouched to
   guarantee bit-exactness. That's the next lever on this kernel if anyone
   wants it; not attempted. Default on, `Q36_VK_IQ2S_EXTRA=0` disables.
   A/B'd in isolated worktrees (`q36-wt/iq2s-tuned-kernel{,-base}`) built
   from clean `44143d8`; two profiling runs had to `flock`-queue behind the
   same concurrent instance from 1e, still mid-flight on HANDOFF item 2.
   Evidence: `evidence/raw/ab-iq2s-tuned-kernel{,-7rep}.csv`,
   `ab-iq2s-tuned-kernel-summary.txt`, `iq2s-tuned-kernel-parity.txt`,
   `iq2s-tuned-kernel-shader-hashes.txt`,
   `iq2s-tuned-prof-{on,off}-{gen1,gen65,prefill}.txt`.
   Remaining open levers for IQ2_M: (1) small-batch 2..31 token f32 kernels
   (HANDOFF §5 item 2) — **closed by 1g**, landed by the concurrent instance;
   (2) the dequant
   arithmetic itself inside `dense_extra_decode_iq2s`/`dense_extra_mmq_iq2s`
   (1f's own `reconsider_if`) — the residue's generic-vs-tuned kernel gap is
   now closed, what's left is genuine ALU cost, not misrouting. Evidence:
   `karpathy/HANDOFF-iq2s-20260920.md`, `evidence/raw/iq2s-*`,
   `evidence/raw/kquant-moe-*`, `evidence/raw/ab-kquant-moe-decode*`,
   `AlreadyTried.md`.
   **1g. IQ2_S in the small-batch (2..127 token) fused MoE expert pair —
   ACCEPTED (2026-09-21).** HANDOFF §5 item 2, and the last known q8-route
   residue on IQ2_M outside the GEMM range. `moe_gate_up_f32b` /
   `moe_down_q2k_f32b` were IQ2_XXS/Q2_K-only builds, so the admission
   predicate rejected IQ2_S for every `n_tok` between the 1-token identity pair
   and the GEMM range and the whole MoE FFN took q8. Ported the same IQ2_S
   mapping the decode/GEMM ports use (82-byte block, grid at table word 544)
   into both kernels behind `#ifdef Q36_MOE_IQ2S`, added the `tables` binding
   the down kernel needed (6 vs 5), and widened the predicate to
   `if (iq2s && !(gemm || down_iq2s || (identity && down_sum_decode))) return 0;`
   — coverage is `n_tok` 2..127, wider than HANDOFF's stated 2..31
   (`Q36_VK_MOE_PAIR_TILE = 8`, `Q36_VK_MOE_GEMM_MIN` default 128). Scope is
   `down_iq2s`-only so the IQ3_S-down layers (1c's `il=0,1,2`) keep their q8
   fallback and no IQ2_S kernel can ever read a 110-byte block.
   Route misses (`Q36_VK_MOE_ROUTE_DEBUG=1`, ctx 512, `--prefill-chunk 16`,
   `--gen-tokens 0`): **1280 -> 96**, the 96 being exactly `il=0,1,2 × 32
   forwards` — i.e. the change is provably live, which is what makes the chunk
   16 A/B legible. 7-rep interleaved A/B (ctx 512, gen 128, gate 3.4%):
   `--prefill-chunk 16` (the arm that actually dispatches `n_tok=16`) prefill
   **72.92 -> 116.21 t/s (+59.37%, MAD 0.440/0.130)**, decode -1.63% (A MAD
   1.310, inside the 9.4% spread and with no mechanism — `n_tok==1` routing is
   untouched). The `--prefill-chunk 256` null control reads +3.23% with B MAD
   17.08 and fully overlapping ranges: inside the 3.4% floor, no signal, the
   correct result for a path whose dispatch site is not reached at
   `n_tok >= Q36_VK_MOE_GEMM_MIN`. Parity (frontier 512, three arms) f32b vs q8
   `max_abs` 0.779015 / top-1 match / top64 61/64, vs the accepted GEMM arm
   0.980383 / 62/64 — the new pair is *closer* to q8 than the accepted arm, and
   61/64 is this file's existing `n_tok>1` floor, not a new defect; guard
   0.796737 / 63/64 is the pre-existing IQ2_XXS f32b baseline. Base builds
   byte-identical (`moe_down_q2k_f32b.spv` `0ce317941dbef148…`,
   `moe_gate_up_f32b.spv` `9701f0330b5b5d73…`); `compat_gate.sh` PASS.
   One bug caught before landing: the first draft hand-derived the IQ2_S
   slotting and measured `max_abs` 7.86 / top64 41-of-64; isolated at
   `n_tok=1` via `Q36_VK_MOE_DOWN_SUM_DECODE=1` vs `=0`, then replaced with the
   accepted sum-decode loop copied verbatim (7.86 -> 0.858). Attribution
   disclosed: the B arm was the shared main worktree and also carried the
   concurrent instance's `dense_extra_*` specialization (1f, committed as
   `f36daf4` after this run), so the +59.37% is that pair vs `44143d8` — bounded
   independently by 1f's own measured ~0 and by the chunk-256 control above.
   Caveat: the real
   target is a batched server and `bench_ab.sh` has no concurrency knob, so
   chunk 16 is a routing proxy, not a serving measurement. Evidence:
   `evidence/raw/iq2s-smallbatch-VERDICT.md`,
   `ab-iq2s-smallbatch-chunk{16,256}.{csv,summary}`,
   `iq2s-smallbatch-parity.txt`, `route-miss-{A,B}.txt`.
2. **Wave32 for the non-mmq paths — MEASURED NEGATIVE, closed (2026-09-17,
   fully closed 2026-09-20).**
   The cheap bound was run first, before any build: `RADV_PERFTEST=cswave32` vs
   stock on Swift 27B, 7 interleaved reps, ctx 1024 -> prefill **171.14 ->
   169.88 = -0.74%** (MAD 0.600/1.540), negative in all seven pairs; decode a
   tie in steady state (the raw -5.18% is A's cold-start tail). n=7 per side.
   The reason is structural, not mysterious: the hot kernels are *already*
   wave32-forced (`dense_*_mmq` at `q36_vulkan.c:1575`), so the flag only moved
   off-critical-path kernels. **Do not re-run the env lever.** A per-kernel
   source widening of the decode kernels (each has exactly one cross-lane op)
   is still *possible*, but the measured ceiling is ~0, so it is demoted out of
   the ranked list. Evidence: `evidence/raw/ab-cswave32.csv`; runner pitfall in
   §9.
   **2b. The same audit found a real BUG, not a lever — FIXED (2026-09-17).**
   `dense_iq3_s_mmq_r4.spv` escapes the force predicate (suffix test
   `"_mmq.spv"` against a name ending `_mmq_r4.spv`), so it ran at wave64 while
   its `gl_SubgroupID`-partitioned token tile *requires* wave32: the formula
   peaks at **79** with 2 subgroups, so tokens 80-127 of every 128-token tile
   were written by nobody. Reachable via `small_rows` (IQ3_S, `out_dim == 48 &&
   n_tok <= 128`), latent for all three local models. Fix + A/B (no change:
   -0.03% / -0.05%, gate PASS) merged as `6e999eb`; row in `AlreadyTried.md`.
   The audit's §3b list — nine clean-scan shaders blocked only by the comment's
   *letter*, including `moe_matvec`, which holds 67% of IQ2_M's GPU time — has
   since been run as **W4 (2026-09-20): REJECTED, NEUTRAL/NO WIN.** Forcing
   wave32 across the §3b set (`moe_matvec`, `moe_matvec_fast`,
   `matmul_q8_0_decode*`, `add_rms_norm`, `rms_norm*`, `kv_store_quant`) on
   IQ2_M/Guard: parity bit-exact, prefill 242.43→240.00 = **-1.00%** (fails
   ≥3.4% MoE gate), decode 46.62→46.62 = **+0.00%** exact tie. Native wave64
   remains optimal for these shaders on GFX1013; closed, no
   `reconsider_if` short of a hardware/compiler revision. Evidence:
   `evidence/raw/ab-w4-wave32-iq2m*.csv`, `evidence/raw/W4-VERDICT.md`,
   `AlreadyTried.md` (W4 entry).
3. **Packed-native shader restructuring.** The only surviving general lever
   class: this driver exposes **no** usable dot-product intrinsics
   (`int dot: 0`, all accelerated variants false) and no bf16, so the wins have
   to come from restructuring shaders so ACO emits packed `v_pk_fma_f16`
   (bit-identical output). Proven prior art: the friend's `mul_mm.comp` +17-24%
   prefill and the iq3_s GEMV port +14%.
   **3b. MEASURED NEUTRAL on the hot kernel (2026-09-17).** `dense_iq3_xxs_mmq`
   already packs f16 (`f16vec2` + packed `fma`, lines 155-159), so the peer
   mechanism has no unclaimed counterpart there. codex's staging restructure
   (+71/-63: double-buffered A, `buf_b` deleted, shuffle-built B operand, one
   barrier dropped) gave prefill 170.93 (MAD 0.080) -> 170.75 (MAD 0.660) =
   **-0.11%**, decode +0.05% -> FAIL, rejected. Staging/barrier/LDS is CLOSED
   *within one projection*; sharing the stage across the gate/up pair is a
   different lever and it paid — see 4b. Evidence: `evidence/raw/ab-w8-mmq.csv`.
   **3c. W2 — IQ3_XXS ISA/ALU ablation — SETTLED, NOT PURELY DRAM-BOUND
   (2026-09-20).** Answers the question 3b opened: `Q36_IQ3XXS_ABLATE` (keep
   loads, drop decode ALU) on `dense_iq3_xxs_decode_r4` measured baseline
   26.578 ms/tok (282.2 GB/s, 62.1% of the 454.4 GB/s roofline) vs ablated
   20.081 ms/tok (373.5 GB/s, 82.2% of roofline). **Exposed ALU cost is 24.44%
   of kernel time**, mostly the 4x DPP cross-lane reduction chains
   (`quad_perm`/`row_half_mirror`/`row_mirror`/`v_permlanex16_b32`/
   `v_readlane_b32`). Diagnostic ceiling is the 20.08 ms/tok DRAM floor; no ALU
   fix beats it. No code change (diagnostic only). Evidence:
   `evidence/raw/iq3xxs-isa/W2-VERDICT.md`.
4. **FA LDS staging — TRIED, NEUTRAL, rejected (2026-09-17).** The peer's row
   claims PP +5.4% / TG +12.4% and was the largest *unread* gap in that doc. It
   has now been read and implemented rather than assumed: codex +20/-5 to
   `vulkan/attn_prefill_qtile2.comp`, built clean, A/B 7 interleaved reps ->
   prefill **171.47 (MAD 0.660) -> 170.75 (MAD 0.480) = -0.42%**, decode
   **18.58 -> 18.58 = +0.00%**, gate +1.50% -> **FAIL**. The 0.42% is inside the
   0.7% dense floor and in the *favourable* direction, so it is drift, not a win.
   Rejected, not merged. Caveat that bounds this negative: attention is not in the
   top-3 of this model's prefill profile, so an e2e A/B is a blunt instrument for
   this kernel — a real kernel-level gain could hide under the floor here. Treat
   the peer's FA number as still unconfirmed on this tree rather than disproven.
   Evidence: `evidence/raw/ab-w10-fa.csv`, patch `evidence/raw/w10-fa.diff`.
4b. **Dense gate/up pair fusion (W6) — LANDED & ACCEPTED (2026-09-21).** The dense
   FFN prefill paid the `b16` activation staging + LUT setup twice per workgroup,
   once for `gate` and again for the identically-shaped `up`. New
   `vulkan/dense_iq3_xxs_mmq_pair.comp` + `q36_gpu_matmul_iq3_xxs_pair_mmq_tensor()`
   serve both matrices from one workgroup, amortizing that over 64 weight rows
   instead of 32, with the unfused kernel's per-matrix arithmetic and store
   mapping (bit-exact, not merely close). Coverage 94/110 layer-instances (85.5%);
   the other 16 are genuinely mixed-type gate/up. **Prefill 169.25 (MAD 0.140) ->
   173.01 (MAD 0.440) = +2.22%**, 7 interleaved reps, ctx 1024; decode +0.55%
   (noise); gate/up kernel time -8.4% at ctx 512; frontier-513 parity
   `max_abs_diff = 0`; IQ2_S guard sha256s unchanged. Default ON,
   `Q36_VK_DENSE_IQ3_PAIR=0` disables. `reconsider_if`: the 16 mixed-type
   instances become same-type (~+2.6% implied at full coverage). Evidence:
   `evidence/raw/W6-VERDICT.md`, `ab-w6-pair-{csv,summary.txt,parity.txt,profile.txt}`.
4c. **Stream-K geometry audit (W7) — AUDITED & REJECTED (2026-09-21).** Evaluated
   wave quantization and partial-wave tail across all MMQ shapes on AMD BC-250
   (40 CUs, GFX1013) at chunk 256. Primary shapes divide cleanly: down-projection
   (5120x17408, 320 WGs) is exactly 4.00 waves (R=2) / 2.00 waves (R=4) with 0.0%
   tail loss; QKV-projection (5120x10240, 640 WGs) is exactly 8.00 waves (R=2) /
   4.00 waves (R=4) with 0.0% tail loss; gate/up (5120x17408, 1088 WGs) has 2.86%
   tail loss. De-homogenizing the 88 IQ4_XS dispatches into their 5 actual tensor
   shapes reveals that 84.1% of IQ4_XS work has <= 2.86% tail (blended 2.23% at R=2,
   4.39% at R=4). Aggregate weighted tail across the active residency bracket
   [R=2, R=4] is **1.78%** (R=2, 80 slots) to **3.32%** (R=4, 160 slots) of prefill
   kernel time (1.93% to 3.61% of MMQ time). Primary compute paths (>80% of GEMM work)
   have tail <= 2.86% (0.0% on down/qkv). Timeline analysis confirms 0.5% idle across
   dispatches; latency is per-workgroup barrier/LDS latency, not scheduling tail.
   Stream-K introduces atomic K-reductions and split-K barriers for < 1.8% theoretical
   headroom. Gate was > 3.0% of kernel time -> **REJECTED (NOT A LEVER)**. Closed
   without code churn. Evidence: `evidence/raw/W7-VERDICT.md`, `evidence/raw/w7_streamk_audit.py`.
4d. **W3 — `delta_net_cols` prefill on BC-250 — REJECTED, FROZEN; latent wave32
   bug FIXED (2026-09-20).** 7-rep A/B via `Q36_VK_DELTA_COL_PREFILL=1`: prefill
   170.77→166.54 = **-2.48%** (negative in all 7 pairs, fails ≥1.5% gate),
   decode -0.72%. Native wave64 register recurrence
   (`delta_net_decode_reg_f16`) beats the "load once/loop/store once" column
   kernel here; the `!q36_vk.bc250` guard stays frozen. Found and fixed in
   passing: `q36_vk_force_wave32`'s pattern matched `"delta_net_cols.spv"` but
   missed the f16 variant `delta_net_cols_f16.spv` actually dispatched for
   Swift's recurrent state, so it ran at native wave64 (2 subgroups instead of
   4) and left half the state columns unwritten — OOB writes to row 159 of a
   128-row buffer, `max_abs` 19.12, top-1 flip. Widening the pattern to
   `"delta_net_cols"` restored parity (`max_abs` 0.767, top-64 61/64). This bug
   was latent in every prior wave32 audit on this file. Evidence:
   `evidence/raw/ab-w3-deltacol*.csv`, `evidence/raw/W3-VERDICT.md`.
4e. **W5 — RMSNorm → q8_K producer fusion — REJECTED, GATE NOT MET
   (2026-09-21).** Fusing `quantize_q8_k` into `add_rms_norm` removes an
   activation write/read round-trip; parity bit-exact (`max_abs_diff = 0`,
   248,320 logits). 7-rep A/B on Swift 27B: prefill 170.38→169.83 =
   **-0.32%** (fails ≥0.70% dense gate), decode -0.83% (noise). Root cause:
   `add_rms_norm` is one 1024-thread workgroup per row; `quantize_q8_k` is
   64-thread/256-element blocks, 20 independent workgroups per row across 40
   CUs. Fusing serializes those 20 blocks into 2 rounds of 10 workgroup-wide
   barriers inside the bigger workgroup, triples LDS (8 KiB → 25.6 KiB), and
   drops subgroup occupancy 40→32 — the barrier/occupancy cost outweighs
   avoiding the L2 round-trip the two separate dispatches already get for
   free. Default stays split (`Q36_VK_FUSED_RMS_Q8=0`); scaffolding kept
   behind the flag, not deleted. Evidence: `evidence/raw/ab-w5-fusedrms*.csv`,
   `evidence/raw/W5-VERDICT.md`.
4f. **W8 — long-context KV dequant redundancy — go/no-go POSITIVE, QT-widening
   mechanism REJECTED, scratch-cache UNEXPLORED (2026-09-21).** Plan's own gate:
   measure attention prefill share at ctx 4096/8192 before touching any shader.
   `attn_prefill_qtile2{,_gqa6}.spv` share climbs from ~3.1% (ctx 1024) to
   **20.7% (guard) / 12.6% (Swift) at ctx 4096** to **31.5% (guard) / 20.7%
   (Swift) at ctx 8192** — clears the informal go/no-go easily. Root cause
   confirmed by reading the source: each workgroup already shares K/V dequant
   across its own `QT=2` tokens x all GQA heads; the redundancy is *across*
   workgroups (every query-tile workgroup for a `kvh` re-dequantizes the same
   early spans under causal attention). First mechanism tried — widen `QT` to
   4 so more query rows share each dequant — was rejected by a cheap
   compile-only probe (scratch `QT`/`ROWS` bump, `glslc` + `./mmq_info`, no
   real build) before any A/B: VGPRs 168→**256** (RDNA's hard ceiling), LDS
   21504→43008 B, occupancy **6→2 subgroups/SIMD**. A 3x occupancy cut is
   very likely to erase the halved-dequant win, matching this campaign's prior
   register/LDS traps (W4, item 3b). Second mechanism — a KV-dequant-once
   scratch cache shared across query-tile dispatches — has no VGPR risk but
   an unresolved bandwidth question the plan itself flagged skepticism about,
   and no cheap probe exists for it; **left unexplored, not built
   speculatively**. Closed for this session rather than sinking a build+A/B
   into an unmeasured design. `reconsider_if`: someone builds and measures the
   scratch-cache mechanism (real open work, not a dead end), per-lane state
   shrinks enough (e.g. f16 accumulators) to fit `QT=4` without the occupancy
   cliff, or a driver/HW change raises the VGPR ceiling. Evidence:
   `evidence/raw/w8-attn-share-{guard,swift}-ctx{4096,8192}.txt`,
   `evidence/raw/w8-vgpr-probe.txt`.
5. **Where the time actually goes** (Swift IQ3_XXS profile): prefill
   `dense_iq3_xxs_mmq` = **68.1%** of prefill GPU and is *instruction/latency*
   bound (~39% of packed-f16 peak); decode `dense_iq3_xxs_decode_r4` = 57.8% and
   is *DRAM-bound at 80% of roofline*. **Prefill is the lever; decode has
   almost no headroom.**

6. **Ternary-Bonsai-2-27B (`prism-ml/Ternary-Bonsai-2-27B-gguf`) — cannot run
   as-is, but the work is now bounded and the layout is measured (2026-09-17).**
   Header probe only (32 MiB range fetch, no weights downloaded): GGUF v3, 851
   tensors, arch `qwen35` hybrid SSM, **402 tensors of off-spec type id 142**
   (`general.file_type = 141`), `prism.hadamard.*` weight-name lists, and a
   derived layout of **34.0000 bytes per 128 weights** — identical min to max
   across all 402 tensors, i.e. 32 B of 2-bit codes + a 2 B f16 group scale =
   **2.125 bpw**, with **no grid LUT** (nothing like this engine's IQ2_S/IQ3_S
   machinery). Blockers in cost order: **(a)** type-142 table entry + dequant +
   GEMV — small; the only unknown is the 4-entry codebook; **(b)**
   inverse-Hadamard at load — `grep -ri hadamard` returns nothing, so it is not
   implemented; **(c)** the `qwen35` hybrid graph — the SSM profile already
   exists (`q36.c:136,151-154`: `n_ssm_conv 4 / n_ssm_state 128 / n_ssm_group
   16 / n_ssm_dt_rank 48`, matching the file's `ssm.*` exactly), so the open
   risk is whether that profile is *exercised* by a model we have run. The
   loader rejects the file cleanly: `tensor_type()` and `tensor_nbytes()` both
   bounds-check, so id 142 produces a **named unsupported-type error, no OOB**.
   The publisher's fork **is public** — org **`PrismML-Eng`** (NOT `prismml`,
   which has zero public repos), repo `llama.cpp` — so the codebook can be
   *read* rather than guessed. Speed case if it loads: 7.206 GB read per token
   against the measured **454.4 GB/s** DRAM = **63.1 tok/s** decode roofline,
   versus the 27B's 23.19 at 63.8% of its own roofline -> **~40 tok/s (+73%)**
   at equal efficiency; and ternary weights need **no multiplies**
   (mask-and-add), which is exactly the axis where IQ3_XXS's decode sits at an
   instruction-bound 59%. Evidence: `evidence/raw/ternary-bonsai-probe.txt`,
   `evidence/raw/ternary-bonsai-layout.txt`.
   **6b. UPDATE (2026-09-23): the DERISKED re-release loads and runs.** See §6.7
   and the §8 model row — the codebook/Hadamard/`qwen35`-graph blockers above are
   resolved in that file (type-142 PQ2_0, which this engine has kernels for).

7. **Ternary-Bonsai-2-27B DERISKED PQ2_0 — runs; PQ2_0/Q2_0 small-batch kernel
   ACCEPTED (2026-09-23).** First non-Qwen-mix file in the campaign that is a live
   decode target: 211.08 / 32.97 t/s at ctx 512 (§8). The landed change is
   `vulkan/dense_extra_small_q2.comp` plus its host dispatch — PQ2_0/Q2_0 batches
   of 2..16 tokens, and the ragged tail past a whole 128-token mmq tile, now
   stream each weight row once instead of paying a full tile for 2 tokens:
   `--prefill-chunk 2` **2.95 -> 54.60 t/s (18.5x)**, prompt tails **pp130
   +71.7%**, **pp140 +45.4%**, pp256 unchanged, and a 2-token batch costs
   1.38-1.6x a one-token decode (the MTP-verify regime). **Bit-exact** against n
   one-token decode calls (`tests/test_pq2_small`, 27/27 cases, 0 mismatching
   bits); no regression where it is inert (7-rep interleaved ctx 1024 A/B:
   prefill -0.03%, decode +0.00%). Default on, `Q36_VK_Q2_SMALL_MAX=0` disables;
   threshold 16 is the conservative middle of the shape crossover. Uncommitted in
   `q36-wt/pq2-persist` (branch `experiment/pq2-smallbatch`). Evidence:
   `evidence/raw/pq2-smallbatch-{parity,chunk-sweep}.txt`,
   `evidence/raw/ab-pq2-smallbatch-ctx1024*`, `AlreadyTried.md`.
   **Not attempted, ranked:** (a) the PQ2_0 decode matvec reads at ~318 GB/s of
   the measured 437 GB/s ceiling and every dispatch-shape rewrite of it was
   neutral or worse (§7) — the remaining headroom is a load layout that needs no
   offline repack, not another dispatch shape; (b) the in-file MTP head (15
   `blk.64` tensors) — the head-swap speculative scheme was parked, not rejected:
   it needs a rework of the speculative core, and the algebra of the two schemes
   (not a measurement) only breaks even near 0.7 acceptance, so draft=1 MTP stays
   the only speculative configuration. Draft depth 2+ remains rejected (ledger).

8. **2026-09-24 audit sweep.** Per-item entries land here as they are closed;
   the audit's own item numbers are used (`P1`..`P9`).
   **P1. MoE IQ2_S down sum-decode — audited levers NEGATIVE, load shape
   CONFIRMED as the lever, not built (2026-09-24).** `moe_iq2s_down_sum_decode`
   is 21.6% of IQ2_M decode (`dispatches=4736 gpu_ms=313.156`, 40.6 GB/s over
   99.4 MB/tok of *unique* bytes).  (a) wave32 premise **false** - `mmq_info`
   says `subgroup=32` already for all three blobs (`wave32-eligibility.md:146`
   excluded it on the `subgroupAdd`, never measured); (c) `fma32` **negative** -
   ACO already emits exactly one `v_fma_f64` per *expert* in the `tid == 0`
   tail, and a profile-only `fma32 -> fma` swap was inside noise; "experts in
   parallel" **negative** - `Q36_VK_MOE_DOWN_SUM_DECODE=0` costs 4.8% decode
   (70.9 vs 74.4 t/s).  (b) byte-granular loads **confirmed**: two in-kernel
   floors decompose the kernel as `315.4 = 97.0 coalesced-load floor + 167.2
   scattered field loads + 51.2 ALU`, i.e. 131 GB/s is achievable for the same
   12.72 GB.  The fix (coalesced lane-strided fetch + `subgroupShuffle`/LDS
   redistribution, per-element fma order unchanged so it stays bit-exact) is a
   real rewrite of the IQ2_S branch and was left unbuilt; target 315 -> 200-230
   ms and +14% IQ2_M decode.  Evidence:
   `evidence/raw/p1-sumdecode/P1-VERDICT.md` + `ab-p1-{fma,loadfloor,alufree}.csv`,
   `summary.csv`, `isa-f64-counts.txt`, `p1-diag-probes.diff`.
   **P2. MTP acceptance measured; the drafted token is always right and MTP is
   still a net loss (2026-09-24).**  Swift, greedy, ctx 512, gen 512, three
   prompts.  Draft acceptance when a draft is carried: **100% at depth 1**
   (172/172 margin 0, 60/60 margin 3, 91/91 code, 77/77 agent), **>= 69.4% at
   depth 2** (100 of at most 144 two-draft calls) - the P7 >= 60% gate is met.
   But decode is **slower in 6/6 pairs, 0.4-7%** (story 22.63 -> 21.08/21.14,
   code 21.89 -> 20.92, agent 21.87 -> 20.98), and structurally so: at
   `draft_cap 1` the verify row is a *new position*, so it is a whole extra
   forward - **47.5 ms per one-token spec call, 94.4 ms per two-token call
   (= 47.2 ms/token) vs 44.19 ms per plain step** (solved from the two margin-0
   arms), with the draft head only +3.3 ms.  `--mtp-draft 3/4` (draft_cap 2/3)
   collapses to **4.17/4.44 t/s** because the moment the verify carries a second
   row it leaves the decode kernel for `dense_iq3_xxs_mmq` +
   `predequant_b16` (`q36_vulkan.c:8687-8730`: only `n_tok == 1` gets a decode
   kernel) - the P6 small-batch gap.  Also: the `MTP stats accept=100.0%` line is
   100% by construction at the published `--mtp-draft 2` (`draft_cap = N - 1`
   makes `verify_n == commit_n == 1`), and the default `--mtp-margin 3` gate only
   costs (margin 0 gives +33% tokens/call at the same t/s).  Evidence:
   `evidence/raw/p2-mtp/*.log`, `evidence/raw/p6-smallbatch/NOTES-cost-model.md`.
   **P4. Bit-exact integer IQ3_XXS decode — REJECTED, no packed signed dot on
   GFX1013 (2026-09-24).** Reopened the 2026-09-17 sign-mask rejection with the
   audit's ISA evidence (690 VALU / 128 weight-MACCs = 5.4 VALU/weight; 6.497
   ms/tok = 24.4% of the kernel is exposed ALU above the 20.081 ms/tok
   loads-only floor).  The arithmetic premise is right — grid <= 62, q8 <= 127,
   partial sums < 2^24, so an int32 dot is bit-exact — but the instruction is
   not there: a `dotPacked4x8EXT` shader compiles and ACO emits **zero**
   `v_dot4_i32_i8`, lowering it to 4x `v_mul_i32_i24_sdwa` (byte select) + 3
   adds + accumulate = 2 VALU/weight for the MAC alone.  Best reachable integer
   layout is 4.75-5.0 VALU/weight vs the shipped 5.4-6.0 = 0.3-1.0 ms/tok,
   1-2%; the <= 2.5 target needs a pre-signed grid word (16-variant 512-entry
   LUT, 32 KB LDS, over budget).  `device-query.txt:165
   has_accelerated_dot_product = 0`.  The Q4_K/Q5_K half inherits the ceiling;
   its only survivor is a *load* lever (source-confirmed double fetch of every
   dword row, `dense_kquant_decode.comp:229-234`), left unbuilt.  Evidence:
   `evidence/raw/p4-iq3xxs/P4-VERDICT.md`, `dot4.comp`, `dot4.isa.txt`,
   `dot4.stats.txt`.
   **P5. Grouped KV-head decode attention — still REJECTED; the premise was
   bandwidth, the kernel is not bandwidth-bound (2026-09-24).**  Reopened the
   2026-09-23 rejection with the audit's bandwidth case (KV rows read 6x/8x,
   guard 3.0 ms/tok at ctx 8192 over ~68 MB = ~22 GB/s).  The shader + harness
   the rejection stored were rebuilt byte-identical and rewired behind
   `Q36_VK_ATTN_DECODE_GQA`, then measured per `pos0`: parity 18/18 bit-exact,
   but the grouped build is **0.47x/0.36x at 1023 keys** (2.1x/2.8x slower),
   0.72x/0.50x at 4095, **1.07x/0.74x at 8191**, 1.01x/1.04x at 16383, 1.04x/
   1.02x at 32767, 1.14x/1.07x at 65535 — it wins only past 8k (dense) / 16k
   (MoE) and by 1.01-1.14x of one attention dispatch.  The re-read is an L2
   phenomenon, not DRAM: per-head at 32767 keys moves 54.5 MB of unique KV in
   1.076 ms = 50 GB/s = 11% of roofline, and one span (852 KB) is walked by the
   6 workgroups of a kv head, so dropping the re-read saves L2 + dequant work,
   not bandwidth.  End-to-end numbers are unchanged from 2026-09-23 (1k -8.45%,
   MAD 0.02); 8k/16k/32k were not re-run because decode-after-long-prefill
   swings 20-32 t/s with an identical binary (§9), far more than a ~1.04x change
   in one dispatch of a 33-36 ms/token decode.  Evidence:
   `evidence/raw/p5-gqa-decode/{gqa-kernel-repro-20260924.txt,host-wiring-20260924.patch}`,
   `evidence/raw/ab-attn-decode-gqa-ctx{1k,2k,8k}.*`.

## 7. Closed lines — do not re-litigate without new hardware evidence

Authoritative detail and per-item "reconsider_if" live in
`karpathy/AlreadyTried.md`; this is only the index.

- **`int dot: 0` is a hardware truth on GFX1013** — all accelerated
  integer-dot properties are `false`; the gate at `ggml-vulkan.cpp:4019` can
  never pass. No flag, pull or shader edit fixes it. (Build-side causes were
  investigated and eliminated.)
- **`bf16: 0` / `VK_VALVE_shader_mixed_float_dot_product` absent** — genuinely
  not exposed.
- **Updating llama.cpp bought nothing** (`0cae430` → `972d231`, full rebuild):
  −2.2% prefill / +3.1% decode, inside the 4.4 stddev.
- **The drain fix is rejected as a speed lever** — +0.45% prefill / +0.43%
  decode, and inert in production (profiler-gated). Keep it only as profiling
  hygiene.
- **The 41-block IQ2_M refusal is fixed, not open:** `f9d537d` loads it with
  `--gpu-cpu-parity` OK and no guard regression. Do not re-derive the root cause
  and do not "relax a check" again — the remaining IQ2_M gap is kernel selection
  (§6.1).
- **Thermal drift is not a factor** — 171.96 vs 171.91 prefill hot vs cooled.
  `bench_ab.sh` deliberately does no cool-down.
- **The roofline is 454.4 GB/s, not ~360** as an earlier report claimed. Do not
  gate against 360.
- **The reported "80/128 token-coverage hole" in `dense_iq3_xxs_mmq` is false** —
  force-wave32 at `q36_vulkan.c:1575` covers all 128 tokens.
- **IQ3_XXS MMQ/decode micro-variants, 512-thread add/RMS, XOR-swizzled staging,
  MTP draft depth 2/3, smaller prefill chunks, integer sign-mask decode** — all
  measured and rejected; numbers in the ledger.
- **The peer BC-250 cluster's nulls** (MOVNTDQA, BK 32→64, `nogttspill`,
  decode-ubatch np=16, MTU 9000, split-heap readback, `GEMM_FORCE_L_PERTYPE`)
  — recorded in the ledger so they are not reopened here.
- **Host-cached readback and shader-core-count levers are already implemented /
  not applicable** in this engine.
- **PQ2_0/Q2_0 decode-matvec workgroup shapes — closed (2026-09-23).** The
  streaming ceiling on this board is **~437 GB/s** (standalone probe,
  `evidence/raw/probe.c` + `stream.comp`); the PQ2_0 decode matvec runs at
  **~318 GB/s**, and the gap is that kernel's own access pattern, not dispatch
  shape: fewer workgroups (caps 320/640/1280/2560), 2/8/16 rows per workgroup,
  and four 4-row wave64s packed into one 256-thread workgroup were all **neutral
  or worse** (best -1.8%, worst +43%; 1385 -> 1381 ms on the packing test). Do
  not retry without a wide-aligned load layout that needs no offline repack.
- **GQA-grouped split-K decode attention — rejected (2026-09-23).** One
  workgroup per (kv head, token, span) sharing each K/V read across the heads
  that share a kv head: **bit-exact** (18/18 cases, 600..65535 keys) but
  **0.3-0.7x at 600-4k keys** and only ~1.0-1.17x at 16k-64k, net negative end to
  end. Its first sweeps read **+11% at ctx 8k** and **+23.6% at ctx 2k**
  (`evidence/raw/ab-attn-decode-gqa-ctx{2k,8k}*`) and neither survived
  re-measurement: at ctx 1k/1.5k/2k/4k the same change measures **-2..-9%**, and
  the wins were the decode-variance pitfall below (§9). Shader, harness and the
  rejected patch are archived under `evidence/raw/attn-decode-gqa-rejected*`.

## 8. Model set — what loads and what does not

| file (`/home/server/q36/gguf/`) | size | arch | q36 | upstream llama.cpp |
|---|---|---|---|---|
| `Swift-Qwen3.8-27B-IQ3_XXS.gguf` | 11952M | qwen3 dense 27B | **171.09 / 23.19** | 105.43 / 22.13 |
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 11484M | qwen3 dense 27B | 166.36 / 21.44 | not measured |
| `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf` | 11194M | qwen35moe 40 blk | **909.49 / 87.37** | 615.08 / 78.72 |
| `RavenX-35B-Q36-IQ2XXS.gguf` | 11194M | qwen35moe 40 blk | 717.36 / 89.47 | 545.46 / 77.96 |
| `Qwen3.8-35B-A3B-IQ2_M.gguf` | 11977M | qwen35moe **41 blk** | 116.51 / 29.29 | 529.90 / 91.53 |
| `TERNARY-BONSAI-2-27B-DERISKED-PQ2_0.gguf` | 6873M | qwen35 dense 27B + MTP head | **211.08 / 32.97** at ctx 512 | not measured |

- The guard is the no-regression reference; `Qwen3.6-35B-A3B-AntirezExperts-
  …gguf` is a **symlink** to it, and `q36moe.gguf` points at the *dense* 27B.
  **Always pass absolute model paths** — several entries are symlinks and
  worktrees have no `gguf/` link.
- RavenX out-decodes the guard while its prefill is 21% lower, on a file 128
  bytes different in length: that is quant mix, not layout. There is no single
  "MoE ≈ 900 t/s prefill" figure.
- IQ2_M **loads** since `f9d537d` but its quant mix (375× `IQ2_S` trunk) misses the
  fast prefill dispatch, so it is the one file where q36 is far *behind* upstream.
  Never quote it as a q36 win; see §6.1.
- `TERNARY-BONSAI-2-27B-DERISKED-PQ2_0.gguf` is the **DERISKED re-release** of the
  file §6.6 could not load: same `qwen35` dense 27B layout, now carrying type-142
  **PQ2_0** tensors this engine has kernels for (402 PQ2_0 tensors + BF16
  `ssm_alpha`/`ssm_beta`). 6873M, published md5 matches. Numbers above are the
  standard Bonsai workload (ctx 512, 128 greedy tokens, `--prefill-chunk 256`);
  the older `Q2_0-g64` file measured ~200 / ~33.3 on it. Raw:
  `evidence/raw/pq2-bonsai-workload.txt`. Unlike the models above it, its decode
  matvec is a *dense* PQ2_0 kernel (`dense_extra_decode_pq2_0`) — see §6.7.

## 9. Known pitfalls (each of these has cost real time)

1. **Shader blobs are untracked and absent in fresh worktrees.** 108 `.spv`
   live in `vulkan/`; a new worktree starts with none. If `VULKAN_SHADERS`
   references a missing `.spv` the build or run fails: `cp -a vulkan/*.spv
   <worktree>/vulkan/`. `make clean` deletes them too.
2. **No committed binary** — rebuild from a pinned commit; verify freshness by
   comparing `q36-bench` mtime against the `.o`/source mtimes.
3. **Never build during a bench, never two model processes at once.**
   `bench_ab.sh` refuses to start if `q36-bench`/`q36_test` is running.
4. **`/tmp` is the scratch convention**, but it is volatile — copy every raw CSV
   into `karpathy/evidence/raw/` when the run ends. Numbers whose raw data is
   gone are not evidence.
5. **Tool stdout gets truncated** on this host: write long output to a file and
   read it back with a narrow range.
6. **Worker quirks:** `opencode run --model union-alpha '<brief>'` (the
   `--model union-alpha` flag is mandatory here) and it auto-rejects reads
   under `/tmp` — copy inputs into the worker's worktree. `codex exec
   --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check -C <dir>`
   hits a usage-quota wall that clears on a schedule. `agy -p` can exit 0 with
   an empty transcript — treat empty output as failure, not success.
7. **`~/Karpathy` is a separate checkout and stays untouched** unless the user
   says otherwise.

- **A wrapper B that is not executable kills an A/B before a model loads.** If
  `bench_ab.sh` returns exit **2** almost instantly with "B binary not
  executable", the run produced no numbers at all — do not read it as a failed
  experiment. `tests/bench_cswave32.sh` was committed without its exec bit and
  cost two launches on 2026-09-17. Fix: `chmod +x`, and commit the bit.
- **Worker availability is not stable — check before assigning the critical
  item (2026-09-17).** `claude --dangerously-skip-permissions -p` failed
  instantly with `You've hit your weekly limit · resets Sep 19, 1pm
  (America/Sao_Paulo)`; `opencode run --model union-alpha` returned
  `UnknownError / Unexpected server error. ref: err_...` on both attempts
  (exit 1, no patch written); `agy` is unusable (`print timeout after 5m0s`).
  **codex is the only CLI worker that has delivered a patch on this box** — give
  it the highest-value file first, then fall back. Always give a worker the
  fallback order explicitly so a dead agent is not the end of the item.
- **Syntax-checking a shader with bare `glslc` gives a false negative.** The
  Makefile compiles with `--target-env=vulkan1.1`; the default target env is
  too low, so builtins such as `subgroupAdd` fail and a worker may conclude its
  own (fine) file is broken. Tell workers the exact permitted command:
  `glslc --target-env=vulkan1.1 -fshader-stage=comp <file>.comp -o /tmp/x.spv`.
  - **A subagent that returns a summary but no file leaves no artifact.** Name an
  explicit output path when delegating and verify the file exists before planning
  around it: one audit subagent here returned 4,820 chars of summary and never
  wrote `evidence/wave32-eligibility.md`, so the table was lost.
- **A decode-only patch cannot be judged on the prefill gate.**
  `bench_ab.sh` declares `FAIL (prefill)` by construction when only a decode
  kernel changed. Judge those on the decode median **with its spread** — the
  decode spread on Swift 27B is up to 9.4% and a cold first rep can move the
  whole median.
- **Decode t/s measured right after a >=1.5k-token prefill does not repeat
  (2026-09-23).** Same binary, same settings, ctx 4096: 30.9 / 21.8 / 25.2 t/s.
  Per-kernel `gpu_ms` moves too (one prefill mmq kernel time changed 12% between
  two runs with identical prefill t/s). Decode at ctx <= 1024 is stable to MAD
  0.02. Judge decode changes at short context, in a kernel harness, or with >=7
  interleaved reps that agree on direction — a single sweep at long context is
  not evidence.

## 10. Handoff checklist for the next agent

- [ ] Read this file, then `git log --oneline -6` + `git status`.
- [ ] Confirm the GPU is idle before any build or bench.
- [ ] Before proposing a change: **grep `AlreadyTried.md` for the mechanism**,
      not just the file you are about to edit.
- [ ] grepped the runtime flag that gates the code you will change (§4.5/4.6).
- [ ] Measured with `bench_ab.sh`, 7 reps, interleaved; report median + MAD.
- [ ] Copied the raw CSV into `karpathy/evidence/raw/`.
- [ ] Wrote the verdict into `AlreadyTried.md` in the house style: what was
      changed, what it measured, why it was accepted or rejected, and the
      `reconsider_if` condition.
- [ ] Committed (**everything**, with the numbers in the message).
