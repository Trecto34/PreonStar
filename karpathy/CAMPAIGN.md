# Campaign handoff — q36 inference optimization on BC-250

**Read this file first; it is the map.** If you have five minutes, read
`## Pick up in 5 minutes` and `## The rules that decide a verdict`, then go
straight to `## Open items, ranked`.

Last updated: 2026-09-17 (late session) · repo `/home/server/q36-opt-27b` ·
branch `experiment/radiance-transfer-bc250` · HEAD `e0b7eb4` (w7 loader fix +
its verdict docs; a docs-only verification commit follows it).

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
| Qwen3.8-35B-A3B-IQ2_M (MoE) | **116.51 / 29.29** (loads since `f9d537d`) | 529.90 / **91.53** | **−78% / −68%** — kernel selection, §6.1 |

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

1. **Quant-aware dispatch for the `IQ2_S` / K-quant MoE — the real IQ2_M lever,
   and now the highest-value item on the board.** The loader fix landed
   (`f9d537d`), so `Qwen3.8-35B-A3B-IQ2_M` loads and decodes at **116.51 /
   29.29** where upstream gets **529.90 / 91.53**. The profile says why: its
   `IQ2_S` experts fall through to generic `moe_matvec` (**11,845 ms = 67% of all
   GPU time**) and its K-quant trunk to `dense_kquant` (**4,662 ms**), while the
   guard reaches `moe_iq2_gate_up_gemm` 878.7 + `moe_q2k_down_gemm` 432.6 and
   `dense_q8_0_p_*` GEMMs (~950 ms). `IQ2_S` support already exists in-tree, so
   the work is routing these shapes into the per-quant prefill GEMM family:
   **in-tree prior art, not new shader work.** Evidence:
   `evidence/raw/w7-iq2m-prof.txt`, ledger §"kernel SELECTION".
2. **Wave32 for the non-mmq paths — MEASURED NEGATIVE, closed (2026-09-17).**
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
   for this kernel. Evidence: `evidence/raw/ab-w8-mmq.csv`.
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
5. **Where the time actually goes** (Swift IQ3_XXS profile): prefill
   `dense_iq3_xxs_mmq` = **68.1%** of prefill GPU and is *instruction/latency*
   bound (~39% of packed-f16 peak); decode `dense_iq3_xxs_decode_r4` = 57.8% and
   is *DRAM-bound at 80% of roofline*. **Prefill is the lever; decode has
   almost no headroom.**

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

## 8. Model set — what loads and what does not

| file (`/home/server/q36/gguf/`) | size | arch | q36 | upstream llama.cpp |
|---|---|---|---|---|
| `Swift-Qwen3.8-27B-IQ3_XXS.gguf` | 11952M | qwen3 dense 27B | **171.09 / 23.19** | 105.43 / 22.13 |
| `Qwen3.8-27B-UD-IQ3_S.gguf` | 11484M | qwen3 dense 27B | 166.36 / 21.44 | not measured |
| `Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf` | 11194M | qwen35moe 40 blk | **909.49 / 87.37** | 615.08 / 78.72 |
| `RavenX-35B-Q36-IQ2XXS.gguf` | 11194M | qwen35moe 40 blk | 717.36 / 89.47 | 545.46 / 77.96 |
| `Qwen3.8-35B-A3B-IQ2_M.gguf` | 11977M | qwen35moe **41 blk** | 116.51 / 29.29 | 529.90 / 91.53 |

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
