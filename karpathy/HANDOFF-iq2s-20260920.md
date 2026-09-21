# Handoff — IQ2_S fast path for `Qwen3.8-35B-A3B-IQ2_M` (2026-09-20)

Workstream: make the engine's tuned MoE kernels accept **IQ2_S** routed experts, so
IQ2_M stops falling into the generic `moe_matvec`. Two commits, both landed, tree
clean, guard byte-identical. Track #1 of the session that opened this line of work;
the sibling track list (#2 lm_head requant, #3 wave32 clean-scan A/B) is untouched.

## 1. State

- HEAD `e8cb6de` — "moe: IQ2_S expert variants for the f32 identity decode pair";
  its parent `6ef9ceb` — "moe: IQ2_S expert variants for the fused gate/up and GEMM
  pair". `git status --porcelain` is empty.
- Model: `/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf` (qwen35moe, 41 blocks,
  10.32 GB IQ2_S total, of which 10.06 GB is routed experts (0.26 GB dense/trunk)).
- Guard model: `/home/server/q36/gguf/Huihui-Qwen3.6-35B-A3B-Abliterated-Q36-IQ2XXS.gguf`
  (IQ2_XXS gate/up + Q2_K down + Q8_0 trunk — i.e. the mix every tuned path was
  built for; the IQ2_S work must leave it bit-identical).

| arm (ctx 512, gen 128, 7-rep median) | prefill t/s | decode t/s |
| --- | --- | --- |
| all-matvec (`MOE_GEMM=0 MOE_GATE_UP=0`) | 108.21 | 29.52 |
| + prefill pair (`6ef9ceb`, old decode posture) | 241.20 | 34.81 |
| + decode pair (`e8cb6de` defaults) | **241.27** | **47.14** |

Cumulative: **2.23× prefill, 1.60× decode**. Decode MAD fell 0.28 → 0.16; prefill
MAD 0.49 → 1.40 (MoE prefill noise floor is 3.4%, so those medians are fine).

Guard, same day, ctx 512: shipped GEMM `709.64 / 88.65`, `MOE_GEMM=0` `522.40 /
87.25` — the shipped IQ2_XXS GEMM is worth ~1.36× there, IQ2_S matvec→GEMM ~2.23×.

## 2. What landed

Kernel sources recompiled from the **same** sources with `-DQ36_MOE_IQ2S`:

| file | IQ2_S variant | new bindings | push |
| --- | --- | --- | --- |
| `vulkan/moe_gate_up.comp` | `moe_gate_up_iq2s.spv` (q8_K acts, fused gate/up+SwiGLU) | — | 28 |
| `vulkan/moe_gate_up_gemm.comp` | `moe_gate_up_gemm_iq2s.spv` (64×32 prefill GEMM) | — | 24 |
| `vulkan/moe_down_gemm.comp` | `moe_down_gemm_iq2s.spv` (64×32, reads f32 mid) | 6th: `tables` | 24 |
| `vulkan/moe_gate_up_decode.comp` | `moe_gate_up_decode_iq2s.spv` (f32 acts, identity) | — (tables already 7) | 24 |
| `vulkan/moe_down_q2k_sum_decode.comp` | `moe_down_q2k_sum_decode_iq2s.spv` (f32 mid + routing sum) | 7th: `tables` | 24 |

Host (`q36_vulkan.c`): five new kernel members + registrations + destroys;
`q36_vk_moe_gate_up_iq2_swiglu` accepts IQ2_S pairs at 82 bytes/block;
`q36_gpu_moe_ffn_f32_tensor` accepts IQ2_S gate/up/down with 82-byte
`gu_stride`/`down_stride` and switches the down to the 7-binding sum-decode kernel.
The IQ2_S admission predicate is:

```c
if (iq2s && !(gemm || (identity && down_sum_decode))) return 0;
```

i.e. IQ2_S is admitted for the prefill GEMM pair and for the 1-token identity decode
pair only. **The 2..31-token small-batch kernels (`moe_gate_up_f32b`,
`moe_down_q2k_f32b`) are still IQ2_XXS/Q2_K-only and would read 82-byte blocks as
Q2_K garbage** — that predicate is the only thing stopping them. Extend it only
together with those kernels.

Layout constants that are now depended on by four kernels: `IQ2_S` block = 82 bytes
(`d` f16 at 0, `qs[64]` at 2, `qh[8]` at 66, `scales[8]` at 74); packed-table u32
index `TAB_IQ2S = 544`; per element `n` in a 256-block: `ib32 = n>>5`,
`lg = (n>>3)&3`, `j = n&7`; value `= d * 0.25 * (0.5 + nibble(lg)) * ±grid[j]`,
nibble from `scales[ib32]` (low for `lg<2`), sign bit `j` of sign byte
`34 + 4*ib32 + lg`, entry `qs[4*ib32+lg] | ((qh[ib32] << (8-2*lg)) & 0x300)`.

## 3. Reproduce

```sh
cd /home/server/q36-opt-27b && pgrep -x q36-bench || echo "GPU idle"
make -j12 q36-bench                       # builds the *_iq2s.spv too; .spv are gitignored

M=/home/server/q36/gguf/Qwen3.8-35B-A3B-IQ2_M.gguf
# single measurement
flock -w 1800 /tmp/q36-gpu.lock ./q36-bench --vulkan -m "$M" \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 512 --ctx-max 512 --ctx-alloc 641 --prefill-chunk 256 --gen-tokens 128

# prefill A/B (all-matvec vs prefill+decode pair), 7 interleaved soak-gated pairs
CTX=512 GEN=128 logs/iq2s-gateup-20260920/ab_iq2s_gate_up.sh 7 /tmp/ab1.csv "$M"

# decode A/B (previous posture vs decode pair), prefill is the null control
CTX=512 GEN=128 logs/iq2s-decode-20260920/ab_decode_iq2s.sh 7 /tmp/ab2.csv "$M"
```

Both drivers gate on a thermal soak (entry ≤ 55 °C, `SOAK_TEMP` overrides), run one
GPU process at a time under `/tmp/q36-gpu.lock`, alternate arm order per pair, and
report median + MAD. Never compare two sequential single runs.

### Parity — the instrument that actually works

`--dump-frontier-logits-dir` + the frontier trick. **A frontier at 513 tokens
exercises the identity decode kernels**: the last prefill chunk is one token, so
`n_tok == 1` → identity → `moe_gate_up_decode` / `moe_down_q2k_sum_decode`. A
frontier at 512 exercises only the gemm pair. This is the only cheap way found to
gate the decode kernels.

```sh
mkdir -p /tmp/ref /tmp/new      # the dump dir must exist
flock -w 1800 /tmp/q36-gpu.lock env Q36_VK_MOE_GEMM=0 Q36_VK_MOE_GATE_UP=0 \
  Q36_VK_MOE_DOWN_SUM_DECODE=0 ./q36-bench --vulkan -m "$M" \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 513 --ctx-max 513 --ctx-alloc 642 --prefill-chunk 256 --gen-tokens 16 \
  --dump-frontier-logits-dir /tmp/ref
# ... same without the env overrides into /tmp/new, then
python3 /tmp/cmp_logits.py /tmp/ref/frontier_000513.logits.json /tmp/new/frontier_000513.logits.json
```

Accepted posture (frontier 513, vs the previous default posture): top-1 identical,
top-64 **62/64**, `max_abs` 0.907; vs the all-matvec arm 60/64 / 1.245. Calibration
for "small enough" is the shipped kernel's own delta, not zero: the IQ2_XXS GEMM
measured that way is 64/64 with `max_abs` 0.42 (0.82 at frontier 512 on IQ2_M).

### Guard byte-identity is checked against the *source*, not the tree

`.spv` are gitignored and `make` overwrites them, so comparing a fresh build to
`vulkan/*.spv` is circular. Do:

```sh
git show HEAD~1:vulkan/moe_gate_up_decode.comp > /tmp/pre.comp
./glslc -O --target-env=vulkan1.1 -o /tmp/a.spv /tmp/pre.comp
./glslc -O --target-env=vulkan1.1 -o /tmp/b.spv vulkan/moe_gate_up_decode.comp
sha256sum /tmp/a.spv /tmp/b.spv      # must match
```

Current guard hashes: gate/up GEMM `b46fa81a…`, down GEMM `93a38508…`, q8 gate/up
`601d56a9…`, gate/up decode `946097d8…`, down sum-decode `3f316c16…`.

## 4. Where IQ2_M's time goes now

Per-token decode, differential profile (`Q36_VK_PROF_OP=1 Q36_VK_PROF_KERNEL=1`,
gen 1 vs gen 65, ctx 512, delta ÷ 64) — **22.235 ms/tok total GPU**, down from
27.637 before the decode pair:

| ms/tok | share | op | note |
| --- | --- | --- | --- |
| 10.723 | 48.2% | `dense_kquant` / `matmul_kquant` | **generic** kquant matvec for the 0.26 GB of non-expert IQ2_S tensors |
| 2.640 | 11.9% | `moe_iq2s_down_sum_decode` | new (was `moe_matvec` 4.571) |
| 2.168 | 9.7% | `dense_extra_decode` | |
| 1.383 | 6.2% | `moe_iq2s_gate_up_decode` | new |
| 0.892 | 4.0% | `moe_iq2s_gate_up` | q8 route residue |
| 0.820 | 3.7% | `attn_decode_split` | |
| 0.565 | 2.5% | `matmul_f32_fast` | |
| 0.427 | 1.9% | `q8_k_quant` | |
| 0.404 / 0.398 | 3.6% | `dense_f32f_d_256x2048` / `add_rms_norm` | |
| 0.364 / 0.139 | 2.3% | `moe_matvec` / `moe_tiles` | both fell hard once identity decode stopped building tiles (1.706 → 0.139) |

Per-token **dispatch** counts (same delta method) pin the routing down: 41
`dense_kquant`, 37 `moe_iq2s_gate_up_decode` + 37 `moe_iq2s_down_sum_decode` (so 37
of 41 layers take the new pair), and only 3 `moe_iq2s_gate_up` + 3 `moe_matvec` + 6
`moe_tiles` still on the q8 route — that residue is what §5 item 3 is about.

Prefill at ctx 512 (before the decode pair, same binary family): 3666 ms total,
`moe_iq2s_gate_up_gemm` 637 ms + `moe_iq2s_down_gemm` 253 ms for 148 dispatches each,
`dense_kquant` 1181 ms — i.e. **after the experts, the trunk/dense path is the
prefill lever too**.

## 5. Ranked next steps

1. **IQ2_S tuned dense decode for `matmul_kquant` (48% of decode, ~1.1 s of prefill).**
   Every other format in this engine has a tuned decode kernel
   (`dense_iq3_xxs_decode_r4`, `dense_iq3_s_decode`, `dense_extra_*`); IQ2_S trunk
   rows run the generic kquant matvec. Same pattern as the expert ports: `#ifdef`
   variant off the existing tuned source if one is close, else a new `dense_iq2s_*`.
   Expected to be the single largest remaining decode win on this file. **Check the
   arithmetic first:** non-expert IQ2_S is only 0.26 GB, so one read per token at
   ~230 GB/s is ~1.1 ms, yet `dense_kquant` costs 10.72 ms/token over exactly 41
   dispatches (0.26 ms each, `groups≈1522`/dispatch). That is ~10× the bandwidth
   floor, i.e. likely ALU/ISA-bound byte-unpacking, not DRAM — the same shape as
   `dense_extra_decode_q2_0` before its LUT fix. Confirm bytes/token with a
   temporary counter before writing the kernel, and note that `matmul_kquant` also
   carries the Q8_0/Q5_K/Q4_K trunk types, so scope the IQ2_S branch specifically.
2. **Small-batch (2..31 token) IQ2_S kernels** — `moe_gate_up_f32b`,
   `moe_down_q2k_f32b`. Without them a batched server workload on IQ2_M still routes
   through q8. Same element mapping as the two ports above; the down one also needs
   the `tables` binding. Then widen the admission predicate.
3. **Explain/patch the q8-route residue** (0.892 + 0.364 + 0.427 ms/tok ≈ 7.6%).
   Something still calls `q36_gpu_moe_ffn_tensor` (which always takes the q8 route)
   on a 1-token forward — candidates: the MTP/nextn head, or layers with the bank
   cache off. Find it with `Q36_VK_PROF_KERNEL=1` + `moe_iq2s_gate_up` dispatch count
   per layer, not by guesswork.
4. **Re-profile prefill with the current binary** (the profile above predates the
   decode pair) and decide whether the trunk or the experts have the next prefill
   lever.
5. **Upstream gap.** llama.cpp runs IQ2_M at 529.90 / 91.53 on this box; we are at
   241 / 47. Only 11.9% + 6.2% of decode is now expert kernels, so item 1 dominates
   any further expert work.

## 6. Coordination

A parallel instance owns `karpathy/PLAN-implementation-2026-09-20.md` and its
W2…W8 (`dense_iq3_xxs_decode_r4.comp` ISA/ALU ablation, `delta_net_cols` gate,
etc.); its §0 declares W1 — this workstream — mine and lists my files. **W1 is now
done, so those files are free**: the five `vulkan/*.comp` above, their Makefile
rules, and the IQ2_S host wiring in `q36_vulkan.c`. Its plan document was swept into
`e8cb6de` by `git add -A` (content unchanged, no edits made to it). Their plan says
they will re-baseline W2–W8 against the new HEAD; the numbers they should use are
the table in §1.

## 7. Traps that cost time here

- `--vulkan-fusion-parity` **does not toggle** `Q36_VK_MOE_GATE_UP` (or the new
  IQ2_S kernels) and runs a 6-token prompt, so GEMM is off anyway: it can "pass"
  while comparing a kernel against itself. Not a gate for this work.
- `--gpu-cpu-parity` decodes 10 GB of IQ2_S on the CPU — minutes per case, wrong
  instrument. The GPU-vs-GPU frontier dump replaced it.
- `--dump-frontier-logits-dir` fails unless the target directory already exists.
- Buffered shell output can come back as a phantom `clean — nothing to commit`
  line; redirect Python stdout to a file and read the file.
- Two real bugs were caught only because a *calibrated* parity arm existed: the
  gate/up GEMM assigned the 8-group's low/high nibble to the wrong half for rows
  whose LDS swizzle has bit 2 set (59/64, `max_abs` 1.89), and the down decode
  applied one grid magnitude to both elements of each `vec2` when only the scale is
  per 8-group and the magnitude/sign are per element (37/64, top-1 flip, `max_abs`
  7.19). Both looked like "small numeric drift" until compared against the shipped
  kernel's own delta.

## 8. Evidence

`karpathy/AlreadyTried.md` carries the two verdict entries (2026-09-20). Raw, in
`karpathy/evidence/raw/`:

- prefill pair: `iq2s-gemm-ab-iq2m.csv`, `iq2s-gemm-ab.sh`,
  `iq2s-gemm-matvec-ref.txt`, `iq2s-gemm-new-prof.txt`, `iq2s-gemm-parity.txt`,
  `iq2s-gemm-parity-prefix.txt`, `iq2s-gemm-guard-calibration.txt`,
  `iq2s-gemm-xxs-identical.txt`
- decode pair: `iq2s-decode-ab-iq2m.csv`, `iq2s-decode-ab.sh`,
  `iq2s-decode-parity.txt`, `iq2s-decode-parity-prefix.txt`,
  `iq2s-decode-xxs-identical.txt`, `iq2s-decode-diffprof.txt`
- drivers kept runnable: `logs/iq2s-gateup-20260920/ab_iq2s_gate_up.sh`,
  `logs/iq2s-decode-20260920/ab_decode_iq2s.sh`
- `/tmp/cmp_logits.py` (max_abs / top-1 / top-64 overlap comparator) and
  `/tmp/diffprof.py` are scratch — re-create from §3/§4 if `/tmp` was cleared.

## 9. Loose ends worth a look (not scheduled)

- `vulkan/dense_extra_mmq.comp`'s IQ2_S branch uses `unpack_i8` on grid bytes that
  the oracles treat as unsigned. Harmless **only** because every byte of
  `q36_iq2s_grid` is ≤ 43 (< 128); a re-quantised table would silently break it.
- The repo has two "correct" IQ2_S decode postures: q8_K-quantised mid (matvec
  reference) and f32 mid (the `*_sum_decode` default). Parity numbers above are
  against both; the ledger records which is which.
