# Ternary-Bonsai-2-27B-PTQ1_0-mtp-procreations.gguf — 2026-10-27

Model: 65-block qwen35 hybrid (16 full-attention + 48 recurrent/SSM), all dense
weights PTQ1_0 (type 143, 1.75 bpw, 5.95 GB), prism Hadamard rotation on every
dense weight except `token_embd`, plus an in-file nextn head at `blk.64`.

Baseline (p36 HEAD, ctx 512, `tests/long_context_story_prompt.txt`,
`--ctx-alloc 641 --prefill-chunk 256 --gen-tokens 128`):
prefill 94-100 t/s, decode 29.4-32.1 t/s (spread is thermal, not code).

## What was broken and fixed

1. `--mtp-draft >= 3` (mtp_cap >= 2, i.e. an actual multi-row verify) aborted
   with `MTP draft failed at pos N`. The in-file head's output projection called
   `p36_gpu_tensor_matmul_scaled`, which does not apply the prism Hadamard
   rotation that `output.weight` carries; the target head has the branch, the
   MTP head did not. Fixed in `p36_mtp_forward_vulkan`: rotate+quantise via
   `p36_hadamard_prepare` and call `p36_gpu_tensor_matmul_q8_or_float_scaled`,
   mirroring the target head. Greedy output with `--mtp-draft 3 --mtp-margin 0`
   is now byte-identical to MTP off.

2. PTQ1_0 had no small-batch matmul: every `n_tok != 1` forward (MTP verify,
   small incremental prefill) went through the 64-token MMQ tile, which costs
   the same for 2 rows as for 64. Added `vulkan/dense_extra_decode_ptq1_0_nx.comp`
   (ROWS=4 output rows, NT=2 tokens, one ternary-block stream per workgroup) and
   wired it for `n_tok <= 8` plus the ragged tail (< = 8 rows) past whole 64-row
   tiles. Per-row arithmetic, k-order and `subgroupAdd` are identical to the
   one-token kernel; verified greedy-identical (`--prefill-chunk 2|4|8` vs `1`).

## Measured (see small_batch.csv)

* 2-row PTQ1_0 forward: **~550 ms (MMQ tile) -> 57-60 ms (nx)** = ~10x. A
  `--mtp-draft 3` run that used to decode at 2.80 t/s now decodes at 22.4 t/s.
* Marginal cost of an extra row in a batched PTQ1_0 forward is **~24 ms**,
  i.e. a full 1-row weight pass (6.39 GB / 24 ms = 266 GB/s). The ternary
  kernel does not get the second row for free.

## MTP verdict: REJECTED for this model

`--mtp-draft 3 --mtp-margin 0` (2-row verify): 20.3-22.4 t/s vs 29.4-32.1 plain
= 0.66-0.70x. `--mtp-draft 4/5` worse (15.9 / 12.4 t/s). `--mtp-draft 3` with
the default margin gate: 28.2 t/s. Acceptance at ctx 512 over 128 tokens:
1.54 tokens/cycle (histogram 1:38, 2:45 of 83 cycles).

Why it loses, with numbers:
* A 2-row verify costs 1.8x a 1-row forward (57-60 ms vs 31-35 ms), because the
  extra row costs a full weight pass (above). MTP needs ~1.1x to break even at
  1.54 tokens/cycle.
* 48 of 64 layers are recurrent, so a partial accept cannot roll the state back;
  the cycle restores the frontier snapshot and replays the accepted prefix, a
  full 28.5 ms forward. 46% of cycles are partial: 83 cycles = 45x57 + 38x85 ms.
* Break-even at the measured acceptance needs a 2-row verify under 35.5 ms
  (13.5% faster than a 1-row forward) *with* the replay still in place, or a
  replay-free rollback plus a <=36 ms verify for ~1.35x.

`reconsider_if`: a trit kernel that adds a verify row for <2 ms (no extra weight
pass), or per-row recurrent-state checkpoints inside the batched SSM kernels so
a partial accept rolls back without replaying the forward.

## Where the 1.75 bpw model fails to convert its byte advantage (2026-10-27)

Verified with a throwaway diagnostic build (trit decode replaced by a constant,
then reverted): prefill **100.03 t/s with the unpack deleted** vs 98.7-100
baseline, so the scalar `(qq*3)>>8 - 1` trit decode is free - prefill is bound
by the packed-int16 MAC loop, not by unpacking.

* PTQ1_0 tile kernel does `sum += i16vec2(a)*i16vec2(b)` = **2 MAC/lane/instr**
  and reaches 5.45 TFLOPS (5.13 s for a 512-token prefill = 28 TFLOP). The
  IQ3_XXS sibling runs int8 `dotPacked4x8` = **4 MAC/lane/instr** at ~180 t/s on
  the same 27B dense shape, i.e. ~2x. Same FLOPs, half the MAC rate.
* Decode reads 5.95 GB per token in 23.5 ms of matmul = **253 GB/s** of one
  pass (not two: 2x would be 506 GB/s, above the 448 GB/s GDDR6 peak). The
  IQ3_XXS decode kernel measured 333 GB/s on the same box, so ~1.3x is on the
  table from the 28-byte trit block load pattern (and ~1.8x if the 448 GB/s
  peak were reachable).
* So there is no duplicated work: the deficit is MAC width (prefill) and sector
  efficiency (decode), and both live in the PTQ1_0 kernels, not in the model.

## Dense-vs-MoE reality check (same box, same engine, ctx 512, 32 gen)

| model | file | prefill t/s | decode t/s |
|---|---:|---:|---:|
| Ternary-Bonsai 27B PTQ1_0 (dense) | 5.95 GB | 99.1 | 33.6 |
| Cyber-Tiel-Coder 35B-A3B APEX-I (MoE, ~3B active) | 12.55 GB | 435.4 | 79.8 |
| Occamy 35B-A3B APEX-I (MoE, ~3B active) | 14.75 GB | 410.0 | 77.5 |

The 13-15 GB models are not dense: they are 35B-A3B MoE, so a token touches
~3B weights, not 35B. Bonsai touches all 27B (5.95 GB) per token. Per token the
MoE does ~9x fewer MACs and ~4x fewer weight bytes, which is why it is 4.4x
faster at prefill and 2.4x at decode *despite* being twice the file size. The
Bonsai numbers are the dense-27B-at-1.75-bpw numbers, not a pathology.

## Kernel speed experiments, both reverted (2026-10-27)

1. **PTQ1_0 prefill tile as int8 `dotPacked4x8` — no win.** Rewrote
   `dense_extra_mmq_ptq1_0.comp` to stage trits as int8 quads (4 trits/word,
   layout derived from `p36_dequantize_row_ptq1_0`: element p = 16n + b, quad =
   4 consecutive b at one n; sections 80/40/8 as in the portable decoder) and
   accumulated with `dotPacked4x8EXT`, cutting the inner loop from 512 packed
   i16 madds to 256 dots per 32-weight slice. Measured prefill **85.7 t/s vs
   98.0** for the i16 kernel on unchanged instruction counts => RADV gfx1013
   `v_dot4` is not wider than 2x `v_mad_i16` in practice. Reverted; the i16
   kernel already runs at 5.4 of ~6.1 T-MAC/s packed-i16 peak.
   Verification note: kernels load from `vulkan/*.spv` at run time, so a
   retained baseline binary inherits whatever `.spv` is on disk - A/B runs must
   restore the `.spv` before comparing text.
2. **One-token kernel ROWS 4 -> 8 (grid /8) — no win.** 33.10 t/s vs 33.10
   baseline at ctx 512, 64 gen. Reverted.

Conclusion: prefill is at its packed-i16 roofline and decode at 253 GB/s of
single-pass trit traffic; neither has an implementation bug to reclaim. Beating
either needs a different data layout (32-byte aligned weight blocks so decode
loads one sector per block) - a format change, not a kernel tweak.

3. **Tile kernel LDS vectorisation (uvec2, 4 weights per 8-byte load) — layout
   rejected, not measured.** Halved the LDS instruction count (12 -> 6 per
   k-step: 4 weight pairs + 4 activation pairs per two k-steps) while keeping
   the proven "slot 2k holds elements 2k,2k+1" mapping, so arithmetically a
   superset of the reference. Output was degenerate ("5000..."), identical to
   the dot4 attempt's failure mode, i.e. the *re-layout* itself is the problem,
   not the MAC width. Both re-layouts touched only buf_a/buf_b element type,
   stride and staging; the reference (i16vec2, stride 65/17) is byte-exact
   again after revert. Do not retry a staging rewrite without a kernel-level
   unit test (build one PTQ1_0 tensor, run the tile kernel against the
   one-token kernel for a few shapes, compare values).

## Tile kernel vs one-token kernel: same answer, different rounding (2026-10-27)

`--prefill-chunk 1|8` (one-token / nx kernels) vs `64|256` (mmq tile) change the
first greedy tokens ("consists" vs "contains"), which looked like a tile-kernel
bug. It is not: `--dump-logprobs` on step 0 gives max |dlogp| = 0.106 over the
top-20 (The -0.0002 vs -0.0003, Let -9.88 vs -9.78), i.e. reassociation-level
drift that flips one borderline argmax over 24 tokens. Consequence: greedy text
is NOT a usable oracle for tile-kernel edits - a numeric unit test (one tensor,
tile kernel vs one-token kernel, compare y) is required before any further
staging rewrite. Building that harness is the next step, not another blind
re-layout.

## Oracle for PTQ1_0 kernel edits - added real tile coverage (2026-10-27)

`tests/p36_test.c:3684` only tested `token_counts[] = {1,2,4,8}`, i.e. always the
small-batch nx path; `ref=ptq1-mmq` there names the candidate, `want` is the CPU
dequantized reference. Extended to `{1,2,4,8,16,32}` so tokens 16/32 take the
`n_tok > 8` mmq tile kernel. Passing on the current kernel:

    PTQ1_0 402 eligible, 3 tensors (output.weight 5120, blk.0.ffn_down 17408,
    blk.0.ssm_out 6144), 18 token cases
    tile cases: max_abs 2.1e-07, rel_rms 1.3-2.6e-07, tol 4.9e-04

So the mmq tile kernel is CPU-exact and this is now a ~40 s CPU-referenced
oracle for staging rewrites (add/remove token counts to steer which path runs).

## Prefill chunk-size sweep (2026-10-27)

ctx 512, 8 gen: chunk 64 = 92.0, 128 = 98.5, 256 = 97.6 t/s. Per-chunk overhead
is ~6% at 64 and already amortised by 128; 256 is not faster than 128. Chunk
size is therefore not a lever - the 2x is not in dispatch/barrier overhead.

## Hazard barriers: 36% of decode is genuine serialisation (2026-10-27)

Profiled the batch path (`P36_VK_PROF=1`): 15076 dispatches / 12 forwards with
~1 barrier per dispatch, all emitted as one global `VkMemoryBarrier`
(`SHADER_WRITE|TRANSFER_WRITE -> SHADER_READ|WRITE`, COMPUTE/TRANSFER ->
COMPUTE) into a *single* batched command buffer (`p36_vk_begin_batch_unlocked`,
flush only at host read), so there is no per-dispatch submit/fence ramp to
remove.

* `P36_VK_SKIP_BARRIERS=1`: decode **43.05 t/s vs 31.63** (+36%) - an upper
  bound only; it removes real RAW ordering.
* Tried narrowing scope: per-`VkBufferMemoryBarrier` (WHOLE_SIZE) for just the
  roots bound to the dispatch, entries left pending when not all written roots
  were covered, global fallback otherwise. Correct (oracle OK, greedy text
  identical) but **31.42/30.43/29.88 vs 33.16/31.95/30.99 t/s = -4%** in
  interleaved A/B. Reverted.
* Conclusion: the +36% is pipeline drain / loss of dispatch overlap on genuine
  read-after-write edges in the per-token dataflow chain, not cache-flush scope.
  Recovering it needs the dependencies removed (fuse `hadamard_prepare` - 254
  dispatches + barriers per token - into the consuming matmul, or double-buffer
  the activation chain), not a cheaper barrier form. Prefill chunk sweep already
  ruled out plain dispatch overhead (92.0/98.5/97.6 at chunk 64/128/256).

## Hadamard fusion: ceiling +21% decode, two cheaper routes rejected (2026-10-27)

Diagnostic switch added (env-gated, default off, mirrors P36_VK_SKIP_BARRIERS):
`P36_VK_SKIP_HADAMARD=1` leaves `rt->had_q8` stale, i.e. it removes the
`hadamard_prepare` dispatches *and* their barrier edges while keeping every
consumer and all weight streaming intact.

    decode: 37.51 / 37.47 t/s with the stub vs 31.22 / 30.74 baseline = +21%
    (254 prepare dispatches+barriers per token, ~25 us each)

Cheaper barrier-level routes tried and rejected under the 8% rule:

* buffer-scoped `VkBufferMemoryBarrier` instead of the global barrier: correct
  but -4% (reverted earlier).
* execution-only barrier (access masks 0, ordering only): **wrong output**
  greedy text diverges from the reference build, so RADV does need the cache
  masks for cross-dispatch visibility. Reverted.

So the +21% is only reachable by removing the dependency, i.e. fusion. Design
(ready to implement, no cheaper variant exists):

* Call site: `p36_forward_full_attn_vulkan_model` (core_15 ~line 660)
  `p36_hadamard_prepare(rt, inp, P36_N_EMBD, n_tok)` -> `rt->had_q8`, consumed by
  `p36_gpu_tensor_matmul_q8_or_float_scaled` for q/k/v with in_dim = P36_N_EMBD.
  Same pattern for ffn_shared_mid (P36_N_FF_SHARED), attn_out (P36_N_SSM_INNER),
  recur_proj (grouped variant).
* Fusion target: the PTQ1_0 consumer kernels (`dense_extra_decode_ptq1_0.comp`
  first, then the mmq tile kernel), because a consumer workgroup already reads
  the *whole* activation row - a row is spread across many producer workgroups,
  so the producer side cannot own the transform.
* What to port from `vulkan/hadamard_prepare.comp`: signs x inv_sqrt_block
  multiply, the 10-stage in-LDS FWHT over 1024 elements, the per-32 scale with
  fp64 division, `nearest_int` + `pack_i8x4`, and the bsums packing (74-word
  q8_K block layout). 1024 elements = 4 q8_K blocks = 4 KB LDS, plus the
  2 KB of trit LUTs the decode kernel already keeps.
* Keep the push-constant flag so the host can select fused vs dispatch per
  tensor, and verify in this order:
  `./p36_test --q8k-activation-parity` (bit-exact quantizer),
  `./p36_test --dense-quant-model-rows` (kernel numerics, 18 PTQ1_0 cases),
  greedy text vs /tmp/ref_out.txt, then the interleaved bench with the 8% rule.

## Prepare call-site audit (2026-10-27) - what fusion must cover

All `p36_hadamard_prepare` call sites in core_15 (width, source tensor):

   331  inp           P36_N_EMBD       (pre-attention)
   411  ffn_shared_mid N_FF_SHARED     (after shared-FFN mid)
   664  inp           P36_N_EMBD       (pre-attention, other block type)
   760  attn_out      P36_N_SSM_INNER  (after attention out)
   829  inp           P36_N_EMBD       (pre-attention)
  1065  recur_proj    P36_N_SSM_INNER  (grouped variant, after recurrent proj)
  1369  out_in        P36_N_EMBD       (multi-row)
  2051  mtp_head      P36_N_EMBD       (MTP head, rows=1)

None of these sources is `rt->norm`: the only fused norm+q8 dispatches
(`p36_gpu_add_rms_norm_q8_k_tensor`, core_15:1250 and :1663) produce
`rt->norm` + `rt->inp_q8` for the *post-attention* norm. So the norm-fusion
sketch only covers the three `inp` sites and only if a pre-attention norm+k8
runs on that exact tensor - that producer must be traced before writing code.

Kernel facts for the implementation (verified in vulkan/add_rms_norm_q8_k.comp,
188 lines): layout(local_size_x = 1024), one workgroup per row, keeps the
normalized row in `shared float sh_val[1024]`, quantizes 16 Q8_K blocks per
round (64 threads/block), loops rounds when width > 4096. A 1024-element FWHT
therefore maps 1:1 onto one `sh_val` chunk for EMBD 5120 (5 chunks) and
N_SSM_INNER 6144 (6 chunks), no extra LDS needed beyond signs staging - the
reuse the spec asked for. `nearest_int`, the fp64 scale and `pack_i8x4` already
live in that kernel, so the fused epilogue is a copy of hadamard_prepare's
phase 2 plus a 10-stage butterfly over sh_val.

Recommended shape of the patch (not yet written): add a second kernel
`add_rms_norm_q8_k_had.comp` (copy + fused epilogue, new `signs`/`had_q8`
bindings, `fused` push field) so the existing 188-line kernel and every current
call site stay untouched, and select it from the host only where the rotate flag
is set and the traced producer is the norm.
