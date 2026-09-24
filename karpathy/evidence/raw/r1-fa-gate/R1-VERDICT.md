# R1 — the red `--vulkan-kernels` gate is a stale test expectation, not a kernel bug

## What was red

`./q36_test --vulkan-kernels` fails at `tests/q36_test.c:4958`, the
`n_tok`-invariance memcmp inside `test_vulkan_attn_case`: a batch (prefill)
attention row must reproduce the per-row decode step bit for bit. CAMPAIGN §1b/1c
record the same test as PASS, so the gate went red between those records and HEAD.

## Bisect

```
git bisect run /tmp/r1/check.sh HEAD 8e8e788      # 8e8e788 = last recorded PASS
```

`bisect-check.sh` rebuilds `q36_test` and exits 1 iff `q36_test.c:4958` appears in
the log. Result — first bad commit:

```
9d54dc7  vulkan: flash-attention prefill for GQA 8 MoE too (guard prefill +27.15% at ctx 8192)
```

Last good: `8e8e788`. Full log: `bisect.log`.

## Kernel level — which arm, and how big

An env-free `fprintf` diagnostic was added to `test_vulkan_attn_case`
(temporary, removed) printing, per arm, the batch-vs-decode diff count and the
max error against the f64 CPU reference that the test already computes.

Only one shape fails: `pos0=129 n_tok=3`, `k_type=Q36_KV_CACHE_Q8_0 (1)`,
`v_type=Q36_KV_CACHE_Q4_0 (2)` — i.e. the production KV pair, on the arm the FA
prefill kernel covers. `Q36_VK_ATTN_FA=0` → 0 of 12288 floats differ.

| arm | ndiff / 12288 | max batch-vs-decode | max_err vs f64 CPU ref |
|---|---|---|---|
| FA on  | 11137 | 1.02445e-08 | **1.18406e-08** |
| FA off | 0 | 0 | 1.62379e-08 |

FA's batch output is *closer* to the f64 reference than the qtile2 path it
replaced. The old bitwise assertion only held because qtile2 happened to reuse
the decode reduction order for its V accumulation; FA uses an LDS-staged tile
order and does not. Pure f32 reassociation.

## Gate after the fix

`./q36_test --vulkan-kernels` **exit 0, "vulkan-kernels: OK"**, with FA on and
with `Q36_VK_ATTN_FA=0` (`gate-fa-on.log`, `gate-fa-off.log`). Printed drift:

```
pos0=129  n_tok=3  ratio=8 = 1.02445e-08
pos0=129  n_tok=3  ratio=6 = 1.02445e-08
pos0=2100 n_tok=12 ratio=6 = 1.00117e-08
pos0=2100 n_tok=12 ratio=8 = 1.00117e-08
```

Test changes: `test_vulkan_attn_case` takes `n_head` (so both FA ratios are
reachable), a `fa_batch` predicate mirrors the dispatcher's condition
(`q36_vk_use_attn_fa()` is static to `q36_vulkan.o`, so the env default is
repeated), FA arms require `drift <= 4.0e-5f` — ~4000x the measured drift, four
orders inside the test's own 2e-3 quality bound — and every other arm keeps the
bitwise memcmp. Three arms added at the production KV pair
(`129/3` and `2100/12` at ratio 6, `2100/12` at ratio 8).

## End to end — is the drift inert?

Greedy, temp 0, seed 42, 64 tokens, ctx 1400, `tests/q36_test.c`-independent CLI
(`run-r1.s`, `e2e/`):

| model | FA on vs off, greedy 64 tokens |
|---|---|
| guard (ratio 8) | **byte-identical** |
| Swift (ratio 6) | differs at token ~35 |

Frontier dumps, chunk 256, ctx 512/520 (`e2e/frontier-cmp.txt`):

| model | ctx | max_abs_diff | top-1 | top64 overlap |
|---|---|---|---|---|
| guard | 512 | 0.650295 | same | 64/64 |
| guard | 520 | 0.614791 | same | 62/64 |
| Swift | 512 | 0.667795 | same | 61/64 |
| Swift | 520 | 0.854534 | same | 63/64 |

Determinism control (`ctl/`, 5 arms x 2 runs): the same arm reproduces its token
text exactly run to run (only the `q36: prefill/generation` timing line moves),
so Swift's token-35 divergence is a real FA effect, not run-to-run noise. The
`--prefill-chunk 128` arm is byte-identical to FA-on: a 32-token prompt never
reaches the chunking, so that control is a repeat, not an independent reference —
the reference for this drift remains the accepted FA landing's own NLL table
(`evidence/raw/fa-parity.txt`): FA moves teacher-forced NLL *less* than a plain
chunk 256→128 change does (Swift 3.08 vs 3.58, guard 2.01 vs 2.60 mean |dNLL|),
and at ctx 8192 the chunk-128 control is the arm that flips top-1, not FA.

`./karpathy/compat_gate.sh`: **PASS** (Swift 1024/177.46 prefill tps).

## Verdict

Stale test expectation. Keep FA (faster, and nearer the f64 reference); make the
test FA-aware under a tight numeric bound; widen it to the production KV pair at
both FA ratios. No kernel change. No performance delta (test-only).
