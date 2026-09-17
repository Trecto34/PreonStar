# Local Swift campaign ledger

This ledger starts from the completed Swift bring-up in `changelog.txt`.

| Result | Finding |
|---|---|
| IQ4_XS rejected | Bounded streaming ran, but dense-weight turnover was about 0.11 tok/s; resident loading triggered the host OOM/SIGKILL path. |
| IQ3_XXS accepted as baseline | Completed 11.67 GiB model; short protected smoke passed; q36-bench measured about 180 tok/s prefill and 23 tok/s generation at context 1024. |
| MTP draft=3 rejected as default | Slower than draft=1 on the measured workload. |
| attention span=128 rejected as default | Helped at context 1024 but regressed at context 2048; span 512 remains the stable default. |

Do not repeat these hypotheses without a new hardware or runtime reason.
Preserve MoE routing parity for the Qwen3.5/Qwen3.6 reference on every
accepted change, even when the primary target is dense Swift Qwen3.8.

## Interrupted experiment: 2026-09-17 12:00 (DeepSeek worker)

- **IQ3_XXS dense MMQ lane split — REJECTED/INTERRUPTED:** split the
  `dense_iq3_xxs_mmq` dequant staging across all 128 lanes. The first build
  failed because `half` is a GLSL reserved word; after renaming it, the
  candidate was bit-exact but regressed the down-projection microbenchmark by
  about 2.7% and did not establish an end-to-end win. The compatibility gate
  passed, but the worker was terminated before its required final report.
  Do not retry this lane-split mechanism without a measured explanation for
  the down-projection regression.

## BC-250 transfer audit: 2026-09-17

- **512-thread add/RMS — REJECTED:** Swift kernel 23.64 -> 30.13 ms.
- **Integer sign-mask IQ3_XXS decode — REJECTED:** Swift decode kernel
  373.83 -> 407.97 ms over 16 generation tokens.
- **XOR-swizzled IQ3_XXS MMQ staging — REJECTED:** Swift prefill MMQ kernel
  940.28 -> 1309.31 ms at 128-token context. The existing 17-element padded
  stride is faster than the attempted 16-element XOR layout on BC-250.
- **Global IQ2 grid lookup in MoE gate/up GEMM — REJECTED:** 128-token kernel
  time improved slightly (197.71 -> 194.26 ms), but 1024-token whole-model
  prefill was not reliably faster. Shared staging remains the default.
- **Smaller prefill chunks — REJECTED:** on 1024-token prompts, the existing
  watchdog-safe caps (Swift 256, Qwen3.6 1024) were fastest among tested
  chunk sizes. Do not raise Swift above 256 without a watchdog-safety proof.
