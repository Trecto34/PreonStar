# BC-250 Bonsai campaign evidence manifest (2026-09-18 continuation)

"Untracked" is not an archive.  This file pins the location and checksum of the
large artifacts that are intentionally kept on disk (gitignored or too large to
commit), plus the configuration needed to reproduce them.  Small text evidence
is committed alongside this manifest.

All paths are relative to the repo root `/home/server/q36-opt-27b`.

## Configuration / provenance

- Model: `gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf`
  sha256 `4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e`.
- Baseline source `99596b3`; retained P4 commit `b6e8de8`; report/evidence
  `1bfc787`, `ab7291a`.  Baseline binary
  `logs/bc250-sustained-20260918/baseline/q36-bench`
  sha256 `50f71990f4ffaef40e9bfcb842a32e93a2ecf39c8ee803bc7735253f558a7ff1`.
- Retained Q2_0 MMQ shader `vulkan/dense_extra_mmq_q2_0.spv`
  (gitignored; `make` rebuilds it from `vulkan/dense_extra_mmq.comp`):
  P4 sha256 `1f4b5c63ee2a5ff5624ff96e25267772c2b99dc43ed59301abe828aea151f76a`;
  baseline sha256 `a3bd761d6f831976e28a9a73b966a7aff2a6ea3b7baf5da688bbf6e28e07759c`.
- Primary command: `./q36-bench --vulkan -m gguf/Ternary-Bonsai-2-27B-Q2_0-g64.gguf
  --prompt-file tests/long_context_story_prompt.txt --ctx-start 512 --ctx-max 512
  --ctx-alloc 641 --prefill-chunk 256 --gen-tokens 128` under
  `flock -w 900 /tmp/q36-gpu.lock`, entry edge temp <= 55 C.
- Telemetry sampled from `/sys/class/drm/card*/device/hwmon/hwmon*/{temp1_input,freq1_input,power1_average}`
  and `/sys/class/drm/card0/device/mem_info_gtt_used`.
- No firmware, driver, voltage or clock changes were made.

## Large artifacts kept on disk (with sha256)

Run the checksum refresh with:
`(cd logs/bc250-sustained-20260918 && sha256sum $(cat MANIFEST.artifacts) > MANIFEST.sha256)`

Run `sha256sum` to verify against `MANIFEST.sha256` (generated from the same
file list).  Artifact list in `MANIFEST.artifacts`:

- `runs/cont-timeline-decode.txt.gz` — per-dispatch GPU timeline, gzip text.
  Config: primary command with `Q36_VK_PROF_KERNEL=1 Q36_VK_PROF_DECODE=1
  Q36_VK_PROF_TIMELINE=<path>`; timestamp period 10 ns.
- `runs/par-base/frontier_000{200,400,600,800,1000}.logits.json` — baseline
  Q2_0 MMQ extended-shape frontier logits (n_tok=200 tails).
- `runs/par-p4/frontier_000{200,400,600,800,1000}.logits.json` — P4 counterpart.
- `runs/p1-logits-{base,candidate}`, `runs/p2-logits-{base,candidate}`,
  `runs/p3-logits-{base,chunk512}`, `runs/f1-logits-{base,candidate}`,
  `runs/d2-diag-{base,span256}` — earlier campaign logits retained from the
  prior commit set.
- `baseline/q36-bench`, `baseline/vulkan/*.spv`, `hadamard_parity`,
  `read_copy_bw`, `swiglu_hadamard_parity`, `mmq_info` — preserved binaries
  (see `baseline/sha256.txt` for the baseline set).

## Reproduce the key measurements

- P4 A/B: raw `runs/ab-p4-lane-remap.csv`; procedure in
  `runs/p4-lane-remap-parity.txt`.
- Context A/B: `runs/ctx-ab-2048-2560.csv`, `runs/ctx-ab-1024-3072.csv`,
  `runs/ctxsweep-r{1,2,3}.csv` (+ `-tele.csv` clocks).
- Profiling overhead: `runs/prof-overhead.csv`.
- Secondary benchmarks: `runs/secondary.csv`, `runs/memgrowth-frontiers.csv`.
- Resource stats: `runs/mmq-resource-stats.txt` from `mmq_info.c`.
- Preparation audit + trim: `runs/audit-profile-512-128.txt` (baseline
  dispatch/time table), `prep-trim-rejected.patch` (the default-off
  `Q36_VK_PREP_TRIM` path, source reverted), `runs/ab-prep-trim.csv`,
  `runs/prep-trim-profile-decode.txt`, `runs/prep-trim-gen-{base,candidate}.txt`.
- Decode coalescing/latency audit (2026-09-18): `shader_isa.c` (ACO ISA dump via
  `VK_KHR_pipeline_executable_properties`; build `cc -O2 -o shader_isa
  shader_isa.c -lvulkan`, run `./shader_isa <shader.spv> <push_bytes>`),
  disassembly `isa/dense_extra_decode_q2_0.{baseline,clamp}.asm` and
  `isa/dense_extra_decode_q2_0.baseline.aco-ir.txt`, rejected candidate
  `decode-clamp-rejected.patch`, A/B raw `runs/ab-decode-clamp.csv`
  (10 soak-gated interleaved pairs, decode median paired +0.34 % -> FAIL).
  Parity for that candidate: frontier-512 logits `max_abs_diff = 0`
  (n=248,320) and greedy generation byte-identical, sha256
  `7f316c5cd7b7d5d560c49fe39129254dfbd78f83cdc822a1baa28ff33f7710dc`.
  Analysis: `reports/decode_coalescing_audit.md`.
- Attention parity / D2 regression: `attn_parity.c` + `attn_parity_cmp.py`
  + `attn_parity_test.sh`; raw span dumps `runs/attn-512.bin`,
  `runs/attn-256.bin`, reference lines `runs/attn-parity-ref.txt`, cross-span
  comparison `runs/attn-pair-512-256.txt`.  Build/run:
  `logs/bc250-sustained-20260918/attn_parity_test.sh` (needs `.o` files from
  `make`).
