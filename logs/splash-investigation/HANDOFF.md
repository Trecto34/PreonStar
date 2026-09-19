# Splash / Qwen3.6-Qwen3.8 Investigation — HANDOFF

## Context

Track B (PTQ1_0/PQ2_0 ternary kernel optimization, branch `trackB-ptq1_0`,
worktree `/home/server/q36-opt-27b`) is PAUSED as of 2026-09-19: diminishing
returns, PQ2_0 already faster than PTQ1_0. Full history in that worktree's
`logs/bc250-sustained-20260918/{ptq1_0,pq2_0}-validation/HANDOFF.md` and the
Maestri canvas note "Track B Coordination". Closing state also persisted to
mem0 (user_id `q36-opt-27b`), memory ids:
- `99024eda-9584-4f8e-a836-401bd42dcacd` — pause/pivot decision
- perf win (commit 43dcf03) and negative-results (commit 8d527f4) entries also
  written the same session.

Active focus moves to Qwen3.6/Qwen3.8 quantized-model optimization, informed
by techniques from https://github.com/incoai/splash.

## Splash — what we know so far (README/metadata only, no source read yet)

- Apple-silicon-only local inference engine, Apache-2.0.
- Per-model specialized kernels (not a generic ggml-style dispatcher).
- Ships a draft model alongside the target model — implies speculative
  decoding is core to its throughput story, not incidental.
- Has an explicit "memory plan" concept (unclear yet what this controls —
  KV cache layout? weight residency? needs source read).
- Claims 2x next-fastest engine decoding Qwen3.8-27B on a 48GB M5 Pro; 282ms
  TTFT at 32K cached context.
- OpenAI/Anthropic-compatible API server: `/v1/chat/completions`,
  `/v1/responses`, `/v1/messages`, `/tokenize`, `/apply-template`.
- Install: `brew install incoai/tap/splash`; serve:
  `splash serve --model incoai/Qwen3.8-27B-Splash`.

Note there is an existing separate worktree/branch for Qwen3.8 work already:
`/home/server/q36-opt-qwen38` (`feat/qwen38-opt`, marked prunable) — check
that before duplicating effort; this investigation may fold into it rather
than stay standalone.

## Team / workflow carried forward from Track B

Maestri canvas roster: Builder, Mcode, Reviewer, Grendel (ruthless Principal
Engineer judge), Big Pickle (mem0 memory archivist). Same discipline applies:
independent review before any GPU test/commit, never trust an agent's
self-reported benchmark numbers without reading the actual log file, profile
before optimizing, one GPU/benchmark job running at a time (this box OOM'd
twice under concurrent q36-bench + 5 agent CLIs — 14GB RAM), and favor fast
capped-scope results over long thorough runs.

## Status

Investigation just started. No source clone of splash yet, no concrete
kernel/technique picked to port, no benchmarking done in this worktree yet.
