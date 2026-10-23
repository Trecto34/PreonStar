# Backends

P36 is multi-runtime at the source and API level, but Metal and Vulkan are
separate native builds rather than one fat executable. Each build produces the
same command names—`p36`, `p36-server`, `p36-bench`, `p36-agent`, `p36-eval`,
and `p36_test`—linked to the selected graph runtime:

| Platform/runtime | Build | Explicit invocation |
| --- | --- | --- |
| Linux / generic Vulkan | `make` or `make vulkan-generic` | `./p36 --vulkan -p "Hello"` |
| Linux / BC-250 checked Vulkan | `make vulkan-bc250` | `./p36 --vulkan -p "Hello"` |
| macOS / Metal | `make metal` | `./p36 --metal -p "Hello"` |
| CPU reference | `make cpu` | `./p36 --cpu -p "Hello"` |

Building another runtime overwrites those local executable names. Use separate
worktrees or copy build artifacts if Metal, Vulkan, and CPU binaries must be
kept side by side. The `--backend metal|vulkan|cpu` spelling is equivalent to
the short backend switches and is accepted by the CLI, server, agent,
benchmark, and evaluation harness. A Metal-linked binary rejects `--vulkan`,
and a Vulkan-linked binary rejects `--metal`, so deployment mistakes fail
before model loading.

## Vulkan device and compatibility policy

The generic and BC-250 targets contain the same Vulkan backend and SPIR-V.
`vulkan-generic` queries device capabilities and uses portable fallbacks when
the subgroup or 16-bit features required by a tuned kernel are absent. A
BC-250 is recognized at runtime and keeps the current optimized path.
`vulkan-bc250` adds a vendor/device check so a board-specific release artifact
cannot silently run on different hardware.

The generic target removes the hard device identity restriction, not the need
to validate a new GPU. Before calling another Vulkan device supported, run the
isolated kernel suite and the short CPU/GPU parity gate, then record prefill
and decode throughput at the intended context sizes.

### Prefill chunk safety clamp

On GFX1013 the prefill chunk is clamped (to 1024, or 256 when the model's GQA
ratio is not 8) after any `--prefill-chunk` override, because the override is
exactly the value that can push one attention dispatch past the kernel watchdog
and destroy the GPU context. The clamp also feeds the context-memory estimate,
so scratch is sized for the chunk that will actually run rather than the one
requested — scratch grows with chunk width, and sizing for an unclamped request
reserves memory that is never touched.

`P36_VK_PREFILL_CHUNK_UNSAFE=1` restores the raw value. It is not recommended:
beyond the watchdog risk, a chunk wider than the clamp has measured *slower* on
this hardware as well as larger, so the clamp costs nothing in practice. Sweep
`--prefill-chunk` on your own board if you want to confirm the optimum for it.

### Optional GPU fast paths

Two kernel variants trade bit-exactness for speed. Both are **off by default**,
both are available as flags on every front end (`p36`, `p36-server`, `p36-bench`,
`p36-eval`, `p36-agent`), and both also read an equivalent `P36_VK_*` environment
variable.

| flag | what it changes | exactness |
| --- | --- | --- |
| `--f32-fast-wide` | Runs the f32 matvec with a 256-thread workgroup instead of 64. The narrow form dispatches a single wave for the router and shared-expert gates, which cannot cover memory latency. | **Not bit-exact, and known to fail in production use.** Four subgroup partials are combined through shared memory instead of one subgroup reducing alone, and this kernel computes the MoE router gate, so a rounding flip can change *which experts* a token selects. Treat it as a benchmarking switch only. Validate with the evaluation harness, not with a logits diff. |
| `--attn-span N` | Split-K span width in keys for decode attention (default 512). Narrower spans raise occupancy, since decode dispatches `n_head * spans` workgroups. | **Not bit-exact.** `attn_combine` reduces the per-span partials sequentially in f32, so a different span count regroups that sum. |

`--f32-fast-wide` is the more dangerous of the two and the smaller: isolated
per-kernel profiling puts it at about **half a percent of GPU time** on one
BC-250 (whole-run comparisons overstated it). Its danger is not hypothetical —
it perturbs the router softmax input, so a rounding flip can change which
experts a token selects; on one BC-250 it caused a degenerate failure in long
tool-calling chat (sampled distribution collapsed, generation ran to the token
budget). A short-context logits diff will not catch this. Enable it only for
measurement runs, never for a served endpoint.

Prefill attention uses the flash-attention kernel by default (see
[Speed](#long-context-prefill-flash-attention)). It is **not bit-exact** with
the previous `attn_prefill_qtile2` kernel: f32 sums are added in a different
order. On a teacher-forced next-token check at 8k it moved scores less than
changing `--prefill-chunk` from 256 to 128 does. Set `P36_VK_ATTN_FA=0` to go
back for A/B comparisons.

`--attn-span` trades sequential `attn_combine` rescale cost against split-K
occupancy: narrower spans help at short context and can hurt at long context
(measured **+2.2% decode at ctx 2048, +1.2% at 8192, -1.9% at 16384** on one
BC-250 at span 128 vs. the default 512). Prefill is unaffected. Measure both
flags on your own board at the context length you actually run — benefit
depends on CU count, thermal headroom, and context.

## Metal device and compatibility policy

Like DS4, a normal local Metal process uses `MTLCreateSystemDefaultDevice`.
Apple Silicon exposes its unified GPU as one logical Metal device, so P36
needs no device list for M1/M2/M3/M4+; one process owns one default device
and one model, with no layer split or multi-Mac mode.

The Metal binary targets macOS 11; newer facilities (macOS 15 residency sets,
`MTLMathMode`) are selected only after a runtime availability check, and host
code defaults to the Apple M1 instruction baseline so a binary built on a
newer Mac stays usable on M1 (`NATIVE_CPU_FLAG=-mcpu=native make metal` for a
local-only build). Routed MoE prefill uses DS4's expert-major batch-MM
dispatch where the shape is supported, falling back to the exact matvec path
otherwise (`P36_METAL_MOE_MM=0` forces that fallback). Cache auto-sizing uses
`recommendedMaxWorkingSetSize`; all model buffers use unified storage, with
no discrete-GPU/eGPU path.

Metal shaders compile at runtime from `metal/*.metal` — run frontends from
the project tree and ship the `metal` directory with binary packages. A
frontend that changes working directory before model startup must set
absolute paths via `P36_METAL_DENSE_SOURCE`, `P36_METAL_MOE_SOURCE`,
`P36_METAL_NORM_SOURCE`, `P36_METAL_OPS_SOURCE`,
`P36_METAL_RECURRENT_SOURCE`, `P36_METAL_KV_SOURCE`, and
`P36_METAL_ATTN_SOURCE`.

## CPU reference runtime

Do not treat the CPU path as the production target. The CLI and `p36-server`
support the CPU backend for reference/debug use and share the same KV
session and snapshot format as Metal and Vulkan, but normal inference should
use a GPU graph runtime.
