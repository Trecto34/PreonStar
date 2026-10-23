# Architecture

PreonStar is a single-binary-per-role inference engine written in C99. There is no
framework underneath it and no Python in the hot path: the model loader, the
quantization kernels, the graph executors, the HTTP server, and the agent all
live in this repository.

## Design constraints

Three decisions explain most of the code layout.

**One translation unit per binary.** `p36.c`, `p36_server.c`, `p36_agent.c` and
`p36_vulkan.c` are manifests that #include ordered fragments from `src/core`,
`src/server`, `src/agent` and `src/vulkan`. The fragments are not separate
compilation units on purpose: this build has no LTO, so promoting a function into
its own `.c` file would force it to be exported and would give up inlining on the
decode and prefill paths. The split is for navigation; the compiler still sees
one file.

**The backend is chosen at build time, not run time.** `make` builds with Vulkan,
`make metal` builds with Metal. Both produce the same six command names linked
against a different graph runtime, so the CLI, server, agent, benchmark, and
evaluation harness are backend-agnostic above that line.

**Specialization beats generality.** The engine implements the graph shapes it
needs for the supported model families instead of a general GGML runner. That is
what keeps the kernels small enough to tune, and it is why model support is
opportunistic rather than a matrix.

## Modules

| module | manifest | owns |
|---|---|---|
| core | `p36.c` | GGUF loading and metadata, quantization and dequant kernels, CPU forward pass, RoPE, sampling, sessions |
| server | `p36_server.c` | HTTP and SSE, Anthropic and OpenAI request parsing, tool schemas, trace events, disk KV cache |
| agent | `p36_agent.c` | agent loop, tool dispatch, workspace jail, editor, history |
| vulkan | `p36_vulkan.c` | device and pipeline setup, tensor arena, attention and MoE dispatch |

Each directory has a `README.md` listing every fragment. The manifests must not be
reordered: the include order is the declaration order, and later fragments use
types declared in earlier ones.

## Execution pipeline

1. **Load.** The GGUF is memory-mapped. Tensor metadata is read directly from the
   file; nothing is copied until a tensor is needed.
2. **Bind.** `weights_bind` resolves each named tensor to a typed view and records
   its quantization format and row layout.
3. **Plan memory.** Resident mode pins all weights in device-visible memory.
   Streaming mode keeps a bounded expert cache and faults layers in as the forward
   pass reaches them (see [streaming.md](streaming.md)).
4. **Forward.** Per token: embedding, per-layer attention and MoE, final norm,
   logits. On GPU the whole prefill is dispatched as a small number of graph
   submissions; decode runs one token per submission.
5. **Sample.** Greedy, top-k, top-p, min-p, repeat penalty, with optional MTP
   speculative decoding.

## KV cache

Two independent choices:

- **Type.** Keys and values can each be `f16`, `q8_0` or `q4_0`. The GPU default is
  q8_0 / q4_0; CPU defaults to f16. See [configuration.md](configuration.md).
- **Where it lives.** In process memory, or on disk via `--kv-disk-dir` so a
  restarted server or agent can resume without re-prefilling. Disk entries are
  content-addressed by prompt prefix.

The agent holds the transcript and the live KV state in one session rather than
talking to itself over a socket, which is why switching a saved session can skip
prefill entirely.

## Concurrency

- `p36-server` runs one graph worker by default, with a single mutable checkpoint.
- `--batched-session N` creates N independent resident sessions that share the
  scheduler; `--mixed-prefill-quantum` controls how many prefill tokens a session
  runs per turn when another session is decoding.
- The CLI and agent are single-session by construction.

## Where to look next

- [backends.md](backends.md) -- Vulkan, Metal and CPU differences and requirements.
- [benchmarks.md](benchmarks.md) -- how to measure a change.
- [configuration.md](configuration.md) -- every runtime flag.
