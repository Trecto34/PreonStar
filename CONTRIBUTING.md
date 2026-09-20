# Contributing

PreonStar changes should be tested against the failure mode they can
realistically affect. The project has two regression tracks: correctness and
speed. Include the commands you ran, the machine/backend, the model quant, and
any notable failures in the PR or commit notes.

Do not send changes affecting inference, Vulkan kernels, prompt rendering, KV
cache, or the server API without checking that the resulting code is still
correct. Speed regressions are acceptable only when they fix an important
correctness bug and the tradeoff is explicit.

## Correctness Regression Tests

Build both Vulkan targets first. The generic binary still selects the BC-250
fast path automatically when it sees that device; the checked target rejects
other devices at startup.

```sh
make clean
make vulkan-generic
make vulkan-bc250
```

The C test runner is `q36_test`. Running it without arguments is equivalent to
`--all` and includes model-backed groups. The default Makefile target runs the
model-independent evaluator and agent tests plus selected Q36 unit, cache,
protocol, fixture, and server groups:

```sh
make test                  # default model-independent suite
./q36_test --all           # every registered test; model/Vulkan required
```

Useful narrower checks:

```sh
./q36_test --server
./q36_test --logprob-vectors
./q36_test --long-context
./q36_test --tool-call-quality
./q36_test --qwen-tool-call-quality
./q36_test --vulkan-kernels
./q36_test --gpu-cpu-parity
```

What they cover:

- `--server`: request parsing, Qwen chat rendering, streaming, native tool-call
  parsing, thinking controls, disk-KV cache bookkeeping, and other server-side
  logic. This is the best quick check for API and prompt-rendering changes.
- `--logprob-vectors`: compares local token bytes and top-logprob slices against
  the llama.cpp reference fixtures in `tests/test-vectors/`. This catches
  tokenizer, template, attention, and logits regressions.
- `--long-context`: runs the long-context security continuation regression from
  `tests/long_context_security_prompt.txt` unless `Q36_TEST_LONG_PROMPT` is set.
- `--tool-call-quality`: exercises actual model behavior for tool-call emission
  in both fast and exact paths.
- `--qwen-tool-call-quality`: exercises native Qwen tool-call emission through
  the OpenAI-compatible request path.
- `--vulkan-kernels`: isolated Vulkan kernel numeric checks.
- `--gpu-cpu-parity`: compares live CPU and Metal/Vulkan logits on the same
  generated suffix with Q36 top-1 and top-k overlap gates.

The runner defaults to the q36 model path compiled into `q36.h`. Override paths
and fixtures when needed:

```sh
Q36_TEST_MODEL=/path/to/model.gguf ./q36_test --logprob-vectors
Q36_TEST_VECTOR_FILE=/path/to/llama.vec ./q36_test --logprob-vectors
Q36_TEST_LONG_PROMPT=/path/to/prompt.txt ./q36_test --long-context
Q36_TEST_VECTOR_CASE=short ./q36_test --gpu-cpu-parity
```

For CPU portability, verify that the CPU-only target still builds:

```sh
make cpu
```

The CPU backend is a reference/debug path, not the production performance
target. Avoid large CPU inference runs unless that is the change being tested.

## Reference Vector Refresh

`tests/test-vectors/qwen3.6-35b-a3b` contains committed short and long-context
continuation vectors captured from llama.cpp with the Qwen3.6 Q8_0 reference
GGUF, using greedy decoding, thinking disabled, and `top_logprobs=20`.
Checkpoint-scoped directories keep fixtures from different model revisions
separate; comparing by token bytes catches tokenizer/template/attention
regressions before they become long generation failures. They're the default
offline reference, so building and testing Q36 does not require llama.cpp or
GGML libraries.

The default reference test consumes committed llama.cpp results and does not
link llama.cpp:

```sh
make test-reference
```

Keep experimental captures out of Git:

```sh
make test-vectors-local LLAMA_BUILD_DIR=llama.cpp/build
make test-reference-local
```

The local capture lives under the ignored `tests/test-vectors/local/`
directory. To run the live CPU-vs-llama.cpp gates, build the optional runner:

```sh
make test-llama LLAMA_BUILD_DIR=llama.cpp/build
make test-llama-long LLAMA_BUILD_DIR=llama.cpp/build
```

Only `make test-vectors-refresh` updates the tracked reference corpus. Review
its JSON, manifest, and compact vector diff together.

Hosted official-model comparisons use OpenRouter and the native Q36 scorer;
see `gguf-tools/quality-testing/README.md`. The API key and private response
captures stay outside version control.

## Release QA

The release checks are split so quick source tests do not require loading the
model:

```sh
make test                 # unit, parser, protocol, cache and fixture tests
make test-vulkan          # isolated Vulkan kernel coverage
make test-model           # generation, CPU/Vulkan and fusion parity
make test-metal           # Metal unit and isolated numeric kernel coverage
make test-metal-model     # Metal generation, CPU parity, state, and streaming
make test-session-batch   # Vulkan 1/2/4/8-session full-logit/state oracle
make test-server-live     # live HTTP, CORS and Responses API smoke test
make test-server-live-metal # live Metal HTTP/CORS/Responses smoke test
make test-server-live-metal-ssd # same live Metal surface through SSD streaming
make test-server-batching # concurrent requests against one 4-slot server
make test-server-batching-metal # same concurrent server gate on Metal
make test-server-batching-metal-ssd # Metal batching through bounded SSD cache
make benchmark-session-batch # old, 1/2/4/8-slot, and ordered-fallback server runs
make test-streaming       # resident/warm/cold/pressure/full-layer matrix
make benchmark-gate       # conservative BC-250 throughput floor
make release-build-check  # generic/BC-250 Vulkan and CPU builds with -Werror
make release-build-check-metal # Metal release build with -Werror
```

`make test-release` runs the complete sequence, including reference vectors.
The manual hardware, server, agent, long-context, power and sign-off checklist
is in [`QA_BEFORE_RELEASES.md`](QA_BEFORE_RELEASES.md). Distributed inference
is outside Q36's release scope.

`q36-eval --suite hard-smoke` selects 12 harder cases; `--suite hard` selects
all 50. The original 92 cases remain the default `core` suite. Use
`--list-cases`, `--validate-cases`, and `--source`, `--domain`, or `--case-id`
to inspect or filter cases; `--retry-incomplete` retries answers that hit
their output limit. [EVAL_DATA.md](EVAL_DATA.md) records sources and
licenses. Server requests may set `ignore_eos: true` with an explicit
`temperature: 0` for fixed-length greedy generation.

See [tests/REGRESSIONS.md](tests/REGRESSIONS.md) for serial model and client
checks.

## Speed Regression Tests

Use `q36-bench` for throughput regressions. It reports instantaneous prefill and
generation speed at context frontiers, not one whole-run average. Prefill is
incremental: each row measures only the newly processed suffix since the
previous frontier.

Short smoke benchmark:

```sh
./q36-bench \
  --prompt-file tests/long_context_story_prompt.txt \
  --ctx-start 32 \
  --ctx-max 64 \
  --ctx-alloc 128 \
  --gen-tokens 8
```

Before/after backend benchmark:

```sh
./q36-bench \
  -m gguf/Qwen3.6-35B-A3B-AntirezExperts-IQ2XXS-gateup-Q2K-down-Q8rest.gguf \
  --prompt-file tests/long_context_security_prompt.txt \
  --ctx-start 2048 \
  --ctx-max 32768 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/q36-speed.csv
```

Use the same machine, backend, model file, context sweep, power/thermal state,
and background load when comparing two commits. For Vulkan work, run at least
one before/after CSV and compare both `prefill_tps` and `gen_tps`. Generation is
greedy and skips EOS so each frontier gets the same number of generated tokens.

## Reporting Session Bugs

For debugging a failing generation, keep the trace:

```sh
./q36-server --trace /tmp/q36-trace.txt ...
```

Attach the trace, command line, model file, backend, context size, and whether
disk KV cache was enabled.

### GPU device loss

A GPU hang — most often the kernel watchdog firing on a dispatch that ran too
long — destroys the Vulkan context; every subsequent call against that device
fails. The backend latches this the first time a submit-path call reports
`VK_ERROR_DEVICE_LOST`, prints one diagnostic, refuses further GPU work, and
fails the forward pass rather than returning uncomputed results. The latch is
one-way: dispatches in flight when the context died produced nothing, so there
is no safe way to resume — restart the process.

If it reproduces, lower `--prefill-chunk`; a chunk large enough to push one
dispatch past the watchdog is the usual cause. On Linux with amdgpu, check
`journalctl -k` for a ring timeout. `Q36_VK_FAULT_INJECT_LOST=<n>` forces the
n-th submit to report device loss, so the unwind path can be exercised without
a real hang (diagnostic only).

## Disk KV Cache File Format

The practical disk-KV-cache guide (what it does, cache-key scheme, tuning
knobs) is in the README's [Disk KV Cache](README.md#disk-kv-cache) section.
This is the exact on-disk byte layout, for anyone changing the format itself.

A cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
Q36 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     representative model tensor quant bits: 1-6 or 8
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    Q36 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of
the incoming rendered prompt. After load, the exact checkpoint tokens from
the Q36 payload remain authoritative, and only the incoming text suffix after
the cached bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
It stores unguessable API tool call IDs back to the exact `<tool_call>` block
the model sampled, for tool calls whose block is present in the rendered
cached text, so a restarted server can still render later client history
byte-for-byte even if the client reorders JSON arguments:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled block byte length
8   bytes  tool id
... bytes  exact sampled <tool_call> block
```

This section is auxiliary replay memory, not model state. A cache hit
restores the session payload first, then loads the map if present. Before
rendering a request, the server can also scan cache files for the tool IDs
present in the client history and load just those mappings, so exact replay
can survive server restarts even when the matching KV snapshot isn't the one
ultimately used for the rendered-prefix hit.

The Q36 session payload starts with fourteen little-endian `u32` fields for
version 2, or sixteen fields for typed-KV version 3:

```text
0   magic = "Q36 "
1   payload version = 2 (f16 KV) or 3 (typed KV)
2   saved context size
3   prefill chunk size
4   checkpoint token count
5   vocabulary size
6   layer count
7   KV head count
8   key head dimension
9   value head dimension
10  recurrent convolution width
11  recurrent convolution dimension
12  recurrent state dimension
13  recurrent dt rank
14  K cache type (version 3 only)
15  V cache type (version 3 only)
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- For each full-attention layer: a `u32` row count followed by all K and V
  rows in the selected `f16`, `q8_0`, or `q4_0` cache type.
- For each recurrent layer: its `float32` convolution history and recurrent
  state tensors.

Version 1 is a legacy token-only input, rebuilt by prefilling the saved token
sequence on load; current writes use version 2 or 3 and persist the complete
Qwen full-attention and recurrent state. Logits are raw IEEE-754 `float32`
values saved immediately after the checkpoint tokens, so a loaded snapshot
can sample or continue from the exact next-token distribution without one
extra decode step. MTP draft logits/state are not persisted; after loading a
disk checkpoint the draft state is invalidated and rebuilt by normal
generation. The tensor payload is q36-specific KV/session state, not a
generic inference graph dump, and is expected to be portable only across
compatible `q36.c` builds for this model layout.
