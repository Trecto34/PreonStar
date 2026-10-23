# Server

A local HTTP server exposing OpenAI- and Anthropic-compatible endpoints, backed by the same
engine as the CLI.

Start a local OpenAI/Anthropic-compatible server:

```sh
./p36-server --vulkan --ctx 32768 \
  --kv-disk-dir /tmp/p36-kv --kv-disk-space-mb 8192

./p36-server --metal --ctx 32768 \
  --kv-disk-dir /tmp/p36-kv --kv-disk-space-mb 8192
```

On an 8 GB Mac, reduce context and use the bounded Metal cache:

```sh
./p36-server --metal --ssd-streaming \
  --ssd-streaming-cache-experts 512 --ctx 4096
```

Without extra options the server keeps one mutable backend/KV checkpoint and
uses the original single graph worker. Stateless clients that resend a longer
version of the same prompt can reuse that prefix instead of pre-filling from
token zero.

`--batched-session N` opts into `N` independent resident sessions:

```sh
./p36-server --ctx 32768 --batched-session 4
```

Use `--mixed-prefill-quantum N` to tune how many prefill tokens a batched
session runs per scheduling turn while another session is generating. The
default is 128; smaller values favor decode latency, larger values favor
prefill throughput.

Each active request owns one slot until it finishes; excess requests wait for
an idle slot. Assignment prefers the resident live/token prefix with the
longest match. When disk KV caching is enabled, an unmatched idle slot is
persisted before reuse. Thinking state, tool continuations, RNG, logits,
recurrent state, and full-attention KV remain session-local.

One coordinator owns model execution. Decode-ready slots coalesce for up to
2 ms and advance in one model step. Prefills run round-robin in bounded
quanta: 2048 tokens while no generation is active and 128 while any generation
is active. This prevents one large prompt from blocking every active decoder.
MTP speculative decoding is disabled in batched mode.

Resident context memory is multiplied by `N`; startup prints both the
per-session estimate and the total. Choose `N` and `--ctx` so all session KV,
recurrent state, and graph scratch fit. Backend behavior is:

| Backend | Multi-session execution |
| --- | --- |
| Vulkan resident, 2-8 decode-ready rows | Native row-batched shared projections and FFN work with private positions, recurrent state, and typed KV when graph scratch can hold every row. F16/F16 and Q8_0/Q4_0 KV are supported; other KV pairs and unsupported shapes use ordered fallback. |
| Vulkan SSD streaming | Deterministic ordered fallback, preserving expert-cache ownership. |
| Metal resident, 2-8 decode-ready rows | Native row-batched graph execution where the kernel shape is supported; ordered exact fallback otherwise. |
| Metal SSD streaming | Deterministic ordered fallback, preserving bounded expert-cache ownership. |
| CPU, batches above 8, or forced `P36_VK_SESSION_BATCH=0` | Deterministic ordered fallback. The compatibility environment variable also controls Metal session batching. |

Batch size one calls `p36_session_eval()` directly. If a batched step fails,
all members are invalidated so none can silently continue from a partially
advanced frontier. Ordered fallback preserves concurrency and scheduling
fairness, but does not provide the aggregate throughput gain of native GPU
batching.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/<loaded-model-id>`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`,
`min_p`, `presence_penalty`, `frequency_penalty`, `repeat_penalty` (alias
`repetition_penalty`), `seed`, `stream`,
`stream_options.include_usage`, `tools`, and `tool_choice`. Tool schemas and
calls use the native Qwen3 Coder tagged format, and generated calls are mapped
back to OpenAI tool calls.

Flash attention is built in, so there is no `-fa` switch (unlike llama.cpp).
Vulkan prefill uses the flash-attention kernels `attn_prefill_fa_gqa6/gqa8`
(on by default; `P36_VK_ATTN_FA=0` falls back to the qtile2 kernel for A/B
checks), and decode uses split-K flash-decoding (`attn_decode_split` +
`attn_combine`). llama.cpp flags such as `-fa on` are not accepted.

Server sampling defaults apply when a request omits a field; explicit request
values always win:

```
p36-server --temp 0.2 --top-p 0.9 --min-p 0.05 --top-k 40 --repeat-penalty 1.05
```

`--temp` (0..100), `--top-k` (0 = off), `--top-p` (0..1), `--min-p` (0..1) and
`--repeat-penalty` (0.5..2.0, 1 = off) override the model's built-in defaults.
`repeat_penalty` is llama.cpp-style: the logit of every token already generated
is divided by the penalty when positive and multiplied by it when negative.
Out-of-range request values are clamped to 0.5..2.0. It is disabled inside
native tool-call syntax, like the other penalties. For coding, keep the V cache
at `-ctv q8_0` or `f16`; the Vulkan/Metal default is q4_0.

Both Qwen3.6 and Qwen3.8 accept the fixed-template controls:

```json
"chat_template_kwargs": {
  "enable_thinking": false,
  "preserve_thinking": true
}
```

`preserve_thinking` defaults to `true`, retaining earlier assistant reasoning
verbatim so later prompts remain a prefix-cache match. Set it to `false` to
strip reasoning before the latest user query. System and user messages may also
contain `<|think_on|>` or `<|think_off|>`; P36 removes the control marker before
rendering and applies it to subsequent turns.

When omitted by the client, Qwen uses `temperature=1`, `top_p=1`, no top-k cap,
and `min_p=0.05`. CLI, agent, and server use the loaded model's defaults. Eval
stays fixed at the Qwen sampling defaults for comparable runs. Explicit values
always win.

`/v1/responses` accepts string or message-array input, instructions, direct
tool schemas, function-call continuations, function-call outputs, sampling
controls, reasoning controls, and `max_output_tokens`. It returns native
Responses API message, reasoning, and function-call output items. With
`stream:true`, it emits Responses API SSE events through
`response.completed`.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code
style clients. It accepts `system`, `messages`, `tools`, `tool_choice`,
`max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`,
and thinking controls. Tool uses are returned as Anthropic `tool_use`
blocks.

Both APIs support SSE streaming. In thinking mode, reasoning is streamed in
the native API shape instead of being mixed into final text. OpenAI chat
streaming also streams tool calls as soon as the `<tool_call>` opening is
recognized: the tool header is sent first, then each completed native
parameter is forwarded as a `tool_calls[].function.arguments` delta while
generation continues. The Anthropic endpoint streams thinking and text live, then emits
structured `tool_use` blocks when the generated tool block is complete.

Pass `--cors` to add `Access-Control-Allow-Origin`, methods, and headers and
to answer browser `OPTIONS` preflight requests. CORS headers are disabled by
default.

## Tool call handling and canonicalization

Qwen3.6-35B-A3B emits tool calls in its native tagged format. Tool definitions
are provided in the system prompt inside `<tools>...</tools>`. A call has one
`<function=name>` block inside `<tool_call>`, with one
`<parameter=name>` block per argument:

```text
<tool_call>
<function=list_files>
<parameter=pattern>
*.c
</parameter>
<parameter=max_depth>
2
</parameter>
</function>
</tool_call>
```

Agent clients do not send that same text back on the next request: they send
normalized OpenAI/Anthropic JSON tool-call objects. **If the server
re-rendered those objects slightly differently, the rendered byte prefix
would no longer match the live KV checkpoint** and the next turn would have
to be rebuilt.

All markers are plain ASCII. P36 keeps the exact replay and canonicalization
machinery inherited from `ds4`, because a sampled call and its next-turn API
rendering must still be byte-identical to avoid silent KV drift.

The first line of defense is exact replay. Every tool call gets an
unguessable API tool ID, and the server remembers `tool id -> exact sampled
<tool_call> block` in a bounded in-memory map backed by radix trees. When
the client later sends that tool ID back, the prompt renderer uses the exact
bytes the model sampled, not a freshly formatted approximation. This map
can also be saved inside KV cache files, so exact replay survives server
restarts for cached histories.

**Canonicalization is only the backup path**. If the exact sampled block is
missing, or exact replay is disabled with
`--disable-exact-tool-replay`, the server renders deterministic native Qwen
tags from the JSON tool object, following schema property order. After a
tool-call turn, it compares the live sampled token stream with the prompt
that the next client request will render. If needed, it rewrites the live
checkpoint, or falls back to an older disk KV snapshot and replays only the
suffix. This keeps the model continuation aligned with the stateless API
transcript.

During generation, the server also treats native Qwen syntax differently from
payload. When the model is emitting stable protocol structure — tool,
function, and parameter tags — sampling is forced to `temperature=0` so
the tool call stays parseable. This greedy mode does **not** apply to
argument values: string contents inside the arguments JSON, including file
contents and edit text, use the request's normal sampling settings. That
separation is important: deterministic decoding is helpful for syntax, but
can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.6-35b-a3b",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

## Agent Client Usage

`p36-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit
no higher than the `--ctx` value you started the server with:

```sh
./p36-server --ctx 32768 --kv-disk-dir /tmp/p36-kv --kv-disk-space-mb 8192
```

On a BC-250 with 16 GB of unified memory, weights take ~10–11 GB at Q2,
which leaves roughly 2–4 GB for KV cache, scratch buffers, OS and your
client. The model's native context is 256K tokens, but the live context must
still fit in available memory. Disk KV checkpoints avoid repeated prefill and
preserve sessions across restarts; they do not enlarge the active context
window.

The `262144` output limit in the configs below matches the model's native
context ceiling. The server stops earlier when its configured context window
is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "p36": {
      "name": "p36.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "p36-local"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen 3.6 MoE (p36.c local)",
          "limit": {
            "context": 32768,
            "output": 262144
          }
        }
      }
    }
  },
  "agent": {
    "p36": {
      "description": "Qwen 3.6 MoE served by local p36-server",
      "model": "p36/qwen3.6-35b-a3b",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "p36": {
      "name": "p36.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "p36-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "qwen",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen 3.6 MoE (p36.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 32768,
          "maxTokens": 262144,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "p36",
  "defaultModel": "qwen3.6-35b-a3b"
}
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like
this matches the local `~/bin/claude-p36` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${P36_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${P36_API_KEY:-p36-local}"
export ANTHROPIC_MODEL="qwen3.6-35b-a3b"

export ANTHROPIC_CUSTOM_MODEL_OPTION="qwen3.6-35b-a3b"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="Qwen 3.6 MoE local p36"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="p36.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3.6-35b-a3b"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3.6-35b-a3b"
export CLAUDE_CODE_SUBAGENT_MODEL="qwen3.6-35b-a3b"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens,
before it starts doing useful work. Keep `--kv-disk-dir` enabled: after the
first expensive prefill, the disk KV cache lets later continuations or
restarted sessions reuse the saved prefix instead of processing the whole
prompt again.

## Server extension fields (`/v1/chat/completions`)

All optional and ignored unless sent:

- `thinking_budget` (int): hard cap; `</think>` is forced at N reasoning tokens.
- `action_budget` (int): leak abort as above; the reply ends with
  `finish_reason: "action_budget"`.
- `stop_after_tool_call` (bool): decoding ends at the first complete tool call.
- Replies carry `usage.completion_tokens_details.reasoning_tokens` and
  `content_tokens`, and `usage.timings.{prefill_ms,decode_ms}` next to the
  existing `cached_tokens`.

Truncation semantics (`max_tokens`, `action_budget` or end of context): reasoning
that is cut off stays in `reasoning_content` and `content` is empty; delimiters
(`<think>`, `</think>`, `<tool_call>` or a fragment of one) never reach `content`.
Complete tool calls are always returned as structured `tool_calls`; a tool call
that did not finish is discarded, never returned as text, and the choice carries
`"incomplete_tool_call": true` (finish_reason stays `length`, or `stop` if the
model ended inside the call). A stream may send argument fragments before the
cut, but its final chunk then has `finish_reason: "length"` plus
`incomplete_tool_call`, so a client must discard the assembled call. Prose after
a complete call is dropped. The remote agent treats a flagged call, or output cut
off before any content, as a failed generation: it is not added to the
transcript and goes through `--max-recoveries`, ending `BLOCKED:` rather than
retrying forever.

Tests: `make test` (unit), `make test-server`, `tests/test_server_agent_controls.py URL`
(live server), `tests/agent_regression.py --mode local|remote` (real agent runs,
tasks A-F) and `--compare` for local/remote parity, `tests/bench_server_plain.py`
and `tests/bench_agent_loop.py` for overhead checks.
