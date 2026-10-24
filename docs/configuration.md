# Configuration

How to run each binary, and every knob that changes what it does. Options that
are common to all platforms are marked so; the few that are specific to one
backend or one board are called out and explained in
[hardware docs](hardware/) or [backends.md](backends.md).

## CLI

One-shot prompt:

```sh
./p36 -p "Explain Redis streams in one paragraph."
```

Without `-p`, P36 starts an interactive multi-turn chat:

```sh
./p36
p36>
```

The CLI keeps the rendered transcript and live graph KV checkpoint, so each
turn extends the previous conversation. Useful commands are `/help`, `/think`,
`/think-max`, `/nothink`, `/ctx N`, `/read FILE [PROMPT]`, and `/quit`. Ctrl+C interrupts
the current generation and returns to `p36>`.

Thinking mode is enabled by default. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path for greedy decoding. It uses `--mtp-margin` as a confidence gate and is
currently an experimental slight-speedup path.

## Thinking Modes

Qwen3.6-35B-A3B and Qwen3.8-27B have distinct non-thinking and thinking modes, controlled
natively by `<think>...</think>` blocks rendered into the prompt. The server
defaults to thinking mode.

Mapping of API thinking controls to prompt rendering:

- Anthropic `thinking: {"type":"enabled"}` → thinking on (default).
- Anthropic `thinking: {"type":"disabled"}` → thinking off.
- Qwen3.8-27B supports `low`, `medium`, and `xhigh` in its chat template.
  `high` and the default select `xhigh`; `minimal` selects `low`.
- Qwen3.6-35B-A3B has only thinking on or off. Positive efforts all enable
  thinking without a model-specific effort instruction.
- Numeric effort strings map `"0"` to thinking off, `"1"`–`"33"` to `low`,
  `"34"`–`"66"` to `medium`, and `"67"`–`"100"` to `xhigh`. On Qwen3.6,
  all positive values enable the same thinking mode. Only `"max"` selects
  Think Max.
- `thinking.budget_tokens` on chat or Anthropic messages starts adaptive
  `</think>` ranking at that token count. On Qwen3.8 it also selects `low`
  through 8000, `medium` through 16000, `high` through 24000, and `xhigh`
  above 24000. Explicit `none` and `max` efforts keep their mode.
- Explicit non-thinking: `thinking:{"type":"disabled"}`, `think:false`, or
  `chat_template_kwargs:{"enable_thinking":false}`. Both profiles use the
  `<think>\n\n</think>\n\n` non-thinking prefix.

## KV Cache Quantization

P36 supports `f16`, `q8_0`, and `q4_0` KV rows. Select key and value types
independently with `-ctk` and `-ctv`:

```sh
./p36 -ctk q8_0 -ctv q4_0 -p "Explain radix trees."
./p36-server -ctk f16 -ctv f16
```

Resident Metal and Vulkan frontends default to Q8_0 keys and Q4_0 values. CPU
and SSD streaming default to F16 for both. Explicit flags always override
these defaults. KV quantization reduces context memory; it does not change
model weight quantization or extend Qwen3.6's 262144-token native context.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `p36-server` first tries the cheap exact
token-prefix check, then falls back to comparing rendered prompt bytes with
decoded checkpoint bytes. The live in-memory checkpoint covers the current
session; the disk KV cache makes useful prefixes survive session switches
and server restarts.

The default server has one live KV cache. With `--batched-session N`, each
resident slot has one and active slots are never evicted. When an unrelated
request reuses an idle slot, its old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache.

Enable it with:

```sh
./p36-server --kv-disk-dir /tmp/p36-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The P36 payload still stores the exact token IDs and graph
state for that prefix. This matters for continued chats: the model may have
generated one token whose decoded text is later sent back by a client as two
canonical prompt tokens. A rendered byte-prefix hit can still reuse the
checkpoint and tokenize only the new suffix. The file is intentionally
written with ordinary `read`/`write` I/O, not `mmap`, so restoring cache
entries does not add more VM mappings to a process that already maps the
model.

Tool calls also keep a bounded exact-replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text.
The RAM map keeps up to 100000 IDs by default; tune it with
`--tool-memory-max-ids`. Use `--disable-exact-tool-replay` to disable this
and fall back to canonical JSON-to-Qwen rendering.

Each cache file (`<sha1>.kv`) holds a small header, the human-readable
rendered prefix text (both the lookup key source and the check against BPE
re-tokenization drift), the full P36 session payload (checkpoint tokens,
next-token logits, and per-layer K/V or recurrent state), and an optional
tool-call exact-replay map so restarted servers can still render client tool
history byte-for-byte. See
[CONTRIBUTING.md](../CONTRIBUTING.md#disk-kv-cache-file-format) for the exact
on-disk byte layout.

Checkpoints are written at four points — `cold` (first stable prefix),
`continued` (roughly every 10k tokens by default), `evict` (before an idle
slot is reused), and `shutdown` — with conservative defaults (512-token
minimum, 30000-token cold cap, 32-token tail trim, 2048-token alignment)
chosen to avoid retokenization misses at chunk boundaries. Tunable knobs:
`--kv-cache-min-tokens`, `--kv-cache-cold-max-tokens`,
`--kv-cache-continued-interval-tokens`, `--kv-cache-boundary-trim-tokens`,
`--kv-cache-boundary-align-tokens`, `--tool-memory-max-ids`,
`--disable-exact-tool-replay`.

The cache directory is disposable — stop the server and remove it if
behavior looks suspicious. Files use plain `read`/`write` I/O (not `mmap`)
and store the verbatim cached prompt text, so `hexdump` can inspect them
directly.

### System+tools checkpoint (`--kv-system-checkpoint`)

Agents start every session with the same system prompt and tool schemas
(~1.6-1.8k tokens). With `--kv-disk-dir` and `--kv-system-checkpoint`, the first
request prefills the system turn on its own, stores that state, and every later
conversation with the same prefix restores it instead of recomputing it.

- **Identity.** Every cache key is `sha1(model fingerprint || sha1(rendered prefix text))`.
  The rendered text already pins the system prompt, tool schemas, template flags and
  think mode; the model fingerprint pins the weights, tokenizer and any
  `--directional-steering-file` (size, the first 16 MiB, 32 spread 64 KiB samples, the
  last MiB, plus the steering file) together with a key-format version. A changed prompt, tool schema,
  model or steering file is a clean miss. Entries written by older builds never match
  and age out through the normal disk budget. If the model cannot be read the cache is
  disabled rather than keyed by a guess. Limitation: a patch confined to unsampled
  bytes of an equal-size file would not change the fingerprint.
- **Exactness.** The checkpoint is floored to a multiple of the prefill chunk, so a
  restored session continues on exactly the chunk schedule of a cold prompt. Measured:
  restored and cold greedy replies (reasoning, content, tool-call arguments) were
  identical in 12 of 12 sessions, at chunk 1024 and at 512; a checkpoint inside a chunk
  diverged in 2 of 12 and is therefore not used. With the default 1024 chunk a
  ~1.7k-token prefix caches 1024 tokens; `--prefill-chunk 512` caches 1536.
- **Cost.** About 80 MiB per checkpoint on disk (dominated by fixed recurrent state, not
  token count), bounded by `--kv-disk-space-mb` with the usual eviction. The first
  session pays one extra save (~0.1 s); a disk-full store fails quietly and the request
  runs cold.
- **Metrics.** Chat usage gains
  `timings.system_checkpoint: {hit, tokens, restore_ms, build_ms}` when enabled.

## Migrating from q36

The project, its binaries and its on-disk state were renamed from `q36` to
`p36`. Nothing is migrated for you and there is no fallback path: until you move
the directories, an existing install simply looks new, so `/list` shows no saved
sessions and shell history starts empty.

```bash
mv ~/.q36                ~/.p36                  # kv cache, browser profile, server-kv
mv ~/.q36_history        ~/.p36_history
mv ~/.q36_agent_history  ~/.p36_agent_history
mv <workspace>/.q36-tmp  <workspace>/.p36-tmp   # if the agent ever ran there
```

The agent cache directory follows the same rename: `Q36_AGENT_CACHE_DIR` is now
`P36_AGENT_CACHE_DIR`. Saved sessions, the disk KV cache and both command
histories resolve again once the directories are in place; nothing needs to be
re-prefilled.

## Environment switches

Runtime knobs that are not CLI flags. Set them in the shell that starts `p36`
or `p36-server`; they are read once at first use.

| Variable | Default | Effect |
|---|---|---|
| `P36_VK_MOE_PAIR_DECODE` | on | `=0` disables the fused expert pair at decode and restores the bit-exact q8 route (decode 74-76 -> 52-54 t/s on the BC-250). Use `=0` when you need token-for-token reproducibility against builds from before the pair route. |

