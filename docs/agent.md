# Native agent

P36 includes a native coding agent. Inference is controlled inside the agent
itself, without a socket or API boundary, so the transcript and live KV state
are one session. The tools and system prompt use Qwen3.6's native tagged tool
format directly. This provides a few advantages:

- Low latency for generated text, tool calls, and new sessions.
- Live progress during long prefills.
- No OpenAI, Anthropic, or Hermes conversion in the model loop.
- The transcript and KV state cannot drift apart.
- Built-in file, search, shell, process, and web tools tuned for the model.
- Saved sessions can be switched without prefill when their KV payload is
  present.

Start the agent in the current directory, another project, or one-shot mode:

```sh
./p36-agent
./p36-agent --chdir /path/to/project
./p36-agent --non-interactive -p "Inspect the tests and fix the failure."
```

`--thinking-budget` is a hard maximum for each thinking block and defaults to
50000 thinking tokens. Adaptive `</think>` rank bias starts at 75% of the hard
budget; at 100%, the agent emits `</think>` unconditionally and continues with
its normal action or final response. The closing token itself is not counted as
a thinking token. `--tokens` remains the separate hard limit for the complete
assistant generation.

The agent conservatively stops obvious repeated-generation loops. Twelve
identical consecutive nontrivial lines, 32 near-identical consecutive lines,
or an exact 16-256 byte suffix period repeated at least eight times and across
at least 256 bytes triggers recovery. Repetitive thinking is closed and normal
generation continues. A repetitive partial tool call is discarded before it
can execute, and the model is asked to express the same change compactly.
`--frequency-penalty` (also accepted as `--default-frequency-penalty`) is an
optional low-strength sampler aid; it defaults to 0 and is not used for greedy
native tool syntax. `--repeat-penalty F` (0.5..2.0, default 1 = off) adds the
multiplicative llama.cpp-style penalty, in local mode and forwarded to a
`--server` as `repeat_penalty`.

With Qwen3.8-27B, `--thinking-budget` also selects the starting effort:
up to 8000 tokens is `low`, up to 16000 is `medium`, up to 24000 is `high`,
and larger budgets use `xhigh`. The default is `xhigh` with a 50000-token
hard thinking budget. `high` uses Qwen3.8's native `xhigh` instruction. Explicit
`--think-low`, `--think-medium`, `--think`, or `--think-xhigh` overrides the
starting mode without changing the budget. At an empty idle agent prompt,
Tab cycles `low → medium → high → xhigh` for Qwen3.8 and `off ↔ on` for
Qwen3.6. Tab leaves the closure budget unchanged; the footer shows both.

Agent user and system messages accept `<|think_on|>` and `<|think_off|>`.
The marker is removed before rendering and remains in effect for later turns.
Historical thinking stays in the append-only transcript until compaction.

Resident Metal and Vulkan both use Q8_0 keys with Q4_0 values and default to a
100000-token agent context. The backend's automatic resident GPU prefill width
resolves to 1024 tokens. Compact Metal attention scratch keeps this faster
chunk width practical at long contexts. CPU uses F16 KV. SSD-streamed model
weights also default to a 100000-token context with F16 KV.
Explicit `--ctx`, `-ctk`, and `-ctv` values override the preset.

`p36-agent --chdir` loads its model and runtime assets from the launch directory,
then changes to the requested project for agent tools. Built-in path tools also
expand `~` to the current user's home directory.

When a command such as `sudo` requests a terminal password, the interactive
agent opens a private prompt with input hidden. Type the password there;
Ctrl+C cancels the command. Password input is sent directly to the command,
without entering chat history, traces, or captured tool output. Other command
stdin reads receive EOF; `--non-interactive` cannot prompt for passwords.

Use `/hints on` for occasional short explanations of programming concepts behind
the current work, rendered as teal blockquotes. `/hints off` disables them. This
setting lasts for the current process and is reapplied after context compaction.

Both `p36` and `p36-agent` accept `--prefix-file FILE`. The file contains alternating
`USER:` and `ASSISTANT:` lines, starting with a user and ending with an assistant.
Turn content may span multiple lines. The CLI prefills this conversation before
the first interactive prompt; the agent keeps it through reset and compaction.
For example:

```text
USER: Our project uses C99 and has no external dependencies.
ASSISTANT: I will follow those constraints.
```

Sessions are stored in `~/.p36/kvcache`. Use `/save` to persist the current
session, `/list` to show saved sessions, and `/switch <sha>` to resume one.
The session ID remains stable across later saves. `/del <sha>` removes a saved
session. `/strip <sha>` keeps its transcript and title but removes the KV
payload; switching to a stripped session rebuilds the KV cache by prefilling
the saved text. `/compact` compacts the current context immediately.
Exiting during generation stops the worker before asking whether to save.
Sessions containing images cannot be saved yet; declining to exit after a save
failure returns to the current chat.

## Agent execution control

These controls live in the harness and server, not in the prompt; none of them
adds prompt text. Defaults are conservative and every control can be turned off.

| Flag | Default | Meaning |
|------|---------|---------|
| `--action-budget N` | 384 | After thinking ends (or from token 0 without thinking) the model must act or answer. Once N content tokens pass with no tool call started, the content is scanned for reasoning restarts (`Wait,` `Actually,` `Let me re...`, 3 or more) or repetition; past 4N a single restart is enough. A real final answer has neither and is never cut. `0` disables. |
| `--max-recoveries N` | 3 | A leaked round is dropped (the monologue is never fed back), `ACTION REQUIRED...` is added and the retry runs with thinking off. After N recoveries the turn ends `BLOCKED:`. |
| `--max-repeat-tool N` | 1 | Identical read-only calls allowed against an unchanged workspace. Any write/edit/bash resets the memory. The same call back to back is blocked for every tool. Blocked calls return `REPEAT_CALL_BLOCKED`. |
| `--max-stagnant-turns N` | 2 | Blocked or identically failing calls in a row before a recovery. |
| `--max-verify-steps N` | 8 | The controller is in phase `IMPLEMENT` (default), `VERIFY` or `FINALIZE_REQUIRED`. Reads, searches and listings never enter `VERIFY`, so recovering after compaction is free; a known test/check command (`node --check`, `pytest`, `make test`, lint, headless browser), a verification-helper run or a page visit after a target edit does, and the agent then gets N steps to check its work. A target edit returns to `IMPLEMENT` and resets the count, as does a tool result with a `MISSING:` / `- item` block (concrete evidence work is absent), which also blocks `VERIFY`; edits to disposable verification helpers (temp-dir scripts, or files the agent created itself named like `verify`, `probe`, `mock`) do not. A repeated or near-identical check costs 2. When spent, the state is `FINALIZE_REQUIRED`: further tool calls are refused with `VERIFICATION_BUDGET_EXHAUSTED` and the model must report what is confirmed working, confirmed failing and unresolved. If it keeps calling tools the controller ends the turn with a stop report built from tracked facts (not a success claim). `0` disables. Read-only investigations before the first edit are not limited. |
| `--workspace DIR` | launch dir | Workspace jail root. File tools (`read`, `write`, `edit`, `search`, `list`, `view_image`) are refused with `WORKSPACE_ESCAPE_BLOCKED` unless the canonical path is inside it (`..`, absolute paths and symlink escapes included). `bash` runs under bubblewrap: cwd is the root, only the root is writable, system dirs are read-only, other home/project directories do not exist, scratch lives in `DIR/.p36-tmp`. The root is shown as `WORKSPACE_ROOT` in the compacted task state and cannot be changed by the model. |
| `--yolo` | off | Explicit opt-out of the whole jail (file tools and shell); prints a startup warning. |
| `--unsafe-filesystem` | off | File tools may leave the workspace; the shell stays confined. |
| `--max-implementation-steps N` | 96 | Emergency ceiling on tool steps spent in the IMPLEMENT phase per turn. Unlike the progress counters, target edits do not reset it, so mutation thrashing ends in an INCOMPLETE/STOPPED report (never VERIFY, never BLOCKED). `0` disables. |
| `--max-no-progress-steps N` | 16 | Ceiling on verification steps without the double cost, for when `--max-verify-steps` is raised; 4x it bounds non-check steps while implementing. `0` disables. The session summary lists phase transitions and the final phase. |
| `--think auto\|low\|medium\|high` | off | `auto` picks the thinking budget per turn (256/1024/2048) from the prompt and raises it after tool failures or recoveries; never lowered within a turn. Explicit `--thinking-budget` stays a ceiling. The bare `--think*` flags are unchanged. |
| `--compact-llm` | off | Compaction renders harness-tracked task state (GOAL, VERIFIED FACTS, COMPLETED, FILES, FAILED ATTEMPTS, CURRENT EDIT STATE, NEXT REQUIRED ACTION, DO NOT REPEAT; after the first edit also VERIFICATION STATE with LAST_TARGET_MUTATION, VERIFY_STEPS_USED, MAX_VERIFY_STEPS, NO_PROGRESS_STEPS and UNRESOLVED) instead of asking the model for a narrative, and strips closed thinking from the retained tail. `--compact-llm` restores the model-written summary. A restored session with no tracked state also falls back to it. |

Tool changes: `write` takes `append=true` and is limited to 12 KiB per call (an
oversized call is cut while streaming and answered with
`WRITE_PAYLOAD_TOO_LARGE`; the result reports `total=` and `max_payload=`);
`edit` takes `start_line`/`end_line` without `old` (whitespace-safe) and
reports `EDIT_MATCH_NOT_FOUND` / `EDIT_MATCH_NOT_UNIQUE` / `EDIT_NO_CHANGE` /
`EDIT_LINE_RANGE_INVALID` with counts, plus `hint=whitespace_differs` when only
whitespace differs.

With `--trace FILE` the trace tags `[THINK]`, `[ACTION]`, `[TOOL CALL]`,
`[TOOL RESULT]`, `[RECOVERY]`, `[COMPACTION]`, `[FINAL]` and `[SUMMARY]`;
non-interactive runs end with a session summary (steps, tools, reasoning and
action tokens, recoveries, repeated calls blocked, compactions, cached prompt
share, prefill/decode tok/s).

### Remote agent

```sh
p36-agent --server http://192.168.3.50:8000 --server-model MODEL \
  --think --thinking-budget 768 --action-budget 384 --chdir repo
```

Inference runs on a `p36-server`; tools, shell, watchdog, recovery and
compaction state stay local. The server owns the chat template, tokenizer,
reasoning split, KV and exact tool-call replay; the client sends OpenAI-format
messages with structured `tools` and streams the reply. `--api-key` adds a
Bearer header, `--server-timeout` bounds silence (default 600 s), and
`--server-protocol` accepts `auto` or `chat`. The agent sends its own `reasoning_content` back with each assistant turn (the server keeps it for turns after the last user message, as the local transcript does), and a hard `thinking_budget` closes thinking softly from 75% of the budget and forcibly at 100%, the same schedule as the local agent. No local model is loaded and the
context size is clamped to the server's. Limits: plain `http://` only, tool
images (`view_image`) and saved sessions are not available remotely, and only
the first tool call of a reply is executed (the server is asked to stop there).
