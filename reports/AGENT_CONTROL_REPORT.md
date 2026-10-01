# Agent control + remote agent: results (2026-10-01)

Model for all runs: Ornith-1.5-35B-A3B IQ2XXS (Vulkan, BC-250). Baseline = commit 1c9ed06.

## Prompt size (scripts/agent_prompt_tokens.sh, "hi", no-think)
| | tokens |
|-|-|
| baseline local initial prompt | 2667 (system+tools 2607) |
| now local | 2750 (+83, +3.1%): `append`, `start_line/end_line`, one contract sentence |
| now remote (server-templated tools) | 1585 |
All controller features (budgets, leak detection, watchdog, recovery, compaction state, metrics) add 0 prompt tokens.

## Overhead (3 reps, short)
| | baseline | now |
|-|-|-|
| server plain decode | 91.8 tok/s | 92.3 tok/s |
| server TTFT (368-token prompt) | 0.445 s | 0.428 s |
| local tool loop (3 steps, ~120 tokens) | 1.93 s | 1.90 s |
| remote tool loop (same task) | n/a | 3.57 s (146 tokens; includes ~1.7 s cold prefill) |
Remote within-session KV reuse: 2752 of 2804 prompt tokens cached on step 2 (suffix prefill 181 ms).

## Phase 9 (stateful sessions): not implemented
Within a session the server's prefix/KV reuse already serves ~98% of the prompt; replay cost is ~0.2 s/step.
Remaining cost is the cold start: a new conversation never reuses the shared system+tools prefix (cached=0,
~1 s per 1000 tokens), because the hybrid model cannot rewind recurrent state and disk checkpoints sit at
end-of-conversation boundaries, not at the system-prompt boundary. Follow-up: checkpoint at the system/tools boundary.

## Regression suite (tests/agent_regression.py, real executions)
| | local | remote |
|-|-|-|
| A cube (<=3 steps, valid HTML) | pass, 2 steps | pass, 3 steps |
| B large write via append | pass | pass (3 compactions, 1 real action_budget recovery) |
| C known edit | pass, 3 steps | pass, 4 steps |
| D whitespace edit | pass, 5 steps | pass, 13 steps |
| E thinking budget 16, -n 20000 | pass, 2 steps | pass, 2 steps |
| F repeated read | pass, 1 blocked | pass, 1 blocked |
Verdict parity 6/6; tool-sequence parity only on E and F (remote uses more verification steps; prompts and
sampling defaults differ). Post-think leak and recovery are covered deterministically by unit tests
(`test_leak_recovery_protocol`); the real runs hit one genuine `action_budget` abort that recovered.

## Bugs found while testing
- Remote: thinking and context size were uninitialized in my first remote cut (reasoning 0, compaction never fired). Fixed.
- Watchdog: ten identical succeeding edits bypassed it because writes reset it; now any back-to-back identical call is blocked, no-op edits return EDIT_NO_CHANGE.
- Server (pre-existing): a tool request cut by `max_tokens` returns the model's thinking and a raw `</think>` as `content`.
- `q36moe.gguf` is a dangling symlink (start-server.sh cannot start).

## Limitations
- Remote is plain http, no vision/saved sessions, first tool call per reply only, interactive TUI path untested.
- `tool_choice` on the server still only honors "none".
- `--repetition-threshold` and a nearly-identical-consecutive-output check were not added (leak detector + repetition detector cover the observed cases).
- Leak detection is heuristic (restart markers + repetition); a prose loop with no markers and no repetition is bounded only by --tokens.
- Apart from E and F, local and remote tool sequences differ.

# Round 2 (truncation fix, system checkpoint, parity)

## Truncated tool call / reasoning leak
Root cause: a tool call cut by max_tokens fails `parse_generated_message_ex`; the fallback in
`parse_generated_message_for_response` then returned the raw text (reasoning, `</think>`, partial
`<tool_call>`) as `content`, and a complete call followed by prose failed the same way.
Now (chat completions, stream and not): complete calls are kept, the unfinished tail is discarded and flagged
`incomplete_tool_call: true`, reasoning/content are split before the first call, partial delimiters are stripped,
whitespace-only content is empty, prose after a complete call is dropped, finish_reason stays `length`/`stop`.
Tests: 2 server unit tests (all 10 required cases + a streaming prefix sweep; mutation-checked: 15 failures with
the fix disabled), live sweep on the real model (60 cut points x stream/non-stream), fake-server agent test
(bounded recovery, BLOCKED, no endless retry, nothing poisonous appended).

## System+tools checkpoint: implemented (optional `--kv-system-checkpoint`)
Cache keys now include a model content fingerprint (closes a pre-existing hole: same-shape models sharing a
cache dir could restore each other's state). Checkpoint is floored to a prefill-chunk multiple: mid-chunk
checkpoints diverged from cold in 2/12 sessions, chunk-aligned ones were identical 12/12 (chunk 1024 and 512).
New-session TTFT, ~1.76k-token agent-sized prefix, Ornith 35B:
| | cached | prefill | TTFT |
|-|-|-|-|
| no cache | 0 | 1.64-1.74 s | 1.63-1.73 s (1.99 at chunk 512) |
| checkpoint, chunk 1024 | 1024 | 0.72 s | 0.86-1.41 s |
| checkpoint, chunk 512 | 1536 | 0.31-0.32 s | 0.38-0.52 s (restore ~28 ms warm) |
Misses (clean): different system prompt, changed tool schema, different model (Qwen3.8-27B in the same dir).
First session pays ~0.1-1 s extra (build + save). ~80 MiB per checkpoint on disk. A full disk degrades to cold.
Recommendation: use with `--prefill-chunk 512`.

## Local vs remote parity (tasks A,C,D,E,F x3, D x9 more for R0/R2; mean steps)
| task | local | remote before | +reasoning kept | +75% soft close |
|-|-|-|-|-|
| A | 3.3 | 5.7 | 3.7 | 4.0 |
| C | 3.0 | 3.0 | 3.7 | 3.7 |
| D | 6.0 (3,10 range) | 5.7 (n=9: 6.4) | 6.7 | 8.7 (n=9: 7.2) |
| E | 2.3 | 2.3 | 2.3 | 2.7 |
| F | 3.7 | 3.7 | 3.7 | 3.0 |
Decode speed is the same (87-89 tok/s). Differences found and fixed for parity: the remote agent dropped its own
reasoning between tool steps; the server's soft thinking close started at 100% instead of 75%. Neither moved
step counts beyond run-to-run noise (D alone ranged 3-17 steps). Not differences: sampling defaults (same
function), tool-response wrapper (same), tool-call format (same), decode speed. The earlier "5 vs 13 steps" on D
was noise. Remaining real difference: A needs a post-write verification step more often remotely (pass 4/10 vs
3/4 local at the <=3-step bound); adding the local prompt's two behavioral bullets did not change it (3/6 vs 3/6),
so it was not adopted. Prompt: local 2750 tokens (2690 system+tools), remote 1568 (699 system text, 856 tool
schemas); both unchanged by this round.
