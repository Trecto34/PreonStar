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
