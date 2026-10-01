# Agent control / remote-agent plan (Phase 0 audit, 2026-10-01)

## ALREADY EXISTS (do not duplicate)
- Qwen-native tool parser; generation already stops at first complete `<tool_call>` (agent: `AGENT_QWEN_TOOL_DONE` break in `worker_run_turn`).
- Thinking budget (`--thinking-budget`, default 50000), soft close at 75% (rank-window forcing `</think>`), hard close at 100%, `--think/--think-low/-medium/-xhigh/-max/--nothink`.
- Byte-level repetition detector (exact lines, near-identical lines, n-gram) inside thinking and inside tool calls; partial tool rollback; frequency penalty.
- LLM-written compaction at 70% ctx; mid-generation compaction; `/compact`.
- Exact-match `edit` (+ optional `--edit-upto` range), atomic write/edit, preflight check of `old`.
- Live KV prefix reuse + disk KV cache in the server; `cached_tokens` usage fields; exact tool-call replay map; OpenAI chat/responses/anthropic endpoints with structured tools, tool_calls, reasoning_content, SSE.
- `--trace FILE` events in agent.

## PARTIALLY EXISTS
- Budgets: only global `-n` bounds post-think content (the reported bug). Server `thinking_budget` is a soft rank push, only from `thinking.budget_tokens`.
- Repetition: not applied to plain assistant content; no cross-round duplicate tool-call blocking.
- Tool errors: `Tool error: ...` free text, not machine-actionable codes.
- Server `tool_choice`: only `"none"`.
- Observability: trace + footer, no end-of-run summary.

## MISSING
- action_budget (agent + server), post-think leak detection, recovery messages, max-recoveries -> BLOCKED.
- Cross-turn watchdog (same call, same read range, stagnant turns, repeated tool error).
- Deterministic structured compaction state.
- Chunked/append write; large-write size communication.
- Remote mode (`--server`): no HTTP client in agent.
- Server: stop-at-tool-call-complete (decode continues to EOS), repetition abort, finish_reason `action_budget`/`repetition_abort`, `timings`, session ids.
- `--think auto`, deterministic thinking policy.

## SHOULD NOT BE IMPLEMENTED (existing mechanism better)
- Stateful server sessions (Phase 9): server already does live prefix + disk KV reuse; benchmark first, add only if replay cost is measurable.
- A manual Qwen template in the remote client: server owns it.
- A second tool-call parser for remote: server's `parse_generated_message_ex` returns structured `tool_calls`.
- Separate patch-apply tool: extend `edit` with optional `lines` range instead (token-cheap).

## Order
1 agent budgets+leak (P1-3) -> 2 watchdog (P4) -> 3 deterministic compaction (P5) -> 4 tools (P6) -> 5 server budgets/finish reasons (P8,10) -> 6 remote client (P7) -> 7 think policy (P11) -> 8 metrics (P12) -> 9 tests/bench (P13,14).
Prompt-size rule: controller-only; no added system prompt text; baseline measured before editing.
