# agent module

The interactive agent: the worker loop and turn handling, tool dispatch and
result rendering, the workspace jail around bash and file tools, the editor,
and the session and history stores.

The manifest is `p36_agent.c` (at the repository root). The fragments
are included in order into one translation unit; **do not reorder them**, the
order is the program declaration order and several fragments use declarations
made in earlier ones.

| fragment | lines | begins with |
|---|---:|---|
| `agent_00_set_nonblock.inc` | 780 | `linenoiseEditInsert` |
| `agent_01_parse_options.inc` | 425 | `parse_options` |
| `agent_02_tool_response_is_error.inc` | 608 | `agent_tool_response_is_error` |
| `agent_03_publishf.inc` | 931 | `agent_publishf` |
| `agent_04_syntax_alias_match.inc` | 869 | `agent_syntax_alias_match` |
| `agent_05_tool_viz_prefix.inc` | 911 | `agent_tool_viz_prefix` |
| `agent_06_kv_session_meta_free.inc` | 899 | `agent_kv_session_meta_free` |
| `agent_07_history_ptrs_push.inc` | 906 | `agent_history_ptrs_push` |
| `agent_08_parse_int_default.inc` | 905 | `agent_parse_int_default` |
| `agent_09_edit_find_old_span.inc` | 936 | `agent_edit_find_old_span` |
| `agent_10_tool_search.inc` | 975 | `agent_tool_search` |
| `agent_11_bash_publish_observation.inc` | 509 | `agent_bash_publish_observation` |
| `agent_12_watchdog_blocks.inc` | 508 | `agent_watchdog_blocks` |
| `agent_13_vw_force_final.inc` | 928 | `agent_vw_force_final` |
| `agent_14_worker_set_greedy_sampling.inc` | 492 | `worker_set_greedy_sampling` |
| `agent_15_worker_run_turn.inc` | 691 | `worker_run_turn` |
| `agent_16_worker_request_save.inc` | 747 | `worker_request_save` |
| `agent_17_editor_flush_cpr_candidate.inc` | 975 | `editor_flush_cpr_candidate` |
| `agent_18_yes_no_auto_name.inc` | 958 | `agent_yes_no_auto_name` |

Total: 19 fragments, 14953 lines.
