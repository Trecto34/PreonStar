/* p36_agent.c - manifest for the agent module.
 *
 * The implementation lives in ordered fragments under src/agent/ so each file
 * stays navigable.  They are included into this single translation unit,
 * in this order, because the module is dense with file-scope statics:
 * splitting it into separate TUs would export those statics and cost
 * inlining on the hot paths (this build has no LTO).  Codegen is unchanged
 * by construction.
 *
 * Do not reorder these lines; the order is the program declaration order.
 */
#include "agent_00_set_nonblock.inc"
#include "agent_01_parse_options.inc"
#include "agent_02_tool_response_is_error.inc"
#include "agent_03_publishf.inc"
#include "agent_04_syntax_alias_match.inc"
#include "agent_05_tool_viz_prefix.inc"
#include "agent_06_kv_session_meta_free.inc"
#include "agent_07_history_ptrs_push.inc"
#include "agent_08_parse_int_default.inc"
#include "agent_09_edit_find_old_span.inc"
#include "agent_10_tool_search.inc"
#include "agent_11_bash_publish_observation.inc"
#include "agent_12_watchdog_blocks.inc"
#include "agent_13_vw_force_final.inc"
#include "agent_14_worker_set_greedy_sampling.inc"
#include "agent_15_worker_run_turn.inc"
#include "agent_16_worker_request_save.inc"
#include "agent_17_editor_flush_cpr_candidate.inc"
#include "agent_18_yes_no_auto_name.inc"
