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
#include "agent_00_parse_options.inc"
#include "agent_01_agent_tool.inc"
#include "agent_02_agent_publishf.inc"
#include "agent_03_agent_syntax.inc"
#include "agent_04_agent_tool.inc"
#include "agent_05_agent_kv.inc"
#include "agent_06_agent_history.inc"
#include "agent_07_agent_parse.inc"
#include "agent_08_agent_edit.inc"
#include "agent_09_agent_tool.inc"
#include "agent_10_agent_bash.inc"
#include "agent_11_agent_watchdog.inc"
#include "agent_12_agent_vw.inc"
#include "agent_13_worker_set.inc"
#include "agent_14_worker_run.inc"
#include "agent_15_worker_request.inc"
#include "agent_16_editor_flush.inc"
#include "agent_17_agent_yes.inc"
#include "agent_18_eof.inc"
