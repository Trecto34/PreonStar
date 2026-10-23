/* p36_server.c - manifest for the server module.
 *
 * The implementation lives in ordered fragments under src/server/ so each file
 * stays navigable.  They are included into this single translation unit,
 * in this order, because the module is dense with file-scope statics:
 * splitting it into separate TUs would export those statics and cost
 * inlining on the hot paths (this build has no LTO).  Codegen is unchanged
 * by construction.
 *
 * Do not reorder these lines; the order is the program declaration order.
 */
#include "server_00_request_free.inc"
#include "server_01_parse_anthropic.inc"
#include "server_02_render_qwen.inc"
#include "server_03_parse_chat.inc"
#include "server_04_parse_anthropic.inc"
#include "server_05_json_escape.inc"
#include "server_06_context_length.inc"
#include "server_07_qwen_tool.inc"
#include "server_08_final_response.inc"
#include "server_09_anthropic_sse.inc"
#include "server_10_tool_memory.inc"
#include "server_11_kv_cache.inc"
#include "server_12_kv_cache.inc"
#include "server_13_trace_event.inc"
#include "server_14_generate_job.inc"
#include "server_15_read_http.inc"
#include "server_16_parse_options.inc"
#include "server_17_eof.inc"
