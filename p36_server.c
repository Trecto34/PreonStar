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
#include "server_00_stop_signal_handler.inc"
#include "server_01_request_free.inc"
#include "server_02_parse_anthropic_image_source.inc"
#include "server_03_render_qwen_chat_prompt_text.inc"
#include "server_04_parse_chat_request.inc"
#include "server_05_parse_anthropic_request.inc"
#include "server_06_json_escape.inc"
#include "server_07_context_length_error_param.inc"
#include "server_08_qwen_tool_decode_tracker_upd.inc"
#include "server_09_final_response.inc"
#include "server_10_anthropic_sse_stream_update.inc"
#include "server_11_tool_memory_link_head.inc"
#include "server_12_kv_cache_text_sha.inc"
#include "server_13_kv_cache_open.inc"
#include "server_14_trace_event.inc"
#include "server_15_generate_job.inc"
#include "server_16_read_http_request.inc"
#include "server_17_parse_options.inc"
