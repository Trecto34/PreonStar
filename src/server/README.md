# server module

The HTTP server: request parsing for both the Anthropic and the OpenAI shapes,
SSE streaming, tool schemas and tool memory, trace events, and the on-disk KV
cache. It owns the wire format; model work is handed to the core module.

The manifest is `p36_server.c` (at the repository root). The fragments
are included in order into one translation unit; **do not reorder them**, the
order is the program declaration order and several fragments use declarations
made in earlier ones.

| fragment | lines | begins with |
|---|---:|---|
| `server_00_stop_signal_handler.inc` | 989 | `stop_signal_handler` |
| `server_01_request_free.inc` | 989 | `request_free` |
| `server_02_parse_anthropic_image_source.inc` | 960 | `parse_anthropic_image_source` |
| `server_03_render_qwen_chat_prompt_text.inc` | 411 | `render_qwen_chat_prompt_text` |
| `server_04_parse_chat_request.inc` | 634 | `parse_chat_request` |
| `server_05_parse_anthropic_request.inc` | 521 | `parse_anthropic_request` |
| `server_06_json_escape.inc` | 536 | `json_escape` |
| `server_07_context_length_error_param.inc` | 528 | `context_length_error_param` |
| `server_08_qwen_tool_decode_tracker_upd.inc` | 524 | `qwen_tool_decode_tracker_update` |
| `server_09_final_response.inc` | 492 | `final_response` |
| `server_10_anthropic_sse_stream_update.inc` | 576 | `anthropic_sse_stream_update` |
| `server_11_tool_memory_link_head.inc` | 539 | `tool_memory_link_head` |
| `server_12_kv_cache_text_sha.inc` | 527 | `kv_cache_text_sha` |
| `server_13_kv_cache_open.inc` | 938 | `kv_cache_open` |
| `server_14_trace_event.inc` | 946 | `trace_event` |
| `server_15_generate_job.inc` | 940 | `generate_job` |
| `server_16_read_http_request.inc` | 936 | `read_http_request` |
| `server_17_parse_options.inc` | 4487 | `parse_options` |

Total: 18 fragments, 16473 lines.
