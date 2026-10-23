# core module

Everything that runs on the CPU: GGUF loading and metadata, the quantized
matmul and dequant kernels, the forward pass, RoPE, sampling, and session
lifetime. Nothing here touches Vulkan.

The manifest is `p36.c` (at the repository root). The fragments
are included in order into one translation unit; **do not reorder them**, the
order is the program declaration order and several fragments use declarations
made in earlier ones.

| fragment | lines | begins with |
|---|---:|---|
| `core_00_f16_to_f32.inc` | 1000 | `p36_f16_to_f32` |
| `core_01_q8_0_row_bytes.inc` | 959 | `p36_q8_0_row_bytes` |
| `core_02_j_skip_ws.inc` | 949 | `j_skip_ws` |
| `core_03_j_eval_cmp.inc` | 502 | `j_eval_cmp` |
| `core_04_think_mode_for_context.inc` | 510 | `p36_think_mode_for_context` |
| `core_05_get_scale_min_k4.inc` | 964 | `p36_get_scale_min_k4` |
| `core_06_matmul_batch_worker.inc` | 495 | `p36_matmul_batch_worker` |
| `core_07_rms_norm_rows.inc` | 508 | `p36_rms_norm_rows` |
| `core_08_model_open.inc` | 921 | `model_open` |
| `core_09_weights_bind.inc` | 1013 | `weights_bind` |
| `core_10_hadamard_width_index.inc` | 17 | `p36_hadamard_width_index` |
| `core_11_hadamard_fwht_host.inc` | 471 | `p36_hadamard_fwht_host` |
| `core_12_cpu_runtime_free.inc` | 791 | `p36_cpu_runtime_free` |
| `core_13_apply_rope_axes.inc` | 738 | `p36_apply_rope_axes` |
| `core_14_forward_full_attn_batch.inc` | 203 | `p36_forward_full_attn_batch` |
| `core_15_forward_tokens_cpu.inc` | 1781 | `p36_forward_tokens_cpu` |
| `core_16_next_pow2.inc` | 976 | `next_pow2` |
| `core_17_sample_top_p_min_p.inc` | 964 | `p36_sample_top_p_min_p` |
| `core_18_session_free.inc` | 954 | `p36_session_free` |
| `core_19_mtp_timing_report.inc` | 963 | `p36_mtp_timing_report` |

Total: 20 fragments, 15679 lines.
