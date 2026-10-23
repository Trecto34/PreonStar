# vulkan module

The Vulkan backend: instance and device setup, the tensor arena, pipeline and
descriptor management, and the dispatch paths for attention, MoE and the
elementwise kernels.

The manifest is `p36_vulkan.c` (at the repository root). The fragments
are included in order into one translation unit; **do not reorder them**, the
order is the program declaration order and several fragments use declarations
made in earlier ones.

| fragment | lines | begins with |
|---|---:|---|
| `vk_00_now_ns.inc` | 618 | `enum` |
| `vk_01_prof_reset.inc` | 618 | `p36_vk_prof_reset` |
| `vk_02_use_attn_qtile2.inc` | 599 | `p36_vk_use_attn_qtile2` |
| `vk_03_tensor_alloc_kind_unlocked.inc` | 637 | `p36_vk_tensor_alloc_kind_unlocked` |
| `vk_04_arena_plan_unlocked.inc` | 634 | `p36_vk_arena_plan_unlocked` |
| `vk_05_stream_weight_tensor_unlocke.inc` | 631 | `p36_vk_stream_weight_tensor_unlocked` |
| `vk_06_prof_report.inc` | 604 | `p36_vk_prof_report` |
| `vk_07_gpu_set_dense_model.inc` | 608 | `p36_gpu_set_dense_model` |
| `vk_08_q8_0_quant_wait_name.inc` | 616 | `p36_vk_q8_0_quant_wait_name` |
| `vk_09_predequant_b16_unlocked.inc` | 611 | `p36_vk_predequant_b16_unlocked` |
| `vk_10_gpu_v_grouped_permute_tensor.inc` | 724 | `p36_gpu_v_grouped_permute_tensor` |
| `vk_11_conv_silu_rows.inc` | 534 | `p36_vk_conv_silu_rows` |
| `vk_12_attn_host_rows.inc` | 926 | `p36_vk_attn_host_rows` |
| `vk_13_moe_build_tiles.inc` | 1552 | `p36_vk_moe_build_tiles` |

Total: 14 fragments, 9912 lines.
