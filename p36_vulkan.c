/* p36_vulkan.c - manifest for the vk module.
 *
 * The implementation lives in ordered fragments under src/vulkan/ so each file
 * stays navigable.  They are included into this single translation unit,
 * in this order, because the module is dense with file-scope statics:
 * splitting it into separate TUs would export those statics and cost
 * inlining on the hot paths (this build has no LTO).  Codegen is unchanged
 * by construction.
 *
 * Do not reorder these lines; the order is the program declaration order.
 */
#include "vk_00_now_ns.inc"
#include "vk_01_prof_reset.inc"
#include "vk_02_use_attn_qtile2.inc"
#include "vk_03_tensor_alloc_kind_unlocked.inc"
#include "vk_04_arena_plan_unlocked.inc"
#include "vk_05_stream_weight_tensor_unlocke.inc"
#include "vk_06_prof_report.inc"
#include "vk_07_gpu_set_dense_model.inc"
#include "vk_08_q8_0_quant_wait_name.inc"
#include "vk_09_predequant_b16_unlocked.inc"
#include "vk_10_gpu_v_grouped_permute_tensor.inc"
#include "vk_11_conv_silu_rows.inc"
#include "vk_12_attn_host_rows.inc"
#include "vk_13_moe_build_tiles.inc"
