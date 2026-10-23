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
#include "vk_00_vk_prof.inc"
#include "vk_01_vk_use.inc"
#include "vk_02_vk_tensor.inc"
#include "vk_03_vk_arena.inc"
#include "vk_04_vk_stream.inc"
#include "vk_05_vk_prof.inc"
#include "vk_06_gpu_set.inc"
#include "vk_07_vk_q8.inc"
#include "vk_08_vk_predequant.inc"
#include "vk_09_gpu_v.inc"
#include "vk_10_vk_conv.inc"
#include "vk_11_vk_attn.inc"
#include "vk_12_vk_moe.inc"
#include "vk_13_eof.inc"
