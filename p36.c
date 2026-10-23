/* p36.c - manifest for the core module.
 *
 * The implementation lives in ordered fragments under src/core/ so each file
 * stays navigable.  They are included into this single translation unit,
 * in this order, because the module is dense with file-scope statics:
 * splitting it into separate TUs would export those statics and cost
 * inlining on the hot paths (this build has no LTO).  Codegen is unchanged
 * by construction.
 *
 * Do not reorder these lines; the order is the program declaration order.
 */
#include "core_00_f16_to_f32.inc"
#include "core_01_q8_0_row_bytes.inc"
#include "core_02_j_skip_ws.inc"
#include "core_03_j_eval_cmp.inc"
#include "core_04_think_mode_for_context.inc"
#include "core_05_get_scale_min_k4.inc"
#include "core_06_matmul_batch_worker.inc"
#include "core_07_rms_norm_rows.inc"
#include "core_08_model_open.inc"
#include "core_09_weights_bind.inc"
#include "core_10_hadamard_width_index.inc"
#include "core_11_hadamard_fwht_host.inc"
#include "core_12_cpu_runtime_free.inc"
#include "core_13_apply_rope_axes.inc"
#include "core_14_forward_full_attn_batch.inc"
#include "core_15_forward_tokens_cpu.inc"
#include "core_16_next_pow2.inc"
#include "core_17_sample_top_p_min_p.inc"
#include "core_18_session_free.inc"
#include "core_19_mtp_timing_report.inc"
