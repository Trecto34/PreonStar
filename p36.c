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
#include "core_00_q8_0.inc"
#include "core_01_j_skip.inc"
#include "core_02_j_eval.inc"
#include "core_03_think_mode.inc"
#include "core_04_get_scale.inc"
#include "core_05_matmul_batch.inc"
#include "core_06_rms_norm.inc"
#include "core_07_model_open.inc"
#include "core_08_weights_bind.inc"
#include "core_09_hadamard_width.inc"
#include "core_10_hadamard_fwht.inc"
#include "core_11_cpu_runtime.inc"
#include "core_12_apply_rope.inc"
#include "core_13_forward_full.inc"
#include "core_14_forward_tokens.inc"
#include "core_15_next_pow2.inc"
#include "core_16_sample_top.inc"
#include "core_17_session_free.inc"
#include "core_18_mtp_timing.inc"
#include "core_19_eof.inc"
