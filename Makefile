comma := ,
CC ?= cc
CXX ?= c++
GLSLC ?= ./glslc
.DEFAULT_GOAL := all
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
# Release-safe Apple Silicon baseline. Local-only builds may override this with
# NATIVE_CPU_FLAG=-mcpu=native, but published binaries must remain M1-capable.
NATIVE_CPU_FLAG ?= -mcpu=apple-m1
MACOSX_DEPLOYMENT_TARGET ?= 11.0
DARWIN_MIN_FLAG := -mmacosx-version-min=$(MACOSX_DEPLOYMENT_TARGET)
else
NATIVE_CPU_FLAG ?= -march=native
DARWIN_MIN_FLAG :=
endif

CFLAGS ?= -O3 -ffast-math $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
LLAMA_CPP_DIR ?= llama.cpp
LLAMA_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build
REFERENCE_MODEL ?= gguf/Qwen3.6-35B-A3B-Q8_0.gguf
QWEN27B_MODEL ?= gguf/Qwen3.8-27B-UD-IQ3_S.gguf
QWEN27B_PROMPT ?= tests/long_context_story_prompt.txt
QWEN27B_GEN ?= 128
QWEN27B_GATE_GEN ?= 32
QWEN27B_LLAMA_PROMPT ?= 64
QWEN27B_LLAMA_BENCH ?= $(LLAMA_CPP_DIR)/build-vulkan/bin/llama-bench
SAMPLING_TEST := tests/test_sampling
OPENROUTER_MODEL ?= qwen/qwen3.6-35b-a3b
OPENROUTER_OUT ?= gguf-tools/quality-testing/local/openrouter
OPENROUTER_COUNT ?= 100
LLAMA_INCLUDE := -I"$(LLAMA_CPP_DIR)/include" -I"$(LLAMA_CPP_DIR)/ggml/include"
LLAMA_LIBS := \
	"$(LLAMA_BUILD_DIR)/src/libllama.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml-cpu.a" \
	"$(LLAMA_BUILD_DIR)/ggml/src/libggml-base.a"
LLAMA_LDLIBS := -lstdc++ -fopenmp
# Split translation units keep their fragments under src/<module>/. The fragments
# include project headers by bare name, so the tree root and each module directory
# must be on the quoted-include search path for every compile.
SRC_INCS := -I. -Isrc/core -Isrc/server -Isrc/agent -Isrc/vulkan
CPU_CFLAGS := $(filter-out -ffast-math,$(CFLAGS)) $(DARWIN_MIN_FLAG) -D_GNU_SOURCE -fno-finite-math-only -DP36_NO_GPU $(SRC_INCS)
GPU_CFLAGS := $(CFLAGS) $(DARWIN_MIN_FLAG) -D_GNU_SOURCE -fno-finite-math-only $(SRC_INCS)
VULKAN_CFLAGS ?=
LDLIBS ?= -lm -pthread
GPU_LDLIBS := $(LDLIBS) -ldl -lvulkan
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
METAL_LDFLAGS := $(LDFLAGS) $(DARWIN_MIN_FLAG)
METAL_SRCS := $(wildcard metal/*.metal)
VULKAN_SHADERS := \
	vulkan/gdn_front.spv \
	vulkan/matmul_f16.spv \
	vulkan/matmul_bf16.spv \
	vulkan/vision_matmul_f16.spv \
	vulkan/vision_attention.spv \
	vulkan/matmul_f32.spv \
	vulkan/matmul_f32_fast.spv \
	vulkan/matmul_f32_fast_w256.spv \
	vulkan/add.spv \
	vulkan/hadamard_prepare.spv \
	vulkan/directional_steering.spv \
	vulkan/add_rms_norm.spv \
	vulkan/add_rms_norm_q8_k.spv \
	vulkan/rms_norm.spv \
	vulkan/swiglu.spv \
	vulkan/swiglu_q8_k.spv \
	vulkan/rope_qwen.spv \
	vulkan/rope_qwen_mrope.spv \
	vulkan/rms_norm_rope_qwen.spv \
	vulkan/rms_norm_rope_kv_qwen.spv \
	vulkan/copy_rows.spv \
	vulkan/embed_row.spv \
	vulkan/recur_window.spv \
	vulkan/conv_silu.spv \
	vulkan/recur_conv_silu_decode.spv \
	vulkan/recur_norm_gate.spv \
	vulkan/recur_norm_gate_q8_k.spv \
	vulkan/delta_qk.spv \
	vulkan/delta_qkv.spv \
	vulkan/delta_gates.spv \
	vulkan/quantize_q8_0.spv \
	vulkan/quantize_q8_k.spv \
	vulkan/predequant_b16.spv \
	vulkan/matmul_q8_0.spv \
	vulkan/matmul_q8_0_p36.spv \
	vulkan/matmul_q8_0_f32b.spv \
	vulkan/matmul_q8_0_f32b_pair.spv \
	vulkan/shared_gate_up_decode.spv \
	vulkan/shared_down_tail_decode.spv \
	vulkan/matmul_q8_0_mm.spv \
	vulkan/matmul_q8_0_mm_f16.spv \
	vulkan/matmul_q8_0_mm_f16_out32.spv \
	vulkan/matmul_q8_0_f32b_nx.spv \
	vulkan/matmul_q8_0_decode.spv \
	vulkan/matmul_q8_0_decode_p36.spv \
	vulkan/matmul_q8_0_decode_b64.spv \
	vulkan/matmul_kquant.spv \
	vulkan/matmul_kquant_mmq.spv \
	vulkan/matmul_q5k_mmq.spv \
	vulkan/matmul_q6k_mmq.spv \
	vulkan/matmul_q5k_mmq_fast.spv \
	vulkan/matmul_q6k_mmq_fast.spv \
	vulkan/delta_net.spv \
	vulkan/delta_net_fast.spv \
	vulkan/delta_net_cols.spv \
	vulkan/delta_net_cols_f16.spv \
	vulkan/delta_net_decode.spv \
	vulkan/delta_net_decode_reg.spv \
	vulkan/delta_net_decode_reg_f16.spv \
	vulkan/attn_scores.spv \
	vulkan/attn_post.spv \
	vulkan/attn_reduce.spv \
	vulkan/attn_decode_fused.spv \
	vulkan/attn_decode_split.spv \
	vulkan/attn_prefill_qtile.spv \
	vulkan/attn_prefill_qtile2.spv \
	vulkan/attn_prefill_qtile2_gqa6.spv \
	vulkan/attn_prefill_fa_gqa6.spv \
	vulkan/attn_prefill_fa_gqa8.spv \
	vulkan/attn_combine.spv \
	vulkan/moe_gate_up.spv \
	vulkan/moe_gate_up_iq2s.spv \
	vulkan/router_topk.spv \
	vulkan/moe_tiles.spv \
	vulkan/kv_store.spv \
	vulkan/kv_store_quant.spv \
	vulkan/rms_norm_rope_kv_qwen_quant.spv \
	vulkan/top2.spv \
	vulkan/topk8.spv \
	vulkan/logits_minp_pack.spv \
	vulkan/moe_gate_up_f32b.spv \
	vulkan/moe_gate_up_f32b_iq2s.spv \
	vulkan/moe_down_q2k_f32b_iq2s.spv \
	vulkan/moe_gate_up_decode.spv \
	vulkan/moe_gate_up_decode_iq2s.spv \
	vulkan/moe_gate_up_decode_iq3xxs.spv \
	vulkan/moe_gate_up_decode_q3k.spv \
	vulkan/moe_gate_up_decode_iq2s_iq2xxs.spv \
	vulkan/moe_down_q2k_sum_decode_iq2s.spv \
	vulkan/moe_down_q2k_sum_decode_iq3s.spv \
	vulkan/moe_down_q2k_sum_decode_iq3xxs.spv \
	vulkan/moe_down_q2k_sum_decode_q3k.spv \
	vulkan/moe_down_q2k_f32b.spv \
	vulkan/moe_down_q2k_sum_decode.spv \
	vulkan/moe_gate_up_q4k_f32b.spv \
	vulkan/moe_down_q4k_sum_decode.spv \
	vulkan/moe_gate_up_gemm.spv \
	vulkan/moe_gate_up_gemm_iq2s.spv \
	vulkan/moe_gate_up_gemm_iq3xxs.spv \
	vulkan/moe_gate_up_gemm_q3k.spv \
	vulkan/moe_gate_up_gemm_iq2s_iq2xxs.spv \
	vulkan/moe_down_gemm_iq2s.spv \
	vulkan/moe_down_gemm_iq3s.spv \
	vulkan/moe_down_gemm_iq3xxs.spv \
	vulkan/moe_down_gemm_q3k.spv \
	vulkan/moe_down_gemm.spv \
	vulkan/moe_matvec.spv \
	vulkan/moe_matvec_fast.spv \
	vulkan/dense_iq3_xxs_decode.spv \
	vulkan/dense_iq3_xxs_decode_r4.spv \
	vulkan/dense_iq3_xxs_decode_nx.spv \
	vulkan/dense_iq3_xxs_mmq.spv \
	vulkan/dense_iq3_xxs_mmq_pair.spv \
	vulkan/dense_iq3_s_decode.spv \
	vulkan/dense_iq3_s_decode_full.spv \
	vulkan/dense_iq3_s_decode_r4.spv \
	vulkan/dense_iq3_s_decode_r1.spv \
	vulkan/dense_iq3_s_mmq.spv \
	vulkan/dense_iq3_s_bm64_mmq.spv \
	vulkan/dense_iq3_s_mmq_r4.spv \
	vulkan/dense_iq4_xs.spv \
	vulkan/dense_iq4_xs_decode.spv \
	vulkan/dense_iq4_xs_mmq.spv \
	vulkan/dense_iq1_m.spv \
	vulkan/dense_extra_decode.spv \
	vulkan/dense_extra_decode_q2_0.spv \
	vulkan/dense_extra_decode_pq2_0.spv \
	vulkan/dense_extra_small_q2_0.spv \
	vulkan/dense_extra_small_pq2_0.spv \
	vulkan/dense_extra_mmq_pq2_0.spv \
	vulkan/dense_extra_decode_ptq1_0.spv \
	vulkan/dense_extra_mmq_ptq1_0.spv \
	vulkan/dense_extra_decode_iq2s_pair.spv \
	vulkan/dense_extra_decode_iq2s.spv \
	vulkan/dense_extra_mmq_iq2s.spv \
	vulkan/dense_extra_mmq.spv \
	vulkan/dense_extra_mmq_q2_0.spv \
	vulkan/dense_kquant_mmq.spv \
	vulkan/dense_kquant_decode.spv \
	vulkan/dense_q4k_decode.spv \
	vulkan/dense_q5k_decode.spv \
	vulkan/dense_q3k_decode.spv \
	vulkan/moe_reduce.spv \
	vulkan/ffn_tail.spv

vulkan/moe_matvec_fast.spv: vulkan/moe_matvec_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/embed_row.spv: vulkan/embed_row.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/top2.spv: vulkan/top2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/topk8.spv: vulkan/topk8.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/logits_minp_pack.spv: vulkan/logits_minp_pack.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/recur_norm_gate.spv: vulkan/recur_norm_gate.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/recur_norm_gate_q8_k.spv: vulkan/recur_norm_gate_q8_k.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/swiglu_q8_k.spv: vulkan/swiglu_q8_k.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_decode.spv: vulkan/dense_iq3_xxs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_decode_r4.spv: vulkan/dense_iq3_xxs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_ROWS=4 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_xxs_decode_nx.spv: vulkan/dense_iq3_xxs_decode_nx.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_mmq.spv: vulkan/dense_iq3_xxs_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_xxs_mmq_pair.spv: vulkan/dense_iq3_xxs_mmq_pair.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_decode.spv: vulkan/dense_iq3_s_decode.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_decode_full.spv: vulkan/dense_iq3_s_decode.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_decode_r4.spv: vulkan/dense_iq3_s_decode.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_ROWS=4 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_decode_r1.spv: vulkan/dense_iq3_s_decode.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_ROWS=1 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_iq3_s_mmq.spv: vulkan/dense_iq3_s_mmq.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq3_s_bm64_mmq.spv: vulkan/dense_iq3_s_mmq.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_BM=64 -o $@ $<

vulkan/dense_iq3_s_mmq_r4.spv: vulkan/dense_iq3_s_mmq.comp p36_iq3s_grid_values.inc
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_BM=4 -DP36_BK=64 -o $@ $<

vulkan/dense_iq4_xs.spv: vulkan/dense_iq4_xs.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq4_xs_decode.spv: vulkan/dense_iq4_xs_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_iq4_xs_mmq.spv: vulkan/dense_iq4_xs_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_iq1_m.spv: vulkan/dense_iq1_m.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_decode.spv: vulkan/dense_extra_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_decode_q2_0.spv: vulkan/dense_extra_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q2_0_ONLY=1 -o $@ $<

vulkan/dense_extra_decode_pq2_0.spv: vulkan/dense_extra_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q2_0_ONLY=1 -DP36_PQ2_0=1 -o $@ $<

vulkan/dense_extra_small_q2_0.spv: vulkan/dense_extra_small_q2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_small_pq2_0.spv: vulkan/dense_extra_small_q2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_PQ2_0=1 -o $@ $<

vulkan/dense_extra_mmq_pq2_0.spv: vulkan/dense_extra_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q2_0_ONLY=1 -DP36_PQ2_0=1 -o $@ $<

vulkan/dense_extra_decode_ptq1_0.spv: vulkan/dense_extra_decode_ptq1_0.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_decode_iq2s_pair.spv: vulkan/dense_extra_decode_iq2s_pair.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_decode_iq2s.spv: vulkan/dense_extra_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_IQ2S_ONLY=1 -o $@ $<

vulkan/dense_extra_mmq_iq2s.spv: vulkan/dense_extra_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_IQ2S_ONLY=1 -o $@ $<

vulkan/dense_extra_mmq_ptq1_0.spv: vulkan/dense_extra_mmq_ptq1_0.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_mmq.spv: vulkan/dense_extra_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_extra_mmq_q2_0.spv: vulkan/dense_extra_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q2_0_ONLY=1 -o $@ $<

vulkan/predequant_b16.spv: vulkan/predequant_b16.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_kquant_mmq.spv: vulkan/dense_kquant_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_kquant_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/dense_q4k_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q4K_ONLY=1 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_q3k_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q3K_ONLY=1 -DP36_BOUNDS=0 -o $@ $<

vulkan/dense_q5k_decode.spv: vulkan/dense_kquant_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_Q5K_ONLY=1 -DP36_BOUNDS=0 -o $@ $<

vulkan/matmul_q8_0_f32b.spv: vulkan/matmul_q8_0_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_f32b_pair.spv: vulkan/matmul_q8_0_f32b_pair.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/shared_gate_up_decode.spv: vulkan/shared_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/shared_down_tail_decode.spv: vulkan/shared_down_tail_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm.spv: vulkan/matmul_q8_0_mm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm_f16.spv: vulkan/matmul_q8_0_mm_f16.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_mm_f16_out32.spv: vulkan/matmul_q8_0_mm_f16_out32.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q8_0_f32b_nx.spv: vulkan/matmul_q8_0_f32b_nx.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_f32b.spv: vulkan/moe_gate_up_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# IQ2_S expert variants of the small-batch (2..127 token) fused kernels, same
# sources as the IQ2_XXS/Q2_K builds which stay byte-identical.
vulkan/moe_gate_up_f32b_iq2s.spv: vulkan/moe_gate_up_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

vulkan/moe_down_q2k_f32b_iq2s.spv: vulkan/moe_down_q2k_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

# Same source as moe_gate_up.comp with IQ2_S weight decode (Qwen3.8-35B-A3B-IQ2_M
# and other IQ2_S expert files).  The IQ2_XXS build stays byte-identical.
vulkan/moe_gate_up_iq2s.spv: vulkan/moe_gate_up.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

vulkan/moe_gate_up_decode.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_decode_iq2s.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

vulkan/moe_gate_up_decode_iq3xxs.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3XXS -o $@ $<

vulkan/moe_gate_up_decode_q3k.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_Q3K -o $@ $<

vulkan/moe_gate_up_decode_iq2s_iq2xxs.spv: vulkan/moe_gate_up_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_MIX_IQ2S_IQ2XXS -o $@ $<

vulkan/moe_down_q2k_sum_decode_iq2s.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

# IQ3_S down projection (Qwen3.8-35B-A3B-IQ2_M layers 0-2 carry IQ3_S experts
# under IQ2_S gate/up).  Same sources as the Q2_K/IQ2_S variants.
vulkan/moe_down_q2k_sum_decode_iq3s.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3S -o $@ $<

vulkan/moe_down_q2k_sum_decode_iq3xxs.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3XXS -o $@ $<

vulkan/moe_down_q2k_sum_decode_q3k.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_Q3K -o $@ $<

vulkan/moe_down_q2k_f32b.spv: vulkan/moe_down_q2k_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_q2k_sum_decode.spv: vulkan/moe_down_q2k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_q4k_f32b.spv: vulkan/moe_gate_up_q4k_f32b.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_down_q4k_sum_decode.spv: vulkan/moe_down_q4k_sum_decode.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/moe_gate_up_gemm.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# IQ2_S expert variants (Qwen3.8-35B-A3B-IQ2_M prefill).  Same sources as the
# IQ2_XXS/Q2_K kernels; the base builds stay byte-identical.
vulkan/moe_gate_up_gemm_iq2s.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

vulkan/moe_gate_up_gemm_iq3xxs.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3XXS -o $@ $<

vulkan/moe_gate_up_gemm_q3k.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_Q3K -o $@ $<

vulkan/moe_gate_up_gemm_iq2s_iq2xxs.spv: vulkan/moe_gate_up_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_MIX_IQ2S_IQ2XXS -o $@ $<

vulkan/moe_down_gemm_iq2s.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ2S -o $@ $<

vulkan/moe_down_gemm_iq3s.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3S -o $@ $<

vulkan/moe_down_gemm_iq3xxs.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_IQ3XXS -o $@ $<

vulkan/moe_down_gemm_q3k.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MOE_Q3K -o $@ $<

vulkan/moe_down_gemm.spv: vulkan/moe_down_gemm.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/delta_net_cols.spv: vulkan/delta_net_cols.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/delta_net_cols_f16.spv: vulkan/delta_net_cols.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_STATE_F16=1 -o $@ $<

# P36_COLS=32: a workgroup covers 32 j columns, so its state reads consume
# whole 128-byte cache lines instead of 32 bytes of each.  Must stay in step
# with the reg_cols default in p36_vulkan.c, which sizes the dispatch grid.
vulkan/delta_net_decode_reg.spv: vulkan/delta_net_decode_reg.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_COLS=32 -o $@ $<

vulkan/delta_net_decode_reg_f16.spv: vulkan/delta_net_decode_reg.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_STATE_F16=1 -DP36_COLS=32 -o $@ $<

vulkan/attn_decode_fused.spv: vulkan/attn_decode_fused.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_decode_split.spv: vulkan/attn_decode_split.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_qtile.spv: vulkan/attn_prefill_qtile.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_qtile2.spv: vulkan/attn_prefill_qtile2.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_qtile2_gqa6.spv: vulkan/attn_prefill_qtile2_gqa6.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/attn_prefill_fa_gqa6.spv: vulkan/attn_prefill_fa.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DHQ=6 -o $@ $<

vulkan/attn_prefill_fa_gqa8.spv: vulkan/attn_prefill_fa.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DHQ=8 -o $@ $<

# Vulkan is the default backend. CORE_OBJS holds the GPU engine; CPU_CORE_OBJS
# is the -DP36_NO_GPU reference build selected by `make cpu`.
CORE_OBJS := p36_gpu_core.o p36_vulkan.o p36_image.o
METAL_CORE_OBJS := p36_gpu_core_metal.o p36_metal.o p36_image.o
CPU_CORE_OBJS := p36_cpu.o p36_image.o

# Fragment lists. These are prerequisites so that editing a fragment rebuilds the
# translation unit that #includes it; without them an incremental build silently
# links a stale object.
CORE_INCS := $(wildcard src/core/*.inc)
SERVER_INCS := $(wildcard src/server/*.inc)
AGENT_INCS := $(wildcard src/agent/*.inc)
VK_INCS := $(wildcard src/vulkan/*.inc)

.PHONY: test-quality test-cli all help cpu gpu vulkan vulkan-generic vulkan-bc250 metal p36-quality-score test test-metal test-metal-model test-qwen27b-metal-parity test-qwen27b-vulkan-parity test-qwen27b-llama-parity test-quick test-all test-unit test-vulkan test-streaming test-mtp test-model test-session-batch test-server-live test-server-live-metal test-server-live-metal-ssd test-server-batching test-server-batching-metal test-server-batching-metal-ssd test-release release-build-check release-build-check-metal benchmark-gate benchmark-session-batch benchmark-qwen27b benchmark-qwen27b-gate benchmark-qwen27b-metal-gate benchmark-qwen27b-vulkan benchmark-qwen27b-metal benchmark-qwen27b-llama test-reference test-reference-local test-vectors-local reference-openrouter test-llama test-llama-long test-llama-batch test-llama-all clean

all: p36 p36-server p36-bench p36-agent p36-eval p36_test

help:
	@echo "P36 build targets:"
	@echo "  make              Build generic Vulkan with automatic BC-250 fast-path selection (default)"
	@echo "  make vulkan-generic  Build generic Vulkan for runtime capability detection"
	@echo "  make vulkan-bc250    Build Vulkan and require an AMD BC-250 at runtime"
	@echo "  make metal        Build the same binaries with the Metal backend (macOS)"
	@echo "  make test-metal   Build Metal and run its model-independent unit and kernel tests"
	@echo "  make test-metal-model  Run Metal model, parity, streaming, and state tests"
	@echo "  make test-qwen27b-metal-parity  Run short CPU/Metal parity for the IQ3_S 27B model"
	@echo "  make test-qwen27b-vulkan-parity  Run short CPU/Vulkan parity for the IQ3_S 27B model"
	@echo "  make test-qwen27b-llama-parity  Run short llama.cpp/CPU parity for the IQ3_S 27B model"
	@echo "  make cpu          Build CPU-only ./p36, ./p36-server, ./p36-bench, ./p36-agent, ./p36-eval, and ./p36_test"
	@echo "  make p36-quality-score  Build the OpenRouter/P36 local scorer"
	@echo "  make test         Build and run tests"
	@echo "  make test-release Run the complete model, Vulkan, reference, streaming, benchmark, and build gates"
	@echo "  make benchmark-session-batch  Benchmark old, 1/2/4/8-slot, and ordered-fallback server decode"
	@echo "  make benchmark-qwen27b  Benchmark p36 at 128, 2048, and 4096 prompt tokens, then llama.cpp at its fitting context"
	@echo "  make benchmark-qwen27b-gate  Quick 128/2048 comparison gate with 32 decode tokens"
	@echo "  make benchmark-qwen27b-metal-gate  Quick Metal 128/2048 gate with 32 decode tokens"
	@echo "  make test-reference       Compare P36 CPU against tracked llama.cpp results"
	@echo "  make test-reference-local Compare P36 CPU against ignored local results"
	@echo "  make test-vectors-local   Capture ignored local llama.cpp results"
	@echo "  make reference-openrouter Capture ignored official API results"
	@echo "  make test-llama           Build optional live CPU-vs-llama.cpp tests"
	@echo "  make clean        Remove build outputs"

gpu: vulkan-generic

vulkan: vulkan-generic

vulkan-generic:
	$(MAKE) -B all VULKAN_CFLAGS=

vulkan-bc250:
	$(MAKE) -B all VULKAN_CFLAGS=-DP36_VULKAN_REQUIRE_BC250

metal: p36_cli_metal.o p36_server.o p36_bench.o p36_agent.o p36_eval.o p36_eval_cases.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o rax.o p36_test_metal.o p36_gpu_core_metal_test.o $(METAL_CORE_OBJS)
	$(CC) $(METAL_LDFLAGS) -o p36 p36_cli_metal.o p36_ssd.o p36_prompt_prefix.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o p36-server p36_server.o rax.o p36_ssd.o p36_prompt_prefix.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o p36-bench p36_bench.o p36_ssd.o p36_prompt_prefix.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o p36-agent p36_agent.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o p36-eval p36_eval.o p36_eval_cases.o p36_help.o p36_ssd.o p36_prompt_prefix.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)
	$(CC) $(METAL_LDFLAGS) -o p36_test p36_test_metal.o rax.o p36_ssd.o p36_prompt_prefix.o p36_image.o p36_gpu_core_metal_test.o p36_metal.o $(METAL_LDLIBS)

p36: p36_cli.o p36_ssd.o p36_prompt_prefix.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_cli.o p36_ssd.o p36_prompt_prefix.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

p36-server: p36_server.o rax.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_server.o rax.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS) $(GPU_LDLIBS)

p36-bench: p36_bench.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_bench.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS) $(GPU_LDLIBS)

p36-agent: p36_agent.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_agent.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

p36-eval: p36_eval.o p36_eval_cases.o p36_help.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_eval.o p36_eval_cases.o p36_help.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS) $(GPU_LDLIBS)

p36_test: p36_test.o rax.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_test.o rax.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS) $(GPU_LDLIBS)

TOPK8_TEST := tests/test_topk8

tests/test_topk8.o: tests/test_topk8.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(TOPK8_TEST): tests/test_topk8.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ATTN_DECODE_TEST := tests/test_attn_decode

tests/test_attn_decode.o: tests/test_attn_decode.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(ATTN_DECODE_TEST): tests/test_attn_decode.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ATTN_PREFILL_QTILE2_TEST := tests/test_attn_prefill_qtile2

tests/test_attn_prefill_qtile2.o: tests/test_attn_prefill_qtile2.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(ATTN_PREFILL_QTILE2_TEST): tests/test_attn_prefill_qtile2.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

PQ2_SMALL_TEST := tests/test_pq2_small

tests/test_pq2_small.o: tests/test_pq2_small.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(PQ2_SMALL_TEST): tests/test_pq2_small.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

ATTN_FA_TEST := tests/test_attn_fa

tests/test_attn_fa.o: tests/test_attn_fa.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(ATTN_FA_TEST): tests/test_attn_fa.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

NORM_ROPE_KV_TEST := tests/test_norm_rope_kv

tests/test_norm_rope_kv.o: tests/test_norm_rope_kv.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(NORM_ROPE_KV_TEST): tests/test_norm_rope_kv.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

MATMUL_Q8_TEST := tests/test_matmul_q8

tests/test_matmul_q8.o: tests/test_matmul_q8.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(MATMUL_Q8_TEST): tests/test_matmul_q8.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

MOE_GATE_UP_TEST := tests/test_moe_gate_up

tests/test_moe_gate_up.o: tests/test_moe_gate_up.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(MOE_GATE_UP_TEST): tests/test_moe_gate_up.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

MOE_DOWN_TEST := tests/test_moe_down

tests/test_moe_down.o: tests/test_moe_down.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(MOE_DOWN_TEST): tests/test_moe_down.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

IQ3XXS_NX_TEST := tests/test_dense_iq3xxs_nx

tests/test_dense_iq3xxs_nx.o: tests/test_dense_iq3xxs_nx.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(IQ3XXS_NX_TEST): tests/test_dense_iq3xxs_nx.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)

IQ1M_MMQ_TEST := tests/test_dense_iq1m_mmq

tests/test_dense_iq1m_mmq.o: tests/test_dense_iq1m_mmq.c p36_gpu.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ $<

$(IQ1M_MMQ_TEST): tests/test_dense_iq1m_mmq.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(GPU_LDLIBS)


p36-quality-score: gguf-tools/quality-testing/score_openrouter

gguf-tools/quality-testing/score_openrouter: gguf-tools/quality-testing/score_openrouter.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -I. -o $@ gguf-tools/quality-testing/score_openrouter.o p36_ssd.o p36_prompt_prefix.o $(CORE_OBJS) $(GPU_LDLIBS)

# `make cpu` rebuilds the same binary names from the -DP36_NO_GPU objects.
cpu: p36_cli_cpu.o linenoise_cpu.o p36_server_cpu.o rax_cpu.o p36_bench_cpu.o p36_agent_cpu.o p36_eval_cpu.o p36_eval_cases.o p36_help_cpu.o p36_kvstore_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o p36_web_cpu.o p36_test_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o p36 p36_cli_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o linenoise_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o p36-server p36_server_cpu.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o p36-bench p36_bench_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o p36-agent p36_agent_cpu.o p36_help_cpu.o p36_kvstore_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o p36_web_cpu.o linenoise_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o p36-eval p36_eval_cpu.o p36_eval_cases.o p36_help_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CPU_CFLAGS) -o p36_test p36_test_cpu.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LDLIBS)

# --- GPU (default) objects ---
p36_gpu_core.o: p36.c p36.h p36_image.h p36_gpu.h p36_quant.h p36_ssd.h p36_iq_tables.h p36_iq3s_grid_values.inc p36_streaming_hotlist.inc $(CORE_INCS)
	$(CC) $(GPU_CFLAGS) -c -o $@ p36.c

p36_gpu_core_metal.o: p36.c p36.h p36_image.h p36_gpu.h p36_quant.h p36_ssd.h p36_iq_tables.h p36_iq3s_grid_values.inc p36_streaming_hotlist.inc $(CORE_INCS)
	$(CC) $(GPU_CFLAGS) -DP36_METAL -c -o $@ p36.c

p36_gpu_core_metal_test.o: p36.c p36.h p36_gpu.h p36_quant.h p36_ssd.h p36_iq_tables.h p36_iq3s_grid_values.inc p36_streaming_hotlist.inc $(CORE_INCS)
	$(CC) $(GPU_CFLAGS) -DP36_METAL -DP36_METAL_TEST_COMPAT -c -o $@ p36.c

p36_cli_metal.o: p36_prompt_prefix.h p36_cli.c p36.h p36_cli_args.h p36_ssd.h linenoise.h
	$(CC) $(GPU_CFLAGS) -DP36_METAL -c -o $@ p36_cli.c

p36_vulkan.o: p36_vulkan.c p36_gpu.h p36_quant.h p36_iq2_tables_vulkan.inc p36_iq_tables.h p36_iq3s_grid_values.inc $(VULKAN_SHADERS) $(VK_INCS)
	$(CC) $(GPU_CFLAGS) $(VULKAN_CFLAGS) -c -o $@ p36_vulkan.c

p36_metal.o: p36_metal.m p36_gpu.h p36_quant.h p36_ssd.h $(METAL_SRCS)
	$(CC) $(GPU_CFLAGS) -fobjc-arc -c -o $@ p36_metal.m

p36_cli.o: p36_prompt_prefix.h p36_cli.c p36.h p36_cli_args.h p36_ssd.h linenoise.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_cli.c

p36_server.o: p36_loopguard.h p36_server.c p36.h p36_cli_args.h rax.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_server.c

p36_bench.o: p36_bench.c p36.h p36_cli_args.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_bench.c

p36_agent.o: p36_loopguard.h p36_remote.h p36_prompt_prefix.h p36_agent.c p36.h p36_cli_args.h p36_help.h p36_kvstore.h p36_ssd.h p36_web.h linenoise.h p36_tool_text.h $(AGENT_INCS)
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_agent.c

p36_eval.o: p36_eval_cases.h p36_eval.c p36.h p36_cli_args.h p36_help.h p36_ssd.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_eval.c

p36_help.o: p36_help.c p36_help.h p36.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_help.c

p36_kvstore.o: p36_kvstore.c p36_kvstore.h p36.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_kvstore.c

p36_ssd.o: p36_ssd.c p36_ssd.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_ssd.c

p36_web.o: p36_web.c p36_web.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_web.c

p36_remote.o: p36_remote.c p36_remote.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_remote.c

p36_test.o: tests/p36_test.c p36_server.c p36.h p36_cli_args.h rax.h p36_gpu.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(GPU_CFLAGS) -Wno-unused-function -c -o $@ tests/p36_test.c

p36_test_metal.o: tests/p36_test.c p36_server.c p36.h p36_cli_args.h rax.h p36_gpu.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(GPU_CFLAGS) -DP36_METAL -Wno-unused-function -c -o $@ tests/p36_test.c

gguf-tools/quality-testing/score_openrouter.o: gguf-tools/quality-testing/score_openrouter.c p36.h p36_cli_args.h p36_ssd.h p36_prompt_prefix.h
	$(CC) $(GPU_CFLAGS) -I. -c -o $@ gguf-tools/quality-testing/score_openrouter.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(GPU_CFLAGS) -c -o $@ linenoise.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(GPU_CFLAGS) -c -o $@ rax.c

vulkan/%.spv: vulkan/%.comp
	$(GLSLC) -O -o $@ $<

vulkan/gdn_front.spv: vulkan/gdn_front.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_f32_fast.spv: vulkan/matmul_f32_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# Wide variant: faster but not bit-exact (router gate). Opt-in at runtime.
vulkan/matmul_f32_fast_w256.spv: vulkan/matmul_f32_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -DP36_MFF_LOCAL=256 -o $@ $<

vulkan/matmul_q8_0.spv: vulkan/matmul_q8_0.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q5k_mmq_fast.spv: vulkan/matmul_q5k_mmq_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q6k_mmq_fast.spv: vulkan/matmul_q6k_mmq_fast.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q5k_mmq.spv: vulkan/matmul_q5k_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/matmul_q6k_mmq.spv: vulkan/matmul_q6k_mmq.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

vulkan/rms_norm_rope_kv_qwen_quant.spv: vulkan/rms_norm_rope_kv_qwen_quant.comp
	$(GLSLC) -O --target-env=vulkan1.1 -o $@ $<

# --- CPU-only objects (-DP36_NO_GPU) ---
p36_cpu.o: p36.c p36.h p36_image.h p36_gpu.h p36_quant.h p36_ssd.h p36_iq_tables.h p36_iq3s_grid_values.inc p36_streaming_hotlist.inc $(CORE_INCS)
	$(CC) $(CPU_CFLAGS) -c -o $@ p36.c

p36_image.o: p36_image.c p36_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(GPU_CFLAGS) -c -o $@ p36_image.c

p36_cli_cpu.o: p36_prompt_prefix.h p36_cli.c p36.h p36_cli_args.h p36_ssd.h linenoise.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_cli.c

p36_server_cpu.o: p36_loopguard.h p36_server.c p36.h p36_cli_args.h rax.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_server.c

p36_bench_cpu.o: p36_bench.c p36.h p36_cli_args.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_bench.c

p36_agent_cpu.o: p36_loopguard.h p36_remote.h p36_prompt_prefix.h p36_agent.c p36.h p36_cli_args.h p36_help.h p36_kvstore.h p36_ssd.h p36_web.h linenoise.h p36_tool_text.h $(AGENT_INCS)
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_agent.c

p36_eval_cpu.o: p36_eval_cases.h p36_eval.c p36.h p36_cli_args.h p36_help.h p36_ssd.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_eval.c

p36_help_cpu.o: p36_help.c p36_help.h p36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_help.c

p36_kvstore_cpu.o: p36_kvstore.c p36_kvstore.h p36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_kvstore.c

p36_ssd_cpu.o: p36_ssd.c p36_ssd.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_ssd.c

p36_web_cpu.o: p36_web.c p36_web.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_web.c

p36_test_cpu.o: tests/p36_test.c p36_server.c p36.h p36_cli_args.h rax.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(CPU_CFLAGS) -Wno-unused-function -c -o $@ tests/p36_test.c

p36_sampling_test_core.o: p36.c p36.h p36_gpu.h p36_quant.h p36_ssd.h p36_iq_tables.h p36_iq3s_grid_values.inc p36_streaming_hotlist.inc $(CORE_INCS)
	$(CC) $(CPU_CFLAGS) -Wno-unused-function -DP36_TEST_HOOKS -c -o $@ p36.c

tests/test_sampling.o: tests/test_sampling.c p36.h p36_cli_args.h
	$(CC) $(CPU_CFLAGS) -DP36_TEST_HOOKS -I. -c -o $@ $<

$(SAMPLING_TEST): tests/test_sampling.o p36_sampling_test_core.o p36_ssd_cpu.o p36_prompt_prefix.o p36_image.o
	$(CC) $(LDFLAGS) $(DARWIN_MIN_FLAG) -o $@ $^ $(LDLIBS)

linenoise_cpu.o: linenoise.c linenoise.h
	$(CC) $(CPU_CFLAGS) -c -o $@ linenoise.c

rax_cpu.o: rax.c rax.h rax_malloc.h
	$(CC) $(CPU_CFLAGS) -c -o $@ rax.c

test-cli: all p36_agent_test p36-quality-score $(SAMPLING_TEST) $(TOPK8_TEST) test-quality
	$(MAKE) -C gguf-tools qwen36-quantize
	python3 tests/test_cli_arguments.py

test: all p36_agent_test $(SAMPLING_TEST) $(TOPK8_TEST) test-quality test-cli
	./p36-eval --self-test-extractors
	./p36_agent_test
	python3 tests/test_agent_password.py ./p36_agent_test
	python3 tests/test_agent_terminal.py ./p36_agent_test
	python3 tests/test_agent_verify_watchdog.py ./p36-agent
	python3 tests/test_agent_workspace_jail.py ./p36-agent
	./tests/test_sampling
	./$(TOPK8_TEST)
	./p36_test --quant-primitives --ssd-cache-shrink --qwen-tool-call-format --vector-fixtures --server

test-quick: test

test-unit: test

# CI gate: builds every binary and runs only the tests that need neither a GGUF
# model nor a Vulkan device, so a red run means the tree is broken rather than
# the machine is bare.  This is what .github/workflows/ci.yml runs; the
# model-backed suites in "test" stay on a workstation.
check: all
	./p36_test --quant-primitives
	./p36_test --ssd-cache-shrink
	./p36_test --qwen-tool-call-format
	./p36_test --vector-fixtures
	./p36_test --server
	@echo "check: ok"

test-metal: metal p36_agent_test_metal $(SAMPLING_TEST)
	./p36-eval --self-test-extractors
	./p36_agent_test_metal
	python3 tests/test_agent_password.py ./p36_agent_test_metal
	./tests/test_sampling
	./p36_test --quant-primitives --ssd-cache-shrink --qwen-tool-call-format --vector-fixtures --server
	./p36_test --vulkan-kernels

test-metal-model: metal
	@echo "=== Metal model-dependent tests (requires p36moe.gguf) ==="
	./p36_test --tool-call-quality --qwen-tool-call-quality --thinking-generation --kv-cache-save-restore --session-sync-resume --gpu-cpu-parity --vulkan-fusion-parity --ssd-streaming-parity --mtp-verifier

test-qwen27b-metal-parity: metal
	./p36_test --dense-quant-model-rows --gpu-cpu-parity --model $(QWEN27B_MODEL) --case short

test-qwen27b-vulkan-parity: p36_test
	./p36_test --vulkan-cpu-parity --model $(QWEN27B_MODEL) --case short

test-qwen27b-llama-parity: p36_llama_test
	./p36_llama_test --llama-parity-seq --model $(QWEN27B_MODEL) --case short

test-vulkan: p36_test
	./p36_test --vulkan-kernels

test-streaming: p36_test
	./p36_test --ssd-streaming-parity

test-mtp: p36_test
	./p36_test --mtp-verifier

test-all: test test-vulkan

test-model: p36_test
	@echo "=== Model-dependent tests (requires p36moe.gguf) ==="
	./p36_test --tool-call-quality --qwen-tool-call-quality --thinking-generation --kv-cache-save-restore --session-sync-resume --gpu-cpu-parity --vulkan-fusion-parity --vulkan-queue-parity --mtp-verifier

test-session-batch: p36_test
	./p36_test --vulkan-session-batch
	P36_TEST_BATCH_KV=f16 ./p36_test --vulkan-session-batch
	P36_TEST_BATCH_KV=ssd ./p36_test --vulkan-session-batch

test-server-live: p36-server
	./tests/server_live_smoke.sh

test-server-live-metal: metal
	./tests/server_live_smoke.sh --metal

test-server-live-metal-ssd: metal
	./tests/server_live_smoke.sh --metal --ssd-streaming

test-server-batching: p36-server
	./tests/server_batching.sh

test-server-batching-metal: metal
	./tests/server_batching.sh --metal

test-server-batching-metal-ssd: metal
	./tests/server_batching.sh --metal --ssd-streaming

benchmark-session-batch: p36-server
	./tests/benchmark_server_batching.sh

benchmark-gate: p36-bench
	./tests/release_bench_gate.sh

benchmark-qwen27b: benchmark-qwen27b-vulkan benchmark-qwen27b-llama

benchmark-qwen27b-gate: p36-bench
	./p36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GATE_GEN)
	./p36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GATE_GEN)

benchmark-qwen27b-metal-gate: metal
	./p36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GATE_GEN)
	./p36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GATE_GEN)

benchmark-qwen27b-vulkan: p36-bench
	./p36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GEN)
	./p36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GEN)
	./p36-bench --vulkan -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 4096 --ctx-max 4096 --ctx-alloc 4352 --gen-tokens $(QWEN27B_GEN)

benchmark-qwen27b-metal: metal
	./p36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 128 --ctx-max 128 --ctx-alloc 512 --prefill-chunk 128 --gen-tokens $(QWEN27B_GEN)
	./p36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 2048 --ctx-max 2048 --ctx-alloc 2304 --gen-tokens $(QWEN27B_GEN)
	./p36-bench --metal -m $(QWEN27B_MODEL) --prompt-file $(QWEN27B_PROMPT) --ctx-start 4096 --ctx-max 4096 --ctx-alloc 4352 --gen-tokens $(QWEN27B_GEN)

benchmark-qwen27b-llama:
	$(QWEN27B_LLAMA_BENCH) -m $(QWEN27B_MODEL) -p $(QWEN27B_LLAMA_PROMPT) -n 0 -r 1 -b $(QWEN27B_LLAMA_PROMPT) -ub $(QWEN27B_LLAMA_PROMPT) -fa off --no-warmup

release-build-check:
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS= all
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS=-DP36_VULKAN_REQUIRE_BC250 all
	$(MAKE) -B CFLAGS='$(filter-out -ffast-math,$(CFLAGS)) -Werror' cpu
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' VULKAN_CFLAGS= all

release-build-check-metal:
	$(MAKE) -B CFLAGS='$(CFLAGS) -Werror' metal

test-release:
	$(MAKE) test
	$(MAKE) test-vulkan
	$(MAKE) test-model
	$(MAKE) test-server-live
	$(MAKE) test-reference
	$(MAKE) test-streaming
	$(MAKE) benchmark-gate
	$(MAKE) release-build-check

test-kv-cache: p36_test
	./p36_test --kv-cache-save-restore

test-thinking: p36_test
	./p36_test --thinking-generation

test-qwen-tool: p36_test
	./p36_test --qwen-tool-call-quality --qwen-tool-call-format

test-tool: p36_test
	./p36_test --tool-call-quality --qwen-tool-call-quality

test-reference: p36_reference_test
	./p36_reference_test --vector-fixtures --logprob-vectors

test-reference-local: p36_reference_test
	@test -f tests/test-vectors/local/llama.vec || { echo "missing tests/test-vectors/local/llama.vec; run make test-vectors-local"; exit 2; }
	P36_TEST_VECTOR_FILE=tests/test-vectors/local/llama.vec ./p36_reference_test --logprob-vectors

test-vectors-refresh: .opencode-tmp/llama_qwen_logprobs_capture
	python3 tests/test-vectors/fetch_llama_vectors.py --model $(REFERENCE_MODEL) --threads 8 --capture-bin .opencode-tmp/llama_qwen_logprobs_capture

test-vectors-local: .opencode-tmp/llama_qwen_logprobs_capture
	python3 tests/test-vectors/fetch_llama_vectors.py --out tests/test-vectors/local --model $(REFERENCE_MODEL) --threads 8 --capture-bin .opencode-tmp/llama_qwen_logprobs_capture

reference-openrouter:
	python3 gguf-tools/quality-testing/collect_openrouter.py --model $(OPENROUTER_MODEL) --out $(OPENROUTER_OUT) --count $(OPENROUTER_COUNT) --max-tokens 24 --top-logprobs 20 --provider-order parasail/fp8 --require-parameters --reasoning-effort none

test-llama: p36_llama_test
	./p36_llama_test --llama-parity-seq

test-llama-long: p36_llama_test
	./p36_llama_test --llama-parity-seq-long

test-llama-batch: p36_llama_test
	./p36_llama_test --llama-parity-batch-loose

test-llama-all: p36_llama_test
	./p36_llama_test --llama-parity-seq --llama-parity-seq-long

test-server: p36_test
	./p36_test --server

test-long: p36_test
	./p36_test --long-context

.opencode-tmp/llama_qwen_logprobs_capture: tests/test-vectors/llama_qwen_logprobs_capture.c p36_cli_args.h
	@mkdir -p .opencode-tmp
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) $(LLAMA_INCLUDE) \
		-o $@ tests/test-vectors/llama_qwen_logprobs_capture.c \
		$(LLAMA_LIBS) $(LDLIBS) $(LLAMA_LDLIBS)

p36_llama_test.o: tests/p36_test.c p36_server.c p36.h rax.h p36_gpu.h p36_tool_text.h $(SERVER_INCS)
	$(CC) $(CPU_CFLAGS) $(LLAMA_INCLUDE) -DP36_WITH_LLAMA -Wno-unused-function -c -o $@ tests/p36_test.c

p36_llama_test: p36_llama_test.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o $@ p36_llama_test.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LLAMA_LIBS) $(LDLIBS) $(LLAMA_LDLIBS)

p36_reference_test: p36_test_cpu.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS)
	$(CC) $(CPU_CFLAGS) -o $@ p36_test_cpu.o rax_cpu.o p36_ssd_cpu.o p36_prompt_prefix.o $(CPU_CORE_OBJS) $(LDLIBS)

p36_agent_test.o: p36_loopguard.h p36_remote.h p36_prompt_prefix.h tests/p36_agent_test.c p36_agent.c p36.h p36_cli_args.h p36_help.h p36_kvstore.h p36_ssd.h p36_web.h linenoise.h p36_tool_text.h $(AGENT_INCS)
	$(CC) $(GPU_CFLAGS) -Wno-unused-function -c -o $@ tests/p36_agent_test.c

p36_agent_test: p36_agent_test.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(CORE_OBJS)
	$(CC) $(GPU_CFLAGS) -o $@ p36_agent_test.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(CORE_OBJS) $(GPU_LDLIBS)

p36_agent_test_metal.o: p36_prompt_prefix.h tests/p36_agent_test.c p36_agent.c p36.h p36_cli_args.h p36_help.h p36_kvstore.h p36_ssd.h p36_web.h linenoise.h p36_tool_text.h $(AGENT_INCS)
	$(CC) $(GPU_CFLAGS) -DP36_METAL -Wno-unused-function -c -o $@ tests/p36_agent_test.c

p36_agent_test_metal: p36_agent_test_metal.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(METAL_CORE_OBJS)
	$(CC) $(METAL_LDFLAGS) -o $@ p36_agent_test_metal.o p36_help.o p36_kvstore.o p36_ssd.o p36_prompt_prefix.o p36_web.o p36_remote.o linenoise.o $(METAL_CORE_OBJS) $(METAL_LDLIBS)

clean:
	rm -f p36 p36-server p36-bench p36-agent p36-eval p36_test p36_reference_test p36_llama_test p36_agent_test p36_agent_test_metal $(SAMPLING_TEST) tests/test_sampling.o tests/test_prompt_prefix tests/test_quality_api gguf-tools/quality-testing/score_openrouter gguf-tools/quality-testing/score_openrouter.o *.o $(VULKAN_SHADERS)

p36_prompt_prefix.o: p36_prompt_prefix.c p36_prompt_prefix.h p36.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_prompt_prefix.c

tests/test_prompt_prefix: tests/test_prompt_prefix.c p36_prompt_prefix.o p36_cli_args.h
	$(CC) $(CPU_CFLAGS) -I. -o $@ $< p36_prompt_prefix.o $(LDLIBS)

tests/test_quality_api: tests/test_quality_api.c gguf-tools/quality-testing/score_openrouter.c p36.h p36_cli_args.h
	$(CC) $(CPU_CFLAGS) -ffunction-sections -fdata-sections -I. $(if $(filter Darwin,$(shell uname -s)),-Wl$(comma)-dead_strip,-Wl$(comma)--gc-sections) -o $@ $< $(LDLIBS)

test-quality: tests/test_prompt_prefix tests/test_quality_api
	./tests/test_prompt_prefix
	./tests/test_quality_api
	python3 -m unittest discover -s gguf-tools/quality-testing/tests

p36_eval_cases.o: p36_eval_cases.c p36_eval_cases.h
	$(CC) $(CPU_CFLAGS) -c -o $@ p36_eval_cases.c
