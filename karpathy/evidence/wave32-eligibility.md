# Wave32 eligibility audit — q36 compute shaders

Correctness/safety audit of forcing `requiredSubgroupSize = 32` on the compute pipelines built
from `vulkan/*.comp`. Read-only analysis; no build, no benchmark, no model load, no engine run.

## 0. Scope and method

| item | value |
|---|---|
| corpus | `/home/server/q36-opt-27b/vulkan/*.comp` — **92 files** |
| scanned for | `subgroup*(` calls, `gl_Subgroup*` reads, `barrier()`, `local_size_x/y/z`, `GL_KHR_shader_subgroup_*` declares |
| force list source | `q36_vulkan.c:1556-1582` |
| bucket rule | `CROSS-LANE-HEAVY` = any reduction-class call; `NEEDS-REVIEW` = non-reduction call or `gl_Subgroup*` read; `SAFE-WAVE32` = neither |
| hot flag | substring match on the profile list in the request, with `_r4` added (covers `dense_iq3_xxs_decode_r4`, `dense_iq3_s_mmq_r4`, ...) |

Bucket counts: SAFE-WAVE32 = 57, NEEDS-REVIEW = 5, CROSS-LANE-HEAVY = 30.

Everything below is derived by scanning the `.comp` sources and the `Makefile`; every claim carries
a `file:line`. No performance claim is made anywhere in this document.

## 1. The eligibility rule as written in the engine

`q36_vulkan.c:1561-1568` (comment), quoted verbatim:

```c
/* Shaders that declare local_size_x = 32 and use no cross-lane operation
 * at all.  On wave64 the upper half of every wave is inactive, and since
 * nothing reads a neighbouring lane the wave width cannot change a
 * single result bit -- forcing wave32 only stops wasting half of each
 * wave.  Each entry was checked for zero cross-lane builtin uses in its
 * .comp source; do not add a shader here without that check, because
 * for a shader that does reduce across lanes the two widths are not
 * interchangeable. */
```

`q36_vulkan.c:1569-1574` — the list itself, quoted verbatim:

```c
    static const char *const q36_vk_force_wave32[] = {
        "delta_net_cols.spv",
        "rope_qwen.spv",
        "rope_qwen_mrope.spv",
        "quantize_q8_0.spv",
    };
```

The list is not the whole force set. `q36_vulkan.c:1575-1578` adds a substring rule that covers a
whole family without being listed:

```c
    bool force_wave32 = strstr(k->path, "dense_") && strstr(k->path, "_mmq.spv");
    for (size_t i = 0; !force_wave32 && i < sizeof(q36_vk_force_wave32) / sizeof(*q36_vk_force_wave32); i++) {
        force_wave32 = strstr(k->path, q36_vk_force_wave32[i]) != NULL;
    }
```

and `q36_vulkan.c:1579-1582` applies it, with full-subgroup requirement:

```c
    if (q36_vk.subgroup_size_control && force_wave32) {
        stage.pNext = &subgroup_size;
        stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
```

**Discrepancy worth knowing about.** The request paraphrases the rule as "(a) declares
`local_size_x = 32` **or** has no cross-lane data exchange, and (b) uses no cross-lane operation at
all". The source comment states the conjunctive form: *declare* `local_size_x = 32` **and** use no
cross-lane operation. This audit keeps the source comment as authoritative (§3 marks any candidate
that clears the scan but does not declare `local_size_x = 32`).

A second, sharper reading of the comment's own rationale: "nothing reads a neighbouring lane" is
not sufficient on its own. `gl_SubgroupID` / `gl_SubgroupInvocationID` are lane-*identity* reads
(not cross-lane exchanges) that nonetheless change meaning when the number of subgroups in a
workgroup changes — `local_size_x / width` goes from 2 to 4 for a 128-lane workgroup. §6 documents
this concretely in the `_mmq` family, which the substring rule already forces.

## 2. All 92 shaders

`local_size` = declared workgroup size as `x*y*z` (`n` = `1*1*1` abbreviated `n`). The ops column
lists every hit with its line number: `subgroup*` calls first, then `gl_Subgroup*` reads, then
`barrier()` (uniform-control-flow note, not itself a cross-lane exchange), then declared-but-maybe-
unused `GL_KHR_shader_subgroup_*` extensions. `H` marks a shader on the profile's hot list.

| shader | local_size | cross-lane ops (file line numbers) | bucket |
|---|---|---|---|
| `add.comp` | 256 | **none** | SAFE-WAVE32 |
| `add_rms_norm.comp` **H** | 1024 | *(uniform)* barrier@67,70 | SAFE-WAVE32 |
| `attn_combine.comp` | 256 | **none** | SAFE-WAVE32 |
| `attn_decode_fused.comp` **H** | 256 | subgroupClusteredAdd@204 subgroupClusteredAdd@283 subgroupClusteredAdd@364<br>*(uniform)* barrier@185,206,214,216,244,268,285,293,295,322,336,366,376,378,409<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `attn_decode_split.comp` **H** | 256 | subgroupClusteredAdd@205 subgroupClusteredAdd@276 subgroupClusteredAdd@345<br>*(uniform)* barrier@184,207,215,217,245,278,286,288,315,347,355,357,388<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `attn_post.comp` | 1 | **none** | SAFE-WAVE32 |
| `attn_prefill_qtile.comp` | 256 | subgroupClusteredAdd@250<br>*(uniform)* barrier@156,165,256,269,275,295,317<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `attn_prefill_qtile2.comp` **H** | 128 | subgroupClusteredAdd@203<br>*(uniform)* barrier@118,208,223,230,253,290<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `attn_prefill_qtile2_gqa6.comp` **H** | 128 | subgroupClusteredAdd@205<br>*(uniform)* barrier@120,210,225,232,255,292<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `attn_reduce.comp` | 64 | **none** | SAFE-WAVE32 |
| `attn_scores.comp` | 64 | **none** | SAFE-WAVE32 |
| `conv_silu.comp` | 256 | **none** | SAFE-WAVE32 |
| `copy_rows.comp` | 256 | **none** | SAFE-WAVE32 |
| `delta_gates.comp` | 64 | **none** | SAFE-WAVE32 |
| `delta_net.comp` | 128 | *(uniform)* barrier@94,116 | SAFE-WAVE32 |
| `delta_net_cols.comp` | 128 | subgroupAdd@96 subgroupAdd@105<br>ids: gl_SubgroupID@66 gl_SubgroupInvocationID@68<br>*(uniform)* barrier@86,107<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `delta_net_decode.comp` | 32 | *(uniform)* barrier@75,92 | SAFE-WAVE32 |
| `delta_net_decode_reg.comp` | **not a literal** | subgroupClusteredAdd@68 subgroupClusteredAdd@75<br>declares: subgroup_basic,subgroup_clustered | CROSS-LANE-HEAVY |
| `delta_net_fast.comp` | 128 | *(uniform)* barrier@59,78 | SAFE-WAVE32 |
| `delta_qk.comp` | 64 | *(uniform)* barrier@51,57 | SAFE-WAVE32 |
| `delta_qkv.comp` | 64 | *(uniform)* barrier@44,50 | SAFE-WAVE32 |
| `dense_extra_decode.comp` **H** | 64 | subgroupAdd@241<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_extra_mmq.comp` **H** | 128 | ids: gl_SubgroupID@153 gl_SubgroupInvocationID@154<br>*(uniform)* barrier@175,275,296<br>declares: subgroup_basic | NEEDS-REVIEW |
| `dense_iq1_m.comp` | 32 | subgroupAdd@72<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_iq3_s_decode.comp` **H** | 64 | subgroupAdd@126<br>*(uniform)* barrier@71<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_iq3_s_mmq.comp` | 128 | ids: gl_SubgroupID@78 gl_SubgroupInvocationID@79<br>*(uniform)* barrier@89,98,158,185<br>declares: subgroup_basic | NEEDS-REVIEW |
| `dense_iq3_xxs_decode.comp` **H** | 64 | subgroupAdd@119<br>*(uniform)* barrier@62<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_iq3_xxs_mmq.comp` **H** | 128 | ids: gl_SubgroupID@71 gl_SubgroupInvocationID@72<br>*(uniform)* barrier@82,91,140,161<br>declares: subgroup_basic | NEEDS-REVIEW |
| `dense_iq4_xs.comp` | 32 | subgroupAdd@64<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_iq4_xs_decode.comp` **H** | 64 | subgroupAdd@95<br>*(uniform)* barrier@52<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_iq4_xs_mmq.comp` **H** | 128 | ids: gl_SubgroupID@69 gl_SubgroupInvocationID@70<br>*(uniform)* barrier@78,91,145,166<br>declares: subgroup_basic | NEEDS-REVIEW |
| `dense_kquant_decode.comp` **H** | 64 | subgroupAdd@327<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `dense_kquant_mmq.comp` **H** | 128 | ids: gl_SubgroupID@205 gl_SubgroupInvocationID@206<br>*(uniform)* barrier@250,276,293<br>declares: subgroup_basic | NEEDS-REVIEW |
| `directional_steering.comp` | 256 | *(uniform)* barrier@33,36 | SAFE-WAVE32 |
| `ffn_tail.comp` | 256 | **none** | SAFE-WAVE32 |
| `kv_store.comp` | 256 | **none** | SAFE-WAVE32 |
| `kv_store_quant.comp` **H** | 256 | *(uniform)* barrier@64,69,85,114,158 | SAFE-WAVE32 |
| `matmul_f16.comp` | 64 | **none** | SAFE-WAVE32 |
| `matmul_f32.comp` | 64 | *(uniform)* barrier@44,47 | SAFE-WAVE32 |
| `matmul_f32_fast.comp` | **not a literal** | subgroupAdd@68 subgroupElect@70<br>ids: gl_NumSubgroups@20 gl_NumSubgroups@69 gl_SubgroupID@70 gl_NumSubgroups@73<br>*(uniform)* barrier@71<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `matmul_kquant.comp` | 64 | *(uniform)* barrier@202,346,353 | SAFE-WAVE32 |
| `matmul_kquant_mmq.comp` | 64 | *(uniform)* barrier@189,255,262 | SAFE-WAVE32 |
| `matmul_q5k_mmq.comp` | 64 | *(uniform)* barrier@131,169,176 | SAFE-WAVE32 |
| `matmul_q5k_mmq_fast.comp` | 64 | subgroupClusteredAdd@161 subgroupClusteredAdd@161<br>*(uniform)* barrier@116,154<br>declares: subgroup_basic,subgroup_arithmetic,subgroup_clustered | CROSS-LANE-HEAVY |
| `matmul_q6k_mmq.comp` | 64 | *(uniform)* barrier@127,159,163 | SAFE-WAVE32 |
| `matmul_q6k_mmq_fast.comp` | 64 | subgroupClusteredAdd@152<br>*(uniform)* barrier@113,145<br>declares: subgroup_basic,subgroup_arithmetic,subgroup_clustered | CROSS-LANE-HEAVY |
| `matmul_q8_0.comp` | 64 | subgroupClusteredAdd@111<br>*(uniform)* barrier@87,107<br>declares: subgroup_basic,subgroup_arithmetic,subgroup_clustered | CROSS-LANE-HEAVY |
| `matmul_q8_0_decode.comp` **H** | 128 | *(uniform)* barrier@94 | SAFE-WAVE32 |
| `matmul_q8_0_decode_b64.comp` **H** | 128 | *(uniform)* barrier@98 | SAFE-WAVE32 |
| `matmul_q8_0_decode_q36.comp` **H** | 128 | *(uniform)* barrier@77 | SAFE-WAVE32 |
| `matmul_q8_0_f32b.comp` | 128 | subgroupAdd@66 subgroupElect@68<br>ids: gl_NumSubgroups@67 gl_SubgroupID@68 gl_NumSubgroups@72<br>*(uniform)* barrier@69<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `matmul_q8_0_f32b_nx.comp` | 64 | subgroupAdd@70 subgroupElect@72<br>ids: gl_NumSubgroups@71 gl_SubgroupID@73 gl_NumSubgroups@79<br>*(uniform)* barrier@75<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `matmul_q8_0_f32b_pair.comp` | 128 | subgroupAdd@63 subgroupAdd@64 subgroupElect@66<br>ids: gl_NumSubgroups@65 gl_SubgroupID@67 gl_SubgroupID@68 gl_NumSubgroups@74<br>*(uniform)* barrier@70<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `matmul_q8_0_mm.comp` | 256 | *(uniform)* barrier@88,116 | SAFE-WAVE32 |
| `matmul_q8_0_mm_f16.comp` | 256 | *(uniform)* barrier@136,163 | SAFE-WAVE32 |
| `matmul_q8_0_mm_f16_out32.comp` | 256 | *(uniform)* barrier@85,99 | SAFE-WAVE32 |
| `matmul_q8_0_q36.comp` | 64 | *(uniform)* barrier@77,97,103 | SAFE-WAVE32 |
| `moe_down_gemm.comp` | 256 | *(uniform)* barrier@85,128,148 | SAFE-WAVE32 |
| `moe_down_q2k_f32b.comp` | 32 | subgroupAdd@141 subgroupElect@143<br>ids: gl_NumSubgroups@142 gl_SubgroupID@144 gl_NumSubgroups@148<br>*(uniform)* barrier@146<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_down_q2k_sum_decode.comp` | 32 | subgroupAdd@121<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_down_q4k_sum_decode.comp` | 32 | subgroupAdd@98 subgroupAdd@99<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_gate_up.comp` | 64 | *(uniform)* barrier@116,169,176 | SAFE-WAVE32 |
| `moe_gate_up_decode.comp` **H** | 64 | subgroupAdd@96 subgroupAdd@97 subgroupElect@99<br>ids: gl_NumSubgroups@98 gl_SubgroupID@100 gl_SubgroupID@101<br>*(uniform)* barrier@47,103<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_gate_up_f32b.comp` | 64 | subgroupAdd@135 subgroupAdd@136 subgroupElect@139<br>ids: gl_NumSubgroups@138 gl_SubgroupID@141 gl_SubgroupID@142 gl_NumSubgroups@147<br>*(uniform)* barrier@80,145<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_gate_up_gemm.comp` | 256 | *(uniform)* barrier@105,164,197 | SAFE-WAVE32 |
| `moe_gate_up_q4k_f32b.comp` | 64 | subgroupAdd@110 subgroupAdd@111 subgroupAdd@112 subgroupAdd@113 subgroupElect@115<br>ids: gl_NumSubgroups@114 gl_SubgroupID@116 gl_SubgroupID@117 gl_SubgroupID@118 gl_SubgroupID@119 gl_NumSubgroups@123<br>*(uniform)* barrier@121<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `moe_matvec.comp` **H** | 64 | *(uniform)* barrier@266,512,517 | SAFE-WAVE32 |
| `moe_matvec_fast.comp` **H** | 64 | *(uniform)* barrier@202,425,430<br>declares: subgroup_basic,subgroup_arithmetic,subgroup_clustered | SAFE-WAVE32 |
| `moe_reduce.comp` | 256 | **none** | SAFE-WAVE32 |
| `moe_tiles.comp` | 256 | *(uniform)* barrier@51,56,69 | SAFE-WAVE32 |
| `predequant_b16.comp` | 256 | **none** | SAFE-WAVE32 |
| `quantize_q8_0.comp` **H** | 32 | *(uniform)* barrier@47,58 | SAFE-WAVE32 |
| `quantize_q8_k.comp` | 64 | *(uniform)* barrier@74,86,112,117 | SAFE-WAVE32 |
| `recur_conv_silu_decode.comp` | 256 | **none** | SAFE-WAVE32 |
| `recur_norm_gate.comp` | 256 | *(uniform)* barrier@27,30 | SAFE-WAVE32 |
| `recur_norm_gate_q8_k.comp` | 256 | *(uniform)* barrier@44,49,59,72,80,98 | SAFE-WAVE32 |
| `recur_window.comp` | 256 | **none** | SAFE-WAVE32 |
| `rms_norm.comp` **H** | 256 | *(uniform)* barrier@34,37 | SAFE-WAVE32 |
| `rms_norm_rope_kv_qwen.comp` **H** | 256 | *(uniform)* barrier@44,47,51,77 | SAFE-WAVE32 |
| `rms_norm_rope_kv_qwen_quant.comp` **H** | 256 | subgroupClusteredMax@93 subgroupShuffleDown@130 subgroupShuffleDown@131 subgroupShuffle@138 subgroupClusteredAdd@149 subgroupClusteredAdd@150<br>ids: gl_SubgroupInvocationID@138<br>*(uniform)* barrier@61,64,68,85,101,119,143,162<br>declares: subgroup_basic,subgroup_clustered,subgroup_shuffle | CROSS-LANE-HEAVY |
| `rms_norm_rope_qwen.comp` **H** | 256 | *(uniform)* barrier@38,41,46 | SAFE-WAVE32 |
| `rope_qwen.comp` | 32 | **none** | SAFE-WAVE32 |
| `rope_qwen_mrope.comp` | 32 | **none** | SAFE-WAVE32 |
| `router_topk.comp` | 256 | *(uniform)* barrier@43,48,58,64,66 | SAFE-WAVE32 |
| `shared_down_tail_decode.comp` | 64 | subgroupAdd@66 subgroupElect@68<br>ids: gl_NumSubgroups@67 gl_SubgroupID@68 gl_NumSubgroups@72<br>*(uniform)* barrier@69<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `shared_gate_up_decode.comp` | 64 | subgroupAdd@67 subgroupAdd@68 subgroupElect@69<br>ids: gl_SubgroupID@70 gl_SubgroupID@71 gl_NumSubgroups@75<br>*(uniform)* barrier@73<br>declares: subgroup_basic,subgroup_arithmetic | CROSS-LANE-HEAVY |
| `swiglu.comp` | 256 | **none** | SAFE-WAVE32 |
| `swiglu_q8_k.comp` | 64 | *(uniform)* barrier@71,83,103,108 | SAFE-WAVE32 |
| `top2.comp` | 256 | *(uniform)* barrier@70,81 | SAFE-WAVE32 |
| `topk8.comp` | 256 | *(uniform)* barrier@100,110 | SAFE-WAVE32 |
| `vision_attention.comp` | 128 | *(uniform)* barrier@26,29,41 | SAFE-WAVE32 |
| `vision_matmul_f16.comp` | 16*16*1 | *(uniform)* barrier@38,42 | SAFE-WAVE32 |

## 3. RECOMMENDED TO ADD TO FORCE LIST

Hot-path shaders whose scan is clean — zero `subgroup*` calls and zero `gl_Subgroup*` reads, which is
the condition the comment's rationale actually rests on. Split by whether the declaration also
matches the comment's *letter* (`local_size_x = 32`).

### 3a. Add — hot path, clean scan, `local_size_x = 32`

* `quantize_q8_0.comp` — `local_size_x = 32` (`quantize_q8_0.comp:3`), no `subgroup*` call and no `gl_Subgroup*` read anywhere in the file: no lane-derived index and no cross-lane value can change with the wave width.

### 3b. Clean scan but `local_size_x != 32` — outside the comment's letter, decide explicitly

* `add_rms_norm.comp` — 1024*1*1 (`add_rms_norm.comp:26`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `kv_store_quant.comp` — 256*1*1 (`kv_store_quant.comp:7`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `matmul_q8_0_decode.comp` — 128*1*1 (`matmul_q8_0_decode.comp:6`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `matmul_q8_0_decode_b64.comp` — 128*1*1 (`matmul_q8_0_decode_b64.comp:8`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `matmul_q8_0_decode_q36.comp` — 128*1*1 (`matmul_q8_0_decode_q36.comp:3`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `moe_matvec.comp` — 64*1*1 (`moe_matvec.comp:36`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `moe_matvec_fast.comp` — 64*1*1 (`moe_matvec_fast.comp:12`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `rms_norm.comp` — 256*1*1 (`rms_norm.comp:2`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `rms_norm_rope_kv_qwen.comp` — 256*1*1 (`rms_norm_rope_kv_qwen.comp:3`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.
* `rms_norm_rope_qwen.comp` — 256*1*1 (`rms_norm_rope_qwen.comp:2`), no `subgroup*` call and no `gl_Subgroup*` read; bits cannot move, but the comment says "declare local_size_x = 32", so adding it is a rule-text change, not just an entry.

### 3c. Do NOT add (hot path, but not clean)

* `attn_decode_fused.comp` (CROSS-LANE-HEAVY) — calls subgroupClusteredAdd.
* `attn_decode_split.comp` (CROSS-LANE-HEAVY) — calls subgroupClusteredAdd.
* `attn_prefill_qtile2.comp` (CROSS-LANE-HEAVY) — calls subgroupClusteredAdd.
* `attn_prefill_qtile2_gqa6.comp` (CROSS-LANE-HEAVY) — calls subgroupClusteredAdd.
* `dense_extra_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd.
* `dense_extra_mmq.comp` (NEEDS-REVIEW) — reads gl_SubgroupID,gl_SubgroupInvocationID.
* `dense_iq3_s_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd.
* `dense_iq3_xxs_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd.
* `dense_iq3_xxs_mmq.comp` (NEEDS-REVIEW) — reads gl_SubgroupID,gl_SubgroupInvocationID.
* `dense_iq4_xs_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd.
* `dense_iq4_xs_mmq.comp` (NEEDS-REVIEW) — reads gl_SubgroupID,gl_SubgroupInvocationID.
* `dense_kquant_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd.
* `dense_kquant_mmq.comp` (NEEDS-REVIEW) — reads gl_SubgroupID,gl_SubgroupInvocationID.
* `moe_gate_up_decode.comp` (CROSS-LANE-HEAVY) — calls subgroupAdd,subgroupElect; reads gl_NumSubgroups,gl_SubgroupID.
* `rms_norm_rope_kv_qwen_quant.comp` (CROSS-LANE-HEAVY) — calls subgroupClusteredAdd,subgroupClusteredMax,subgroupShuffle,subgroupShuffleDown; reads gl_SubgroupInvocationID.

## 4. `delta_net_cols.spv` — existing entry vs the stated rule

`delta_net_cols.spv` **does violate the rule as written**, on both clauses, and the entry is
nonetheless required — because the shader is already wave32-only by construction. Evidence from
`vulkan/delta_net_cols.comp`:

| line | code | why it matters |
|---|---|---|
| 13 | `layout(local_size_x = 128) in;` | clause (a): the comment asks for `local_size_x = 32`. |
| 56-58 | `const uint N = 128u; const uint LANES = 32u; const uint ROWS_PER_LANE = N / LANES;` | the lane count is a hard 32: row index `i = r * LANES + lane`. |
| 66 | `uint j = gl_WorkGroupID.y * COLS + gl_SubgroupID;` | output column `j` comes from the **subgroup ordinal**. `COLS = 4` (line 59), so all 4 columns need `128/width = 4` subgroups, i.e. width 32. |
| 68 | `uint lane = gl_SubgroupInvocationID;` | lane identity (not an exchange) that feeds the row index. |
| 96 | `float sk = subgroupAdd(sk_part);` | clause (b): a cross-lane **reduction**, explicitly the case the comment forbids. |
| 105 | `float outj = subgroupAdd(out_part);` | second reduction; result written by `if (lane == 0u) y[vec + j] = outj * norm;` (line 106). |
| 86, 107 | `barrier();` | uniform control flow (workgroup-uniform guard at line 69); not an exchange. |

Why the violation is unavoidable rather than an oversight:

* `j` is derived from `gl_SubgroupID` (line 66) and `COLS = 4` (line 59). With 128 lanes, `width = 32
  gives 4 subgroups (columns 0-3 covered); `width = 64` gives 2 subgroups and columns 2-3 are never
  computed by any workgroup.
* the reduction at line 96/105 sums `shard[r]` over the 32 lanes built at lines 75-78/91-95 by
  `i = r * LANES + lane` with `LANES = 32` (line 57). At `width = 64` lanes 32-63 of each wave
  would contribute `i` values outside the intended row range, so the sum is not the sum the shader
  is written to compute.

**Conclusion:** the entry breaks the comment's stated criteria (b) and (a), yet must stay: removing
it or widening it would change results (and, at width 64, the set of columns produced). This is a
counter-example to the comment as *written* — the operative requirement is "width 32 by construction",
not "no cross-lane operation". The comment should be amended to say so, or the entry annotated.

## 5. Makefile: one `.comp`, several `.spv` (`-D` variants)

A wave32 decision is per **`.spv`** (`strstr` on `k->path`, `q36_vulkan.c:1575-1578`), while the code
audited here is per **`.comp`**. The Makefile builds multiple `.spv` from one `.comp`, differing only
by `-D` flags, so sibling variants of the same source can end up on opposite sides of the force rule.

| `.comp` | `.spv` variants built from it | `-D` flags (from the recipe) | forced? |
|---|---|---|---|
| `dense_iq3_xxs_decode.comp` | `dense_iq3_xxs_decode.spv` | *(none)* | no |
| ↳ `dense_iq3_xxs_decode.comp` | `dense_iq3_xxs_decode_r4.spv` | `-DQ36_BOUNDS=0 -DQ36_ROWS=4` | no |
| `dense_iq3_s_decode.comp` | `dense_iq3_s_decode.spv` | *(none)* | no |
| ↳ `dense_iq3_s_decode.comp` | `dense_iq3_s_decode_full.spv` | `-DQ36_BOUNDS=0` | no |
| ↳ `dense_iq3_s_decode.comp` | `dense_iq3_s_decode_r4.spv` | `-DQ36_BOUNDS=0 -DQ36_ROWS=4` | no |
| ↳ `dense_iq3_s_decode.comp` | `dense_iq3_s_decode_r1.spv` | `-DQ36_BOUNDS=0 -DQ36_ROWS=1` | no |
| `dense_iq3_s_mmq.comp` | `dense_iq3_s_mmq.spv` | *(none)* | **yes** |
| ↳ `dense_iq3_s_mmq.comp` | `dense_iq3_s_bm64_mmq.spv` | `-DQ36_BM=64` | **yes** |
| ↳ `dense_iq3_s_mmq.comp` | `dense_iq3_s_mmq_r4.spv` | `-DQ36_BK=64 -DQ36_BM=4` | no |
| `dense_kquant_decode.comp` | `dense_kquant_decode.spv` | *(none)* | no |
| ↳ `dense_kquant_decode.comp` | `dense_q4k_decode.spv` | `-DQ36_BOUNDS=0 -DQ36_Q4K_ONLY=1` | no |
| ↳ `dense_kquant_decode.comp` | `dense_q5k_decode.spv` | `-DQ36_BOUNDS=0 -DQ36_Q5K_ONLY=1` | no |
| `delta_net_cols.comp` | `delta_net_cols.spv` | *(none)* | **yes** |
| ↳ `delta_net_cols.comp` | `delta_net_cols_f16.spv` | `-DQ36_STATE_F16=1` | no |
| `delta_net_decode_reg.comp` | `delta_net_decode_reg.spv` | `-DQ36_COLS=32` | no |
| ↳ `delta_net_decode_reg.comp` | `delta_net_decode_reg_f16.spv` | `-DQ36_COLS=32 -DQ36_STATE_F16=1` | no |
| `matmul_f32_fast.comp` | `matmul_f32_fast.spv` | *(none)* | no |
| ↳ `matmul_f32_fast.comp` | `matmul_f32_fast_w256.spv` | `-DQ36_MFF_LOCAL=256` | no |

**Mixed-state variants (same `.comp`, different force decision):**

* `dense_iq3_s_mmq.comp` — forced: `dense_iq3_s_mmq.spv`, `dense_iq3_s_bm64_mmq.spv` | *not* forced: `dense_iq3_s_mmq_r4.spv`
* `delta_net_cols.comp` — forced: `delta_net_cols.spv` | *not* forced: `delta_net_cols_f16.spv`

Each line above is a variant pair that shares every line of shader code except the `-D`-selected
paths, yet takes a different subgroup width at pipeline creation. Where the source's lane math
depends on the subgroup width (see §6), that is a correctness bug regardless of which variant is
wrong; where it does not depend on it, the pair is merely inconsistent with no numeric effect.

## 6. The substring rule at `q36_vulkan.c:1575` — what it already forces, and the gap

`strstr(k->path, "dense_") && strstr(k->path, "_mmq.spv")` forces every `dense_*_mmq.spv`. The
outputs that match, from the `Makefile`:

| `.spv` | `.comp` | forced by |
|---|---|---|
| `dense_iq3_xxs_mmq.spv` | `dense_iq3_xxs_mmq.comp` | substring rule |
| `dense_iq3_s_mmq.spv` | `dense_iq3_s_mmq.comp` | substring rule |
| `dense_iq3_s_bm64_mmq.spv` | `dense_iq3_s_mmq.comp` | substring rule |
| `dense_iq4_xs_mmq.spv` | `dense_iq4_xs_mmq.comp` | substring rule |
| `dense_extra_mmq.spv` | `dense_extra_mmq.comp` | substring rule |
| `dense_kquant_mmq.spv` | `dense_kquant_mmq.comp` | substring rule |
| `delta_net_cols.spv` | `delta_net_cols.comp` | list entry |

### 6.1 The `_mmq` tile is partitioned by `gl_SubgroupID` — subgroup **count**, not just lane ops

These shaders have zero `subgroup*` calls, so a scan for cross-lane *operations* clears them (§2:
bucket `NEEDS-REVIEW`, not `SAFE-WAVE32`, precisely because of the `gl_SubgroupID` reads). That
clearance would be wrong. In `vulkan/dense_iq3_s_mmq.comp`:

| line | code |
|---|---|
| 32 | `const uint BN = 128u;` |
| 78-79 | `uint wave = gl_SubgroupID;` / `uint lane = gl_SubgroupInvocationID;` |
| 81-83 | `uint warp_tok = wave;` … `uint thread_tok = lane >> 2u;` |
| 190-191 | `uint tok = tok0 + warp_tok * 32u + (tt >> 1u) * 16u + thread_tok * 2u + (tt & 1u);` |
| 175-176 | `bv[st * 2u + t] = buf_b[(warp_tok * 32u + st * 16u + thread_tok * 2u + t) * STRIDE + k];` |

The token index a lane stores is `warp_tok * 32 + (tt>>1)*16 + thread_tok*2 + (tt&1)`, with
`warp_tok = gl_SubgroupID` and `thread_tok = lane >> 2`. With a 128-lane workgroup:

* `width = 32` → 4 subgroups → `warp_tok ∈ 0..3`, `lane ∈ 0..31` → `thread_tok ∈ 0..7` → reachable
  token offsets `0 .. 3*32+16+14+1 = 127`, i.e. the whole `BN = 128` tile.
* `width = 64` → 2 subgroups → `warp_tok ∈ 0..1`, `lane ∈ 0..63` → `thread_tok ∈ 0..15` → reachable
  offsets `0 .. 1*32+16+30+1 = 79`: tokens 80-127 of each 128-token tile are written by nobody.

So for this source the wave32 setting is not an optimisation at all — it is what makes the token
tile complete. The same 32-stride token formula from `gl_SubgroupID` is present in the other
`_mmq` sources; §6.2 lists what the scan found in each so the same arithmetic can be checked per
file rather than assumed.

### 6.2 The `_mmq` family, one row per source

| `.comp` | `local_size_x` | `BN` / tile line | `gl_Subgroup*` reads | token-index lines | bucket |
|---|---|---|---|---|---|
| `dense_extra_mmq.comp` | 128 | `28`: `const uint BM = 32u;`; `29`: `const uint BN = 128u;` | `gl_SubgroupID`@153, `gl_SubgroupInvocationID`@154 | `286`: `bv[st * 2u + t] = buf_b[(wave * 32u + st * 16u +`; `301`: `uint tok = tok0 + wave * 32u + (t >> 1u) * 16u +` | NEEDS-REVIEW |
| `dense_iq3_s_mmq.comp` | 128 | `31`: `const uint BM = Q36_BM;`; `32`: `const uint BN = 128u;` | `gl_SubgroupID`@78, `gl_SubgroupInvocationID`@79 | `175`: `bv[st * 2u + t] = buf_b[(warp_tok * 32u + st * 16u +`; `190`: `uint tok = tok0 + warp_tok * 32u + (tt >> 1u) * 16u +` | NEEDS-REVIEW |
| `dense_iq3_xxs_mmq.comp` | 128 | `23`: `const uint BM = 32u;`; `24`: `const uint BN = 128u;` | `gl_SubgroupID`@71, `gl_SubgroupInvocationID`@72 | `151`: `bv[st * 2u + t] = buf_b[(warp_tok * 32u + st * 16u +`; `166`: `uint tok = tok0 + warp_tok * 32u + (tt >> 1u) * 16u +` | NEEDS-REVIEW |
| `dense_iq4_xs_mmq.comp` | 128 | `21`: `const uint BM = 32u;`; `22`: `const uint BN = 128u;` | `gl_SubgroupID`@69, `gl_SubgroupInvocationID`@70 | `156`: `bv[st * 2u + t] = buf_b[(warp_tok * 32u + st * 16u +`; `171`: `uint tok = tok0 + warp_tok * 32u + (tt >> 1u) * 16u +` | NEEDS-REVIEW |
| `dense_kquant_mmq.comp` | 128 | `29`: `const uint BM = 32u;`; `30`: `const uint BN = 128u;` | `gl_SubgroupID`@205, `gl_SubgroupInvocationID`@206 | `210`: `uint out_tok0 = tok0 + wave * 32u + lane_tok * 8u;`; `284`: `bv[t] = buf_b[(wave * 32u + lane_tok * 8u + t) * STRIDE + k]` | NEEDS-REVIEW |
| `matmul_kquant_mmq.comp` | 64 | - | none | *(no 32-stride token formula found)* | SAFE-WAVE32 |
| `matmul_q5k_mmq.comp` | 64 | - | none | *(no 32-stride token formula found)* | SAFE-WAVE32 |
| `matmul_q5k_mmq_fast.comp` | 64 | - | none | *(no 32-stride token formula found)* | CROSS-LANE-HEAVY |
| `matmul_q6k_mmq.comp` | 64 | - | none | *(no 32-stride token formula found)* | SAFE-WAVE32 |
| `matmul_q6k_mmq_fast.comp` | 64 | - | none | *(no 32-stride token formula found)* | CROSS-LANE-HEAVY |

### 6.3 Gap: `dense_iq3_s_mmq_r4.spv` escapes the substring rule

`dense_iq3_s_mmq_r4.spv` does not contain `_mmq.spv` (it ends `_mmq_r4.spv`), so neither the
substring rule (`q36_vulkan.c:1575`) nor the list (`1569-1574`) covers it — while `dense_iq3_s_mmq.spv`
and `dense_iq3_s_bm64_mmq.spv`, built from the *same* `dense_iq3_s_mmq.comp`, are forced. The
subgroup-counted token formula at `dense_iq3_s_mmq.comp:190-191` is not guarded by any `-D` in the
`_r4` recipe, so the variant inherits the same 4-subgroup requirement. It is selected at runtime
(`q36_vulkan.c:8165`, `dense_iq3_s_mmq_r4` for `out_dim == 48 && n_tok <= 128`).

## 7. `barrier()` users (uniform control flow — noted separately)

`barrier()` is not a cross-lane data exchange and is not used as a disqualifier above; it matters
only if control flow reaching it is non-uniform, where behaviour is undefined at any width. Files
using it, with line numbers:

* `add_rms_norm.comp` — barrier@67, 70
* `attn_decode_fused.comp` — barrier@185, 206, 214, 216, 244, 268, 285, 293, 295, 322, 336, 366, 376, 378, 409
* `attn_decode_split.comp` — barrier@184, 207, 215, 217, 245, 278, 286, 288, 315, 347, 355, 357, 388
* `attn_prefill_qtile.comp` — barrier@156, 165, 256, 269, 275, 295, 317
* `attn_prefill_qtile2.comp` — barrier@118, 208, 223, 230, 253, 290
* `attn_prefill_qtile2_gqa6.comp` — barrier@120, 210, 225, 232, 255, 292
* `delta_net.comp` — barrier@94, 116
* `delta_net_cols.comp` — barrier@86, 107
* `delta_net_decode.comp` — barrier@75, 92
* `delta_net_fast.comp` — barrier@59, 78
* `delta_qk.comp` — barrier@51, 57
* `delta_qkv.comp` — barrier@44, 50
* `dense_extra_mmq.comp` — barrier@175, 275, 296
* `dense_iq3_s_decode.comp` — barrier@71
* `dense_iq3_s_mmq.comp` — barrier@89, 98, 158, 185
* `dense_iq3_xxs_decode.comp` — barrier@62
* `dense_iq3_xxs_mmq.comp` — barrier@82, 91, 140, 161
* `dense_iq4_xs_decode.comp` — barrier@52
* `dense_iq4_xs_mmq.comp` — barrier@78, 91, 145, 166
* `dense_kquant_mmq.comp` — barrier@250, 276, 293
* `directional_steering.comp` — barrier@33, 36
* `kv_store_quant.comp` — barrier@64, 69, 85, 114, 158
* `matmul_f32.comp` — barrier@44, 47
* `matmul_f32_fast.comp` — barrier@71
* `matmul_kquant.comp` — barrier@202, 346, 353
* `matmul_kquant_mmq.comp` — barrier@189, 255, 262
* `matmul_q5k_mmq.comp` — barrier@131, 169, 176
* `matmul_q5k_mmq_fast.comp` — barrier@116, 154
* `matmul_q6k_mmq.comp` — barrier@127, 159, 163
* `matmul_q6k_mmq_fast.comp` — barrier@113, 145
* `matmul_q8_0.comp` — barrier@87, 107
* `matmul_q8_0_decode.comp` — barrier@94
* `matmul_q8_0_decode_b64.comp` — barrier@98
* `matmul_q8_0_decode_q36.comp` — barrier@77
* `matmul_q8_0_f32b.comp` — barrier@69
* `matmul_q8_0_f32b_nx.comp` — barrier@75
* `matmul_q8_0_f32b_pair.comp` — barrier@70
* `matmul_q8_0_mm.comp` — barrier@88, 116
* `matmul_q8_0_mm_f16.comp` — barrier@136, 163
* `matmul_q8_0_mm_f16_out32.comp` — barrier@85, 99
* `matmul_q8_0_q36.comp` — barrier@77, 97, 103
* `moe_down_gemm.comp` — barrier@85, 128, 148
* `moe_down_q2k_f32b.comp` — barrier@146
* `moe_gate_up.comp` — barrier@116, 169, 176
* `moe_gate_up_decode.comp` — barrier@47, 103
* `moe_gate_up_f32b.comp` — barrier@80, 145
* `moe_gate_up_gemm.comp` — barrier@105, 164, 197
* `moe_gate_up_q4k_f32b.comp` — barrier@121
* `moe_matvec.comp` — barrier@266, 512, 517
* `moe_matvec_fast.comp` — barrier@202, 425, 430
* `moe_tiles.comp` — barrier@51, 56, 69
* `quantize_q8_0.comp` — barrier@47, 58
* `quantize_q8_k.comp` — barrier@74, 86, 112, 117
* `recur_norm_gate.comp` — barrier@27, 30
* `recur_norm_gate_q8_k.comp` — barrier@44, 49, 59, 72, 80, 98
* `rms_norm.comp` — barrier@34, 37
* `rms_norm_rope_kv_qwen.comp` — barrier@44, 47, 51, 77
* `rms_norm_rope_kv_qwen_quant.comp` — barrier@61, 64, 68, 85, 101, 119, 143, 162
* `rms_norm_rope_qwen.comp` — barrier@38, 41, 46
* `router_topk.comp` — barrier@43, 48, 58, 64, 66
* `shared_down_tail_decode.comp` — barrier@69
* `shared_gate_up_decode.comp` — barrier@73
* `swiglu_q8_k.comp` — barrier@71, 83, 103, 108
* `top2.comp` — barrier@70, 81
* `topk8.comp` — barrier@100, 110
* `vision_attention.comp` — barrier@26, 29, 41
* `vision_matmul_f16.comp` — barrier@38, 42

## 8. Summary counts

| bucket | all | hot |
|---|---|---|
| SAFE-WAVE32 | 57 | 11 |
| NEEDS-REVIEW | 5 | 4 |
| CROSS-LANE-HEAVY | 30 | 11 |
| **total** | **92** | **26** |

Clean (SAFE-WAVE32) shaders, all 57 of them, so the set can be compared against the force list at a
glance:

```
add.comp                                 local_size_x=256      
add_rms_norm.comp                        local_size_x=1024     HOT
attn_combine.comp                        local_size_x=256      
attn_post.comp                           local_size_x=1        
attn_reduce.comp                         local_size_x=64       
attn_scores.comp                         local_size_x=64       
conv_silu.comp                           local_size_x=256      
copy_rows.comp                           local_size_x=256      
delta_gates.comp                         local_size_x=64       
delta_net.comp                           local_size_x=128      
delta_net_decode.comp                    local_size_x=32       
delta_net_fast.comp                      local_size_x=128      
delta_qk.comp                            local_size_x=64       
delta_qkv.comp                           local_size_x=64       
directional_steering.comp                local_size_x=256      
ffn_tail.comp                            local_size_x=256      
kv_store.comp                            local_size_x=256      
kv_store_quant.comp                      local_size_x=256      HOT
matmul_f16.comp                          local_size_x=64       
matmul_f32.comp                          local_size_x=64       
matmul_kquant.comp                       local_size_x=64       
matmul_kquant_mmq.comp                   local_size_x=64       
matmul_q5k_mmq.comp                      local_size_x=64       
matmul_q6k_mmq.comp                      local_size_x=64       
matmul_q8_0_decode.comp                  local_size_x=128      HOT
matmul_q8_0_decode_b64.comp              local_size_x=128      HOT
matmul_q8_0_decode_q36.comp              local_size_x=128      HOT
matmul_q8_0_mm.comp                      local_size_x=256      
matmul_q8_0_mm_f16.comp                  local_size_x=256      
matmul_q8_0_mm_f16_out32.comp            local_size_x=256      
matmul_q8_0_q36.comp                     local_size_x=64       
moe_down_gemm.comp                       local_size_x=256      
moe_gate_up.comp                         local_size_x=64       
moe_gate_up_gemm.comp                    local_size_x=256      
moe_matvec.comp                          local_size_x=64       HOT
moe_matvec_fast.comp                     local_size_x=64       HOT
moe_reduce.comp                          local_size_x=256      
moe_tiles.comp                           local_size_x=256      
predequant_b16.comp                      local_size_x=256      
quantize_q8_0.comp                       local_size_x=32       HOT
quantize_q8_k.comp                       local_size_x=64       
recur_conv_silu_decode.comp              local_size_x=256      
recur_norm_gate.comp                     local_size_x=256      
recur_norm_gate_q8_k.comp                local_size_x=256      
recur_window.comp                        local_size_x=256      
rms_norm.comp                            local_size_x=256      HOT
rms_norm_rope_kv_qwen.comp               local_size_x=256      HOT
rms_norm_rope_qwen.comp                  local_size_x=256      HOT
rope_qwen.comp                           local_size_x=32       
rope_qwen_mrope.comp                     local_size_x=32       
router_topk.comp                         local_size_x=256      
swiglu.comp                              local_size_x=256      
swiglu_q8_k.comp                         local_size_x=64       
top2.comp                                local_size_x=256      
topk8.comp                               local_size_x=256      
vision_attention.comp                    local_size_x=128      
vision_matmul_f16.comp                   local_size_x=(16, 16, 1, 3) 
```

Generated 2026-09-17 by read-only scan of `/home/server/q36-opt-27b/vulkan` and `/home/server/q36-opt-27b/Makefile`.
