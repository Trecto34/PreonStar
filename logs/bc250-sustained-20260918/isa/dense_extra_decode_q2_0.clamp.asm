===== executable 0 'compute' / IR 'Assembly' (Final Assembly) =====
BB0:
	v_and_b32_e32 v1, 12, v0                                    ; 3602008c
	v_and_b32_e32 v2, 48, v0                                    ; 360400b0
	v_add_nc_u32_e32 v3, 64, v0                                 ; 4a0600c0
	s_movk_i32 s0, 0xc0                                         ; b00000c0
	s_mov_b32 s1, 0x80808080                                    ; be8103ff 80808080
	s_mov_b32 s8, 0xfefefeff                                    ; be8803ff fefefeff
	v_lshlrev_b32_e32 v1, 6, v1                                 ; 34020286
	v_lshlrev_b32_e32 v6, 2, v0                                 ; 340c0082
	v_and_b32_e32 v4, 12, v3                                    ; 3608068c
	v_and_b32_e32 v5, 48, v3                                    ; 360a06b0
	v_and_or_b32 v1, 3, v0, v1                                  ; d7710001 04060083
	v_lshlrev_b32_e32 v4, 6, v4                                 ; 34080886
	v_lshl_or_b32 v2, v2, 12, v1                                ; d76f0002 04051902
	v_and_or_b32 v4, 3, v3, v4                                  ; d7710004 04120683
	v_and_b32_e32 v3, s0, v3                                    ; 36060600
	v_or_b32_e32 v2, s1, v2                                     ; 38040401
	v_lshl_or_b32 v5, v5, 12, v4                                ; d76f0005 04111905
	v_add_nc_u32_e32 v2, s8, v2                                 ; 4a040408
	v_lshl_or_b32 v3, v3, 18, v5                                ; d76f0003 04152503
	v_xor_b32_e32 v2, s1, v2                                    ; 3a040401
	v_or_b32_e32 v3, s1, v3                                     ; 38060601
	v_add_nc_u32_e32 v3, s8, v3                                 ; 4a060608
	v_xor_b32_e32 v3, s1, v3                                    ; 3a060601
	ds_write2st64_b32 v6, v2, v3 offset1:1                      ; d83c0100 00030206
	v_add_nc_u32_e32 v7, 0x80, v0                               ; 4a0e00ff 00000080
	v_add_nc_u32_e32 v10, s0, v0                                ; 4a140000
	v_and_b32_e32 v9, 48, v7                                    ; 36120eb0
	v_and_b32_e32 v8, 12, v7                                    ; 36100e8c
	v_and_b32_e32 v11, 12, v10                                  ; 3616148c
	v_and_b32_e32 v12, 48, v10                                  ; 361814b0
	v_lshlrev_b32_e32 v8, 6, v8                                 ; 34101086
	v_lshlrev_b32_e32 v11, 6, v11                               ; 34161686
	v_and_or_b32 v8, 3, v7, v8                                  ; d7710008 04220e83
	v_and_b32_e32 v7, s0, v7                                    ; 360e0e00
	v_and_or_b32 v11, 3, v10, v11                               ; d771000b 042e1483
	v_and_b32_e32 v10, s0, v10                                  ; 36141400
	v_lshl_or_b32 v9, v9, 12, v8                                ; d76f0009 04211909
	v_lshl_or_b32 v12, v12, 12, v11                             ; d76f000c 042d190c
	v_lshl_or_b32 v7, v7, 18, v9                                ; d76f0007 04252507
	v_lshl_or_b32 v10, v10, 18, v12                             ; d76f000a 0431250a
	v_or_b32_e32 v7, s1, v7                                     ; 380e0e01
	v_or_b32_e32 v10, s1, v10                                   ; 38141401
	v_add_nc_u32_e32 v7, s8, v7                                 ; 4a0e0e08
	v_add_nc_u32_e32 v10, s8, v10                               ; 4a141408
	v_xor_b32_e32 v7, s1, v7                                    ; 3a0e0e01
	v_xor_b32_e32 v10, s1, v10                                  ; 3a141401
	ds_write2_b32 v6, v7, v10 offset0:128 offset1:192           ; d838c080 000a0706
	v_and_b32_e32 v13, 7, v0                                    ; 361a0087
	s_lshl_b32 s9, s7, 2                                        ; 8f098207
	v_lshrrev_b32_e32 v14, 3, v0                                ; 2c1c0083
	v_lshrrev_b64 v[1:2], 0, 0                                  ; d7000001 02010080
	v_lshrrev_b64 v[3:4], 0, 0                                  ; d7000003 02010080
	s_branch BB1                                                ; bf8202b2
BB6:
	s_mov_b64 exec, s[0:1]                                      ; befe0400
	s_mov_b32 s0, s2                                            ; be800302
	s_movk_i32 s1, 0x8000                                       ; b0018000
	s_load_dwordx8 s[16:23], s[0:1], null                       ; f40c0400 fa000000
	v_mul_lo_u32 v5, 0x4a, v14                                  ; d5690005 02021cff 0000004a
	v_mul_lo_u32 v6, 0x128, v14                                 ; d5690006 02021cff 00000128
	v_lshl_add_u32 v5, v13, 3, v5                               ; d7460005 0415070d
	v_lshlrev_b32_e32 v5, 2, v5                                 ; 340a0a82
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	s_clause 0x2                                                ; bfa10002
	buffer_load_dwordx4 v[8:11], v5, s[20:23], 0 offen offset:8 ; e0381008 80050805
	buffer_load_dwordx4 v[16:19], v5, s[20:23], 0 offen offset:24 ; e0381018 80051005
	buffer_load_dword v6, v6, s[20:23], 0 offen                 ; e0301000 80050606
	s_add_u32 s0, s3, -1                                        ; 8000c103
	v_and_b32_e32 v15, 1, v0                                    ; 361e0081
	v_lshrrev_b32_e32 v7, 1, v13                                ; 2c0e1a81
	s_min_u32 s1, s9, s0                                        ; 83810009
	s_add_u32 s8, s9, 1                                         ; 80088109
	s_mul_i32 s1, s1, s5                                        ; 93010501
	v_lshlrev_b32_e32 v15, 3, v15                               ; 341e1e83
	s_min_u32 s8, s8, s0                                        ; 83880008
	v_lshl_add_u32 v7, v14, 2, v7                               ; d7460007 041d050e
	s_mul_i32 s8, s8, s5                                        ; 93080508
	v_mul_lo_u32 v7, 18, v7                                     ; d5690007 02020e92
	v_add_nc_u32_e32 v12, s1, v7                                ; 4a180e01
	v_add_nc_u32_e32 v23, s8, v7                                ; 4a2e0e08
	v_add_nc_u32_e32 v20, 2, v12                                ; 4a281882
	v_and_b32_e32 v22, -4, v12                                  ; 362c18c4
	v_and_b32_e32 v26, -4, v23                                  ; 36342ec4
	v_add_nc_u32_e32 v24, 2, v23                                ; 4a302e82
	v_add_nc_u32_e32 v21, v20, v15                              ; 4a2a1f14
	v_and_b32_e32 v20, -4, v20                                  ; 362828c4
	v_add_nc_u32_e32 v25, v24, v15                              ; 4a321f18
	v_and_b32_e32 v24, -4, v24                                  ; 363030c4
	v_add_nc_u32_e32 v20, v20, v15                              ; 4a281f14
	v_add_nc_u32_e32 v24, v24, v15                              ; 4a301f18
	s_clause 0x3                                                ; bfa10003
	buffer_load_dwordx3 v[27:29], v20, s[16:19], 0 offen        ; e03c1000 80041b14
	buffer_load_dword v22, v22, s[16:19], 0 offen               ; e0301000 80041616
	buffer_load_dwordx3 v[30:32], v24, s[16:19], 0 offen        ; e03c1000 80041e18
	buffer_load_dword v26, v26, s[16:19], 0 offen               ; e0301000 80041a1a
	v_and_b32_e32 v12, 3, v12                                   ; 36181883
	v_and_b32_e32 v21, 3, v21                                   ; 362a2a83
	v_and_b32_e32 v23, 3, v23                                   ; 362e2e83
	v_lshlrev_b32_e32 v21, 3, v21                               ; 342a2a83
	s_waitcnt vmcnt(3)                                          ; bf8c3f73
	v_alignbit_b32 v27, v28, v27, v21                           ; d54e001b 0456371c
	v_alignbit_b32 v29, v29, v28, v21                           ; d54e001d 0456391d
	v_lshlrev_b32_sdwa v34, 2, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 344436f9 00860682
	v_lshlrev_b32_sdwa v35, 2, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34463af9 00860682
	ds_read_b32 v34, v34                                        ; d8d80000 22000022
	ds_read_b32 v35, v35                                        ; d8d80000 23000023
	s_add_u32 s10, s9, 2                                        ; 800a8209
	v_lshlrev_b32_sdwa v24, 2, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 343036f9 01860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_and_b32_e32 v25, 3, v25                                   ; 36323283
	v_lshlrev_b32_e32 v12, 3, v12                               ; 34181883
	v_lshlrev_b32_e32 v23, 3, v23                               ; 342e2e83
	s_min_u32 s10, s10, s0                                      ; 838a000a
	s_mul_i32 s10, s10, s5                                      ; 930a050a
	v_lshlrev_b32_e32 v25, 3, v25                               ; 34323283
	v_add_nc_u32_e32 v33, s10, v7                               ; 4a420e0a
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v20, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122810f9 09090622
	v_mul_i32_i24_sdwa v5, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 120a10f9 08080622
	v_mul_i32_i24_sdwa v21, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a10f9 0a0a0622
	v_mul_i32_i24_sdwa v34, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124410f9 0b0b0622
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v28, sext(v35), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123820f9 08080623
	v_add_nc_u32_e32 v5, v5, v20                                ; 4a0a2905
	v_mul_i32_i24_sdwa v20, sext(v35), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122820f9 0a0a0623
	v_add3_u32 v5, v5, v21, v34                                 ; d76d0005 048a2b05
	v_mul_i32_i24_sdwa v34, sext(v35), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124420f9 09090623
	v_mul_i32_i24_sdwa v35, sext(v35), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124620f9 0b0b0623
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v21, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122a12f9 08080618
	v_add_nc_u32_e32 v28, v28, v34                              ; 4a38451c
	v_mul_i32_i24_sdwa v34, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124412f9 09090618
	v_add3_u32 v28, v28, v20, v35                               ; d76d001c 048e291c
	v_mul_i32_i24_sdwa v35, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124612f9 0a0a0618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123012f9 0b0b0618
	v_add_nc_u32_e32 v21, v21, v34                              ; 4a2a4515
	v_add3_u32 v21, v21, v35, v24                               ; d76d0015 04624715
	v_lshlrev_b32_sdwa v35, 2, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34463af9 01860682
	v_add3_u32 v21, v21, v28, v5                                ; d76d0015 04163915
	v_lshlrev_b32_sdwa v5, 2, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 340a36f9 02860682
	ds_read_b32 v35, v35                                        ; d8d80000 23000023
	ds_read_b32 v5, v5                                          ; d8d80000 05000005
	v_lshlrev_b32_sdwa v27, 2, v27 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 343636f9 03860682
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v20, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122822f9 08080623
	v_mul_i32_i24_sdwa v28, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123822f9 0a0a0623
	v_mul_i32_i24_sdwa v24, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123022f9 09090623
	v_mul_i32_i24_sdwa v35, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124622f9 0b0b0623
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v34, sext(v5), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124414f9 08080605
	v_add_nc_u32_e32 v20, v20, v24                              ; 4a283114
	v_add3_u32 v20, v20, v28, v35                               ; d76d0014 048e3914
	v_mul_i32_i24_sdwa v35, sext(v5), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124614f9 09090605
	v_add_nc_u32_e32 v34, v34, v35                              ; 4a444722
	v_mul_i32_i24_sdwa v35, sext(v5), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124614f9 0a0a0605
	v_mul_i32_i24_sdwa v5, sext(v5), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 120a14f9 0b0b0605
	v_add3_u32 v34, v34, v35, v5                                ; d76d0022 04164722
	v_lshlrev_b32_sdwa v35, 2, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34463af9 02860682
	v_lshlrev_b32_sdwa v29, 2, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 343a3af9 03860682
	ds_read_b32 v35, v35                                        ; d8d80000 23000023
	ds_read_b32 v27, v27                                        ; d8d80000 1b00001b
	ds_read_b32 v29, v29                                        ; d8d80000 1d00001d
	v_add3_u32 v34, v34, v20, v21                               ; d76d0022 04562922
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v20, sext(v35), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122824f9 09090623
	v_mul_i32_i24_sdwa v5, sext(v35), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 120a24f9 08080623
	v_mul_i32_i24_sdwa v21, sext(v35), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a24f9 0a0a0623
	v_mul_i32_i24_sdwa v35, sext(v35), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124624f9 0b0b0623
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v28, sext(v27), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123816f9 0909061b
	v_mul_i32_i24_sdwa v24, sext(v27), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123016f9 0808061b
	v_add_nc_u32_e32 v5, v5, v20                                ; 4a0a2905
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v20, sext(v29), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122826f9 0909061d
	v_add_nc_u32_e32 v24, v24, v28                              ; 4a303918
	v_mov_b32_e32 v28, v19                                      ; 7e380313
	v_add3_u32 v5, v5, v21, v35                                 ; d76d0005 048e2b05
	v_mul_i32_i24_sdwa v35, sext(v27), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124616f9 0a0a061b
	v_mul_i32_i24_sdwa v27, sext(v27), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123616f9 0b0b061b
	v_mul_i32_i24_sdwa v21, sext(v29), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a26f9 0a0a061d
	v_add3_u32 v24, v24, v35, v27                               ; d76d0018 046e4718
	v_and_b32_e32 v27, -4, v33                                  ; 363642c4
	v_add3_u32 v24, v24, v5, v34                                ; d76d0018 048a0b18
	v_add_nc_u32_e32 v34, 2, v33                                ; 4a444282
	v_mul_i32_i24_sdwa v5, sext(v29), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 120a26f9 0808061d
	v_mul_i32_i24_sdwa v29, sext(v29), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123a26f9 0b0b061d
	v_add_nc_u32_e32 v35, v34, v15                              ; 4a461f22
	v_and_b32_e32 v34, -4, v34                                  ; 364444c4
	v_add_nc_u32_e32 v5, v5, v20                                ; 4a0a2905
	v_add_nc_u32_e32 v34, v34, v15                              ; 4a441f22
	v_add3_u32 v5, v5, v21, v29                                 ; d76d0005 04762b05
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[19:21], v34, s[16:19], 0 offen        ; e03c1000 80041322
	buffer_load_dword v27, v27, s[16:19], 0 offen               ; e0301000 80041b1b
	v_and_b32_e32 v33, 3, v33                                   ; 36424283
	s_add_u32 s11, s9, 3                                        ; 800b8309
	s_waitcnt vmcnt(4)                                          ; bf8c3f74
	v_bfe_u32 v22, v22, v12, 16                                 ; d5480016 02421916
	s_waitcnt vmcnt(3)                                          ; bf8c3f73
	v_alignbit_b32 v30, v31, v30, v25                           ; d54e001e 04663d1f
	v_and_b32_e32 v35, 3, v35                                   ; 36464683
	v_alignbit_b32 v32, v32, v31, v25                           ; d54e0020 04663f20
	v_lshlrev_b32_sdwa v31, 2, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 343e3cf9 00860682
	v_lshlrev_b32_sdwa v34, 2, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 344440f9 00860682
	ds_read_b32 v31, v31                                        ; d8d80000 1f00001f
	ds_read_b32 v34, v34                                        ; d8d80000 22000022
	v_add_nc_u32_e32 v5, v5, v24                                ; 4a0a3105
	s_min_u32 s11, s11, s0                                      ; 838b000b
	v_lshlrev_b32_e32 v33, 3, v33                               ; 34424283
	v_lshlrev_b32_sdwa v24, 2, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34303cf9 01860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_fma_mix_f32 v22, v22, v6, neg(0) op_sel_hi:[1,0,0]        ; cc200016 8a020d16
	v_lshlrev_b32_e32 v35, 3, v35                               ; 34464683
	s_mul_i32 s11, s11, s5                                      ; 930b050b
	v_cvt_f32_i32_e32 v5, v5                                    ; 7e0a0b05
	s_waitcnt vmcnt(2)                                          ; bf8c3f72
	v_bfe_u32 v26, v26, v23, 16                                 ; d548001a 02422f1a
	v_add_nc_u32_e32 v7, s11, v7                                ; 4a0e0e0b
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v12, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121810f9 0909061f
	v_mac_f32_e32 v1, v22, v5                                   ; 3e020b16
	v_mul_i32_i24_sdwa v22, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c10f9 0a0a061f
	v_mul_i32_i24_sdwa v5, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 120a10f9 0808061f
	v_mul_i32_i24_sdwa v31, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123e10f9 0b0b061f
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v25, sext(v34), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123220f9 08080622
	v_fma_mix_f32 v26, v26, v6, neg(0) op_sel_hi:[1,0,0]        ; cc20001a 8a020d1a
	v_add_nc_u32_e32 v29, 2, v7                                 ; 4a3a0e82
	v_add_nc_u32_e32 v5, v5, v12                                ; 4a0a1905
	v_mul_i32_i24_sdwa v12, sext(v34), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121820f9 0a0a0622
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v23, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122e12f9 08080618
	v_add3_u32 v5, v5, v22, v31                                 ; d76d0005 047e2d05
	v_mul_i32_i24_sdwa v31, sext(v34), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123e20f9 09090622
	v_mul_i32_i24_sdwa v34, sext(v34), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124420f9 0b0b0622
	v_add_nc_u32_e32 v25, v25, v31                              ; 4a323f19
	v_mul_i32_i24_sdwa v31, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123e12f9 09090618
	v_add3_u32 v25, v25, v12, v34                               ; d76d0019 048a1919
	v_mul_i32_i24_sdwa v34, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124412f9 0a0a0618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123012f9 0b0b0618
	v_lshlrev_b32_sdwa v12, 2, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34183cf9 02860682
	v_add_nc_u32_e32 v23, v23, v31                              ; 4a2e3f17
	v_add3_u32 v23, v23, v34, v24                               ; d76d0017 04624517
	v_add_nc_u32_e32 v22, v29, v15                              ; 4a2c1f1d
	v_and_b32_e32 v29, -4, v29                                  ; 363a3ac4
	v_lshlrev_b32_sdwa v30, 2, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 343c3cf9 03860682
	v_lshlrev_b32_sdwa v31, 2, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 343e40f9 02860682
	v_add3_u32 v23, v23, v25, v5                                ; d76d0017 04163317
	v_lshlrev_b32_sdwa v5, 2, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 340a40f9 01860682
	v_lshlrev_b32_sdwa v32, 2, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 344040f9 03860682
	ds_read_b32 v5, v5                                          ; d8d80000 05000005
	ds_read_b32 v12, v12                                        ; d8d80000 0c00000c
	ds_read_b32 v31, v31                                        ; d8d80000 1f00001f
	ds_read_b32 v30, v30                                        ; d8d80000 1e00001e
	ds_read_b32 v32, v32                                        ; d8d80000 20000020
	v_add_nc_u32_e32 v29, v29, v15                              ; 4a3a1f1d
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v25, sext(v5), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123222f9 0a0a0605
	v_mul_i32_i24_sdwa v24, sext(v5), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123022f9 09090605
	v_mul_i32_i24_sdwa v15, sext(v5), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e22f9 08080605
	v_mul_i32_i24_sdwa v5, sext(v5), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 120a22f9 0b0b0605
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v34, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124414f9 0808060c
	v_add_nc_u32_e32 v15, v15, v24                              ; 4a1e310f
	v_mul_i32_i24_sdwa v24, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123014f9 0a0a060c
	v_add3_u32 v15, v15, v25, v5                                ; d76d000f 0416330f
	v_mul_i32_i24_sdwa v5, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a14f9 0909060c
	v_mul_i32_i24_sdwa v12, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 121814f9 0b0b060c
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v25, sext(v31), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123224f9 0808061f
	v_add_nc_u32_e32 v34, v34, v5                               ; 4a440b22
	v_mul_i32_i24_sdwa v5, sext(v31), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a24f9 0909061f
	v_add3_u32 v34, v34, v24, v12                               ; d76d0022 04323122
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v24, sext(v30), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123016f9 0a0a061e
	v_add_nc_u32_e32 v25, v25, v5                               ; 4a320b19
	v_mov_b32_e32 v5, v22                                       ; 7e0a0316
	v_add3_u32 v34, v34, v15, v23                               ; d76d0022 045e1f22
	v_mul_i32_i24_sdwa v15, sext(v30), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e16f9 0808061e
	v_mul_i32_i24_sdwa v23, sext(v30), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122e16f9 0909061e
	v_mul_i32_i24_sdwa v30, sext(v30), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123c16f9 0b0b061e
	v_add_nc_u32_e32 v15, v15, v23                              ; 4a1e2f0f
	v_add3_u32 v15, v15, v24, v30                               ; d76d000f 047a310f
	v_and_b32_e32 v30, -4, v7                                   ; 363c0ec4
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[22:24], v29, s[16:19], 0 offen        ; e03c1000 8004161d
	buffer_load_dword v30, v30, s[16:19], 0 offen               ; e0301000 80041e1e
	v_mul_i32_i24_sdwa v12, sext(v31), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121824f9 0a0a061f
	v_mul_i32_i24_sdwa v31, sext(v31), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123e24f9 0b0b061f
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v29, sext(v32), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123a38f9 0a0a0620
	v_and_b32_e32 v5, 3, v5                                     ; 360a0a83
	v_and_b32_e32 v7, 3, v7                                     ; 360e0e83
	v_add_nc_u32_e32 v14, 8, v14                                ; 4a1c1c88
	v_add3_u32 v25, v25, v12, v31                               ; d76d0019 047e1919
	s_waitcnt vmcnt(3)                                          ; bf8c3f73
	v_alignbit_b32 v19, v20, v19, v35                           ; d54e0013 048e2714
	v_mul_i32_i24_sdwa v12, sext(v32), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121838f9 08080620
	v_add3_u32 v15, v15, v25, v34                               ; d76d000f 048a330f
	v_mul_i32_i24_sdwa v25, sext(v32), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123238f9 09090620
	v_mul_i32_i24_sdwa v32, sext(v32), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124038f9 0b0b0620
	v_lshlrev_b32_e32 v5, 3, v5                                 ; 340a0a83
	v_alignbit_b32 v21, v21, v20, v35                           ; d54e0015 048e2915
	v_lshlrev_b32_e32 v7, 3, v7                                 ; 340e0e83
	v_lshlrev_b32_sdwa v31, 2, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 343e26f9 00860682
	v_lshlrev_b32_sdwa v34, 2, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 344426f9 01860682
	v_add_nc_u32_e32 v12, v12, v25                              ; 4a18330c
	v_lshlrev_b32_sdwa v35, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34462af9 01860682
	v_add3_u32 v12, v12, v29, v32                               ; d76d000c 04823b0c
	v_lshlrev_b32_sdwa v32, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34402af9 00860682
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_cvt_f32_i32_e32 v12, v12                                  ; 7e180b0c
	v_mac_f32_e32 v2, v26, v12                                  ; 3e04191a
	v_lshlrev_b32_sdwa v12, 2, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 341826f9 02860682
	ds_read_b32 v31, v31                                        ; d8d80000 1f00001f
	ds_read_b32 v32, v32                                        ; d8d80000 20000020
	ds_read_b32 v34, v34                                        ; d8d80000 22000022
	ds_read_b32 v35, v35                                        ; d8d80000 23000023
	ds_read_b32 v12, v12                                        ; d8d80000 0c00000c
	v_lshlrev_b32_sdwa v19, 2, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342626f9 03860682
	s_waitcnt vmcnt(2)                                          ; bf8c3f72
	v_bfe_u32 v27, v27, v33, 16                                 ; d548001b 0242431b
	v_fma_mix_f32 v27, v27, v6, neg(0) op_sel_hi:[1,0,0]        ; cc20001b 8a020d1b
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v15, sext(v34), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e12f9 08080622
	v_mul_i32_i24_sdwa v29, sext(v34), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123a12f9 09090622
	v_mul_i32_i24_sdwa v33, sext(v34), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124212f9 0a0a0622
	v_mul_i32_i24_sdwa v34, sext(v34), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124412f9 0b0b0622
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v25, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123222f9 09090623
	v_mul_i32_i24_sdwa v26, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123422f9 0a0a0623
	v_mul_i32_i24_sdwa v20, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122822f9 08080623
	v_mul_i32_i24_sdwa v35, sext(v35), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124622f9 0b0b0623
	v_add_nc_u32_e32 v15, v15, v29                              ; 4a1e3b0f
	v_add_nc_u32_e32 v20, v20, v25                              ; 4a283314
	v_mul_i32_i24_sdwa v25, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123210f9 0909061f
	v_lshlrev_b32_sdwa v29, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 343a2af9 02860682
	v_add3_u32 v15, v15, v33, v34                               ; d76d000f 048a430f
	ds_read_b32 v29, v29                                        ; d8d80000 1d00001d
	ds_read_b32 v19, v19                                        ; d8d80000 13000013
	v_mul_i32_i24_sdwa v34, sext(v32), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124420f9 08080620
	v_add3_u32 v20, v20, v26, v35                               ; d76d0014 048e3514
	v_mul_i32_i24_sdwa v35, sext(v32), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124620f9 09090620
	v_mul_i32_i24_sdwa v26, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123410f9 0a0a061f
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v33, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124214f9 0a0a060c
	v_lshlrev_b32_sdwa v21, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342a2af9 03860682
	v_add_nc_u32_e32 v34, v34, v35                              ; 4a444722
	ds_read_b32 v21, v21                                        ; d8d80000 15000015
	v_mul_i32_i24_sdwa v35, sext(v32), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124620f9 0a0a0620
	v_mul_i32_i24_sdwa v32, sext(v32), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124020f9 0b0b0620
	v_add3_u32 v34, v34, v35, v32                               ; d76d0022 04824722
	v_mul_i32_i24_sdwa v35, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124610f9 0808061f
	v_mul_i32_i24_sdwa v31, sext(v31), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123e10f9 0b0b061f
	v_add_nc_u32_e32 v35, v35, v25                              ; 4a463323
	v_add3_u32 v35, v35, v26, v31                               ; d76d0023 047e3523
	v_add3_u32 v15, v15, v34, v35                               ; d76d000f 048e450f
	s_waitcnt vmcnt(1)                                          ; bf8c3f71
	v_alignbit_b32 v22, v23, v22, v5                            ; d54e0016 04162d17
	v_lshlrev_b32_sdwa v34, 2, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34442cf9 00860682
	ds_read_b32 v34, v34                                        ; d8d80000 22000022
	v_mul_i32_i24_sdwa v31, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123e14f9 0808060c
	v_mul_i32_i24_sdwa v32, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124014f9 0909060c
	v_mul_i32_i24_sdwa v12, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 121814f9 0b0b060c
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v35, sext(v29), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124624f9 0808061d
	v_alignbit_b32 v24, v24, v23, v5                            ; d54e0018 04162f18
	v_mul_i32_i24_sdwa v5, sext(v29), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a24f9 0909061d
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v23, sext(v19), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122e16f9 0a0a0613
	v_lshlrev_b32_sdwa v25, 2, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 343230f9 00860682
	ds_read_b32 v25, v25                                        ; d8d80000 19000019
	v_add_nc_u32_e32 v31, v31, v32                              ; 4a3e411f
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v26, sext(v21), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123438f9 08080615
	v_add_nc_u32_e32 v35, v35, v5                               ; 4a460b23
	v_add3_u32 v31, v31, v33, v12                               ; d76d001f 0432431f
	v_mul_i32_i24_sdwa v12, sext(v29), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121824f9 0a0a061d
	v_mul_i32_i24_sdwa v29, sext(v29), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123a24f9 0b0b061d
	v_add3_u32 v31, v31, v20, v15                               ; d76d001f 043e291f
	v_mul_i32_i24_sdwa v15, sext(v19), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e16f9 08080613
	v_mul_i32_i24_sdwa v20, sext(v19), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122816f9 09090613
	v_lshlrev_b32_sdwa v32, 2, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34402cf9 01860682
	ds_read_b32 v32, v32                                        ; d8d80000 20000020
	v_mul_i32_i24_sdwa v19, sext(v19), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122616f9 0b0b0613
	v_add3_u32 v35, v35, v12, v29                               ; d76d0023 04761923
	v_mul_i32_i24_sdwa v29, sext(v21), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123a38f9 09090615
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v33, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124210f9 08080622
	v_add_nc_u32_e32 v15, v15, v20                              ; 4a1e290f
	v_add_nc_u32_e32 v26, v26, v29                              ; 4a343b1a
	v_add3_u32 v15, v15, v23, v19                               ; d76d000f 044e2f0f
	v_add3_u32 v15, v15, v35, v31                               ; d76d000f 047e470f
	v_mul_i32_i24_sdwa v35, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124610f9 09090622
	v_add_nc_u32_e32 v33, v33, v35                              ; 4a424721
	v_mul_i32_i24_sdwa v35, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124610f9 0a0a0622
	v_mul_i32_i24_sdwa v34, sext(v34), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124410f9 0b0b0622
	v_add3_u32 v33, v33, v35, v34                               ; d76d0021 048a4721
	v_lshlrev_b32_sdwa v34, 2, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 344430f9 01860682
	v_lshlrev_b32_sdwa v35, 2, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34462cf9 02860682
	v_mul_i32_i24_sdwa v31, sext(v21), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123e38f9 0a0a0615
	v_mul_i32_i24_sdwa v21, sext(v21), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122a38f9 0b0b0615
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v5, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 120a20f9 08080619
	v_lshlrev_b32_sdwa v22, 2, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342c2cf9 03860682
	v_add3_u32 v26, v26, v31, v21                               ; d76d001a 04563f1a
	v_add_nc_u32_e32 v26, v26, v15                              ; 4a341f1a
	v_lshlrev_b32_sdwa v15, 2, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 341e30f9 02860682
	ds_read_b32 v34, v34                                        ; d8d80000 22000022
	ds_read_b32 v35, v35                                        ; d8d80000 23000023
	ds_read_b32 v15, v15                                        ; d8d80000 0f00000f
	ds_read_b32 v22, v22                                        ; d8d80000 16000016
	v_mul_i32_i24_sdwa v8, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121020f9 09090619
	v_mul_i32_i24_sdwa v12, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121820f9 0a0a0619
	v_mul_i32_i24_sdwa v25, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123220f9 0b0b0619
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v16, sext(v32), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122012f9 08080620
	v_mul_i32_i24_sdwa v19, sext(v32), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122612f9 09090620
	v_mul_i32_i24_sdwa v20, sext(v32), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122812f9 0a0a0620
	v_mul_i32_i24_sdwa v32, sext(v32), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124012f9 0b0b0620
	v_cvt_f32_i32_e32 v26, v26                                  ; 7e340b1a
	v_add_nc_u32_e32 v5, v5, v8                                 ; 4a0a1105
	v_add_nc_u32_e32 v16, v16, v19                              ; 4a202710
	v_mac_f32_e32 v3, v27, v26                                  ; 3e06351b
	v_add3_u32 v5, v5, v12, v25                                 ; d76d0005 04661905
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v21, sext(v34), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122a22f9 08080622
	v_mul_i32_i24_sdwa v23, sext(v34), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122e22f9 09090622
	v_lshlrev_b32_sdwa v24, 2, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 343030f9 03860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_mul_i32_i24_sdwa v25, sext(v34), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123222f9 0a0a0622
	v_mul_i32_i24_sdwa v34, sext(v34), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124422f9 0b0b0622
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v27, sext(v35), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123614f9 09090623
	v_mul_i32_i24_sdwa v29, sext(v35), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123a14f9 0a0a0623
	v_mul_i32_i24_sdwa v26, sext(v35), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123414f9 08080623
	v_mul_i32_i24_sdwa v35, sext(v35), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 124614f9 0b0b0623
	v_add3_u32 v16, v16, v20, v32                               ; d76d0010 04822910
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v32, sext(v15), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124024f9 0909060f
	v_mul_i32_i24_sdwa v31, sext(v15), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123e24f9 0808060f
	v_add_nc_u32_e32 v21, v21, v23                              ; 4a2a2f15
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_bfe_u32 v30, v30, v7, 16                                  ; d548001e 02420f1e
	v_add_nc_u32_e32 v26, v26, v27                              ; 4a34371a
	v_add3_u32 v16, v16, v5, v33                                ; d76d0010 04860b10
	v_mul_i32_i24_sdwa v33, sext(v15), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124224f9 0a0a060f
	v_mul_i32_i24_sdwa v15, sext(v15), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 121e24f9 0b0b060f
	v_add_nc_u32_e32 v31, v31, v32                              ; 4a3e411f
	v_add3_u32 v21, v21, v25, v34                               ; d76d0015 048a3315
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v34, sext(v22), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124416f9 08080616
	v_fma_mix_f32 v30, v30, v6, neg(0) op_sel_hi:[1,0,0]        ; cc20001e 8a020d1e
	v_add3_u32 v26, v26, v29, v35                               ; d76d001a 048e3b1a
	v_mul_i32_i24_sdwa v35, sext(v22), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 124616f9 09090616
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v6, sext(v24), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 120c38f9 0a0a0618
	v_mul_i32_i24_sdwa v5, sext(v24), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a38f9 09090618
	v_add3_u32 v31, v31, v33, v15                               ; d76d001f 043e431f
	v_add3_u32 v26, v26, v21, v16                               ; d76d001a 04422b1a
	v_add_nc_u32_e32 v34, v34, v35                              ; 4a444722
	v_mul_i32_i24_sdwa v35, sext(v22), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 124616f9 0a0a0616
	v_mul_i32_i24_sdwa v22, sext(v22), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122c16f9 0b0b0616
	v_add3_u32 v34, v34, v35, v22                               ; d76d0022 045a4722
	v_mul_i32_i24_sdwa v35, sext(v24), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 124638f9 08080618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123038f9 0b0b0618
	v_add3_u32 v34, v34, v31, v26                               ; d76d0022 046a3f22
	v_add_nc_u32_e32 v35, v35, v5                               ; 4a460b23
	v_add3_u32 v35, v35, v6, v24                                ; d76d0023 04620d23
	v_add_nc_u32_e32 v35, v35, v34                              ; 4a464523
	v_cvt_f32_i32_e32 v35, v35                                  ; 7e460b23
	v_mac_f32_e32 v4, v30, v35                                  ; 3e08471e
BB1:
	s_mov_b64 s[0:1], exec                                      ; be80047e
	s_waitcnt_depctr 0xfffe                                     ; bfa3fffe
	v_cmpx_le_u32_e32 s4, v14                                   ; 7da61c04
BB2:
	s_andn2_b64 s[0:1], s[0:1], exec                            ; 8a807e00
	s_cbranch_scc1 BB6                                          ; bf85fd49
BB7:
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[4:5], -1                                ; be8425c1
	s_cmp_lt_u32 s9, s3                                         ; bf0a0309
	v_cndmask_b32_e64 v35, 0, v1, s[4:5]                        ; d5010023 00120280
	s_mov_b64 s[4:5], 1                                         ; be840481
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff04b123
	v_add_f32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff044e23
	v_add_f32_dpp v35, v35, v35 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054123
	v_add_f32_dpp v35, v35, v35 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054023
	v_permlanex16_b32 v34, v35, 0, 0                            ; d7780022 02010123
	v_add_f32_e32 v35, v35, v34                                 ; 06464523
	v_readlane_b32 s0, v35, 0                                   ; d7600000 02010123
	v_add_f32_e32 v35, s0, v35                                  ; 06464600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v35, 63                                  ; d7600000 02017f23
	s_cbranch_execz BB13                                        ; bf88000b
BB8:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0                    ; e0700000 00030080
BB13:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s1, s9, 1                                         ; 80018109
	v_cndmask_b32_e64 v35, 0, v2, s[10:11]                      ; d5010023 002a0480
	s_cmp_lt_u32 s1, s3                                         ; bf0a0301
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff04b123
	v_add_f32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff044e23
	v_add_f32_dpp v35, v35, v35 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054123
	v_add_f32_dpp v35, v35, v35 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054023
	v_permlanex16_b32 v34, v35, 0, 0                            ; d7780022 02010123
	v_add_f32_e32 v35, v35, v34                                 ; 06464523
	v_readlane_b32 s0, v35, 0                                   ; d7600000 02010123
	v_add_f32_e32 v35, s0, v35                                  ; 06464600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v35, 63                                  ; d7600000 02017f23
	s_cbranch_execz BB19                                        ; bf88000b
BB14:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0 offset:4           ; e0700004 00030080
BB19:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s1, s9, 2                                         ; 80018209
	v_cndmask_b32_e64 v35, 0, v3, s[10:11]                      ; d5010023 002a0680
	s_cmp_lt_u32 s1, s3                                         ; bf0a0301
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff04b123
	v_add_f32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff044e23
	v_add_f32_dpp v35, v35, v35 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054123
	v_add_f32_dpp v35, v35, v35 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054023
	v_permlanex16_b32 v34, v35, 0, 0                            ; d7780022 02010123
	v_add_f32_e32 v35, v35, v34                                 ; 06464523
	v_readlane_b32 s0, v35, 0                                   ; d7600000 02010123
	v_add_f32_e32 v35, s0, v35                                  ; 06464600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v35, 63                                  ; d7600000 02017f23
	s_cbranch_execz BB25                                        ; bf88000b
BB20:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0 offset:8           ; e0700008 00030080
BB25:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s9, s9, 3                                         ; 80098309
	v_cndmask_b32_e64 v35, 0, v4, s[10:11]                      ; d5010023 002a0880
	s_cmp_lt_u32 s9, s3                                         ; bf0a0309
	s_cselect_b64 s[8:9], -1, 0                                 ; 858880c1
	v_add_f32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff04b123
	v_add_f32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff044e23
	v_add_f32_dpp v35, v35, v35 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054123
	v_add_f32_dpp v35, v35, v35 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 064646fa ff054023
	v_permlanex16_b32 v34, v35, 0, 0                            ; d7780022 02010123
	v_add_f32_e32 v35, v35, v34                                 ; 06464523
	v_readlane_b32 s0, v35, 0                                   ; d7600000 02010123
	v_add_f32_e32 v35, s0, v35                                  ; 06464600
	s_and_b64 exec, s[4:5], s[8:9]                              ; 87fe0804
	v_readlane_b32 s0, v35, 63                                  ; d7600000 02017f23
	s_cbranch_execz BB31                                        ; bf88000a
BB26:
	s_movk_i32 s3, 0x8000                                       ; b0038000
	s_load_dwordx4 s[8:11], s[2:3], 0x20                        ; f4080201 fa000020
	s_lshl_b32 s7, s7, 4                                        ; 8f078407
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[8:11], s7 offset:12           ; e070000c 07020080
BB31:
	s_endpgm                                                    ; bf810000

