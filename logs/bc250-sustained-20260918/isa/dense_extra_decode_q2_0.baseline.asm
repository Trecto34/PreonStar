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
BB1:
	s_mov_b64 s[0:1], exec                                      ; be80047e
	s_waitcnt_depctr 0xfffe                                     ; bfa3fffe
	v_cmpx_le_u32_e32 s4, v14                                   ; 7da61c04
BB2:
	s_andn2_b64 s[0:1], s[0:1], exec                            ; 8a807e00
	s_cbranch_scc0 BB19                                         ; bf8402de
BB6:
	s_mov_b64 exec, s[0:1]                                      ; befe0400
	s_mov_b32 s0, s2                                            ; be800302
	s_movk_i32 s1, 0x8000                                       ; b0018000
	s_load_dwordx4 s[12:15], s[0:1], 0x10                       ; f4080300 fa000010
	v_mul_lo_u32 v5, 0x4a, v14                                  ; d5690005 02021cff 0000004a
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_mul_lo_u32 v6, 0x128, v14                                 ; d5690006 02021cff 00000128
	v_lshl_add_u32 v5, v13, 3, v5                               ; d7460005 0415070d
	v_lshlrev_b32_e32 v5, 2, v5                                 ; 340a0a82
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	s_clause 0x2                                                ; bfa10002
	buffer_load_dwordx4 v[8:11], v5, s[12:15], 0 offen offset:8 ; e0381008 80030805
	buffer_load_dwordx4 v[16:19], v5, s[12:15], 0 offen offset:24 ; e0381018 80031005
	buffer_load_dword v6, v6, s[12:15], 0 offen                 ; e0301000 80030606
	s_cmp_lt_u32 s9, s3                                         ; bf0a0309
	s_cbranch_scc0 BB9                                          ; bf8400ae
BB7:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_load_dwordx4 s[12:15], s[0:1], null                       ; f4080300 fa000000
	v_lshrrev_b32_e32 v5, 1, v13                                ; 2c0a1a81
	s_mul_i32 s8, s9, s5                                        ; 93080509
	v_and_b32_e32 v12, 1, v0                                    ; 36180081
	v_lshl_add_u32 v5, v14, 2, v5                               ; d7460005 0415050e
	v_lshlrev_b32_e32 v12, 3, v12                               ; 34181883
	v_mul_lo_u32 v5, 18, v5                                     ; d5690005 02020a92
	v_add_nc_u32_e32 v7, s8, v5                                 ; 4a0e0a08
	v_and_b32_e32 v5, -4, v5                                    ; 360a0ac4
	v_add_nc_u32_e32 v15, 2, v7                                 ; 4a1e0e82
	v_add_nc_u32_e32 v5, s8, v5                                 ; 4a0a0a08
	v_add_nc_u32_e32 v20, v15, v12                              ; 4a28190f
	v_and_b32_e32 v15, -4, v15                                  ; 361e1ec4
	v_add_nc_u32_e32 v15, v15, v12                              ; 4a1e190f
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[21:23], v15, s[12:15], 0 offen        ; e03c1000 8003150f
	buffer_load_dword v5, v5, s[12:15], 0 offen                 ; e0301000 80030505
	v_and_b32_e32 v7, 3, v7                                     ; 360e0e83
	v_and_b32_e32 v20, 3, v20                                   ; 36282883
	v_lshlrev_b32_e32 v20, 3, v20                               ; 34282883
	s_waitcnt vmcnt(1)                                          ; bf8c3f71
	v_alignbit_b32 v21, v22, v21, v20                           ; d54e0015 04522b16
	v_alignbit_b32 v23, v23, v22, v20                           ; d54e0017 04522d17
	v_lshlrev_b32_sdwa v24, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34302af9 00860682
	v_lshlrev_b32_sdwa v26, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34342af9 01860682
	v_lshlrev_b32_sdwa v12, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34182af9 02860682
	v_lshlrev_b32_sdwa v27, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34362ef9 01860682
	v_lshlrev_b32_sdwa v25, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34322ef9 00860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	ds_read_b32 v25, v25                                        ; d8d80000 19000019
	ds_read_b32 v26, v26                                        ; d8d80000 1a00001a
	ds_read_b32 v27, v27                                        ; d8d80000 1b00001b
	ds_read_b32 v12, v12                                        ; d8d80000 0c00000c
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v20, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122810f9 09090618
	v_mul_i32_i24_sdwa v22, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c10f9 0a0a0618
	v_mul_i32_i24_sdwa v15, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e10f9 08080618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123010f9 0b0b0618
	v_add_nc_u32_e32 v15, v15, v20                              ; 4a1e290f
	v_add3_u32 v15, v15, v22, v24                               ; d76d000f 04622d0f
	v_lshlrev_b32_sdwa v24, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34302ef9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_lshlrev_b32_e32 v7, 3, v7                                 ; 340e0e83
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v20, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122820f9 08080619
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c20f9 09090619
	v_add_nc_u32_e32 v20, v20, v22                              ; 4a282d14
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c20f9 0a0a0619
	v_mul_i32_i24_sdwa v25, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123220f9 0b0b0619
	v_add3_u32 v20, v20, v22, v25                               ; d76d0014 04662d14
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c12f9 0909061a
	v_mul_i32_i24_sdwa v25, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123212f9 0808061a
	v_add_nc_u32_e32 v25, v25, v22                              ; 4a322d19
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c12f9 0a0a061a
	v_mul_i32_i24_sdwa v26, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123412f9 0b0b061a
	v_lshlrev_b32_sdwa v21, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342a2af9 03860682
	v_add3_u32 v25, v25, v22, v26                               ; d76d0019 046a2d19
	ds_read_b32 v21, v21                                        ; d8d80000 15000015
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v26, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123422f9 0808061b
	v_lshlrev_b32_sdwa v23, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342e2ef9 03860682
	ds_read_b32 v23, v23                                        ; d8d80000 17000017
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122c14f9 0808060c
	v_add3_u32 v25, v25, v20, v15                               ; d76d0019 043e2919
	v_mul_i32_i24_sdwa v15, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e22f9 0909061b
	v_mul_i32_i24_sdwa v20, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122822f9 0a0a061b
	v_mul_i32_i24_sdwa v27, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123622f9 0b0b061b
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_bfe_u32 v5, v5, v7, 16                                    ; d5480005 02420f05
	v_add_nc_u32_e32 v26, v26, v15                              ; 4a341f1a
	v_fma_mix_f32 v5, v5, v6, neg(0) op_sel_hi:[1,0,0]          ; cc200005 8a020d05
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v7, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120e16f9 09090615
	v_add3_u32 v26, v26, v20, v27                               ; d76d001a 046e291a
	v_mul_i32_i24_sdwa v27, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123614f9 0909060c
	v_add_nc_u32_e32 v22, v22, v27                              ; 4a2c3716
	v_mul_i32_i24_sdwa v27, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123614f9 0a0a060c
	v_mul_i32_i24_sdwa v12, sext(v12), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 121814f9 0b0b060c
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v15, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e26f9 08080617
	v_mul_i32_i24_sdwa v20, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122826f9 09090617
	v_add3_u32 v22, v22, v27, v12                               ; d76d0016 04323716
	v_mul_i32_i24_sdwa v27, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123624f9 0a0a0618
	v_mul_i32_i24_sdwa v12, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121816f9 0a0a0615
	v_add_nc_u32_e32 v15, v15, v20                              ; 4a1e290f
	v_add3_u32 v22, v22, v26, v25                               ; d76d0016 04663516
	v_mul_i32_i24_sdwa v25, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123224f9 08080618
	v_mul_i32_i24_sdwa v26, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123424f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123024f9 0b0b0618
	v_add_nc_u32_e32 v25, v25, v26                              ; 4a323519
	v_add3_u32 v25, v25, v27, v24                               ; d76d0019 04623719
	v_mul_i32_i24_sdwa v27, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123616f9 08080615
	v_mul_i32_i24_sdwa v21, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122a16f9 0b0b0615
	v_add_nc_u32_e32 v27, v27, v7                               ; 4a360f1b
	v_add3_u32 v27, v27, v12, v21                               ; d76d001b 0456191b
	v_mul_i32_i24_sdwa v21, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a26f9 0a0a0617
	v_mul_i32_i24_sdwa v23, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122e26f9 0b0b0617
	v_add3_u32 v27, v27, v25, v22                               ; d76d001b 045a331b
	v_add3_u32 v15, v15, v21, v23                               ; d76d000f 045e2b0f
	v_add_nc_u32_e32 v15, v15, v27                              ; 4a1e370f
	v_cvt_f32_i32_e32 v15, v15                                  ; 7e1e0b0f
	v_mac_f32_e32 v1, v5, v15                                   ; 3e021f05
BB9:
	s_add_u32 s8, s9, 1                                         ; 80088109
	s_cmp_lt_u32 s8, s3                                         ; bf0a0308
	s_cbranch_scc0 BB12                                         ; bf8400ae
BB10:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_load_dwordx4 s[12:15], s[0:1], null                       ; f4080300 fa000000
	v_lshrrev_b32_e32 v5, 1, v13                                ; 2c0a1a81
	s_mul_i32 s8, s8, s5                                        ; 93080508
	v_and_b32_e32 v7, 1, v0                                     ; 360e0081
	v_lshl_add_u32 v5, v14, 2, v5                               ; d7460005 0415050e
	v_lshlrev_b32_e32 v7, 3, v7                                 ; 340e0e83
	v_mul_lo_u32 v5, 18, v5                                     ; d5690005 02020a92
	v_add_nc_u32_e32 v5, s8, v5                                 ; 4a0a0a08
	v_add_nc_u32_e32 v12, 2, v5                                 ; 4a180a82
	v_and_b32_e32 v20, -4, v5                                   ; 36280ac4
	v_add_nc_u32_e32 v15, v12, v7                               ; 4a1e0f0c
	v_and_b32_e32 v12, -4, v12                                  ; 361818c4
	v_add_nc_u32_e32 v12, v12, v7                               ; 4a180f0c
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[21:23], v12, s[12:15], 0 offen        ; e03c1000 8003150c
	buffer_load_dword v20, v20, s[12:15], 0 offen               ; e0301000 80031414
	v_and_b32_e32 v5, 3, v5                                     ; 360a0a83
	v_and_b32_e32 v15, 3, v15                                   ; 361e1e83
	v_lshlrev_b32_e32 v15, 3, v15                               ; 341e1e83
	s_waitcnt vmcnt(1)                                          ; bf8c3f71
	v_alignbit_b32 v23, v23, v22, v15                           ; d54e0017 043e2d17
	v_alignbit_b32 v21, v22, v21, v15                           ; d54e0015 043e2b16
	v_lshlrev_b32_sdwa v25, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34322ef9 00860682
	v_lshlrev_b32_sdwa v27, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34362ef9 01860682
	v_lshlrev_b32_sdwa v24, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34302af9 00860682
	v_lshlrev_b32_sdwa v26, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34342af9 01860682
	v_lshlrev_b32_sdwa v7, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 340e2af9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	ds_read_b32 v25, v25                                        ; d8d80000 19000019
	ds_read_b32 v26, v26                                        ; d8d80000 1a00001a
	ds_read_b32 v27, v27                                        ; d8d80000 1b00001b
	ds_read_b32 v7, v7                                          ; d8d80000 07000007
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v12, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121810f9 08080618
	v_mul_i32_i24_sdwa v22, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c10f9 0a0a0618
	v_mul_i32_i24_sdwa v15, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e10f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123010f9 0b0b0618
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_add3_u32 v12, v12, v22, v24                               ; d76d000c 04622d0c
	v_lshlrev_b32_sdwa v24, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34302ef9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_lshlrev_b32_e32 v5, 3, v5                                 ; 340a0a83
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v15, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e20f9 08080619
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c20f9 09090619
	v_add_nc_u32_e32 v15, v15, v22                              ; 4a1e2d0f
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c20f9 0a0a0619
	v_mul_i32_i24_sdwa v25, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123220f9 0b0b0619
	v_add3_u32 v15, v15, v22, v25                               ; d76d000f 04662d0f
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c12f9 0909061a
	v_mul_i32_i24_sdwa v25, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123212f9 0808061a
	v_add_nc_u32_e32 v25, v25, v22                              ; 4a322d19
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c12f9 0a0a061a
	v_mul_i32_i24_sdwa v26, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123412f9 0b0b061a
	v_lshlrev_b32_sdwa v21, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342a2af9 03860682
	v_add3_u32 v25, v25, v22, v26                               ; d76d0019 046a2d19
	ds_read_b32 v21, v21                                        ; d8d80000 15000015
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v26, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123422f9 0808061b
	v_lshlrev_b32_sdwa v23, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342e2ef9 03860682
	ds_read_b32 v23, v23                                        ; d8d80000 17000017
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122c14f9 08080607
	v_add3_u32 v25, v25, v15, v12                               ; d76d0019 04321f19
	v_mul_i32_i24_sdwa v12, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121822f9 0909061b
	v_mul_i32_i24_sdwa v15, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121e22f9 0a0a061b
	v_mul_i32_i24_sdwa v27, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123622f9 0b0b061b
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_bfe_u32 v20, v20, v5, 16                                  ; d5480014 02420b14
	v_add_nc_u32_e32 v26, v26, v12                              ; 4a34191a
	v_fma_mix_f32 v20, v20, v6, neg(0) op_sel_hi:[1,0,0]        ; cc200014 8a020d14
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v5, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a16f9 09090615
	v_add3_u32 v26, v26, v15, v27                               ; d76d001a 046e1f1a
	v_mul_i32_i24_sdwa v27, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123614f9 09090607
	v_add_nc_u32_e32 v22, v22, v27                              ; 4a2c3716
	v_mul_i32_i24_sdwa v27, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123614f9 0a0a0607
	v_mul_i32_i24_sdwa v7, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 120e14f9 0b0b0607
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v12, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121826f9 08080617
	v_mul_i32_i24_sdwa v15, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e26f9 09090617
	v_add3_u32 v22, v22, v27, v7                                ; d76d0016 041e3716
	v_mul_i32_i24_sdwa v27, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123624f9 0a0a0618
	v_mul_i32_i24_sdwa v7, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 120e16f9 0a0a0615
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_add3_u32 v22, v22, v26, v25                               ; d76d0016 04663516
	v_mul_i32_i24_sdwa v25, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123224f9 08080618
	v_mul_i32_i24_sdwa v26, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123424f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123024f9 0b0b0618
	v_add_nc_u32_e32 v25, v25, v26                              ; 4a323519
	v_add3_u32 v25, v25, v27, v24                               ; d76d0019 04623719
	v_mul_i32_i24_sdwa v27, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123616f9 08080615
	v_mul_i32_i24_sdwa v21, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122a16f9 0b0b0615
	v_add_nc_u32_e32 v27, v27, v5                               ; 4a360b1b
	v_add3_u32 v27, v27, v7, v21                                ; d76d001b 04560f1b
	v_mul_i32_i24_sdwa v21, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a26f9 0a0a0617
	v_mul_i32_i24_sdwa v23, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122e26f9 0b0b0617
	v_add3_u32 v27, v27, v25, v22                               ; d76d001b 045a331b
	v_add3_u32 v12, v12, v21, v23                               ; d76d000c 045e2b0c
	v_add_nc_u32_e32 v12, v12, v27                              ; 4a18370c
	v_cvt_f32_i32_e32 v12, v12                                  ; 7e180b0c
	v_mac_f32_e32 v2, v20, v12                                  ; 3e041914
BB12:
	s_add_u32 s8, s9, 2                                         ; 80088209
	s_cmp_lt_u32 s8, s3                                         ; bf0a0308
	s_cbranch_scc0 BB15                                         ; bf8400ae
BB13:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_load_dwordx4 s[12:15], s[0:1], null                       ; f4080300 fa000000
	v_lshrrev_b32_e32 v5, 1, v13                                ; 2c0a1a81
	s_mul_i32 s8, s8, s5                                        ; 93080508
	v_and_b32_e32 v7, 1, v0                                     ; 360e0081
	v_lshl_add_u32 v5, v14, 2, v5                               ; d7460005 0415050e
	v_lshlrev_b32_e32 v7, 3, v7                                 ; 340e0e83
	v_mul_lo_u32 v5, 18, v5                                     ; d5690005 02020a92
	v_add_nc_u32_e32 v5, s8, v5                                 ; 4a0a0a08
	v_add_nc_u32_e32 v12, 2, v5                                 ; 4a180a82
	v_and_b32_e32 v20, -4, v5                                   ; 36280ac4
	v_add_nc_u32_e32 v15, v12, v7                               ; 4a1e0f0c
	v_and_b32_e32 v12, -4, v12                                  ; 361818c4
	v_add_nc_u32_e32 v12, v12, v7                               ; 4a180f0c
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[21:23], v12, s[12:15], 0 offen        ; e03c1000 8003150c
	buffer_load_dword v20, v20, s[12:15], 0 offen               ; e0301000 80031414
	v_and_b32_e32 v5, 3, v5                                     ; 360a0a83
	v_and_b32_e32 v15, 3, v15                                   ; 361e1e83
	v_lshlrev_b32_e32 v15, 3, v15                               ; 341e1e83
	s_waitcnt vmcnt(1)                                          ; bf8c3f71
	v_alignbit_b32 v23, v23, v22, v15                           ; d54e0017 043e2d17
	v_alignbit_b32 v21, v22, v21, v15                           ; d54e0015 043e2b16
	v_lshlrev_b32_sdwa v25, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34322ef9 00860682
	v_lshlrev_b32_sdwa v27, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34362ef9 01860682
	v_lshlrev_b32_sdwa v24, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34302af9 00860682
	v_lshlrev_b32_sdwa v26, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34342af9 01860682
	v_lshlrev_b32_sdwa v7, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 340e2af9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	ds_read_b32 v25, v25                                        ; d8d80000 19000019
	ds_read_b32 v26, v26                                        ; d8d80000 1a00001a
	ds_read_b32 v27, v27                                        ; d8d80000 1b00001b
	ds_read_b32 v7, v7                                          ; d8d80000 07000007
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v12, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121810f9 08080618
	v_mul_i32_i24_sdwa v22, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c10f9 0a0a0618
	v_mul_i32_i24_sdwa v15, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e10f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123010f9 0b0b0618
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_add3_u32 v12, v12, v22, v24                               ; d76d000c 04622d0c
	v_lshlrev_b32_sdwa v24, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34302ef9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_lshlrev_b32_e32 v5, 3, v5                                 ; 340a0a83
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v15, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121e20f9 08080619
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c20f9 09090619
	v_add_nc_u32_e32 v15, v15, v22                              ; 4a1e2d0f
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c20f9 0a0a0619
	v_mul_i32_i24_sdwa v25, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123220f9 0b0b0619
	v_add3_u32 v15, v15, v22, v25                               ; d76d000f 04662d0f
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122c12f9 0909061a
	v_mul_i32_i24_sdwa v25, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123212f9 0808061a
	v_add_nc_u32_e32 v25, v25, v22                              ; 4a322d19
	v_mul_i32_i24_sdwa v22, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c12f9 0a0a061a
	v_mul_i32_i24_sdwa v26, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123412f9 0b0b061a
	v_lshlrev_b32_sdwa v21, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342a2af9 03860682
	v_add3_u32 v25, v25, v22, v26                               ; d76d0019 046a2d19
	ds_read_b32 v21, v21                                        ; d8d80000 15000015
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v26, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123422f9 0808061b
	v_lshlrev_b32_sdwa v23, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342e2ef9 03860682
	ds_read_b32 v23, v23                                        ; d8d80000 17000017
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v22, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122c14f9 08080607
	v_add3_u32 v25, v25, v15, v12                               ; d76d0019 04321f19
	v_mul_i32_i24_sdwa v12, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121822f9 0909061b
	v_mul_i32_i24_sdwa v15, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121e22f9 0a0a061b
	v_mul_i32_i24_sdwa v27, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123622f9 0b0b061b
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_bfe_u32 v20, v20, v5, 16                                  ; d5480014 02420b14
	v_add_nc_u32_e32 v26, v26, v12                              ; 4a34191a
	v_fma_mix_f32 v20, v20, v6, neg(0) op_sel_hi:[1,0,0]        ; cc200014 8a020d14
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v5, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a16f9 09090615
	v_add3_u32 v26, v26, v15, v27                               ; d76d001a 046e1f1a
	v_mul_i32_i24_sdwa v27, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123614f9 09090607
	v_add_nc_u32_e32 v22, v22, v27                              ; 4a2c3716
	v_mul_i32_i24_sdwa v27, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123614f9 0a0a0607
	v_mul_i32_i24_sdwa v7, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 120e14f9 0b0b0607
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v12, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121826f9 08080617
	v_mul_i32_i24_sdwa v15, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e26f9 09090617
	v_add3_u32 v22, v22, v27, v7                                ; d76d0016 041e3716
	v_mul_i32_i24_sdwa v27, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123624f9 0a0a0618
	v_mul_i32_i24_sdwa v7, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 120e16f9 0a0a0615
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_add3_u32 v22, v22, v26, v25                               ; d76d0016 04663516
	v_mul_i32_i24_sdwa v25, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123224f9 08080618
	v_mul_i32_i24_sdwa v26, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123424f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123024f9 0b0b0618
	v_add_nc_u32_e32 v25, v25, v26                              ; 4a323519
	v_add3_u32 v25, v25, v27, v24                               ; d76d0019 04623719
	v_mul_i32_i24_sdwa v27, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123616f9 08080615
	v_mul_i32_i24_sdwa v21, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122a16f9 0b0b0615
	v_add_nc_u32_e32 v27, v27, v5                               ; 4a360b1b
	v_add3_u32 v27, v27, v7, v21                                ; d76d001b 04560f1b
	v_mul_i32_i24_sdwa v21, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122a26f9 0a0a0617
	v_mul_i32_i24_sdwa v23, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122e26f9 0b0b0617
	v_add3_u32 v27, v27, v25, v22                               ; d76d001b 045a331b
	v_add3_u32 v12, v12, v21, v23                               ; d76d000c 045e2b0c
	v_add_nc_u32_e32 v12, v12, v27                              ; 4a18370c
	v_cvt_f32_i32_e32 v12, v12                                  ; 7e180b0c
	v_mac_f32_e32 v3, v20, v12                                  ; 3e061914
BB15:
	s_add_u32 s8, s9, 3                                         ; 80088309
	s_cmp_lt_u32 s8, s3                                         ; bf0a0308
	s_cbranch_scc0 BB18                                         ; bf8400af
BB16:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_load_dwordx4 s[12:15], s[0:1], null                       ; f4080300 fa000000
	v_lshrrev_b32_e32 v5, 1, v13                                ; 2c0a1a81
	s_mul_i32 s8, s8, s5                                        ; 93080508
	v_and_b32_e32 v7, 1, v0                                     ; 360e0081
	v_lshl_add_u32 v5, v14, 2, v5                               ; d7460005 0415050e
	v_lshlrev_b32_e32 v7, 3, v7                                 ; 340e0e83
	v_mul_lo_u32 v5, 18, v5                                     ; d5690005 02020a92
	v_add_nc_u32_e32 v5, s8, v5                                 ; 4a0a0a08
	v_add_nc_u32_e32 v12, 2, v5                                 ; 4a180a82
	v_and_b32_e32 v20, -4, v5                                   ; 36280ac4
	v_add_nc_u32_e32 v15, v12, v7                               ; 4a1e0f0c
	v_and_b32_e32 v12, -4, v12                                  ; 361818c4
	v_add_nc_u32_e32 v12, v12, v7                               ; 4a180f0c
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	s_clause 0x1                                                ; bfa10001
	buffer_load_dwordx3 v[21:23], v12, s[12:15], 0 offen        ; e03c1000 8003150c
	buffer_load_dword v20, v20, s[12:15], 0 offen               ; e0301000 80031414
	v_and_b32_e32 v5, 3, v5                                     ; 360a0a83
	v_and_b32_e32 v15, 3, v15                                   ; 361e1e83
	v_lshlrev_b32_e32 v15, 3, v15                               ; 341e1e83
	s_waitcnt vmcnt(1)                                          ; bf8c3f71
	v_alignbit_b32 v23, v23, v22, v15                           ; d54e0017 043e2d17
	v_alignbit_b32 v21, v22, v21, v15                           ; d54e0015 043e2b16
	v_lshlrev_b32_sdwa v25, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34322ef9 00860682
	v_lshlrev_b32_sdwa v27, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34362ef9 01860682
	v_lshlrev_b32_sdwa v24, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0 ; 34302af9 00860682
	v_lshlrev_b32_sdwa v26, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1 ; 34342af9 01860682
	v_lshlrev_b32_sdwa v7, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 340e2af9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	ds_read_b32 v25, v25                                        ; d8d80000 19000019
	ds_read_b32 v26, v26                                        ; d8d80000 1a00001a
	ds_read_b32 v27, v27                                        ; d8d80000 1b00001b
	ds_read_b32 v7, v7                                          ; d8d80000 07000007
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v12, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121810f9 08080618
	v_mul_i32_i24_sdwa v22, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c10f9 0a0a0618
	v_mul_i32_i24_sdwa v15, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e10f9 09090618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123010f9 0b0b0618
	v_add_nc_u32_e32 v12, v12, v15                              ; 4a181f0c
	v_add3_u32 v12, v12, v22, v24                               ; d76d000c 04622d0c
	v_lshlrev_b32_sdwa v24, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_2 ; 34302ef9 02860682
	ds_read_b32 v24, v24                                        ; d8d80000 18000018
	v_lshlrev_b32_e32 v5, 3, v5                                 ; 340a0a83
	s_waitcnt lgkmcnt(4)                                        ; bf8cc47f
	v_mul_i32_i24_sdwa v22, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c20f9 0a0a0619
	v_mul_i32_i24_sdwa v8, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121020f9 08080619
	v_mul_i32_i24_sdwa v15, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e20f9 09090619
	v_mul_i32_i24_sdwa v25, sext(v25), sext(v16) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123220f9 0b0b0619
	s_waitcnt lgkmcnt(3)                                        ; bf8cc37f
	v_mul_i32_i24_sdwa v16, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122012f9 0a0a061a
	v_add_nc_u32_e32 v8, v8, v15                                ; 4a101f08
	v_mul_i32_i24_sdwa v15, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121e12f9 0909061a
	v_add3_u32 v8, v8, v22, v25                                 ; d76d0008 04662d08
	v_mul_i32_i24_sdwa v25, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123212f9 0808061a
	v_mul_i32_i24_sdwa v26, sext(v26), sext(v9) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123412f9 0b0b061a
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v22, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122c22f9 0808061b
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v9, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 121214f9 08080607
	v_lshlrev_b32_sdwa v21, 2, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342a2af9 03860682
	ds_read_b32 v21, v21                                        ; d8d80000 15000015
	v_lshlrev_b32_sdwa v23, 2, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3 ; 342e2ef9 03860682
	ds_read_b32 v23, v23                                        ; d8d80000 17000017
	v_add_nc_u32_e32 v25, v25, v15                              ; 4a321f19
	v_mul_i32_i24_sdwa v15, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121e14f9 0a0a0607
	v_add3_u32 v25, v25, v16, v26                               ; d76d0019 046a2119
	v_mul_i32_i24_sdwa v26, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123422f9 0909061b
	s_waitcnt lgkmcnt(2)                                        ; bf8cc27f
	v_mul_i32_i24_sdwa v16, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 122024f9 08080618
	s_waitcnt vmcnt(0)                                          ; bf8c3f70
	v_bfe_u32 v20, v20, v5, 16                                  ; d5480014 02420b14
	v_add3_u32 v25, v25, v8, v12                                ; d76d0019 04321119
	v_mul_i32_i24_sdwa v8, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 121022f9 0a0a061b
	v_mul_i32_i24_sdwa v27, sext(v27), sext(v17) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123622f9 0b0b061b
	v_mul_i32_i24_sdwa v12, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 121814f9 09090607
	v_mul_i32_i24_sdwa v7, sext(v7), sext(v10) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 120e14f9 0b0b0607
	v_mul_i32_i24_sdwa v17, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 122224f9 09090618
	v_add_nc_u32_e32 v22, v22, v26                              ; 4a2c3516
	v_fma_mix_f32 v20, v20, v6, neg(0) op_sel_hi:[1,0,0]        ; cc200014 8a020d14
	s_waitcnt lgkmcnt(1)                                        ; bf8cc17f
	v_mul_i32_i24_sdwa v26, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 123416f9 0a0a0615
	v_add_nc_u32_e32 v9, v9, v12                                ; 4a121909
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	v_mul_i32_i24_sdwa v5, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 120a26f9 09090617
	v_add_nc_u32_e32 v16, v16, v17                              ; 4a202310
	v_mul_i32_i24_sdwa v6, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 120c26f9 0a0a0617
	v_add3_u32 v22, v22, v8, v27                                ; d76d0016 046e1116
	v_mul_i32_i24_sdwa v27, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123626f9 08080617
	v_mul_i32_i24_sdwa v23, sext(v23), sext(v19) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122e26f9 0b0b0617
	v_add3_u32 v9, v9, v15, v7                                  ; d76d0009 041e1f09
	v_add_nc_u32_e32 v27, v27, v5                               ; 4a360b1b
	v_add3_u32 v9, v9, v22, v25                                 ; d76d0009 04662d09
	v_mul_i32_i24_sdwa v22, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 ; 122c24f9 0a0a0618
	v_mul_i32_i24_sdwa v24, sext(v24), sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 123024f9 0b0b0618
	v_mul_i32_i24_sdwa v25, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 ; 123216f9 09090615
	v_add3_u32 v27, v27, v6, v23                                ; d76d001b 045e0d1b
	v_add3_u32 v16, v16, v22, v24                               ; d76d0010 04622d10
	v_mul_i32_i24_sdwa v24, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 ; 123016f9 08080615
	v_mul_i32_i24_sdwa v21, sext(v21), sext(v11) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 ; 122a16f9 0b0b0615
	v_add_nc_u32_e32 v24, v24, v25                              ; 4a303318
	v_add3_u32 v24, v24, v26, v21                               ; d76d0018 04563518
	v_add3_u32 v24, v24, v16, v9                                ; d76d0018 04262118
	v_add_nc_u32_e32 v27, v27, v24                              ; 4a36311b
	v_cvt_f32_i32_e32 v27, v27                                  ; 7e360b1b
	v_mac_f32_e32 v4, v20, v27                                  ; 3e083714
BB18:
	v_add_nc_u32_e32 v14, 8, v14                                ; 4a1c1c88
	s_branch BB1                                                ; bf82fd1d
BB19:
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[4:5], -1                                ; be8425c1
	s_cmp_lt_u32 s9, s3                                         ; bf0a0309
	v_cndmask_b32_e64 v27, 0, v1, s[4:5]                        ; d501001b 00120280
	s_mov_b64 s[4:5], 1                                         ; be840481
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v27, v27, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff04b11b
	v_add_f32_dpp v27, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff044e1b
	v_add_f32_dpp v27, v27, v27 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05411b
	v_add_f32_dpp v27, v27, v27 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05401b
	v_permlanex16_b32 v26, v27, 0, 0                            ; d778001a 0201011b
	v_add_f32_e32 v27, v27, v26                                 ; 0636351b
	v_readlane_b32 s0, v27, 0                                   ; d7600000 0201011b
	v_add_f32_e32 v27, s0, v27                                  ; 06363600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v27, 63                                  ; d7600000 02017f1b
	s_cbranch_execz BB25                                        ; bf88000b
BB20:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0                    ; e0700000 00030080
BB25:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s1, s9, 1                                         ; 80018109
	v_cndmask_b32_e64 v27, 0, v2, s[10:11]                      ; d501001b 002a0480
	s_cmp_lt_u32 s1, s3                                         ; bf0a0301
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v27, v27, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff04b11b
	v_add_f32_dpp v27, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff044e1b
	v_add_f32_dpp v27, v27, v27 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05411b
	v_add_f32_dpp v27, v27, v27 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05401b
	v_permlanex16_b32 v26, v27, 0, 0                            ; d778001a 0201011b
	v_add_f32_e32 v27, v27, v26                                 ; 0636351b
	v_readlane_b32 s0, v27, 0                                   ; d7600000 0201011b
	v_add_f32_e32 v27, s0, v27                                  ; 06363600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v27, 63                                  ; d7600000 02017f1b
	s_cbranch_execz BB31                                        ; bf88000b
BB26:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0 offset:4           ; e0700004 00030080
BB31:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s1, s9, 2                                         ; 80018209
	v_cndmask_b32_e64 v27, 0, v3, s[10:11]                      ; d501001b 002a0680
	s_cmp_lt_u32 s1, s3                                         ; bf0a0301
	s_cselect_b64 s[10:11], -1, 0                               ; 858a80c1
	v_add_f32_dpp v27, v27, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff04b11b
	v_add_f32_dpp v27, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff044e1b
	v_add_f32_dpp v27, v27, v27 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05411b
	v_add_f32_dpp v27, v27, v27 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05401b
	v_permlanex16_b32 v26, v27, 0, 0                            ; d778001a 0201011b
	v_add_f32_e32 v27, v27, v26                                 ; 0636351b
	v_readlane_b32 s0, v27, 0                                   ; d7600000 0201011b
	v_add_f32_e32 v27, s0, v27                                  ; 06363600
	s_and_b64 exec, s[4:5], s[10:11]                            ; 87fe0a04
	v_readlane_b32 s0, v27, 63                                  ; d7600000 02017f1b
	s_cbranch_execz BB37                                        ; bf88000b
BB32:
	s_mov_b32 s10, s2                                           ; be8a0302
	s_movk_i32 s11, 0x8000                                      ; b00b8000
	s_load_dwordx4 s[12:15], s[10:11], 0x20                     ; f4080305 fa000020
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_lshl_b32 s0, s7, 4                                        ; 8f008407
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[12:15], s0 offset:8           ; e0700008 00030080
BB37:
	s_waitcnt_depctr 0xffe3                                     ; bfa3ffe3
	s_mov_b64 exec, -1                                          ; befe04c1
	s_or_saveexec_b64 s[10:11], -1                              ; be8a25c1
	s_add_u32 s9, s9, 3                                         ; 80098309
	v_cndmask_b32_e64 v27, 0, v4, s[10:11]                      ; d501001b 002a0880
	s_cmp_lt_u32 s9, s3                                         ; bf0a0309
	s_cselect_b64 s[8:9], -1, 0                                 ; 858880c1
	v_add_f32_dpp v27, v27, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff04b11b
	v_add_f32_dpp v27, v27, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff044e1b
	v_add_f32_dpp v27, v27, v27 row_half_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05411b
	v_add_f32_dpp v27, v27, v27 row_mirror row_mask:0xf bank_mask:0xf fi:1 ; 063636fa ff05401b
	v_permlanex16_b32 v26, v27, 0, 0                            ; d778001a 0201011b
	v_add_f32_e32 v27, v27, v26                                 ; 0636351b
	v_readlane_b32 s0, v27, 0                                   ; d7600000 0201011b
	v_add_f32_e32 v27, s0, v27                                  ; 06363600
	s_and_b64 exec, s[4:5], s[8:9]                              ; 87fe0804
	v_readlane_b32 s0, v27, 63                                  ; d7600000 02017f1b
	s_cbranch_execz BB43                                        ; bf88000a
BB38:
	s_movk_i32 s3, 0x8000                                       ; b0038000
	s_load_dwordx4 s[8:11], s[2:3], 0x20                        ; f4080201 fa000020
	s_lshl_b32 s7, s7, 4                                        ; 8f078407
	v_mul_f32_e64 v0, s0, s6                                    ; d5080000 02000c00
	s_waitcnt lgkmcnt(0)                                        ; bf8cc07f
	s_waitcnt_vscnt null, 0x0                                   ; bbfd0000
	buffer_store_dword v0, off, s[8:11], s7 offset:12           ; e070000c 07020080
BB43:
	s_endpgm                                                    ; bf810000

