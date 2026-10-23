static void p36_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

static bool tensor_nbytes(uint32_t type, uint64_t elements, uint64_t *bytes);
static void p36_dequantize_row_q8_0(const p36_block_q8_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_bf16(const uint16_t *x, float *y, uint64_t k);
static void p36_dequantize_row_q2_0(const p36_block_q2_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_pq2_0(const p36_block_pq2_0 *x, float *y, uint64_t k);
static void p36_dequantize_row_ptq1_0(const p36_block_ptq1_0 *x, float *y, uint64_t k);


static void p36_dequantize_row_q2_k(const p36_block_q2_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q3_k(const p36_block_q3_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q4_k(const p36_block_q4_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q5_k(const p36_block_q5_k *x, float *y, uint64_t k);
static void p36_dequantize_row_q6_k(const p36_block_q6_k *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_xxs(const p36_block_iq2_xxs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_xs(const p36_block_iq2_xs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq3_xxs(const p36_block_iq3_xxs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq1_s(const p36_block_iq1_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq4_nl(const p36_block_iq4_nl *x, float *y, uint64_t k);
static void p36_dequantize_row_iq2_s(const p36_block_iq2_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq3_s(const p36_block_iq3_s *x, float *y, uint64_t k);
static void p36_dequantize_row_iq4_xs(const p36_block_iq4_xs *x, float *y, uint64_t k);
static void p36_dequantize_row_iq1_m(const p36_block_iq1_m *x, float *y, uint64_t k);

static const uint8_t *p36_iq2xxs_grid_at(uint32_t idx) {
    return p36_iq2xxs_grid + (size_t)idx * 8u;
}

static const uint8_t *p36_iq2xs_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq2xs_grid[idx];
}

static const uint8_t *p36_iq2s_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq2s_grid[idx];
}

static const uint8_t *p36_iq3s_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq3s_grid[idx];
}

static const uint8_t *p36_iq3xxs_grid_at(uint32_t idx) {
    return (const uint8_t *)&p36_iq3xxs_grid[idx];
}

static bool p36_dequantize_row_from_ptr(uint32_t type, const uint8_t *src, float *dst, uint32_t n) {
    switch (type) {
    case P36_TENSOR_F32:
        memcpy(dst, src, (size_t)n * sizeof(*dst));
        return true;
    case P36_TENSOR_F16: {
        const uint16_t *h = (const uint16_t *)src;
        for (uint32_t i = 0; i < n; i++) dst[i] = p36_f16_to_f32(h[i]);
        return true;
    }
    case P36_TENSOR_BF16:
        p36_dequantize_row_bf16((const uint16_t *)src, dst, n);
        return true;
    case P36_TENSOR_Q8_0:
        p36_dequantize_row_q8_0((const p36_block_q8_0 *)src, dst, n);
        return true;
    case P36_TENSOR_Q2_0:
        p36_dequantize_row_q2_0((const p36_block_q2_0 *)src, dst, n);
        return true;
    case P36_TENSOR_PQ2_0:
        p36_dequantize_row_pq2_0((const p36_block_pq2_0 *)src, dst, n);
        return true;
    case P36_TENSOR_PTQ1_0:
        p36_dequantize_row_ptq1_0((const p36_block_ptq1_0 *)src, dst, n);
        return true;
    case P36_TENSOR_Q2_K:
        p36_dequantize_row_q2_k((const p36_block_q2_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q3_K:
        p36_dequantize_row_q3_k((const p36_block_q3_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q4_K:
        p36_dequantize_row_q4_k((const p36_block_q4_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q5_K:
        p36_dequantize_row_q5_k((const p36_block_q5_k *)src, dst, n);
        return true;
    case P36_TENSOR_Q6_K:
        p36_dequantize_row_q6_k((const p36_block_q6_k *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_XXS:
        p36_dequantize_row_iq2_xxs((const p36_block_iq2_xxs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_XS:
        p36_dequantize_row_iq2_xs((const p36_block_iq2_xs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ3_XXS:
        p36_dequantize_row_iq3_xxs((const p36_block_iq3_xxs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ1_S:
        p36_dequantize_row_iq1_s((const p36_block_iq1_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ4_NL:
        p36_dequantize_row_iq4_nl((const p36_block_iq4_nl *)src, dst, n);
        return true;
    case P36_TENSOR_IQ2_S:
        p36_dequantize_row_iq2_s((const p36_block_iq2_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ3_S:
        p36_dequantize_row_iq3_s((const p36_block_iq3_s *)src, dst, n);
        return true;
    case P36_TENSOR_IQ4_XS:
        p36_dequantize_row_iq4_xs((const p36_block_iq4_xs *)src, dst, n);
        return true;
    case P36_TENSOR_IQ1_M:
        p36_dequantize_row_iq1_m((const p36_block_iq1_m *)src, dst, n);
        return true;
    default:
        return false;
    }
}

bool p36_quant_dequantize(uint32_t type, const void *src, float *out, uint32_t n) {
    return src && out && p36_dequantize_row_from_ptr(type, src, out, n);
}

void p36_quant_pack_iq1_grid(void *dst) {
    memcpy(dst, p36_iq1s_grid, sizeof(p36_iq1s_grid));
}

static void p36_dequantize_row_q8_0(const p36_block_q8_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK8_0) != 0) p36_die("q8_0 row size mismatch");
    nb = k / P36_QK8_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK8_0; j++) *y++ = d * (float)x[i].qs[j];
    }
}

static void p36_dequantize_row_bf16(const uint16_t *x, float *y, uint64_t k) {
    for (uint64_t i = 0; i < k; i++) {
        uint32_t bits = (uint32_t)x[i] << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        y[i] = f;
    }
}

static void p36_dequantize_row_q2_0(const p36_block_q2_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK2_0) != 0) p36_die("q2_0 row size mismatch");
    nb = k / P36_QK2_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK2_0; j++) {
            uint8_t q = (uint8_t)((x[i].qs[j / 4u] >> ((j % 4u) * 2u)) & 0x03u);
            *y++ = (float)((int)q - 1) * d;
        }
    }
}

static void p36_dequantize_row_pq2_0(const p36_block_pq2_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_PQ2_0) != 0) p36_die("pq2_0 row size mismatch");
    nb = k / P36_QK_PQ2_0;
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t j = 0; j < P36_QK_PQ2_0; j++) {
            uint8_t q = (uint8_t)((x[i].qs[j / 4u] >> ((j % 4u) * 2u)) & 0x03u);
            *y++ = (float)((int)q - 1) * d;
        }
    }
}

static void p36_dequantize_row_ptq1_0(const p36_block_ptq1_0 *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_PTQ1_0) != 0) p36_die("ptq1_0 row size mismatch");
    nb = k / P36_QK_PTQ1_0;
    const uint32_t pow3[5] = {1,3,9,27,81};
    for (uint64_t i = 0; i < nb; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t p = 0; p < P36_QK_PTQ1_0; p++) {
            uint8_t byte;
            uint8_t n;
            if (p < 80) {
                byte = x[i].qs[p % 16];
                n = p / 16;
            } else if (p < 120) {
                uint32_t l = p - 80;
                byte = x[i].qs[16 + (l % 8)];
                n = l / 8;
            } else {
                uint32_t l = p - 120;
                byte = x[i].qh[l % 2];
                n = l / 2;
            }
            uint32_t qq = (byte * pow3[n]) & 0xffu;
            uint32_t xi = (qq * 3u) >> 8u;
            *y++ = ((float)xi - 1.0f) * d;
        }
    }
}

static void p36_dequantize_row_q2_k(const p36_block_q2_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q2_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t sc = x[i].scales[is++];
                float dl = d * (float)(sc & 0x0f);
                float ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l] >> shift) & 3u) - ml;

                sc = x[i].scales[is++];
                dl = d * (float)(sc & 0x0f);
                ml = min * (float)(sc >> 4);
                for (int l = 0; l < 16; l++) *y++ = dl * (float)((q[l + 16] >> shift) & 3u) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

static void p36_dequantize_row_q3_k(const p36_block_q3_k *x, float *y, uint64_t k) {
    const uint32_t mask2 = 0x03030303u;
    const uint32_t mask4 = 0x0f0f0f0fu;
    if ((k % P36_QK_K) != 0) p36_die("q3_k row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        uint32_t aux[4] = {0};
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & mask4) | (((tmp >> 4) & mask2) << 4);
        aux[3] = ((aux[1] >> 4) & mask4) | (((tmp >> 6) & mask2) << 4);
        aux[0] = (aux[0] & mask4) | (((tmp >> 0) & mask2) << 4);
        aux[1] = (aux[1] & mask4) | (((tmp >> 2) & mask2) << 4);

        const int8_t *scales = (const int8_t *)aux;
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        float d = p36_f16_to_f32(x[i].d);
        uint8_t m = 1;
        int is = 0;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l] >> shift) & 3u) - ((hm[l] & m) ? 0 : 4));
                dl = d * (float)(scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * (float)((int)((q[l + 16] >> shift) & 3u) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void p36_dequantize_row_q4_k(const p36_block_q4_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q4_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        for (uint32_t j = 0; j < P36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            p36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            p36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void p36_dequantize_row_q5_k(const p36_block_q5_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q5_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        float d = p36_f16_to_f32(x[i].d);
        float min = p36_f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (uint32_t j = 0; j < P36_QK_K; j += 64) {
            uint8_t sc;
            uint8_t m;
            float d1;
            float d2;
            float m1;
            float m2;
            p36_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            d1 = d * (float)sc;
            m1 = min * (float)m;
            p36_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            d2 = d * (float)sc;
            m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)((ql[l] & 0x0f) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

static void p36_dequantize_row_q6_k(const p36_block_q6_k *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("q6_k row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (uint32_t n = 0; n < P36_QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void p36_dequantize_row_iq2_xxs(const p36_block_iq2_xxs *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq2_xxs row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ++ib32) {
            uint32_t aux32[2];
            const uint8_t *aux8;
            memcpy(aux32, x[i].qs + 4 * ib32, 2 * sizeof(uint32_t));
            aux8 = (const uint8_t *)aux32;
            {
                const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t *grid = p36_iq2xxs_grid_at(aux8[l]);
                    const uint8_t signs = p36_ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127u];
                    for (int j = 0; j < 8; ++j) {
                        y[j] = db * (float)grid[j] * ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    }
                    y += 8;
                }
            }
        }
    }
}

static void p36_dequantize_row_iq2_xs(const p36_block_iq2_xs *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq2_xs row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        float d = p36_f16_to_f32(x[i].d);
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            float db[2] = {
                d * (0.5f + (float)(x[i].scales[ib] & 15u)) * 0.25f,
                d * (0.5f + (float)(x[i].scales[ib] >> 4)) * 0.25f,
            };
            for (uint32_t l = 0; l < 4; l++) {
                uint16_t qs = x[i].qs[4 * ib + l];
                const uint8_t *grid = p36_iq2xs_grid_at(qs & 511u);
                uint8_t signs = p36_ksigns_iq2xs[qs >> 9];
                for (uint32_t j = 0; j < 8; j++)
                    y[j] = db[l / 2] * (float)grid[j] *
                        ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                y += 8;
            }
        }
    }
}

static void p36_dequantize_row_iq2_s(const p36_block_iq2_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq2_s row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + P36_QK_K / 8;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ++ib32) {
            float db0 = d * (0.5f + (float)(x[i].scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (float)(x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = l < 2 ? db0 : db1;
                const uint8_t *grid = p36_iq2s_grid_at((uint32_t)(qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * (float)grid[j] * ((signs[l] & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void p36_dequantize_row_iq3_xxs(const p36_block_iq3_xxs *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq3_xxs row size mismatch");
    for (uint64_t i = 0; i < k / P36_QK_K; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_and_signs = qs + P36_QK_K / 4;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ib32++) {
            uint32_t aux;
            memcpy(&aux, scales_and_signs + 4 * ib32, sizeof(aux));
            const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const uint8_t signs = p36_ksigns_iq2xs[(aux >> (7 * l)) & 127u];
                const uint8_t *grid1 = p36_iq3xxs_grid_at(qs[2 * l]);
                const uint8_t *grid2 = p36_iq3xxs_grid_at(qs[2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j] = db * (float)grid1[j] *
                        ((signs & p36_kmask_iq2xs[j]) ? -1.0f : 1.0f);
                    y[j + 4] = db * (float)grid2[j] *
                        ((signs & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void p36_dequantize_row_iq3_s(const p36_block_iq3_s *x, float *y, uint64_t k) {
    uint64_t nb;
    if ((k % P36_QK_K) != 0) p36_die("iq3_s row size mismatch");
    nb = k / P36_QK_K;
    for (uint64_t i = 0; i < nb; i++) {
        const float d = p36_f16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (uint32_t ib32 = 0; ib32 < P36_QK_K / 32; ib32 += 2) {
            const float db1 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] & 0x0f));
            const float db2 = d * (1.0f + 2.0f * (float)(x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[0] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[0] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db1 * (float)grid1[j] * ((signs[l] & p36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db1 * (float)grid2[j] * ((signs[l] & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid1 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 0] | ((qh[1] << (8 - 2 * l)) & 256)));
                const uint8_t *grid2 = p36_iq3s_grid_at((uint32_t)(qs[2 * l + 1] | ((qh[1] << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db2 * (float)grid1[j] * ((signs[l] & p36_kmask_iq2xs[j + 0]) ? -1.0f : 1.0f);
                    y[j + 4] = db2 * (float)grid2[j] * ((signs[l] & p36_kmask_iq2xs[j + 4]) ? -1.0f : 1.0f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

static void p36_dequantize_row_iq4_xs(const p36_block_iq4_xs *x, float *y, uint64_t k) {
    static const int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    if ((k % P36_QK_K) != 0) p36_die("iq4_xs row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        const uint8_t *qs = x[b].qs;
        const float d = p36_f16_to_f32(x[b].d);
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            int ls = ((x[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 15) |
                     (((x[b].scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (float)(ls - 32);
            for (uint32_t j = 0; j < 16; j++) {
                y[j] = dl * values[qs[j] & 15];
                y[j + 16] = dl * values[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

static void p36_dequantize_row_iq1_s(const p36_block_iq1_s *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq1_s row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        float d = p36_f16_to_f32(x[b].d);
        const uint8_t *qs = x[b].qs;
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            uint16_t qh = x[b].qh[ib];
            float dl = d * (float)(2u * ((qh >> 12) & 7u) + 1u);
            float delta = qh & 0x8000u ? -0.125f : 0.125f;
            for (uint32_t l = 0; l < 4; l++) {
                uint32_t idx = qs[l] | (((qh >> (3u * l)) & 7u) << 8);
                const int8_t *grid = (const int8_t *)&p36_iq1s_grid[idx];
                for (uint32_t j = 0; j < 8; j++) y[j] = dl * ((float)grid[j] + delta);
                y += 8;
            }
            qs += 4;
        }
    }
}

static void p36_dequantize_row_iq4_nl(const p36_block_iq4_nl *x, float *y, uint64_t k) {
    static const int8_t values[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113,
    };
    if ((k % 32u) != 0) p36_die("iq4_nl row size mismatch");
    for (uint64_t b = 0; b < k / 32u; b++) {
        float d = p36_f16_to_f32(x[b].d);
        for (uint32_t j = 0; j < 16; j++) {
            y[j] = d * (float)values[x[b].qs[j] & 15u];
            y[j + 16] = d * (float)values[x[b].qs[j] >> 4];
        }
        y += 32;
    }
}

static void p36_dequantize_row_iq1_m(const p36_block_iq1_m *x, float *y, uint64_t k) {
    if ((k % P36_QK_K) != 0) p36_die("iq1_m row size mismatch");
    for (uint64_t b = 0; b < k / P36_QK_K; b++) {
        const uint16_t *sc = (const uint16_t *)x[b].scales;
        uint16_t dh = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                      ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        float d = p36_f16_to_f32(dh);
        const uint8_t *qs = x[b].qs;
        const uint8_t *qh = x[b].qh;
        for (uint32_t ib = 0; ib < P36_QK_K / 32; ib++) {
            float dl[2] = {
                d * (float)(2 * ((sc[ib / 2] >> (6 * (ib & 1))) & 7) + 1),
                d * (float)(2 * ((sc[ib / 2] >> (6 * (ib & 1) + 3)) & 7) + 1),
            };
            for (uint32_t l = 0; l < 4; l++) {
                uint32_t hi = l & 1 ? qh[l / 2] << 4 : qh[l / 2] << 8;
                const int8_t *grid = (const int8_t *)&p36_iq1s_grid[qs[l] | (hi & 0x700)];
                float delta = qh[l / 2] & (l & 1 ? 0x80 : 0x08) ? -0.125f : 0.125f;
                for (uint32_t j = 0; j < 8; j++) y[j] = dl[l / 2] * ((float)grid[j] + delta);
                y += 8;
            }
            qs += 4;
            qh += 2;
        }
    }
}

static bool p36_tensor_row_to_float(const p36_model *m, const p36_tensor *t, uint64_t row, float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t row_count;
    uint64_t row_bytes;
    if (!m || !t || !dst || t->ndim == 0 || t->dim[0] != (uint64_t)n) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &row_bytes)) return false;
    src = m->map + t->abs_offset + row * row_bytes;
    return p36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool p36_tensor_row_ptr(const p36_model *m, const p36_tensor *t, uint64_t row, const uint8_t **src, uint64_t *row_bytes) {
    uint64_t row_count;
    uint64_t got_row_bytes;
    if (!m || !t || !src || t->ndim == 0) return false;
    row_count = t->elements / t->dim[0];
    if (row >= row_count || !tensor_nbytes(t->type, t->dim[0], &got_row_bytes)) return false;
    *src = m->map + t->abs_offset + row * got_row_bytes;
    if (row_bytes) *row_bytes = got_row_bytes;
    return true;
}

static bool p36_tensor_expert_row_ptr(const p36_model *m, const p36_tensor *t, uint32_t expert, uint32_t row,
                                      const uint8_t **src, uint64_t *row_bytes) {
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !src || t->ndim != 3) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    return p36_tensor_row_ptr(m, t, flat_row, src, row_bytes);
}

static P36_MAYBE_UNUSED bool p36_tensor_expert_row_to_float(const p36_model *m, const p36_tensor *t, uint32_t expert, uint32_t row,
                                           float *dst, uint32_t n) {
    const uint8_t *src;
    uint64_t rows_per_expert;
    uint64_t flat_row;
    if (!m || !t || !dst || t->ndim != 3 || t->dim[0] != (uint64_t)n) return false;
    rows_per_expert = t->dim[1];
    if (expert >= t->dim[2] || row >= rows_per_expert) return false;
    flat_row = (uint64_t)expert * rows_per_expert + row;
    if (!p36_tensor_row_ptr(m, t, flat_row, &src, NULL)) return false;
    return p36_dequantize_row_from_ptr(t->type, src, dst, n);
}

static bool p36_tensor_get_plain(const p36_model *m, const p36_tensor *t, float *dst, uint32_t n) {
    if (!m || !t || !dst || t->ndim != 1 || t->dim[0] != (uint64_t)n) return false;
    return p36_dequantize_row_from_ptr(t->type, m->map + t->abs_offset, dst, n);
}

static float p36_tensor_scalar_or(const p36_model *m, const p36_tensor *t, float fallback) {
    float v = fallback;
    if (!t) return fallback;
    if (!p36_tensor_get_plain(m, t, &v, 1)) return fallback;
    return v;
}

static float p36_tensor_index_or(const p36_model *m, const p36_tensor *t, uint32_t idx, float fallback) {
    if (!m || !t || t->ndim != 1 || idx >= t->dim[0]) return fallback;
    if (t->type == P36_TENSOR_F32) {
        const float *src = (const float *)(m->map + t->abs_offset);
        return src[idx];
    }
    if (t->type == P36_TENSOR_F16) {
        const uint16_t *src = (const uint16_t *)(m->map + t->abs_offset);
        return p36_f16_to_f32(src[idx]);
    }
    return fallback;
}

static void p36_scale_inplace(float *x, uint32_t n, float scale) {
    if (!x || scale == 1.0f) return;
    for (uint32_t i = 0; i < n; i++) x[i] *= scale;
}

static bool p36_tensor_type_supports_q8k_dot(uint32_t type) {
    switch (type) {
    case P36_TENSOR_BF16:
    case P36_TENSOR_Q2_0:
    case P36_TENSOR_PQ2_0:
    case P36_TENSOR_PTQ1_0:
    case P36_TENSOR_Q2_K:
    case P36_TENSOR_Q3_K:
    case P36_TENSOR_Q4_K:
    case P36_TENSOR_Q5_K:
    case P36_TENSOR_Q6_K:
    case P36_TENSOR_IQ2_XXS:
    case P36_TENSOR_IQ2_XS:
    case P36_TENSOR_IQ3_XXS:
    case P36_TENSOR_IQ1_S:
    case P36_TENSOR_IQ4_NL:
    case P36_TENSOR_IQ2_S:
    case P36_TENSOR_IQ3_S:
    case P36_TENSOR_IQ4_XS:
    case P36_TENSOR_IQ1_M:
        return true;
    default:
        return false;
    }
}

static bool p36_tensor_type_supports_q8_0_dot(uint32_t type) {
    return type == P36_TENSOR_Q8_0;
}

static bool p36_tensor_type_supports_prequant_dot(uint32_t type) {
    return p36_tensor_type_supports_q8k_dot(type) || p36_tensor_type_supports_q8_0_dot(type);
}

static bool p36_tensor_pair_can_fuse(const p36_tensor *t0,
                                     const p36_tensor *t1,
                                     uint32_t ndim,
                                     uint32_t in_dim,
                                     uint32_t out_dim) {
    if (!t0 || !t1 || t0->type != t1->type || t0->ndim != ndim || t1->ndim != ndim) return false;
    if (t0->dim[0] != in_dim || t1->dim[0] != in_dim || t0->dim[1] != out_dim || t1->dim[1] != out_dim) return false;
    if (ndim == 3 && t0->dim[2] != t1->dim[2]) return false;
    return p36_tensor_type_supports_prequant_dot(t0->type);
}

static bool p36_tensor_row_dot_q8k(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    const p36_block_q8_k *q = (const p36_block_q8_k *)xq;
    float weights[P36_QK_K];
    double acc = 0.0;
    uint32_t block_bytes;
    if (!row || !xq || !out || (n % P36_QK_K) != 0) return false;
    if (type >= sizeof(gguf_types) / sizeof(gguf_types[0]) || !gguf_types[type].block_bytes) return false;
    block_bytes = gguf_types[type].block_bytes *
                  (P36_QK_K / gguf_types[type].block_elems);
    for (uint32_t block = 0; block < n / P36_QK_K; block++) {
        if (!p36_dequantize_row_from_ptr(type, row + (uint64_t)block * block_bytes,
                                         weights, P36_QK_K)) return false;
        for (uint32_t i = 0; i < P36_QK_K; i++)
            acc += (double)weights[i] * (double)(q[block].d * (float)q[block].qs[i]);
    }
    *out = (float)acc;
    return true;
}

static bool p36_tensor_row_dot_q8k_pair(uint32_t type,
                                        const uint8_t *row0,
                                        const uint8_t *row1,
                                        const uint8_t *xq,
                                        uint32_t n,
                                        float *out0,
                                        float *out1) {
    if (!row0 || !row1 || !xq || !out0 || !out1 || (n % P36_QK_K) != 0) return false;
    return p36_tensor_row_dot_q8k(type, row0, xq, n, out0) &&
           p36_tensor_row_dot_q8k(type, row1, xq, n, out1);
}

static bool p36_tensor_row_dot_prequant(uint32_t type, const uint8_t *row, const uint8_t *xq, uint32_t n, float *out) {
    float acc = 0.0f;
    if (type == P36_TENSOR_Q8_0) {
        if (!row || !xq || !out || (n % P36_QK8_0) != 0) return false;
        acc = p36_quant_dot_q8_0(row, xq, (int)n);
        *out = acc;
        return true;
    }
    return p36_tensor_row_dot_q8k(type, row, xq, n, out);
}

static bool p36_tensor_row_dot_prequant_pair(uint32_t type,
                                             const uint8_t *row0,
                                             const uint8_t *row1,
                                             const uint8_t *xq,
                                             uint32_t n,
                                             float *out0,
                                             float *out1) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (type == P36_TENSOR_Q8_0) {
        if (!row0 || !row1 || !xq || !out0 || !out1 || (n % P36_QK8_0) != 0) return false;
        acc0 = p36_quant_dot_q8_0(row0, xq, (int)n);
        acc1 = p36_quant_dot_q8_0(row1, xq, (int)n);
        *out0 = acc0;
        *out1 = acc1;
        return true;
    }
    return p36_tensor_row_dot_q8k_pair(type, row0, row1, xq, n, out0, out1);
}

typedef enum {
    P36_ACTIVATION_QUANT_NONE,
    P36_ACTIVATION_QUANT_Q8_K,
    P36_ACTIVATION_QUANT_Q8_0,
} p36_activation_quant_kind;

static p36_activation_quant_kind p36_activation_quant_kind_for_type(uint32_t type) {
    if (p36_tensor_type_supports_q8_0_dot(type)) return P36_ACTIVATION_QUANT_Q8_0;
    if (p36_tensor_type_supports_q8k_dot(type)) return P36_ACTIVATION_QUANT_Q8_K;
    return P36_ACTIVATION_QUANT_NONE;
}

static bool p36_activation_quant_valid_dim(p36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case P36_ACTIVATION_QUANT_NONE:
        return true;
    case P36_ACTIVATION_QUANT_Q8_K:
        return (n % P36_QK_K) == 0;
    case P36_ACTIVATION_QUANT_Q8_0:
        return (n % P36_QK8_0) == 0;
    default:
        return false;
    }
}

static uint32_t p36_activation_quant_row_bytes(p36_activation_quant_kind kind, uint32_t n) {
    switch (kind) {
    case P36_ACTIVATION_QUANT_Q8_K:
        return p36_q8k_row_bytes(n);
    case P36_ACTIVATION_QUANT_Q8_0:
        return p36_q8_0_row_bytes(n);
    default:
        return 0;
    }
}

static bool p36_quantize_activation_row(p36_activation_quant_kind kind, const float *x, uint8_t *xq, uint32_t n) {
    if (kind == P36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || !p36_activation_quant_valid_dim(kind, n)) return false;
    if (kind == P36_ACTIVATION_QUANT_Q8_0) p36_quant_q8_0(x, xq, (int64_t)n);
    else p36_quant_q8_k(x, xq, (int64_t)n);
    return true;
}

typedef struct {
    const float *x;
    uint8_t *xq;
    p36_activation_quant_kind kind;
    uint32_t in_dim;
    uint32_t row_bytes;
} p36_quantize_activation_batch_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_tensor *t;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} p36_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} p36_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const p36_engine *e;
    const p36_tensor *t;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale;
} p36_expert_matmul_batch_ctx;

typedef struct {
    float *out0;
    float *out1;
    const p36_engine *e;
    const p36_tensor *t0;
    const p36_tensor *t1;
    uint32_t expert;
    const float *x;
    const uint8_t *xq;
    uint32_t n_tok;
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t xq_row_bytes;
    float scale0;
    float scale1;
} p36_expert_matmul_pair_batch_ctx;

typedef struct {
    float *out;
    const float *w;
    const float *x;
    uint32_t n_tok;
    uint32_t n;
} p36_vector_dot_batch_ctx;

typedef struct {
    float *x;
    uint32_t rows;
    uint32_t cols;
    float scale;
} p36_scale_rows_ctx;

typedef struct {
    float *dst;
    const float *a;
    const float *b;
    uint32_t rows;
    uint32_t cols;
} p36_add_rows_ctx;

typedef struct {
    float *dst;
    const float *src;
    const float *weight;
    uint32_t rows;
    uint32_t cols;
    float eps;
} p36_rms_norm_rows_ctx;

typedef struct {
    float *out;
    const float *gate;
    const float *up;
    uint32_t rows;
    uint32_t cols;
} p36_swiglu_rows_ctx;

static void p36_quantize_activation_batch_worker(void *opaque, uint64_t row0, uint64_t row1) {
    p36_quantize_activation_batch_ctx *ctx = (p36_quantize_activation_batch_ctx *)opaque;
    for (uint64_t t = row0; t < row1; t++) {
        p36_quantize_activation_row(ctx->kind,
                                    ctx->x + t * ctx->in_dim,
                                    ctx->xq + t * ctx->row_bytes,
                                    ctx->in_dim);
    }
}

static bool p36_quantize_activation_batch(p36_activation_quant_kind kind,
                                          const float *x,
                                          uint8_t *xq,
                                          uint32_t n_tok,
                                          uint32_t in_dim,
                                          uint32_t n_threads) {
    p36_quantize_activation_batch_ctx ctx;
    if (kind == P36_ACTIVATION_QUANT_NONE) return true;
    if (!x || !xq || n_tok == 0 || !p36_activation_quant_valid_dim(kind, in_dim)) return false;
    ctx.x = x;
    ctx.xq = xq;
    ctx.kind = kind;
    ctx.in_dim = in_dim;
    ctx.row_bytes = p36_activation_quant_row_bytes(kind, in_dim);
    p36_parallel_for_rows(n_tok, 2, n_threads, p36_quantize_activation_batch_worker, &ctx);
    return true;
}

static bool p36_quantize_activation_batch_for_type(uint32_t type,
                                                   const float *x,
                                                   uint8_t *xq,
                                                   uint32_t n_tok,
                                                   uint32_t in_dim,
                                                   uint32_t n_threads) {
    return p36_quantize_activation_batch(p36_activation_quant_kind_for_type(type), x, xq, n_tok, in_dim, n_threads);
}

