Step C (packed-f16 QK) — rejected 2026-11-17. Ledger: karpathy/AlreadyTried.md.
pv_*  = shipped step B (K/V f16 LDS + packed-f16 PV)
qk_*  = pv + -DFA_F16_QK: Q held as f16vec4, per-lane partial t = q0*k0, fma x3
        in f16vec4 (4 products per component), widened to f32 before the
        16-lane clustered add. The QK code was removed from the shader after
        the verdict; the diagnostic variant below shows the shape.
qonly = pv + f16 Q but f32 dot (qk_variant_qonly.comp): isolates Q rounding.
qs6   = P36_TEST_QSCALE=6 (peaked softmax).
