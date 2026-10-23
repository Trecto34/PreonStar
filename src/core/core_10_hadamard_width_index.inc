static int p36_hadamard_width_index(uint32_t width) {
    for (uint32_t i = 0; i < g_p36_hadamard.n_widths; i++) {
        if (g_p36_hadamard.widths[i] == width) return (int)i;
    }
    return -1;
}

static const float *p36_hadamard_signs(uint32_t width) {
    int i = p36_hadamard_width_index(width);
    return i < 0 ? NULL : g_p36_hadamard.signs[i];
}

static P36_MAYBE_UNUSED bool p36_hadamard_rotate_enabled(const p36_tensor *t) {
    return g_p36_hadamard.enabled && t && t->hadamard_rotate;
}

/* Unnormalized in-place Sylvester Walsh-Hadamard butterfly of length n. */
