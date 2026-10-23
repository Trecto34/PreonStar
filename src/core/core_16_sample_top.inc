static uint64_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(str_i32_table *t, uint64_t expected) {
    t->cap = next_pow2(expected * 2 + 16);
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(str_i32_table *t, p36_str key, int value) {
    uint64_t mask = t->cap - 1;
    uint64_t i = hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (p36_str_eq(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1) & mask;
    }
    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
}

static bool table_get(const str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    uint64_t mask;
    uint64_t i;
    if (t->cap == 0) return false;
    mask = t->cap - 1;
    i = hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        p36_str key = t->entry[i].key;
        if (key.len == len && memcmp(key.ptr, ptr, len) == 0) {
            *value = t->entry[i].value;
            return true;
        }
        i = (i + 1) & mask;
    }
    return false;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) return b;
    {
        uint32_t n = 0;
        for (uint32_t x = 0; x < 256; x++) {
            if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || x >= 174) continue;
            if (x == b) return 256 + n;
            n++;
        }
    }
    return b;
}

static char *byte_encode(p36_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4 + 1);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const p36_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1 + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    int rank = -1;
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1, b->ptr, (size_t)b->len);
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const p36_vocab *vocab, p36_str raw_piece, token_vec *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);
    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));

    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }

    for (;;) {
        int best_i = -1;
        int best_rank = INT32_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        {
            owned_str merged;
            merged.len = sym[best_i].len + sym[best_i + 1].len;
            merged.ptr = xmalloc((size_t)merged.len);
            memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
            memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
            free(sym[best_i].ptr);
            free(sym[best_i + 1].ptr);
            sym[best_i] = merged;
            for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
            n_sym--;
        }
    }

    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            p36_tokens_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    p36_tokens_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }
    free(sym);
    free(encoded);
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool qwen_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    if ((uint8_t)s[pos] < 128) return ascii_alpha((uint8_t)s[pos]);
    return true;
}

static bool qwen_punct_symbol_at(const char *s, uint64_t len, uint64_t pos) {
    uint8_t c;
    (void)len;
    c = (uint8_t)s[pos];
    return c < 128 && !ascii_alpha(c) && !ascii_digit(c) && !ascii_space(c);
}

static bool qwen_match_suffix(const char *s, uint64_t len, uint64_t pos, uint64_t *out_pos) {
    static const char *suffixes[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
    if (pos >= len || s[pos] != '\'') return false;
    for (size_t i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); i++) {
        size_t n = strlen(suffixes[i]);
        if (pos + n > len) continue;
        if (strncasecmp(s + pos, suffixes[i], n) == 0) {
            *out_pos = pos + n;
            return true;
        }
    }
    return false;
}

static uint64_t qwen_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && qwen_letter_like_at(s, len, pos)) pos = next_utf8_char(s, len, pos);
    return pos;
}

static void bpe_tokenize_text(const p36_vocab *vocab, const char *text, token_vec *out) {
    uint64_t len = strlen(text);
    uint64_t pos = 0;
    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];

        if (c == ' ' && pos + 1 < len) {
            uint64_t next = pos + 1;
            uint8_t nc = (uint8_t)text[next];
            if (ascii_digit(nc)) {
                int ndigits = 0;
                pos++;
                while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                    pos++;
                    ndigits++;
                }
            } else if (qwen_letter_like_at(text, len, next) ||
                       (!ascii_newline(nc) && !ascii_digit(nc) && !qwen_letter_like_at(text, len, next) &&
                        !ascii_space(nc) && next + 1 < len && qwen_letter_like_at(text, len, next + 1))) {
                uint64_t after = 0;
                pos++;
                if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                pos = qwen_consume_letters(text, len, pos);
                if (qwen_match_suffix(text, len, pos, &after)) pos = after;
            } else if (qwen_punct_symbol_at(text, len, next)) {
                pos++;
                while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
                while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
            }
        } else if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (qwen_letter_like_at(text, len, pos) ||
                   (!ascii_newline(c) && !ascii_digit(c) && !qwen_letter_like_at(text, len, pos) &&
                    !ascii_space(c) && pos + 1 < len && qwen_letter_like_at(text, len, pos + 1))) {
            uint64_t after = 0;
            if (!qwen_letter_like_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            pos = qwen_consume_letters(text, len, pos);
            if (qwen_match_suffix(text, len, pos, &after)) pos = after;
        } else if ((c == ' ' && pos + 1 < len && qwen_punct_symbol_at(text, len, pos + 1)) || qwen_punct_symbol_at(text, len, pos)) {
            if (c == ' ') pos++;
            while (pos < len && qwen_punct_symbol_at(text, len, pos)) pos = next_utf8_char(text, len, pos);
            while (pos < len && ((uint8_t)text[pos] == '/' || ascii_newline((uint8_t)text[pos]))) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (qwen_letter_like_at(text, len, p) || qwen_punct_symbol_at(text, len, p))) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }

        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (p36_str){ text + start, pos - start }, out);
    }
}

static int vocab_lookup(const p36_vocab *vocab, const char *text) {
    int token = -1;
    if (!table_get(&vocab->token_to_id, text, strlen(text), &token)) {
        fprintf(stderr, "p36: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

static void vocab_load(p36_vocab *vocab, const p36_model *model) {
    p36_array_ref tokens;
    p36_array_ref token_types;
    p36_array_ref merges;
    p36_cursor c;
    p36_cursor types;
    int special_cap = 0;
    memset(vocab, 0, sizeof(*vocab));
    if (!model_get_array(model, "tokenizer.ggml.tokens", &tokens) || tokens.type != GGUF_VALUE_STRING || tokens.len > INT32_MAX) {
        p36_die("GGUF tokenizer token table is missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.token_type", &token_types) ||
        token_types.type != GGUF_VALUE_INT32 || token_types.len != tokens.len) {
        p36_die("GGUF tokenizer token types are missing or invalid");
    }
    if (!model_get_array(model, "tokenizer.ggml.merges", &merges) || merges.type != GGUF_VALUE_STRING) {
        p36_die("GGUF tokenizer merge table is missing or invalid");
    }
    vocab->n_vocab = (int)tokens.len;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    table_init(&vocab->token_to_id, tokens.len);
    c = cursor_at(model, tokens.data_pos);
    types = cursor_at(model, token_types.data_pos);
    for (int i = 0; i < vocab->n_vocab; i++) {
        uint32_t type;
        if (!cursor_string(&c, &vocab->token[i])) p36_die(c.error);
        if (!cursor_u32(&types, &type)) p36_die(types.error);
        table_put(&vocab->token_to_id, vocab->token[i], i);
        if (type == 3 || type == 4) {
            if (vocab->n_special == special_cap) {
                special_cap = special_cap ? special_cap * 2 : 32;
                vocab->special = xrealloc(vocab->special,
                    (size_t)special_cap * sizeof(vocab->special[0]));
            }
            vocab->special[vocab->n_special++] =
                (p36_special_token){.text = vocab->token[i], .id = i};
        }
    }
    table_init(&vocab->merge_rank, merges.len);
    c = cursor_at(model, merges.data_pos);
    for (uint64_t i = 0; i < merges.len; i++) {
        p36_str merge;
        if (!cursor_string(&c, &merge)) p36_die(c.error);
        table_put(&vocab->merge_rank, merge, (int)i);
    }
    vocab->bos_id = (int)required_u32(model, "tokenizer.ggml.bos_token_id");
    vocab->eos_id = (int)required_u32(model, "tokenizer.ggml.eos_token_id");
    vocab->add_bos = model_get_bool(model, "tokenizer.ggml.add_bos_token", &vocab->add_bos) ? vocab->add_bos : false;
    vocab->im_start_id = vocab_lookup(vocab, "<|im_start|>");
    vocab->im_end_id = vocab_lookup(vocab, "<|im_end|>");
    vocab->think_start_id = vocab_lookup(vocab, "<think>");
    vocab->think_end_id = vocab_lookup(vocab, "</think>");
    vocab->vision_start_id = vocab_lookup(vocab, "<|vision_start|>");
    vocab->vision_end_id = vocab_lookup(vocab, "<|vision_end|>");
    vocab->image_pad_id = vocab_lookup(vocab, "<|image_pad|>");
    vocab->video_pad_id = vocab_lookup(vocab, "<|video_pad|>");
}

static void vocab_free(p36_vocab *vocab) {
    free(vocab->token);
    free(vocab->special);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

void p36_tokenize_text(p36_engine *e, const char *text, p36_tokens *out) {
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

static bool special_token_at(const p36_vocab *vocab, const char *p,
                             size_t remain, int *token, size_t *len) {
    for (int i = 0; i < vocab->n_special; i++) {
        const p36_special_token *special = &vocab->special[i];
        if (special->text.len <= remain &&
            !memcmp(p, special->text.ptr, special->text.len)) {
            *token = special->id;
            *len = (size_t)special->text.len;
            return true;
        }
    }
    return false;
}

static void tokenize_span(const p36_vocab *vocab, const char *p, size_t n, token_vec *out) {
    char *tmp;
    if (!n) return;
    tmp = xmalloc(n + 1);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const p36_vocab *vocab, const char *text, token_vec *out) {
    const char *span;
    const char *p;
    const char *end;
    if (!text) text = "";
    span = text;
    p = text;
    end = text + strlen(text);
    while (p < end) {
        int token = -1;
        size_t len = 0;
        if (special_token_at(vocab, p, (size_t)(end - p), &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            p36_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(end - span), out);
}

void p36_tokenize_rendered_chat(p36_engine *e, const char *text, p36_tokens *out) {
    tokenize_rendered_chat_vocab(&e->vocab, text, out);
}

void p36_chat_apply_thinking_control(const char *content, bool *thinking) {
    if (!content || !thinking) return;
    if (strstr(content, "<|think_off|>")) *thinking = false;
    else if (strstr(content, "<|think_on|>")) *thinking = true;
}

static char *p36_chat_content_text(const char *text, bool controls) {
    static const char think_off[] = "<|think_off|>";
    static const char think_on[] = "<|think_on|>";
    const char *remove = NULL;
    size_t remove_len = 0;
    const char *start = text ? text : "";
    const char *end = start + strlen(start);

    if (controls && strstr(start, think_off)) {
        remove = think_off;
        remove_len = sizeof(think_off) - 1;
    } else if (controls && strstr(start, think_on)) {
        remove = think_on;
        remove_len = sizeof(think_on) - 1;
    }

    char *plain = xmalloc((size_t)(end - start) + 1);
    size_t used = 0;
    while (remove) {
        const char *p = strstr(start, remove);
        if (!p) break;
        size_t n = (size_t)(p - start);
        memcpy(plain + used, start, n);
        used += n;
        start = p + remove_len;
    }
    size_t tail = (size_t)(end - start);
    memcpy(plain + used, start, tail + 1);
    start = plain;
    end = plain + used + tail;
    while (start < end && isspace((unsigned char)*start)) start++;
    while (end > start && isspace((unsigned char)end[-1])) end--;
    char *trimmed = xmalloc((size_t)(end - start) + 1);
    memcpy(trimmed, start, (size_t)(end - start));
    trimmed[end - start] = '\0';
    free(plain);
    return trimmed;
}

void p36_chat_begin(p36_engine *e, p36_tokens *tokens) {
    if (e->vocab.add_bos && e->vocab.bos_id >= 0)
        p36_tokens_push(tokens, e->vocab.bos_id);
}

static void chat_append_open_role(p36_engine *e, token_vec *tokens, const char *role) {
    p36_tokenize_rendered_chat(e, "<|im_start|>", tokens);
    bpe_tokenize_text(&e->vocab, role, tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

static void chat_append_close_role(p36_engine *e, token_vec *tokens) {
    p36_tokenize_rendered_chat(e, "<|im_end|>", tokens);
    bpe_tokenize_text(&e->vocab, "\n", tokens);
}

void p36_chat_append_max_effort_prefix(p36_engine *e, p36_tokens *tokens) {
    bpe_tokenize_text(&e->vocab, P36_REASONING_EFFORT_MAX_PREFIX, tokens);
}

void p36_chat_append_message(p36_engine *e, p36_tokens *tokens, const char *role, const char *content) {
    if (!role) role = "user";
    if (!content) content = "";
    if (!strcmp(role, "developer")) role = "system";
    if (!strcmp(role, "tool") || !strcmp(role, "function")) {
        char *text = p36_chat_content_text(content, false);
        chat_append_open_role(e, tokens, "user");
        p36_tokenize_rendered_chat(e, "<tool_response>\n", tokens);
        p36_tokenize_rendered_chat(e, text, tokens);
        p36_tokenize_rendered_chat(e, "\n</tool_response>", tokens);
        chat_append_close_role(e, tokens);
        free(text);
        return;
    }
    bool controls = !strcmp(role, "system") || !strcmp(role, "user");
    char *text = p36_chat_content_text(content, controls);
    if (!strcmp(role, "system") && !text[0]) {
        free(text);
        return;
    }
    chat_append_open_role(e, tokens, role);
    p36_tokenize_rendered_chat(e, text, tokens);
    chat_append_close_role(e, tokens);
    free(text);
}

int p36_chat_append_multimodal_message(p36_engine *e, p36_tokens *tokens,
                                       const char *role, const char **parts,
                                       p36_vision_embedding *images, size_t image_count,
                                       p36_vision_span *spans, char *err, size_t errlen) {
    if (!e || !tokens || !role || !parts || (image_count && (!images || !spans)))
        goto invalid;
    for (size_t i = 0; i < image_count; i++) {
        if (!images[i].data || !images[i].token_count ||
            (uint64_t)images[i].grid_width * images[i].grid_height != images[i].token_count ||
            e->vocab.vision_start_id < 0 || e->vocab.vision_end_id < 0 || e->vocab.image_pad_id < 0)
            goto invalid;
    }
    if (!image_count) {
        p36_chat_append_message(e, tokens, role, parts[0]);
        return 1;
    }
    bool tool = !strcmp(role, "tool");
    chat_append_open_role(e, tokens, tool ? "user" : role);
    if (tool) p36_tokenize_rendered_chat(e, "<tool_response>\n", tokens);
    for (size_t i = 0; i <= image_count; i++) {
        if (parts[i]) p36_tokenize_rendered_chat(e, parts[i], tokens);
        if (i < image_count) p36_prompt_append_vision(e, tokens, &spans[i], &images[i], err, errlen);
    }
    if (tool) p36_tokenize_rendered_chat(e, "\n</tool_response>", tokens);
    chat_append_close_role(e, tokens);
    return 1;
invalid:
    if (err && errlen) snprintf(err, errlen, "invalid image observation");
    return 0;
}

int p36_chat_append_vision_message(p36_engine *e, p36_tokens *tokens,
                                    const char *role, const char *content,
                                    p36_vision_span *span,
                                    p36_vision_embedding *embedding,
                                    char *err, size_t errlen) {
    if (!e || !tokens || !span || !embedding) return 0;
    char *text = p36_chat_content_text(content, true);
    chat_append_open_role(e, tokens, role ? role : "user");
    if (text[0]) p36_tokenize_rendered_chat(e, text, tokens);
    free(text);
    if (!p36_prompt_append_vision(e, tokens, span, embedding, err, errlen))
        return 0;
    chat_append_close_role(e, tokens);
    return 1;
}

void p36_chat_append_assistant_prefix(p36_engine *e, p36_tokens *tokens, p36_think_mode think_mode) {
    chat_append_open_role(e, tokens, "assistant");
    if (think_mode == P36_THINK_NONE) {
        p36_tokenize_rendered_chat(e, "<think>\n\n</think>\n\n", tokens);
        return;
    }
    p36_tokenize_rendered_chat(e, "<think>\n", tokens);
    if (think_mode == P36_THINK_MAX && (!e || !e->chat_template))
        p36_chat_append_max_effort_prefix(e, tokens);
}

void p36_encode_chat_prompt(p36_engine *e, const char *system, const char *prompt, p36_think_mode think_mode, p36_tokens *out) {
    bool thinking = p36_think_mode_enabled(think_mode);
    p36_chat_apply_thinking_control(system, &thinking);
    p36_chat_apply_thinking_control(prompt, &thinking);
    if (!thinking) think_mode = P36_THINK_NONE;
    else if (think_mode == P36_THINK_NONE) think_mode = P36_THINK_HIGH;
    if (e && e->chat_template) {
        p36_jinja_message msgs[2] = {0};
        size_t n_msgs = 0;
        if (system && system[0]) {
            char *sys_trimmed = p36_chat_content_text(system, true);
            if (sys_trimmed && sys_trimmed[0]) {
                msgs[n_msgs].role = "system";
                msgs[n_msgs].content = sys_trimmed;
                n_msgs++;
            } else {
                free(sys_trimmed);
            }
        }
        char *usr_trimmed = p36_chat_content_text(prompt ? prompt : "", true);
        msgs[n_msgs].role = "user";
        msgs[n_msgs].content = usr_trimmed ? usr_trimmed : "";
        n_msgs++;
        p36_jinja_render_params jp = {
            .messages = msgs,
            .message_count = n_msgs,
            .tools_json_lines = NULL,
            .add_generation_prompt = true,
            .enable_thinking_set = true,
            .enable_thinking = think_mode != P36_THINK_NONE,
            .reasoning_effort = think_mode != P36_THINK_NONE ? p36_jinja_effort_name(think_mode) : NULL,
        };
        char *rendered = p36_jinja_render(e->chat_template, &jp, NULL, 0);
        for (size_t i = 0; i < n_msgs; i++) free((void *)msgs[i].content);
        if (rendered) {
            out->len = 0;
            p36_tokenize_rendered_chat(e, rendered, out);
            free(rendered);
            return;
        }
    }
    p36_chat_begin(e, out);
    if (system && system[0]) {
        p36_chat_append_message(e, out, "system", system);
    }
    p36_chat_append_message(e, out, "user", prompt ? prompt : "");
    p36_chat_append_assistant_prefix(e, out, think_mode);
}

static void dump_tokens_fp(FILE *fp, const p36_vocab *vocab, const token_vec *tokens) {
    fprintf(fp, "[");
    for (int i = 0; i < tokens->len; i++) {
        if (i) fprintf(fp, ", ");
        fprintf(fp, "%d", tokens->v[i]);
    }
    fprintf(fp, "]\n");
    for (int i = 0; i < tokens->len; i++) {
        int id = tokens->v[i];
        if (id >= 0 && id < vocab->n_vocab) {
            fprintf(fp, "%6d  %.*s\n", id, (int)vocab->token[id].len, vocab->token[id].ptr);
        }
    }
}

static uint32_t utf8_decode_one(const char *s, uint64_t len, uint64_t *pos) {
    uint8_t c = (uint8_t)s[*pos];
    if (c < 0x80 || *pos + 1 >= len) {
        (*pos)++;
        return c;
    }
    if ((c & 0xe0) == 0xc0 && *pos + 1 < len) {
        uint32_t cp = ((uint32_t)(c & 0x1f) << 6) | ((uint8_t)s[*pos + 1] & 0x3f);
        *pos += 2;
        return cp;
    }
    if ((c & 0xf0) == 0xe0 && *pos + 2 < len) {
        uint32_t cp = ((uint32_t)(c & 0x0f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 2] & 0x3f);
        *pos += 3;
        return cp;
    }
    if ((c & 0xf8) == 0xf0 && *pos + 3 < len) {
        uint32_t cp = ((uint32_t)(c & 0x07) << 18) |
                      ((uint32_t)((uint8_t)s[*pos + 1] & 0x3f) << 12) |
                      ((uint32_t)((uint8_t)s[*pos + 2] & 0x3f) << 6) |
                      ((uint8_t)s[*pos + 3] & 0x3f);
        *pos += 4;
        return cp;
    }
    (*pos)++;
    return c;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) return (int)cp;
    {
        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; b++) {
            if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174) continue;
            if (cp == 256 + n) return (int)b;
            n++;
        }
    }
    return -1;
}

static bool vocab_token_is_literal_special(p36_str s) {
    return (s.len >= 2 && s.ptr[0] == '<' && s.ptr[1] == '|') ||
           p36_streq(s, "<think>") ||
           p36_streq(s, "</think>");
}

char *p36_token_text(p36_engine *e, int token, size_t *len) {
    p36_str s;
    char *out;
    size_t n = 0;
    uint64_t pos = 0;
    if (!e || token < 0 || token >= e->vocab.n_vocab) {
        out = xmalloc(1);
        out[0] = '\0';
        if (len) *len = 0;
        return out;
    }
    s = e->vocab.token[token];
    out = xmalloc((size_t)s.len + 1);
    if (vocab_token_is_literal_special(s)) {
        memcpy(out, s.ptr, (size_t)s.len);
        out[s.len] = '\0';
        if (len) *len = (size_t)s.len;
        return out;
    }
    while (pos < s.len) {
        uint32_t cp = utf8_decode_one(s.ptr, s.len, &pos);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
    }
    out[n] = '\0';
    if (len) *len = n;
    return out;
}

int p36_token_eos(p36_engine *e) {
    return e ? e->vocab.eos_id : -1;
}

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    int best = 0;
    float best_v = P36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

#ifndef P36_NO_GPU
static int sample_argmax_margin(const float *logits, uint32_t n_vocab, float *margin) {
    int best = 0;
    float best_v = P36_NEG_INF;
    float second_v = P36_NEG_INF;
    for (uint32_t i = 0; i < n_vocab; i++) {
        float v = logits[i];
        if (v > best_v) {
            second_v = best_v;
            best_v = v;
            best = (int)i;
        } else if (v > second_v) {
            second_v = v;
        }
    }
    if (margin) *margin = best_v - second_v;
    return best;
}
#endif

typedef struct {
    int id;
    float logit;
    float prob;
} p36_sample_candidate;

static uint64_t p36_sample_rng_next(uint64_t *state) {
    uint64_t x = *state;
    if (x == 0) x = 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545f4914f6cdd1dull;
}

static float p36_sample_rng_f32(uint64_t *state) {
    const uint64_t x = p36_sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

static int p36_sample_candidate_cmp_desc(const void *a, const void *b) {
    const p36_sample_candidate *ca = (const p36_sample_candidate *)a;
    const p36_sample_candidate *cb = (const p36_sample_candidate *)b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int p36_sample_full_vocab(const float *logits, uint32_t n_vocab,
                                 float temperature, float top_p, float min_p,
                                 uint64_t *rng, float *prob_scratch) {
    float max_logit = P36_NEG_INF;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        if (min_rel > 1.0f) return best;

        /* Reject only values proven to fail the ordinary expf comparison.
         * The small retreat covers logf/expf boundary rounding. */
        float reject_scaled = P36_NEG_INF;
        bool have_reject_scaled = false;
        if (min_rel > 0.0f && isfinite(min_rel)) {
            float cutoff = logf(min_rel);
            for (int i = 0; i < 8 && isfinite(cutoff); i++) {
                cutoff = nextafterf(cutoff, -FLT_MAX);
                if (expf(cutoff) < min_rel) {
                    reject_scaled = cutoff;
                    have_reject_scaled = true;
                    break;
                }
            }
        }

        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            prob_scratch[i] = -1.0f;
            if (!isfinite(v)) continue;
            const float scaled = (v - max_logit) / temperature;
            if (have_reject_scaled && scaled <= reject_scaled) continue;
            const float p = expf(scaled);
            if (p < min_rel) continue;
            prob_scratch[i] = p;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = p36_sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float p = prob_scratch[i];
            if (p < 0.0f) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    {
        uint32_t n = 0;
        float sum = 0.0f;
        p36_sample_candidate *cand = NULL;
        if (min_p > 0.0f && min_p <= 1.0f) {
            /* Normalization cancels in the min-p comparison. Keep the full
             * softmax sum, but sort only candidates that can survive. */
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                prob_scratch[i] = -1.0f;
                if (!isfinite(v)) continue;
                const float p = expf((v - max_logit) / temperature);
                prob_scratch[i] = p;
                sum += p;
            }
            if (sum <= 0.0f || !isfinite(sum)) return best;

            const float min_prob = (1.0f / sum) * min_p;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float p = prob_scratch[i];
                if (p < 0.0f || p / sum < min_prob) continue;
                n++;
            }
            if (n == 0) return best;
            cand = xmalloc((size_t)n * sizeof(cand[0]));
            uint32_t out = 0;
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float p = prob_scratch[i];
                if (p < 0.0f || p / sum < min_prob) continue;
                cand[out++] = (p36_sample_candidate){
                    .id = (int)i, .logit = logits[i], .prob = p
                };
            }
        } else {
            cand = xmalloc((size_t)finite * sizeof(cand[0]));
            for (uint32_t i = 0; i < n_vocab; i++) {
                const float v = logits[i];
                if (!isfinite(v)) continue;
                const float p = expf((v - max_logit) / temperature);
                cand[n++] = (p36_sample_candidate){
                    .id = (int)i, .logit = v, .prob = p
                };
                sum += p;
            }
        }
        if (sum <= 0.0f || !isfinite(sum)) {
            free(cand);
            return best;
        }

        qsort(cand, (size_t)n, sizeof(cand[0]), p36_sample_candidate_cmp_desc);
        {
            const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
            float filtered_sum = 0.0f;
            uint32_t filtered = 0;
            for (uint32_t i = 0; i < n; i++) {
                const float p = cand[i].prob / sum;
                if (i > 0 && p < min_prob) break;
                filtered_sum += cand[i].prob;
                filtered++;
                if (filtered_sum / sum >= top_p) break;
            }
            if (filtered == 0) {
                free(cand);
                return best;
            }
            {
                float r = p36_sample_rng_f32(rng) * filtered_sum;
                for (uint32_t i = 0; i < filtered; i++) {
                    r -= cand[i].prob;
                    if (r <= 0.0f) {
                        const int id = cand[i].id;
                        free(cand);
                        return id;
                    }
                }
                {
                    const int id = cand[filtered - 1].id;
                    free(cand);
                    return id;
                }
            }
        }
    }
}

/* Shared tail of top-k sampling: given candidate ids/vals already sorted
 * descending by value (length n, n<=1024), apply the temperature softmax,
 * the min-p relative floor and top-p cumulative-mass cutoff, then draw one
 * weighted sample. Used both by the CPU top-k scan below and by the GPU
 * top-8 fast path in p36_session_sample, so the two stay identical past
 * candidate selection. */
static int p36_sample_from_sorted_topk(const int *ids, const float *vals, int n,
                                       float temperature, float top_p, float min_p,
                                       uint64_t *rng) {
    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = p36_sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

