static void kv_cache_text_sha(const char *ident, const char *text, size_t len, char out[41]) {
    char plain[41];
    sha1_bytes_hex(text, len, plain);
    if (!ident || !ident[0]) { memcpy(out, plain, 41); return; }
    char both[80];
    memcpy(both, ident, 40);
    memcpy(both + 40, plain, 40);
    sha1_bytes_hex(both, sizeof(both), out);
}

#define KV_KEY_VERSION "p36-kvkey-v2/chatml-template-v1"

static void sha1_update_file_range(sha1_ctx *c, FILE *fp, uint64_t off, size_t n) {
    char buf_[65536];
    if (fseeko(fp, (off_t)off, SEEK_SET) != 0) return;
    while (n) {
        size_t want = n < sizeof(buf_) ? n : sizeof(buf_);
        size_t got = fread(buf_, 1, want, fp);
        if (!got) break;
        sha1_update(c, buf_, got);
        n -= got;
    }
}

/* Content fingerprint of a GGUF without reading all of it: size, the first
 * 16 MiB (metadata incl. tokenizer and tensor table), 32 spread 64 KiB samples
 * of the weights and the last MiB, plus an optional steering file in full.
 * Any re-quantization, fine-tune or tokenizer edit changes it. */
static bool kv_model_fingerprint(const char *model_path, const char *steering_path,
                                 char out[41]) {
    FILE *fp = model_path ? fopen(model_path, "rb") : NULL;
    if (!fp) return false;
    if (fseeko(fp, 0, SEEK_END) != 0) { fclose(fp); return false; }
    const uint64_t size = (uint64_t)ftello(fp);
    sha1_ctx c;
    sha1_init(&c);
    sha1_update(&c, KV_KEY_VERSION, strlen(KV_KEY_VERSION));
    sha1_update(&c, &size, sizeof(size));
    const size_t head = size < (16u << 20) ? (size_t)size : (16u << 20);
    sha1_update_file_range(&c, fp, 0, head);
    if (size > head + (1u << 20)) {
        for (int i = 0; i < 32; i++) {
            uint64_t off = head + (size - head - 65536) / 31 * (uint64_t)i;
            sha1_update_file_range(&c, fp, off, 65536);
        }
        sha1_update_file_range(&c, fp, size - (1u << 20), 1u << 20);
    }
    fclose(fp);
    if (steering_path && steering_path[0]) {
        FILE *sp = fopen(steering_path, "rb");
        if (!sp) return false;
        sha1_update(&c, "steer", 5);
        char b[65536];
        size_t n;
        while ((n = fread(b, 1, sizeof(b), sp)) > 0) sha1_update(&c, b, n);
        fclose(sp);
    }
    uint8_t digest[20];
    sha1_final(&c, digest);
    hex20(digest, out);
    return true;
}

typedef struct {
    rax *ids;
} id_set;

static void id_set_init(id_set *set) {
    set->ids = raxNew();
    if (!set->ids) die("out of memory");
}

static bool id_set_contains(const id_set *set, const char *id) {
    if (!set || !set->ids || !id || !id[0]) return false;
    return raxFind(set->ids, (unsigned char *)id, strlen(id)) != raxNotFound;
}

static void id_set_add(id_set *set, const char *id) {
    if (!set || !set->ids || !id || !id[0] || id_set_contains(set, id)) return;
    if (!raxInsert(set->ids, (unsigned char *)id, strlen(id), (void *)1, NULL))
        die("out of memory");
}

static void id_set_free(id_set *set) {
    if (set && set->ids) raxFree(set->ids);
    if (set) memset(set, 0, sizeof(*set));
}

static void collect_tool_call_ids(const chat_msgs *msgs, id_set *ids) {
    if (!msgs || !ids) return;
    for (int i = 0; i < msgs->len; i++) {
        const tool_calls *calls = &msgs->v[i].calls;
        for (int j = 0; j < calls->len; j++) {
            id_set_add(ids, calls->v[j].id);
        }
    }
}

static bool sha_hex_name(const char *name, char sha[41]) {
    if (strlen(name) != 43 || strcmp(name + 40, ".kv")) return false;
    for (int i = 0; i < 40; i++) {
        if (!isxdigit((unsigned char)name[i])) return false;
        sha[i] = (char)tolower((unsigned char)name[i]);
    }
    sha[40] = '\0';
    return true;
}

static char *path_join(const char *dir, const char *name) {
    buf b = {0};
    buf_puts(&b, dir);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') buf_putc(&b, '/');
    buf_puts(&b, name);
    return buf_take(&b);
}

static char *kv_path_for_sha(kv_disk_cache *kc, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return path_join(kc->dir, name);
}

static bool mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static void kv_entry_free(kv_entry *e) {
    free(e->path);
    memset(e, 0, sizeof(*e));
}

static void kv_cache_clear(kv_disk_cache *kc) {
    for (int i = 0; i < kc->len; i++) kv_entry_free(&kc->entry[i]);
    free(kc->entry);
    kc->entry = NULL;
    kc->len = 0;
    kc->cap = 0;
}

static void kv_cache_push(kv_disk_cache *kc, kv_entry e) {
    if (kc->len == kc->cap) {
        kc->cap = kc->cap ? kc->cap * 2 : 16;
        kc->entry = xrealloc(kc->entry, (size_t)kc->cap * sizeof(kc->entry[0]));
    }
    kc->entry[kc->len++] = e;
}

static const char *find_next_tool_block(const char *p, const char **end_out) {
    if (end_out) *end_out = NULL;
    if (!p) return NULL;

    const char *best = NULL;
    const char *best_end = NULL;

    const struct {
        const char *start;
        const char *end;
    } syntaxes[] = {
        {P36_TOOL_CALLS_START, P36_TOOL_CALLS_END},
    };

    for (size_t i = 0; i < sizeof(syntaxes) / sizeof(syntaxes[0]); i++) {
        const char *start = strstr(p, syntaxes[i].start);
        if (!start) continue;
        const char *block_start = start;
        while (block_start > p && (block_start[-1] == '\n' || block_start[-1] == '\r')) {
            block_start--;
        }
        const char *end = strstr(start, syntaxes[i].end);
        if (!end) continue;
        if (!best || block_start < best) {
            best = block_start;
            best_end = end + strlen(syntaxes[i].end);
        }
    }

    if (end_out) *end_out = best_end;
    return best;
}

static int tool_memory_count_blocks_in_text(server *s, const char *text) {
    if (!s || s->disable_exact_tool_replay || !text || !text[0]) return 0;
    int count = 0;
    pthread_mutex_lock(&s->tool_mu);
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b) count += b->refs;
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);
    return count;
}

static int tool_memory_count_qwen_tool_in_text(server *s, const char *text) {
    return tool_memory_count_blocks_in_text(s, text);
}

static bool kv_tool_map_write(server *s, FILE *fp, const char *text,
                              uint64_t *written_bytes) {
    if (written_bytes) *written_bytes = 0;
    if (!s || s->disable_exact_tool_replay || !fp || !text || !text[0]) return true;

    pthread_mutex_lock(&s->tool_mu);
    uint32_t count = 0;
    uint64_t bytes = KV_TOOL_MAP_HEADER;
    uint64_t scan = ++s->tool_mem.scan_clock;
    const char *p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_tool_block(p, &end);
        if (!start || !end) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t qwen_tool_len = b->len;
                if (id_len > UINT32_MAX || qwen_tool_len > UINT32_MAX) continue;
                count++;
                bytes += 8u + (uint64_t)id_len + (uint64_t)qwen_tool_len;
            }
        }
        p = end;
    }
    if (count == 0) {
        pthread_mutex_unlock(&s->tool_mu);
        return true;
    }

    uint8_t h[KV_TOOL_MAP_HEADER];
    h[0] = KV_TOOL_MAP_MAGIC0;
    h[1] = KV_TOOL_MAP_MAGIC1;
    h[2] = KV_TOOL_MAP_MAGIC2;
    h[3] = KV_TOOL_MAP_VERSION;
    le_put32(h + 4, count);
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h);

    scan = ++s->tool_mem.scan_clock;
    p = text;
    for (;;) {
        const char *end = NULL;
        const char *start = find_next_tool_block(p, &end);
        if (!start || !end || !ok) break;
        tool_memory_block *b =
            tool_memory_find_block_locked(&s->tool_mem, start, (size_t)(end - start));
        if (b && b->seen != scan) {
            b->seen = scan;
            for (tool_memory_entry *e = b->entries; ok && e; e = e->block_next) {
                size_t id_len = strlen(e->id);
                size_t qwen_tool_len = b->len;
                if (id_len > UINT32_MAX || qwen_tool_len > UINT32_MAX) continue;
                uint8_t lens[8];
                le_put32(lens, (uint32_t)id_len);
                le_put32(lens + 4, (uint32_t)qwen_tool_len);
                ok = fwrite(lens, 1, sizeof(lens), fp) == sizeof(lens) &&
                     fwrite(e->id, 1, id_len, fp) == id_len &&
                     fwrite(b->qwen_tool, 1, qwen_tool_len, fp) == qwen_tool_len;
            }
        }
        p = end;
    }
    pthread_mutex_unlock(&s->tool_mu);

    if (ok && written_bytes) *written_bytes = bytes;
    return ok;
}

static int kv_tool_map_load_from_pos(server *s, FILE *fp, const id_set *wanted) {
    if (!s || s->disable_exact_tool_replay || !fp) return 0;
    uint8_t h[KV_TOOL_MAP_HEADER];
    size_t n = fread(h, 1, sizeof(h), fp);
    if (n == 0 && feof(fp)) return 0;
    if (n != sizeof(h)) return 0;
    if (h[0] != KV_TOOL_MAP_MAGIC0 || h[1] != KV_TOOL_MAP_MAGIC1 ||
        h[2] != KV_TOOL_MAP_MAGIC2 || h[3] != KV_TOOL_MAP_VERSION) return 0;

    uint32_t count = le_get32(h + 4);
    if ((uint64_t)count > (uint64_t)tool_memory_max_entries(&s->tool_mem) * 4u) return 0;
    int loaded = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t lens[8];
        if (fread(lens, 1, sizeof(lens), fp) != sizeof(lens)) return loaded;
        uint32_t id_len = le_get32(lens);
        uint32_t qwen_tool_len = le_get32(lens + 4);
        if (id_len == 0 || id_len > 256 || qwen_tool_len == 0 ||
            qwen_tool_len > P36_TOOL_MEMORY_MAX_BYTES) return loaded;
        char *id = xmalloc((size_t)id_len + 1);
        char *qwen_tool = xmalloc((size_t)qwen_tool_len + 1);
        bool ok = fread(id, 1, id_len, fp) == id_len &&
                  fread(qwen_tool, 1, qwen_tool_len, fp) == qwen_tool_len;
        id[id_len] = '\0';
        qwen_tool[qwen_tool_len] = '\0';
        if (ok && (!wanted || id_set_contains(wanted, id))) {
            tool_memory_put_source(s, id, qwen_tool, TOOL_MEMORY_DISK);
            loaded++;
        }
        free(id);
        free(qwen_tool);
        if (!ok) return loaded;
    }
    return loaded;
}

static void kv_fill_header(uint8_t h[KV_CACHE_FIXED_HEADER], uint8_t quant_bits,
                           uint8_t reason, uint8_t ext_flags,
                           uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                           uint64_t created_at, uint64_t last_used,
                           uint64_t payload_bytes) {
    memset(h, 0, KV_CACHE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    le_put32(h + 8, tokens);
    le_put32(h + 12, hits);
    le_put32(h + 16, ctx_size);
    le_put64(h + 24, created_at);
    le_put64(h + 32, last_used);
    le_put64(h + 40, payload_bytes);
}

static bool kv_quant_bits_valid(int quant_bits) {
    switch (quant_bits) {
    case 1: case 2: case 3: case 4: case 5: case 6: case 8:
        return true;
    default:
        return false;
    }
}

static bool kv_read_header(FILE *fp, kv_entry *e, uint32_t *text_bytes) {
    uint8_t h[KV_CACHE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= KV_REASON_SYSTEM ? h[5] : KV_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->tokens = le_get32(h + 8);
    e->hits = le_get32(h + 12);
    e->ctx_size = le_get32(h + 16);
    e->created_at = le_get64(h + 24);
    e->last_used = le_get64(h + 32);
    e->payload_bytes = le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && kv_quant_bits_valid(e->quant_bits);
}

static bool kv_read_entry_file(const char *path, const char sha[41], kv_entry *out) {
    struct stat st;
    if (stat(path, &st) != 0 || st.st_size < (off_t)(KV_CACHE_FIXED_HEADER + 4)) return false;
    FILE *fp = fopen(path, "rb");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    fclose(fp);
    if (!ok) return false;
    const uint64_t fixed = KV_CACHE_FIXED_HEADER + 4ull;
    if (UINT64_MAX - fixed < (uint64_t)text_bytes ||
        UINT64_MAX - fixed - (uint64_t)text_bytes < e.payload_bytes) return false;
    const uint64_t expected = fixed + (uint64_t)text_bytes + e.payload_bytes;
    if ((uint64_t)st.st_size < expected) return false;
    memcpy(e.sha, sha, 41);
    e.path = xstrdup(path);
    e.file_size = (uint64_t)st.st_size;
    *out = e;
    return true;
}

static void kv_cache_refresh(kv_disk_cache *kc) {
    if (!kc->enabled) return;
    kv_cache_clear(kc);
    DIR *d = opendir(kc->dir);
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        char *path = path_join(kc->dir, de->d_name);
        kv_entry e = {0};
        if (kv_read_entry_file(path, sha, &e)) kv_cache_push(kc, e);
        free(path);
    }
    closedir(d);
}

static bool kv_cache_touch_file(const char *path, uint32_t hits) {
    FILE *fp = fopen(path, "r+b");
    if (!fp) return false;
    kv_entry e = {0};
    uint32_t text_bytes = 0;
    bool ok = kv_read_header(fp, &e, &text_bytes);
    if (ok) {
        uint8_t h[KV_CACHE_FIXED_HEADER];
        uint64_t now = (uint64_t)time(NULL);
        kv_fill_header(h, e.quant_bits, e.reason, e.ext_flags, e.tokens, hits, e.ctx_size,
                       e.created_at, now, e.payload_bytes);
        ok = fseek(fp, 0, SEEK_SET) == 0 &&
             fwrite(h, 1, sizeof(h), fp) == sizeof(h);
    }
    fclose(fp);
    return ok;
}

static void kv_cache_restore_tool_memory_for_messages(server *s, const chat_msgs *msgs) {
    if (!s || s->disable_exact_tool_replay || !s->kv.enabled || !msgs) return;
    id_set wanted = {0};
    id_set_init(&wanted);
    collect_tool_call_ids(msgs, &wanted);
    if (raxSize(wanted.ids) == 0) {
        id_set_free(&wanted);
        return;
    }

    DIR *d = opendir(s->kv.dir);
    if (!d) {
        id_set_free(&wanted);
        return;
    }
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!sha_hex_name(de->d_name, sha)) continue;
        (void)sha;
        char *path = path_join(s->kv.dir, de->d_name);
        FILE *fp = fopen(path, "rb");
        free(path);
        if (!fp) continue;

        kv_entry hdr = {0};
        uint32_t text_bytes = 0;
        bool ok = kv_read_header(fp, &hdr, &text_bytes);
        uint64_t skip = (uint64_t)text_bytes + hdr.payload_bytes;
        if (ok && (hdr.ext_flags & KV_EXT_TOOL_MAP) &&
            skip <= (uint64_t)INT64_MAX &&
            fseeko(fp, (off_t)skip, SEEK_CUR) == 0)
        {
            kv_tool_map_load_from_pos(s, fp, &wanted);
        }
        fclose(fp);
    }
    closedir(d);
    id_set_free(&wanted);
}

static double kv_entry_eviction_score(const kv_entry *e, const p36_tokens *live) {
    if (!e || e->file_size == 0) return 0.0;
    (void)live;
    /*
     * Hits count successful disk reuses, but a fresh snapshot is still useful:
     * it may be the only copy of the session that is about to be evicted from
     * RAM.  Continued checkpoints are deliberately aligned restart frontiers,
     * so do not demote them just because they are a prefix of the live session.
     * Use hits+1 for eviction value so a just-written checkpoint does not get
     * deleted immediately just because its persisted hit counter is still 0.
     */
    return ((double)e->hits + 1.0) * (double)e->tokens / (double)e->file_size;
}

static void kv_cache_evict(kv_disk_cache *kc, const p36_tokens *live) {
    if (!kc->enabled || kc->budget_bytes == 0) return;
    kv_cache_refresh(kc);
    uint64_t total = 0;
    for (int i = 0; i < kc->len; i++) total += kc->entry[i].file_size;
    while (total > kc->budget_bytes && kc->len > 1) {
        /* Never evict the entry written last: a fresh checkpoint has no hits,
         * so by score alone it loses to stale entries from older sessions and
         * is deleted the moment it is stored, leaving nothing to restore. */
        int newest = 0;
        for (int i = 1; i < kc->len; i++)
            if (kc->entry[i].last_used >= kc->entry[newest].last_used) newest = i;
        int victim = newest == 0 ? 1 : 0;
        double victim_score = kv_entry_eviction_score(&kc->entry[victim], live);
        for (int i = 0; i < kc->len; i++) {
            if (i == newest || i == victim) continue;
            double score = kv_entry_eviction_score(&kc->entry[i], live);
            if (score < victim_score ||
                (score == victim_score && kc->entry[i].last_used < kc->entry[victim].last_used))
            {
                victim = i;
                victim_score = score;
            }
        }
        kv_entry e = kc->entry[victim];
        if (unlink(e.path) == 0) {
            server_log(P36_LOG_KVCACHE,
                       "p36-server: kv cache evicted tokens=%u hits=%u size=%.2f MiB",
                       e.tokens, e.hits, (double)e.file_size / (1024.0 * 1024.0));
            if (total >= e.file_size) total -= e.file_size;
            else total = 0;
        } else {
            total = 0;
        }
        kv_entry_free(&e);
        memmove(kc->entry + victim, kc->entry + victim + 1,
                (size_t)(kc->len - victim - 1) * sizeof(kc->entry[0]));
        kc->len--;
    }
}

