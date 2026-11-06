/* Model-free regressions for the implicit-state-machine audit (core slice).
 * Includes p36.c directly (like the engine build) so it can reach the static
 * session / Jinja / hotlist internals.  No GGUF and no GPU dispatch: sessions
 * are hand-built structs, and the cancel hook keeps prefill off the engine.
 *   make tests/test_state_machine && ./tests/test_state_machine */
#define P36_DEBUG_ASSERT 1
#include "../p36.c"

static int failures;
#define CHECK(expr) do { if (!(expr)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); } } while (0)

static bool cancel_always(void *ud) { (void)ud; return true; }

static void session_init(p36_session *s, int ctx, int live_len, bool valid) {
    memset(s, 0, sizeof(*s));
    s->ctx_size = ctx;
    s->mtp_draft_token = -1;
    for (int i = 0; i < live_len; i++) p36_tokens_push(&s->checkpoint, 100 + i);
    s->checkpoint_valid = valid;
    s->cancel = cancel_always;   /* keeps prefill_range off the engine */
}

static void test_rewrite_from_common(void) {
    p36_tokens prompt = {0};
    char err[128] = {0};
    for (int i = 0; i < 5; i++) p36_tokens_push(&prompt, 100 + i);
    /* A live vision checkpoint must be rebuilt by the vision path, never
     * extended as text by the rewrite helper. */
    p36_session sv;
    session_init(&sv, 16, 3, true);
    sv.vision_state = true;
    CHECK(p36_session_rewrite_from_common(&sv, &prompt, 3, err, sizeof(err)) ==
          P36_SESSION_REWRITE_REBUILD_NEEDED);
    p36_tokens_free(&sv.checkpoint);
    /* A BROKEN session (valid=false, len>0) must not be resurrected. */
    p36_session sb;
    session_init(&sb, 4, 3, false);   /* ctx 4: the rebuild sync refuses early */
    (void)p36_session_rewrite_from_common(&sb, &prompt, 3, err, sizeof(err));
    CHECK(!sb.checkpoint_valid);
    p36_tokens_free(&sb.checkpoint);
    p36_tokens_free(&prompt);
}

static void test_invariants(void) {
    p36_session s;
    session_init(&s, 16, 2, true);
    CHECK(p36_session_check_invariants(&s) == NULL);
    s.mtp_draft_valid = true;                 /* valid draft with token -1 */
    CHECK(p36_session_check_invariants(&s) != NULL);
    s.mtp_draft_valid = false;
    s.rope_delta = 3;                         /* rope_delta without vision */
    CHECK(p36_session_check_invariants(&s) != NULL);
    s.rope_delta = 0;
    s.vision_count = 1;                       /* spans without vision_state */
    CHECK(p36_session_check_invariants(&s) != NULL);
    p36_tokens_free(&s.checkpoint);
}

static void test_payload_load_clears_vision(void) {
    p36_engine e;
    p36_session s;
    memset(&e, 0, sizeof(e));
    e.backend = P36_BACKEND_CPU;
    session_init(&s, 16, 0, false);
    s.cancel = NULL;
    s.engine = &e;                            /* runtime NULL: reserve_kv fails mid-load */
    s.vision_state = true;
    s.vision_count = 1;
    s.vision_spans = calloc(1, sizeof(*s.vision_spans));
    FILE *fp = tmpfile();
    CHECK(fp != NULL);
    if (fp) {
        uint32_t h[16] = {P36_PAYLOAD_MAGIC, P36_PAYLOAD_VERSION_TYPED_KV, 16, 0, 2,
                          P36_N_VOCAB, P36_N_LAYER, P36_N_HEAD_KV, P36_N_HEAD_DIM,
                          P36_N_VALUE_DIM, P36_N_SSM_CONV, P36_N_SSM_CONV_DIM,
                          P36_N_SSM_STATE, P36_N_SSM_DT_RANK,
                          P36_KV_CACHE_F16, P36_KV_CACHE_F16};
        fwrite(h, sizeof(h), 1, fp);
        rewind(fp);
        char err[128] = {0};
        CHECK(p36_session_load_payload(&s, fp, sizeof(h), err, sizeof(err)) != 0);
        fclose(fp);
    }
    CHECK(!s.vision_state && s.vision_count == 0 && s.vision_spans == NULL);
    CHECK(!s.checkpoint_valid && s.checkpoint.len == 0);
    p36_tokens_free(&s.checkpoint);
}




int main(void) {
    test_rewrite_from_common();
    test_invariants();
    test_payload_load_clears_vision();
    if (failures) {
        fprintf(stderr, "state-machine tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("state-machine tests: ok");
    return 0;
}
