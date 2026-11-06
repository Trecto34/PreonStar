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





int main(void) {
    test_invariants();
    if (failures) {
        fprintf(stderr, "state-machine tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("state-machine tests: ok");
    return 0;
}
