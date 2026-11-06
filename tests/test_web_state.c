/* Model-free unit tests for p36_web.c state handling (no browser, no GPU).
 * Includes the .c so static helpers are reachable. */
#include "../p36_web.c"

static int failures;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

/* IO-4: with CDP down, a still-running child recorded in chrome_pid must be
 * stopped, not forgotten (forgetting it lets a second Chrome spawn on the same
 * profile and port while the first keeps running). */
static void test_stale_live_browser_is_stopped(void) {
    p36_web_config cfg = {.port = 1};   /* nothing listens on port 1 */
    p36_web *web = p36_web_create(&cfg);
    pid_t child = fork();
    if (child == 0) { for (;;) pause(); }
    web->chrome_pid = child;
    char err[160] = {0};
    CHECK(!web_ensure_browser(web, err, sizeof(err)));   /* no confirm callback: refuses to spawn */
    CHECK(web->chrome_pid == 0);
    CHECK(kill(child, 0) != 0 && errno == ESRCH);        /* gone and reaped, not merely forgotten */
    if (kill(child, 0) == 0) { kill(child, SIGKILL); waitpid(child, NULL, 0); }
    p36_web_free(web);
}

/* An exited child is still reaped and forgotten. */
static void test_exited_browser_is_reaped(void) {
    p36_web_config cfg = {.port = 1};
    p36_web *web = p36_web_create(&cfg);
    pid_t child = fork();
    if (child == 0) _exit(0);
    usleep(100 * 1000);
    web->chrome_pid = child;
    char err[160] = {0};
    CHECK(!web_ensure_browser(web, err, sizeof(err)));
    CHECK(web->chrome_pid == 0);
    CHECK(waitpid(child, NULL, WNOHANG) == -1 && errno == ECHILD);
    p36_web_free(web);
}

int main(void) {
    test_stale_live_browser_is_stopped();
    test_exited_browser_is_reaped();
    if (failures) { fprintf(stderr, "web state tests: %d failure(s)\n", failures); return 1; }
    puts("web state tests: ok");
    return 0;
}
