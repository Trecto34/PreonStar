/* Tiny Vulkan state-machine regressions (VK-1, VK-6).  No model: one 16-byte
 * copy dispatch, one 1 KiB embedding-table upload and one embed_row dispatch.
 * Skips itself (exit 0) when no Vulkan device can be opened.
 *
 * The test #includes p36_vulkan.c so it can read the static latches.  The
 * runtime mutex is swapped for an error-checking one and pthread_mutex_unlock
 * is wrapped to count EPERM, which is how an unlock without a matching lock
 * (VK-6) becomes observable.
 *   make tests/test_vk_state && ./tests/test_vk_state */
#include <pthread.h>
#include <errno.h>

static int test_unlock_errors;
static int test_unlock(pthread_mutex_t *m) {
    int rc = pthread_mutex_unlock(m);
    if (rc == EPERM) test_unlock_errors++;
    return rc;
}
#define pthread_mutex_unlock(m) test_unlock(m)

#include "../p36_vulkan.c"

static int failures;
#define CHECK(expr) do { if (!(expr)) { failures++; \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); } } while (0)

/* Cycle 1: a failed submit must not leave a stale batch_count, and cleanup must
 * clear the one-way device-lost latch so a later init starts clean. */
static void test_submit_failure_and_cleanup(void) {
    float in[4] = {1, 2, 3, 4}, out[4] = {0};
    setenv("P36_VK_FAULT_INJECT_LOST", "1", 1);
    if (!p36_gpu_init()) { fputs("test_vk_state: no Vulkan device, skipping\n", stderr); exit(0); }
    p36_gpu_tensor *a = p36_gpu_tensor_alloc(16), *b = p36_gpu_tensor_alloc(16);
    CHECK(a && b && p36_gpu_tensor_write(a, 0, in, sizeof(in)));
    CHECK(p36_gpu_copy_f32_tensor(b, a, 4));      /* recorded, batch_count == 1 */
    CHECK(!p36_gpu_tensor_read(b, 0, out, sizeof(out)));   /* flush -> injected loss */
    CHECK(p36_vk_device_lost);
    CHECK(p36_vk_batch_count == 0);
    p36_gpu_tensor_free(a);
    p36_gpu_tensor_free(b);
    p36_gpu_cleanup();
    unsetenv("P36_VK_FAULT_INJECT_LOST");
    CHECK(!p36_vk_device_lost);
    CHECK(p36_vk_fault_inject_lost == 0 && p36_vk_submit_calls == 0);
}

/* Cycle 2: embed_row must lock/unlock symmetrically, and cleanup must free the
 * resident embedding table instead of leaving a dangling pointer. */
static void test_embed_row_and_cleanup(void) {
    enum { N_EMBD = 256, ROWS = 4 };
    const uint64_t row_bytes = (N_EMBD / 32) * 34;       /* Q8_0 */
    const uint64_t table_bytes = row_bytes * ROWS;
    unsigned char *map = calloc(1, 1u << 20);
    CHECK(p36_gpu_init());
    p36_gpu_tensor *dst = p36_gpu_tensor_alloc(N_EMBD * sizeof(float));
    CHECK(map && dst);
    test_unlock_errors = 0;
    CHECK(p36_gpu_embed_row_tensor(dst, map, 1u << 20, 0, table_bytes, N_EMBD, 8u, NULL, 1));
    CHECK(test_unlock_errors == 0);      /* unlock without lock => EPERM counted */
    CHECK(p36_vk_embed_table != NULL);
    p36_gpu_tensor_free(dst);
    p36_gpu_cleanup();
    CHECK(p36_vk_embed_table == NULL);
    CHECK(!p36_vk_embed_table_failed);
    free(map);
}

int main(void) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
    pthread_mutex_destroy(&p36_vk_mu);
    pthread_mutex_init(&p36_vk_mu, &attr);
    test_submit_failure_and_cleanup();
    test_embed_row_and_cleanup();
    if (failures) {
        fprintf(stderr, "vk state tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("vk state tests: ok");
    return 0;
}
