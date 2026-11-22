/* p36-perplexity: llama-perplexity-compatible scorer for any model p36 loads
 * (including inkling2_loop APEX files that upstream llama.cpp cannot open).
 *
 * Semantics match llama.cpp tools/perplexity (non-KL mode): tokenize the
 * whole file, split into n_ctx-token chunks, optionally replace each chunk's
 * first token with BOS, and score tokens [n_ctx/2 + 1, n_ctx) of each chunk.
 *
 * Default mode uses p36_session_score_tokens (one batched prefill with
 * all-row logits).  --decode scores the same tokens with one p36_session_eval
 * per position instead: slow, but an independent cross-check of the
 * batched path against the decode path.
 *
 * --dump FILE writes one record per scored token for paired comparisons:
 *   int32 target, int32 argmax, float nll, float p_argmax
 */
#define _POSIX_C_SOURCE 200809L
#include "p36.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return NULL; }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) { fclose(fp); free(buf); return NULL; }
    buf[n] = 0;
    fclose(fp);
    return buf;
}

static void score_logits(const float *lg, int nv, int target, float *nll, int *amax, float *pmax) {
    float mx = -INFINITY;
    int am = 0;
    for (int i = 0; i < nv; i++) if (lg[i] > mx) { mx = lg[i]; am = i; }
    double sum = 0.0;
    for (int i = 0; i < nv; i++) sum += exp((double)lg[i] - mx);
    double lse = mx + log(sum);
    *nll = (float)(lse - lg[target]);
    *amax = am;
    *pmax = (float)exp((double)mx - lse);
}

static void usage(void) {
    fprintf(stderr,
            "usage: p36-perplexity -m MODEL -f TEXT [options]\n"
            "  -c N               chunk size in tokens (default 512)\n"
            "  --chunks N         max chunks (default: all)\n"
            "  --bos ID           replace each chunk's first token with ID (default: none)\n"
            "  --prefill-chunk N  GPU prefill width (default: chunk size)\n"
            "  -ctk/-ctv TYPE     KV cache types (default q8_0 / q4_0)\n"
            "  --decode           score via per-token decode (slow cross-check)\n"
            "  --dump FILE        per-token records: i32 target, i32 argmax, f32 nll, f32 pmax\n");
}

int main(int argc, char **argv) {
    const char *model = NULL, *file = NULL, *dump = NULL;
    int n_ctx = 512, chunks = 0, bos = -1, prefill_chunk = 0;
    bool decode = false;
    p36_kv_cache_type ctk = P36_KV_CACHE_Q8_0, ctv = P36_KV_CACHE_Q4_0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const bool has = i + 1 < argc;
        if ((!strcmp(a, "-m") || !strcmp(a, "--model")) && has) model = argv[++i];
        else if (!strcmp(a, "-f") && has) file = argv[++i];
        else if (!strcmp(a, "-c") && has) n_ctx = atoi(argv[++i]);
        else if (!strcmp(a, "--chunks") && has) chunks = atoi(argv[++i]);
        else if (!strcmp(a, "--bos") && has) bos = atoi(argv[++i]);
        else if (!strcmp(a, "--prefill-chunk") && has) prefill_chunk = atoi(argv[++i]);
        else if ((!strcmp(a, "-ctk") || !strcmp(a, "--cache-type-k")) && has) {
            if (!p36_parse_kv_cache_type(argv[++i], &ctk)) { usage(); return 2; }
        } else if ((!strcmp(a, "-ctv") || !strcmp(a, "--cache-type-v")) && has) {
            if (!p36_parse_kv_cache_type(argv[++i], &ctv)) { usage(); return 2; }
        } else if (!strcmp(a, "--decode")) decode = true;
        else if (!strcmp(a, "--dump") && has) dump = argv[++i];
        else { usage(); return 2; }
    }
    if (!model || !file || n_ctx < 4) { usage(); return 2; }
    if (prefill_chunk <= 0) prefill_chunk = n_ctx;

    p36_engine_options opt = {0};
    opt.model_path = model;
    opt.backend = P36_BACKEND_VULKAN;
    opt.n_threads = 16;
    opt.context_size = n_ctx;
    opt.session_count = 1;
    opt.prefill_chunk = (uint32_t)prefill_chunk;
    opt.power_percent = 100;
    opt.cache_type_k = ctk;
    opt.cache_type_v = ctv;
    p36_engine *e = NULL;
    if (p36_engine_open(&e, &opt) != 0) return 1;

    char *text = read_file(file);
    if (!text) return 1;
    p36_tokens toks = {0};
    p36_tokenize_text(e, text, &toks);
    free(text);
    const int max_chunks = toks.len / n_ctx;
    if (chunks <= 0 || chunks > max_chunks) chunks = max_chunks;
    const int first = n_ctx / 2 + 1;           /* first scored target index */
    const int per_chunk = n_ctx - first;
    fprintf(stderr, "p36-perplexity: %d tokens, %d chunks of %d, %d scored/chunk, bos=%d, mode=%s\n",
            toks.len, chunks, n_ctx, per_chunk, bos, decode ? "decode" : "batched");

    p36_session *s = NULL;
    if (p36_session_create(&s, e, n_ctx) != 0) { fprintf(stderr, "p36-perplexity: session create failed\n"); return 1; }

    FILE *df = NULL;
    if (dump && !(df = fopen(dump, "wb"))) { perror(dump); return 1; }
    int *buf = malloc(sizeof(int) * (size_t)n_ctx);
    float *nll = malloc(sizeof(float) * (size_t)per_chunk);
    float *pmax = malloc(sizeof(float) * (size_t)per_chunk);
    int *amax = malloc(sizeof(int) * (size_t)per_chunk);
    double nll_sum = 0.0, nll2_sum = 0.0;
    long count = 0;
    char err[256];
    const double t0 = now_sec();

    for (int c = 0; c < chunks; c++) {
        memcpy(buf, toks.v + (size_t)c * n_ctx, sizeof(int) * (size_t)n_ctx);
        if (bos >= 0) buf[0] = bos;
        p36_tokens chunk = { .v = buf, .len = n_ctx, .cap = n_ctx };
        if (!decode) {
            if (p36_session_score_tokens(s, &chunk, first, nll, amax, pmax, err, sizeof(err)) != 0) {
                fprintf(stderr, "p36-perplexity: chunk %d: %s\n", c, err);
                return 1;
            }
        } else {
            p36_session_invalidate(s);
            p36_tokens pre = { .v = buf, .len = first, .cap = first };
            if (p36_session_sync(s, &pre, err, sizeof(err)) != 0) {
                fprintf(stderr, "p36-perplexity: sync chunk %d: %s\n", c, err);
                return 1;
            }
            for (int t = first; t < n_ctx; t++) {
                if (t > first && p36_session_eval(s, buf[t - 1], err, sizeof(err)) != 0) {
                    fprintf(stderr, "p36-perplexity: eval chunk %d pos %d: %s\n", c, t - 1, err);
                    return 1;
                }
                int nv = 0;
                const float *lg = p36_session_logits(s, &nv);
                if (!lg || nv <= 0) { fprintf(stderr, "p36-perplexity: no logits\n"); return 1; }
                score_logits(lg, nv, buf[t], &nll[t - first], &amax[t - first], &pmax[t - first]);
            }
        }
        for (int k = 0; k < per_chunk; k++) {
            nll_sum += nll[k];
            nll2_sum += (double)nll[k] * nll[k];
            count++;
            if (df) {
                int32_t ri[2] = { buf[first + k], amax[k] };
                float rf[2] = { nll[k], pmax[k] };
                fwrite(ri, sizeof ri, 1, df);
                fwrite(rf, sizeof rf, 1, df);
            }
        }
        printf("[%d]%.4f,", c + 1, exp(nll_sum / (double)count));
        fflush(stdout);
    }
    const double mean = nll_sum / (double)count;
    const double var = nll2_sum / (double)count - mean * mean;
    const double se = sqrt(var / (double)(count - 1));
    printf("\nFinal estimate: PPL = %.4f +/- %.5f  (tokens=%ld, %.1fs)\n",
           exp(mean), exp(mean) * se, count, now_sec() - t0);
    if (df) fclose(df);
    free(buf); free(nll); free(pmax); free(amax);
    p36_tokens_free(&toks);
    p36_session_free(s);
    p36_engine_close(e);
    return 0;
}
