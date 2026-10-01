/* Shared by q36-agent and q36-server: deterministic loop guards that run outside
 * the model.  Include after xmalloc/xrealloc are defined.  Header-only, static. */
#ifndef Q36_LOOPGUARD_H
#define Q36_LOOPGUARD_H

#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#define AGENT_REPETITION_WINDOW_BYTES 32768
#define AGENT_REPETITION_EXACT_LINE_RUN 12
#define AGENT_REPETITION_NEAR_LINE_RUN 32
#define AGENT_REPETITION_NGRAM_REPEATS 8
#define AGENT_REPETITION_NGRAM_MIN_TOTAL 256

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

static void agent_line_spans_free(agent_line_spans *spans) {
    free(spans->v);
    memset(spans, 0, sizeof(*spans));
}

static void agent_line_spans_push(agent_line_spans *spans, agent_line_span span) {
    if (spans->len == spans->cap) {
        spans->cap = spans->cap ? spans->cap * 2 : 128;
        spans->v = xrealloc(spans->v, (size_t)spans->cap * sizeof(spans->v[0]));
    }
    spans->v[spans->len++] = span;
}

/* Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences. */
static void agent_split_lines(const char *data, size_t len, agent_line_spans *spans) {
    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;
        while (pos < len && data[pos] != '\n' && data[pos] != '\r') pos++;
        size_t content_end = pos;
        if (pos < len) {
            if (data[pos] == '\r' && pos + 1 < len && data[pos + 1] == '\n')
                pos += 2;
            else
                pos++;
        }
        agent_line_spans_push(spans, (agent_line_span){
            .start = start,
            .content_end = content_end,
            .end = pos,
        });
    }
}

typedef struct {
    char text[AGENT_REPETITION_WINDOW_BYTES];
    size_t len;
    size_t unchecked;
} agent_repetition_detector;

static void agent_repetition_trim_line(const char *text, agent_line_span span,
                                       size_t *start, size_t *len) {
    size_t a = span.start;
    size_t b = span.content_end;
    while (a < b && isspace((unsigned char)text[a])) a++;
    while (b > a && isspace((unsigned char)text[b - 1])) b--;
    *start = a;
    *len = b - a;
}

static bool agent_repetition_lines_near(const char *a, size_t an,
                                        const char *b, size_t bn) {
    if (an < 24 || bn < 24 || an + 2 < bn || bn + 2 < an) return false;
    size_t common = an < bn ? an : bn;
    size_t mismatches = an > bn ? an - bn : bn - an;
    size_t allowed = common / 20;
    if (allowed < 2) allowed = 2;
    for (size_t i = 0; i < common && mismatches <= allowed; i++)
        if (a[i] != b[i]) mismatches++;
    return mismatches <= allowed;
}

/* Conservative byte-level loop detection. It only considers a suffix made of
 * many consecutive near-identical lines, or an exact byte period repeated at
 * least eight times. Ordinary repeated syntax separated by distinct content
 * therefore does not trip the detector. */
static bool agent_repetition_text_detected(const char *text, size_t len) {
    if (!text || len < 128) return false;

    agent_line_spans spans = {0};
    agent_split_lines(text, len, &spans);
    int exact_run = 0;
    int near_run = 0;
    size_t reference_start = 0, reference_len = 0;
    for (int i = spans.len - 1; i >= 0 && spans.len - i <= 48; i--) {
        size_t start = 0, line_len = 0;
        agent_repetition_trim_line(text, spans.v[i], &start, &line_len);
        if (line_len == 0) continue;
        if (!reference_len) {
            reference_start = start;
            reference_len = line_len;
            exact_run = near_run = 1;
            continue;
        }
        if (line_len >= 12 && line_len == reference_len &&
            !memcmp(text + start, text + reference_start, line_len)) {
            exact_run++;
            near_run++;
        } else if (agent_repetition_lines_near(
                       text + start, line_len,
                       text + reference_start, reference_len)) {
            near_run++;
            exact_run = 0;
        } else {
            break;
        }
        if (exact_run >= AGENT_REPETITION_EXACT_LINE_RUN ||
            near_run >= AGENT_REPETITION_NEAR_LINE_RUN) {
            agent_line_spans_free(&spans);
            return true;
        }
    }
    agent_line_spans_free(&spans);

    for (size_t period = 16; period <= 256; period++) {
        size_t repeated = period * AGENT_REPETITION_NGRAM_REPEATS;
        if (repeated < AGENT_REPETITION_NGRAM_MIN_TOTAL || repeated > len)
            continue;
        const char *suffix = text + len - repeated;
        bool same = true;
        for (size_t i = period; i < repeated; i++) {
            if (suffix[i] != suffix[i % period]) {
                same = false;
                break;
            }
        }
        if (!same) continue;
        unsigned char distinct[4] = {0};
        int distinct_count = 0;
        for (size_t i = 0; i < period && distinct_count < 4; i++) {
            unsigned char c = (unsigned char)suffix[i];
            bool seen = false;
            for (int j = 0; j < distinct_count; j++)
                if (distinct[j] == c) seen = true;
            if (!seen) distinct[distinct_count++] = c;
        }
        if (distinct_count >= 4) return true;
    }
    return false;
}

static bool agent_repetition_detector_feed(agent_repetition_detector *d,
                                           const char *text, size_t len) {
    if (!d || !text || !len) return false;
    if (len >= sizeof(d->text)) {
        text += len - sizeof(d->text);
        len = sizeof(d->text);
        d->len = 0;
    } else if (d->len + len > sizeof(d->text)) {
        size_t drop = d->len + len - sizeof(d->text);
        memmove(d->text, d->text + drop, d->len - drop);
        d->len -= drop;
    }
    memcpy(d->text + d->len, text, len);
    d->len += len;
    d->unchecked += len;
    bool newline = memchr(text, '\n', len) || memchr(text, '\r', len);
    if (d->unchecked < 32 && !newline) return false;
    d->unchecked = 0;
    return agent_repetition_text_detected(d->text, d->len);
}

/* Post-think leak detector.
 * Post-think leak: after </think> the model must act (tool call) or answer.
 * Once more than action_budget content tokens were produced with no tool
 * started, the content is scanned for reasoning-style restarts ("Wait,",
 * "Actually,", "Let me re...") or repeated text.  A real final answer has
 * neither, so it is never cut; a second reasoning channel is aborted. */
typedef struct {
    int tokens;
    int next_check;
    agent_repetition_detector rep;
} agent_action_leak;

static bool agent_leak_marker_at(const char *p, size_t n) {
    static const char *const m[] = {
        "Wait", "Actually", "Hmm", "Let me re", "Let me think", "Let me check",
        "Let me look", "Let me read", "Let me see", "Let me verify",
        "But wait", "Hold on", "Alternatively", "Maybe I", "I need to re",
        "On second thought", "Okay, so", "OK, so",
    };
    for (size_t i = 0; i < sizeof(m) / sizeof(m[0]); i++) {
        size_t l = strlen(m[i]);
        if (n >= l && !strncmp(p, m[i], l)) return true;
    }
    return false;
}

static int agent_leak_marker_count(const char *t, size_t len) {
    int count = 0;
    bool at_start = true;
    for (size_t i = 0; i < len; i++) {
        if (at_start && t[i] != ' ' && t[i] != '\n' && t[i] != '-' && t[i] != '*') {
            if (agent_leak_marker_at(t + i, len - i)) count++;
            at_start = false;
        }
        if (t[i] == '\n' || ((t[i] == '.' || t[i] == '?' || t[i] == '!') &&
                             i + 1 < len && t[i + 1] == ' '))
            at_start = true;
    }
    return count;
}

/* Feed one post-think content token; true when the content is a reasoning leak. */
static bool agent_action_leak_feed(agent_action_leak *l, int budget,
                                   const char *text, size_t len) {
    l->tokens++;
    bool repeated = agent_repetition_detector_feed(&l->rep, text, len);
    if (budget <= 0 || l->tokens <= budget) return false;
    if (repeated) return true;
    if (l->tokens < l->next_check) return false;
    l->next_check = l->tokens + 32;
    /* Past 4x the budget with still no action, a single restart marker is
     * enough: the model has had ample room to act or answer. */
    int need = l->tokens > 4 * budget ? 1 : 3;
    return agent_leak_marker_count(l->rep.text, l->rep.len) >= need;
}


#endif
