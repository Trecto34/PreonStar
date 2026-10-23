#include "p36.h"
#include "p36_tool_text.h"
#include "p36_prompt_prefix.h"
#include "p36_help.h"
#include "p36_kvstore.h"
#include "p36_web.h"
#include "p36_remote.h"
#include "linenoise.h"
#include "p36_cli_args.h"

#include <errno.h>
#include <ctype.h>
#include <dirent.h>
#include <fnmatch.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <copyfile.h>
#elif defined(__linux__)
#include <sys/xattr.h>
#endif

/* This is intentionally not in linenoise.h, but it is part of the existing
 * multiplexed editor implementation.  The agent uses it only to restore text
 * after Enter is pressed while the model is still busy. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

#define AGENT_RESIDENT_CTX 100000
#define AGENT_STREAMING_CTX 100000
#define AGENT_THINKING_BUDGET_DEFAULT 50000
#define AGENT_ACTION_BUDGET_DEFAULT 384
#define AGENT_WRITE_PAYLOAD_MAX 12288 /* bytes of content per write call */
#define AGENT_MAX_REPEAT_TOOL_DEFAULT 1
#define AGENT_MAX_STAGNANT_DEFAULT 2
#define AGENT_MAX_RECOVERIES_DEFAULT 3
#define AGENT_MAX_VERIFY_DEFAULT 8       /* non-mutating steps after the last target edit */
#define AGENT_MAX_IMPL_DEFAULT 96        /* emergency ceiling on IMPLEMENT-phase steps per user turn */
#define AGENT_MAX_NOPROGRESS_DEFAULT 16  /* hard ceiling on the same, ignoring step cost */
#define AGENT_VW_MAX_REJECTS 2           /* refused tool calls before the controller answers */

static int set_nonblock(int fd, bool on, int *old_flags);
static int agent_replace_file(const char *path, const char *data, size_t len,
                              const char *expected, size_t expected_len,
                              char *err, size_t errlen);
static bool agent_parse_bool_default(const char *s, bool def);

/* ============================================================================
 * Configuration, Worker State, And Streaming Types
 * ============================================================================
 *
 * The agent is intentionally a single process: the UI thread owns terminal
 * input/output, while the worker thread owns the live P36 session and KV state.
 * These types define the shared state and the small streaming state machines
 * used to render sampled assistant text and Qwen tool calls as they arrive.
 */

typedef struct {
    const char *prompt;
    const char *system;
    p36_prompt_prefix prefix;
    const char *trace_path;
    int n_predict;
    int thinking_budget;
    int action_budget;      /* post-think content tokens before a leak check; 0 = off */
    int max_repeat_tool;    /* identical calls allowed per unchanged workspace */
    int max_stagnant_turns; /* blocked/identically-failing calls before recovery */
    int max_recoveries;     /* controller recoveries per user turn before BLOCKED */
    int max_verify_steps;   /* non-mutating steps allowed after the last target edit; 0 = off */
    int max_no_progress_steps; /* absolute ceiling of the same; 0 = off */
    int max_implementation_steps; /* hard ceiling on IMPLEMENT-phase steps, never reset by edits; 0 = off */
    bool think_auto;       /* deterministic per-turn thinking budget (--think auto) */
    bool compact_llm;       /* write compaction summaries with the model, not harness state */
    int ctx_size;
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float frequency_penalty;
    float repeat_penalty;
    uint64_t seed;
    p36_think_mode think_mode;
    bool think_mode_set;
    bool temperature_set;
    bool top_k_set;
    bool top_p_set;
    bool min_p_set;
    bool repeat_penalty_set;
} agent_generation_options;

typedef struct {
    p36_engine_options engine;
    agent_generation_options gen;
    const char *chdir_path;
    /* Workspace jail.  ws_root is controller-owned (set once in main, never by the model);
     * empty = jail not established (unit-test workers).  yolo lifts every restriction,
     * unsafe_fs only the file-tool one. */
    const char *workspace_path;
    char ws_root[PATH_MAX];
    bool yolo, unsafe_fs;
    /* Remote mode: inference on a p36-server, tools/watchdog/compaction local. */
    const char *server_url;
    const char *server_model;
    const char *api_key;
    int server_timeout;
    bool non_interactive;
    bool edit_upto;
} agent_config;

typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    int power_percent;
    p36_think_mode think_mode;
    int thinking_budget;
    bool qwen38;
    char error[256];
} agent_status;

/* Durable task state maintained by the harness (never by the model), so
 * compaction can preserve facts instead of prose.  Fixed-size on purpose. */
#define AGENT_ST_GOALS 4
#define AGENT_ST_FILES 32
#define AGENT_ST_LINES 24
typedef struct {
    char path[200];
    char ranges[80];  /* "read 1-200; 300-340" while unchanged since */
    char note[100];   /* last write/edit outcome; empty if never modified */
    char blob[80];    /* "line 216, 615601 bytes" omitted by read; survives compaction */
} agent_st_file;

/* Verification-convergence state.  Harness-owned, per user turn, and part of
 * the task state so compaction cannot reset it.  "Target" mutations change the
 * artifact being built; "helper" mutations touch disposable verification
 * scripts and never count as progress.  See agent_vw_note(). */
#define AGENT_VW_SIGS 24
#define AGENT_VW_HELPERS 8
#define AGENT_VW_IMPL_CEILING 4   /* IMPLEMENT-phase no-progress ceiling = this x max_noprog */
enum { AGENT_PH_IMPLEMENT = 0, AGENT_PH_VERIFY };   /* zero value = IMPLEMENT: default after task start */
typedef struct {
    int phase;              /* IMPLEMENT until strong evidence of checking a finished target */
    int npending;           /* concrete evidence that required work is missing; blocks VERIFY/finalize */
    char pending[100];      /* first such item */
    bool armed;             /* a target mutation happened this user turn */
    bool finalize;          /* FINALIZE_REQUIRED: further tool calls are refused */
    int step;               /* tool calls seen this user turn */
    int impl_steps, max_impl;  /* IMPLEMENT-phase steps this turn; never reset by edits */
    bool impl_stop;         /* FINALIZE_REQUIRED came from the implementation ceiling */
    int last_target_step;   /* step of the last target mutation; 0 = none */
    char last_target[200];
    int verify_used;        /* budget units since the last target mutation */
    int no_progress;        /* raw non-mutating steps since then (no step cost) */
    int dups;               /* near-identical re-checks since then */
    int helper_since;       /* helper mutations since then */
    int rejected;           /* tool calls refused while FINALIZE_REQUIRED */
    int max_verify, max_noprog;
    uint64_t sigs[AGENT_VW_SIGS];
    int nsigs;
    char helpers[AGENT_VW_HELPERS][200];  /* helper files this turn created */
    int nhelpers;
} agent_verify_state;

typedef struct {
    char *goals[AGENT_ST_GOALS];
    int ngoals;
    agent_st_file files[AGENT_ST_FILES];
    int nfiles;
    char *done[AGENT_ST_LINES];    /* completed actions */
    int ndone;
    char *facts[AGENT_ST_LINES];   /* command outcomes */
    int nfacts;
    char *failed[AGENT_ST_LINES];  /* failed attempts */
    int nfailed;
    char *unresolved[AGENT_ST_LINES];  /* failures not (yet) shown to be product bugs */
    int nunresolved;
    char *dnr[AGENT_ST_LINES];     /* re-checks the controller saw repeated */
    int ndnr;
    bool last_failed;
    char ws_root[PATH_MAX];        /* WORKSPACE_ROOT, copied from the config on every tool call */
    agent_verify_state vw;
} agent_task_state;

/* Session metrics, reported by the end-of-run summary. */
typedef struct {
    int steps, tools, recoveries, blocked_calls, compactions;
    /* verification watchdog */
    int target_mutations, helper_mutations, verify_steps, verify_exhaustions;
    int no_progress_steps, forced_finalizations;
    int impl_steps, impl_limit_hits;
    int ph_iv, ph_vi, ph_vf;   /* controller phase transitions: IMPLEMENT->VERIFY, VERIFY->IMPLEMENT, VERIFY->FINALIZE_REQUIRED */
    long reasoning_tokens, action_tokens, prompt_tokens, cached_tokens;
    double prefill_ms, decode_ms, wall_s;
    char finish[40];
} agent_metrics;

typedef struct agent_bash_job agent_bash_job;

/* Runtime preference only; never part of saved session metadata. */
typedef struct {
    bool enabled;
    bool instructed;
    enum {
        AGENT_HINTS_UNSET,
        AGENT_HINTS_OFF,
        AGENT_HINTS_ON,
        AGENT_HINTS_UNKNOWN,
    } applied;
} agent_hints;

typedef struct {
    int fd;
    int job_id;
    double deadline;
    char prompt[512];
} agent_password_request;

typedef struct {
    p36_engine *engine;
    agent_config *cfg;
    p36_session *session;
    p36_tokens transcript;
    p36_vision_span *images;
    size_t image_count;
    size_t image_cap;
    char *cache_dir;
    char *sysprompt_path;
    char session_sha[41];
    char *session_title;
    uint64_t session_created_at;
    char *legacy_session_path_to_delete;
    bool user_activity;
    bool session_dirty;
    int compact_reserve;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
    bool stop;
    bool interrupt;
    bool initialized;
    bool active;
    bool paused;
    bool save_requested;
    bool compact_requested;
    bool power_requested;
    int requested_power;
    int progress_base;
    double progress_started_at;
    int last_system_prompt_reminder_at;
    char *cmd_text;
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    p36_web *web;
    bool web_approval_pending;
    bool web_approval_answered;
    bool web_approval_result;
    char web_approval_message[256];
    char web_approval_error[160];
    bool password_pending;
    bool password_answered;
    bool password_result;
    agent_password_request password_request;
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    agent_hints hints;
    bool thinking_enabled;
    char more_path[PATH_MAX];
    int more_next_line;
    off_t more_byte_offset;
    bool more_mid_line;
    bool more_bare;
    bool more_valid;
    agent_bash_job *bash_jobs;
    int next_bash_job_id;
    bool raw_mode_needs_restore;
    /* Watchdog: calls executed since the last workspace mutation. */
    struct { uint64_t hash; int count; } wd_calls[32];
    int wd_ncalls;
    int wd_blocked;     /* duplicate calls refused this user turn */
    int wd_stagnant;    /* consecutive blocked / identically failing calls */
    uint64_t wd_last_err;
    uint64_t wd_prev_hash;   /* previous executed call, for back-to-back repeats */
    int wd_prev_run;
    agent_task_state st;
    /* Remote backend: OpenAI-format message objects (JSON text), system first. */
    char **rmsg;
    int nrmsg, rmsg_cap;
    int remote_ctx;
    int remote_used;            /* prompt+completion tokens of the last reply */
    agent_metrics m;
    double session_t0;
} agent_worker;

static unsigned agent_next_prefill_label(void);

typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    p36_engine *engine;
    agent_worker *worker;
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_hint;
    bool md_line_started;
    char md_hint_prefix[sizeof("> **Hint:**") - 1];
    size_t md_hint_prefix_len;
    char md_inline[4096];
    size_t md_inline_len;
    bool md_escape;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    bool md_code_highlight_upto;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

typedef struct {
    char *name;
    char *value;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

typedef struct {
    char *text;
    size_t len;
    size_t cap;
} agent_tool_text_part;

typedef struct {
    agent_tool_text_part *parts;
    size_t part_count;
    size_t part_cap;
    p36_vision_embedding *images;
    size_t image_count;
    size_t image_cap;
} agent_tool_observation;

typedef enum {
    AGENT_QWEN_TOOL_SEARCH,
    AGENT_QWEN_TOOL_STRUCTURAL,
    AGENT_QWEN_TOOL_PARAM_VALUE,
    AGENT_QWEN_TOOL_DONE,
    AGENT_QWEN_TOOL_ERROR,
} agent_qwen_tool_state;

typedef struct {
    agent_qwen_tool_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    agent_tool_call current;
    char *param_name;
    size_t param_value_start;
    bool param_close_prefix;
    agent_tool_calls calls;
    char error[160];
} agent_qwen_tool_parser;

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

typedef struct {
    char tail[32];
    size_t len;
} agent_qwen_tool_marker_detector;

typedef struct {
    agent_token_renderer *renderer;
    agent_qwen_tool_parser *parser;
    agent_tool_visualizer viz;
    bool in_think;
    bool qwen_tool_active;
    bool qwen_tool_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char qwen_tool_start_tail[64];
    size_t qwen_tool_start_len;
    agent_qwen_tool_marker_detector plain_qwen_tool;
    agent_qwen_tool_marker_detector think_qwen_tool;
    bool qwen_tool_in_think;
    bool qwen_tool_in_think_reported;
    bool post_think_gap;
    bool tool_preflight_error;
    char tool_preflight_error_msg[256];
} agent_stream_renderer;

typedef struct {
    bool active;
    bool done;
} agent_edit_upto_forcer;

static volatile sig_atomic_t agent_sigint;
static agent_worker *agent_completion_worker;

static void worker_apply_pending_power(agent_worker *w);
static void agent_trace(agent_worker *w, const char *fmt, ...);
static void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
static void agent_publish_system_status(agent_worker *w, const char *msg);
static int agent_web_confirm(void *privdata, const char *message,
                             char *err, size_t err_len);
static void agent_web_log(void *privdata, const char *message);
static bool agent_preflight_edit_old(agent_worker *w, const agent_tool_call *call,
                                     char *err, size_t err_len);
static int agent_worker_sync_tokens(agent_worker *w, const p36_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
static int agent_compact_reserve_tokens(agent_worker *w);
static void agent_format_welcome_banner(const agent_config *cfg,
                                        char *buf, size_t len);

/* ============================================================================
 * Small Utilities And Command-Line Parsing
 * ============================================================================
 */

static void agent_sigint_handler(int sig) {
    (void)sig;
    agent_sigint = 1;
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("p36-agent: malloc");
        exit(1);
    }
    return p;
}

static char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static bool agent_expand_home_path(const char *path, char *out, size_t out_len) {
    if (!path || !out || out_len == 0) return false;
    const char *home = getenv("HOME");
    if (path[0] != '~' || (path[1] != '/' && path[1] != '\0') ||
        !home || !home[0]) {
        return snprintf(out, out_len, "%s", path) < (int)out_len;
    }
    return snprintf(out, out_len, "%s%s", home, path + 1) < (int)out_len;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        perror("p36-agent: realloc");
        exit(1);
    }
    return p;
}

static void agent_worker_images_clear(agent_worker *w) {
    if (!w) return;
    for (size_t i = 0; i < w->image_count; i++)
        p36_vision_embedding_free(&w->images[i].embedding);
    free(w->images);
    w->images = NULL;
    w->image_count = 0;
    w->image_cap = 0;
}

static void agent_worker_images_append(agent_worker *w,
                                       p36_vision_span *spans,
                                       size_t count) {
    if (!count) return;
    if (w->image_count + count > w->image_cap) {
        size_t cap = w->image_cap ? w->image_cap : 4;
        while (cap < w->image_count + count) cap *= 2;
        w->images = xrealloc(w->images, cap * sizeof(w->images[0]));
        w->image_cap = cap;
    }
    memcpy(w->images + w->image_count, spans, count * sizeof(spans[0]));
    w->image_count += count;
    memset(spans, 0, count * sizeof(spans[0]));
}


static void write_all(int fd, const char *p, size_t n) {
    if (fd == STDOUT_FILENO) {
        linenoiseWrite(fd, p, n);
        return;
    }
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr <= 0) {
            if (wr < 0 && errno == EINTR) continue;
            return;
        }
        p += wr;
        n -= (size_t)wr;
    }
}

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

static void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "p36-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "p36-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

static bool parse_steering_level(const char *arg, float *out) {
    char *end = NULL;
    errno = 0;
    float v = strtof(arg, &end);
    if (!arg[0] || *end || errno == ERANGE || !isfinite(v) ||
        v < -100.0f || v > 100.0f) return false;
    *out = v;
    return true;
}

static bool agent_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}

static bool agent_slash_command_known(const char *cmd) {
    return !strcmp(cmd, "/help") ||
           !strcmp(cmd, "/save") ||
           !strcmp(cmd, "/compact") ||
           !strcmp(cmd, "/list") ||
           !strcmp(cmd, "/quit") ||
           !strcmp(cmd, "/exit") ||
           !strcmp(cmd, "/new") ||
           agent_slash_command_with_args(cmd, "/power") ||
           agent_slash_command_with_args(cmd, "/steer") ||
           agent_slash_command_with_args(cmd, "/hints") ||
           agent_slash_command_with_args(cmd, "/switch") ||
           agent_slash_command_with_args(cmd, "/del") ||
           agent_slash_command_with_args(cmd, "/strip") ||
           agent_slash_command_with_args(cmd, "/history");
}

static uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "p36-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "p36-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static p36_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return P36_BACKEND_METAL;
    if (!strcmp(s, "vulkan")) return P36_BACKEND_VULKAN;
    if (!strcmp(s, "cpu")) return P36_BACKEND_CPU;
    fprintf(stderr, "p36-agent: invalid backend: %s\n", s);
    fprintf(stderr, "p36-agent: valid backends are: metal, vulkan, cpu\n");
    exit(2);
}

static p36_backend default_backend(void) {
#ifdef P36_NO_GPU
    return P36_BACKEND_CPU;
#elif defined(__APPLE__)
    return P36_BACKEND_METAL;
#else
    return P36_BACKEND_VULKAN;
#endif
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static void usage(FILE *fp, const char *topic) {
    p36_help_print(fp, P36_HELP_AGENT, topic);
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc || p36_cli_token_is_option(argv[*i + 1])) {
        fprintf(stderr, "p36-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

