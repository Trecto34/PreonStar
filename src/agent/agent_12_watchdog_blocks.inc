static bool agent_watchdog_blocks(agent_worker *w, const agent_tool_call *c,
                                  int max_repeat) {
    uint64_t h = agent_tool_call_hash(c);
    /* Any tool: the same call again and again back to back is a loop (an edit
     * whose result does not change, a polling bash, ...). */
    int run = h == w->wd_prev_hash ? w->wd_prev_run + 1 : 1;
    if (run > max_repeat) return true;
    w->wd_prev_hash = h;
    w->wd_prev_run = run;
    if (!agent_tool_is_readonly(c->name)) {
        w->wd_ncalls = 0;
        return false;
    }
    for (int i = 0; i < w->wd_ncalls; i++) {
        if (w->wd_calls[i].hash != h) continue;
        if (w->wd_calls[i].count >= max_repeat) return true;
        w->wd_calls[i].count++;
        return false;
    }
    if (w->wd_ncalls == 32) {
        memmove(w->wd_calls, w->wd_calls + 1, 31 * sizeof(w->wd_calls[0]));
        w->wd_ncalls--;
    }
    w->wd_calls[w->wd_ncalls].hash = h;
    w->wd_calls[w->wd_ncalls].count = 1;
    w->wd_ncalls++;
    return false;
}


/* ---- Verification-convergence watchdog ----------------------------------
 * Phases: IMPLEMENT (default; target edits always return here), VERIFY, then
 * FINALIZE_REQUIRED.  Only strong evidence (a known check command, a helper run,
 * a page visit after a target edit) enters VERIFY; reads and searches never do,
 * since they are also how the agent recovers after compaction.  In VERIFY the
 * agent gets a bounded number of steps (--max-verify-steps) to check its work.  Edits to
 * disposable verification helpers (/tmp scripts, files named like verify, probe
 * or mock that the agent itself created) never count as progress, near-identical re-checks cost
 * double, and --max-no-progress-steps is a ceiling that ignores that weighting.
 * When the budget is spent the state flips to FINALIZE_REQUIRED: further tool
 * calls are refused with a short notice and the model must give a final answer.
 * Nothing here adds to the system prompt; the notice is only injected when it
 * fires.  The budget is armed by the first target mutation, so pure
 * read-only investigations are not limited by it.
 *
 * Mutation detection is path based for write/edit and content-hash based for
 * bash (files the agent already touched).  A bash command that creates a brand
 * new file outside those is treated as non-mutating: the bounded fallback. */
typedef enum { AGENT_VW_VERIFY, AGENT_VW_TARGET, AGENT_VW_HELPER } agent_vw_kind;

typedef struct {
    bool existed;
    uint64_t hash;
    char path[200];
    int nfiles;
    uint64_t fhash[AGENT_ST_FILES];
    char fpath[AGENT_ST_FILES][200];
} agent_vw_pre;

static uint64_t agent_vw_file_hash(const char *path) {
    FILE *fp = path && path[0] ? fopen(path, "rb") : NULL;
    if (!fp) return 0;
    uint64_t h = 1469598103934665603ULL;
    unsigned char buf[65536];
    size_t n, total = 0;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= 1099511628211ULL; }
        total += n;
        if (total > (8u << 20)) break;   /* bounded cost; huge files hash their first 8 MiB */
    }
    fclose(fp);
    return h ? h : 1;
}

/* A temp-dir path is a helper only when it is outside the workspace: an agent
 * working in /tmp/project still edits real targets there. */
static bool agent_vw_tmp_path(const char *p) {
    static const char *const roots[] = {"/tmp/", "/var/tmp/", "/dev/shm/"};
    if (!p || p[0] != '/') return false;     /* relative paths are inside the workspace */
    bool tmp = false;
    for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]) && !tmp; i++)
        tmp = !strncmp(p, roots[i], strlen(roots[i]));
    if (!tmp) {
        const char *t = getenv("TMPDIR");
        size_t n = t ? strlen(t) : 0;
        while (n > 1 && t[n - 1] == '/') n--;
        tmp = n > 1 && !strncmp(p, t, n) && p[n] == '/';
    }
    if (!tmp) return false;
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd))) {
        size_t n = strlen(cwd);
        if (!strncmp(p, cwd, n) && (p[n] == '/' || p[n] == '\0')) return false;
    }
    return true;
}

static bool agent_vw_helper_name(const char *p) {
    static const char *const pat[] = {"verif", "probe", "mock", "harness", "scratch",
                                      "repro", "smoke", "sanity", "tmp_", "_tmp", "debug_"};
    if (!p) return false;
    const char *base = strrchr(p, '/');
    base = base ? base + 1 : p;
    char lower[160];
    size_t n = strlen(base);
    if (n >= sizeof(lower)) n = sizeof(lower) - 1;
    for (size_t i = 0; i < n; i++) lower[i] = (char)tolower((unsigned char)base[i]);
    lower[n] = 0;
    for (size_t i = 0; i < sizeof(pat) / sizeof(pat[0]); i++)
        if (strstr(lower, pat[i])) return true;
    return false;
}

static bool agent_vw_known_helper(const agent_verify_state *v, const char *p) {
    for (int i = 0; i < v->nhelpers; i++)
        if (!strcmp(v->helpers[i], p)) return true;
    return false;
}

/* A path is a verification helper when it lives in a temp dir, was created by
 * the agent this turn under a helper-looking name, or is such a file edited
 * again.  An existing product file that merely has "verify" in its name is a
 * target: only files the agent created itself can be helpers by name. */
static bool agent_vw_path_is_helper(const agent_verify_state *v, const char *p, bool existed) {
    if (!p || !p[0]) return false;
    return agent_vw_tmp_path(p) || agent_vw_known_helper(v, p) ||
           (!existed && agent_vw_helper_name(p));
}

static void agent_vw_remember_helper(agent_verify_state *v, const char *p) {
    if (!p || !p[0] || agent_vw_known_helper(v, p)) return;
    if (v->nhelpers == AGENT_VW_HELPERS) {
        memmove(v->helpers[0], v->helpers[1], (AGENT_VW_HELPERS - 1) * sizeof(v->helpers[0]));
        v->nhelpers--;
    }
    snprintf(v->helpers[v->nhelpers++], sizeof(v->helpers[0]), "%s", p);
}

static bool agent_vw_stagnant(const agent_verify_state *v) {
    return v->armed && !v->finalize && v->phase == AGENT_PH_VERIFY && (v->dups >= 3 || v->helper_since >= 3);
}

/* Order-insensitive-ish fingerprint of what a non-mutating call looks at, with
 * helper script names and numbers masked so `node /tmp/verify.mjs` and
 * `node /tmp/verify2.mjs --n 7` count as the same check.  helperish reports
 * whether the call runs or reads a verification helper. */
static uint64_t agent_vw_signature(const agent_verify_state *v, const agent_tool_call *c,
                                   bool *helperish) {
    const char *name = c->name ? c->name : "?";
    uint64_t h = agent_fnv1a(1469598103934665603ULL, name);
    bool helper = false;
    if (!strcmp(name, "bash")) {
        const char *cmd = agent_tool_arg_value(c, "command");
        char tok[256];
        size_t tl = 0;
        for (const char *p = cmd ? cmd : "";; p++) {
            if (*p && !isspace((unsigned char)*p)) {
                if (tl + 1 < sizeof(tok) && *p != '"' && *p != '\'')
                    tok[tl++] = (char)(isdigit((unsigned char)*p) ? '#' : *p);
                continue;
            }
            if (tl) {
                tok[tl] = 0;
                const bool is_h = agent_vw_tmp_path(tok) || agent_vw_known_helper(v, tok) ||
                                  (strchr(tok, '.') && agent_vw_helper_name(tok));
                helper = helper || is_h;
                h = agent_fnv1a(h, is_h ? "<H>" : tok);
                h = agent_fnv1a(h, " ");
                tl = 0;
            }
            if (!*p) break;
        }
    } else {
        static const char *const keys[] = {"path", "query", "url"};
        for (size_t k = 0; k < sizeof(keys) / sizeof(keys[0]); k++) {
            const char *a = agent_tool_arg_value(c, keys[k]);
            if (!a) continue;
            helper = helper || agent_vw_tmp_path(a) || agent_vw_known_helper(v, a);
            h = agent_fnv1a(h, a);
            h = agent_fnv1a(h, ";");
        }
    }
    if (helperish) *helperish = helper;
    return h;
}

/* Strong, deterministic "the agent is checking a finished target" signal: a known
 * test/syntax/lint/browser command, or a run of a verification helper.  Plain
 * reads, greps, listings and wc are implementation tools too, so never count. */
static bool agent_vw_check_command(const char *cmd) {
    static const char *const pat[] = {
        "node --check", "node -c ", "pytest", "unittest", "make test", "make check", "npm test",
        "npm run test", "npm run lint", "npm run build", "cargo test", "cargo check", "go test",
        "go vet", "ctest", "py_compile", "tsc", "eslint", "ruff", "flake8", "mypy", "shellcheck",
        "playwright", "puppeteer", "headless", "chromium", "google-chrome", "firefox"};
    for (size_t i = 0; cmd && i < sizeof(pat) / sizeof(pat[0]); i++)
        if (strstr(cmd, pat[i])) return true;
    return false;
}

/* "MISSING:" followed by a "- item" line in a tool result is concrete evidence that
 * required work is still absent.  Returns the first item (clipped) or NULL. */
static const char *agent_vw_missing_item(const char *res) {
    const char *p = strstr(res, "MISSING:\n- ");
    return p ? p + 11 : NULL;
}

static bool agent_vw_bash_writes(const char *cmd) {
    static const char *const ops[] = {">", "tee ", "sed -i", " mv ", " cp ", " rm ", "touch ", "mkdir ", "install "};
    for (size_t i = 0; cmd && i < sizeof(ops) / sizeof(ops[0]); i++)
        if (strstr(cmd, ops[i])) return true;
    return false;
}

static void agent_vw_before(const agent_task_state *st, const agent_tool_call *c, agent_vw_pre *pre) {
    memset(pre, 0, sizeof(*pre));
    const char *name = c->name ? c->name : "";
    const char *path = agent_tool_arg_value(c, "path");
    if ((!strcmp(name, "write") || !strcmp(name, "edit")) && path) {
        snprintf(pre->path, sizeof(pre->path), "%s", path);
        pre->hash = agent_vw_file_hash(path);
        pre->existed = pre->hash != 0;
    } else if (!strcmp(name, "bash")) {
        pre->nfiles = st->nfiles;
        for (int i = 0; i < st->nfiles; i++) {
            snprintf(pre->fpath[i], sizeof(pre->fpath[i]), "%s", st->files[i].path);
            pre->fhash[i] = agent_vw_file_hash(pre->fpath[i]);
        }
    }
}

/* Advance the counters for one tool call.  Returns true exactly once per turn:
 * on the call that exhausts the budget.  dup_out reports a repeated check. */
static bool agent_vw_step(agent_verify_state *v, agent_metrics *m, agent_vw_kind kind,
                          uint64_t sig, const char *path, bool check, bool *dup_out) {
    bool dup = false;
    v->step++;
    if (dup_out) *dup_out = false;
    /* Emergency ceiling against mutation thrashing: counts every IMPLEMENT step and, unlike
     * the progress counters below, is not reset by target edits. */
    bool impl_hit = false;
    if (v->phase == AGENT_PH_IMPLEMENT && !v->finalize) {
        v->impl_steps++;
        m->impl_steps++;
        if (v->max_impl > 0 && v->impl_steps >= v->max_impl) {
            v->finalize = v->impl_stop = impl_hit = true;
            m->impl_limit_hits++;
        }
    }
    if (kind == AGENT_VW_TARGET) {
        v->armed = true;
        v->last_target_step = v->step;
        snprintf(v->last_target, sizeof(v->last_target), "%s", path ? path : "?");
        v->verify_used = v->no_progress = v->dups = v->helper_since = 0;
        v->nsigs = 0;
        v->npending = 0;
        v->pending[0] = 0;
        if (v->phase == AGENT_PH_VERIFY) m->ph_vi++;
        v->phase = AGENT_PH_IMPLEMENT;
        m->target_mutations++;
        return impl_hit;
    }
    if (kind == AGENT_VW_HELPER) { v->helper_since++; m->helper_mutations++; }
    if (impl_hit) return true;
    if (!v->armed) return false;
    if (v->phase == AGENT_PH_IMPLEMENT) {
        /* Reads, searches and listings are how an agent recovers after compaction, so
         * they never move it to VERIFY; only a check command (or a helper run) does.
         * Uncertain means IMPLEMENT: just a generous ceiling against pathological loops. */
        if (check && !v->npending) {
            v->phase = AGENT_PH_VERIFY;
            m->ph_iv++;
            v->no_progress = v->dups = v->helper_since = 0;
            v->nsigs = 0;
        } else {
            v->no_progress++;
            m->no_progress_steps++;
            if (v->finalize || v->max_noprog <= 0 ||
                v->no_progress < v->max_noprog * AGENT_VW_IMPL_CEILING) return false;
            v->finalize = true;
            m->verify_exhaustions++;
            return true;
        }
    }
    for (int i = 0; i < v->nsigs; i++)
        if (v->sigs[i] == sig) { dup = true; break; }
    if (dup) {
        v->dups++;
    } else {
        if (v->nsigs == AGENT_VW_SIGS) {
            memmove(v->sigs, v->sigs + 1, (AGENT_VW_SIGS - 1) * sizeof(v->sigs[0]));
            v->nsigs--;
        }
        v->sigs[v->nsigs++] = sig;
    }
    if (dup_out) *dup_out = dup;
    v->verify_used += (dup || agent_vw_stagnant(v)) ? 2 : 1;
    v->no_progress++;
    m->verify_steps++;
    m->no_progress_steps++;
    if (v->finalize) return false;
    if (!((v->max_verify > 0 && v->verify_used >= v->max_verify) ||
          (v->max_noprog > 0 && v->no_progress >= v->max_noprog)))
        return false;
    v->finalize = true;
    m->ph_vf++;
    m->verify_exhaustions++;
    return true;
}

static void agent_vw_begin_turn(agent_task_state *st, const agent_generation_options *g) {
    memset(&st->vw, 0, sizeof(st->vw));
    st->vw.max_verify = g ? g->max_verify_steps : AGENT_MAX_VERIFY_DEFAULT;
    st->vw.max_noprog = g ? g->max_no_progress_steps : AGENT_MAX_NOPROGRESS_DEFAULT;
    st->vw.max_impl = g ? g->max_implementation_steps : AGENT_MAX_IMPL_DEFAULT;
    for (int i = 0; i < st->nunresolved; i++) free(st->unresolved[i]);
    for (int i = 0; i < st->ndnr; i++) free(st->dnr[i]);
    st->nunresolved = st->ndnr = 0;
}

static const char agent_vw_notice_tail[] =
    "Verification/investigation budget exhausted.\n"
    "Summarize the current result and unresolved limitations: say what is confirmed "
    "working, what is confirmed failing, and what is inconclusive.\n"
    "Do not call additional tools.\n";

static char *agent_vw_notice(const agent_verify_state *v) {
    char b[512];
    if (v->impl_stop) {
        snprintf(b, sizeof(b),
                 "IMPLEMENTATION_STEP_LIMIT_REACHED\n%d implementation steps this turn without finishing.\n"
                 "Give a concise INCOMPLETE/STOPPED report: which files you modified, what is done, "
                 "and what work remains unresolved. Do not claim success.\n"
                 "Do not call additional tools.\n", v->impl_steps);
        return xstrdup(b);
    }
    snprintf(b, sizeof(b), "VERIFICATION_BUDGET_EXHAUSTED\nNo target files changed since step %d.\n%s",
             v->last_target_step, agent_vw_notice_tail);
    return xstrdup(b);
}

/* Implementation-ceiling variant: an INCOMPLETE report of modified files and open work. */
static char *agent_vw_impl_text(const agent_task_state *st) {
    agent_buf b = {0};
    char tmp[400];
    snprintf(tmp, sizeof(tmp),
             "Stopping: INCOMPLETE. The implementation step limit was reached (%d steps this turn).\n"
             "Target files modified:\n", st->vw.impl_steps);
    agent_buf_puts(&b, tmp);
    int n = 0;
    for (int i = 0; i < st->nfiles; i++) {
        if (!st->files[i].note[0]) continue;
        snprintf(tmp, sizeof(tmp), "- %s: %s\n", st->files[i].path, st->files[i].note);
        agent_buf_puts(&b, tmp);
        n++;
    }
    if (!n) agent_buf_puts(&b, "- none recorded\n");
    agent_buf_puts(&b, "Unresolved:\n");
    if (st->vw.npending) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->vw.pending);
        agent_buf_puts(&b, tmp);
    }
    for (int i = st->nunresolved > 3 ? st->nunresolved - 3 : 0; i < st->nunresolved; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->unresolved[i]);
        agent_buf_puts(&b, tmp);
    }
    agent_buf_puts(&b, "- the task was not shown to be complete\n");
    agent_buf_puts(&b, "This is a stop, not a claim that the task is complete.\n");
    return agent_buf_take(&b);
}

/* Controller-written answer, used only when the model keeps calling tools
 * after being told not to.  It reports tracked facts, not success. */
static char *agent_vw_final_text(const agent_task_state *st) {
    if (st->vw.impl_stop) return agent_vw_impl_text(st);
    agent_buf b = {0};
    char tmp[400];
    snprintf(tmp, sizeof(tmp),
             "Stopping: the verification budget is exhausted (no target change since step %d%s%s).\n",
             st->vw.last_target_step, st->vw.last_target[0] ? ", last edit " : "", st->vw.last_target);
    agent_buf_puts(&b, tmp);
    agent_buf_puts(&b, "Checks that ran:\n");
    for (int i = st->nfacts > 3 ? st->nfacts - 3 : 0; i < st->nfacts; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->facts[i]);
        agent_buf_puts(&b, tmp);
    }
    if (!st->nfacts) agent_buf_puts(&b, "- none recorded\n");
    agent_buf_puts(&b, "Unresolved:\n");
    for (int i = st->nunresolved > 3 ? st->nunresolved - 3 : 0; i < st->nunresolved; i++) {
        snprintf(tmp, sizeof(tmp), "- %s\n", st->unresolved[i]);
        agent_buf_puts(&b, tmp);
    }
    if (!st->nunresolved) agent_buf_puts(&b, "- none recorded\n");
    agent_buf_puts(&b, "This is a stop, not a claim that every check passed.\n");
    return agent_buf_take(&b);
}

/* Classify the call that just ran, advance the budget, record what the
 * controller learned, and append the exhaustion notice when this call spent
 * the budget.  Takes ownership of res.  executed is false for calls the repeat
 * watchdog refused: those still burn budget as duplicate checks. */
static char *agent_vw_after(agent_worker *w, const agent_tool_call *c,
                            const agent_vw_pre *pre, bool executed, char *res) {
    agent_task_state *st = &w->st;
    agent_verify_state *v = &st->vw;
    const char *name = c->name ? c->name : "";
    const bool err = !strncmp(res, "Tool error", 10);
    agent_vw_kind kind = AGENT_VW_VERIFY;
    const char *tpath = NULL;
    char changed[200] = {0};

    if (executed && !err) {
        if ((!strcmp(name, "write") || !strcmp(name, "edit")) && pre->path[0]) {
            if (agent_vw_file_hash(pre->path) != pre->hash) {
                if (agent_vw_path_is_helper(v, pre->path, pre->existed)) {
                    kind = AGENT_VW_HELPER;
                    agent_vw_remember_helper(v, pre->path);
                } else {
                    kind = AGENT_VW_TARGET;
                }
                tpath = pre->path;
            }
        } else if (!strcmp(name, "bash")) {
            for (int i = 0; i < pre->nfiles; i++) {
                if (agent_vw_file_hash(pre->fpath[i]) == pre->fhash[i]) continue;
                if (agent_vw_path_is_helper(v, pre->fpath[i], true)) continue;
                snprintf(changed, sizeof(changed), "%s", pre->fpath[i]);
                kind = AGENT_VW_TARGET;
                tpath = changed;
                break;
            }
            const char *cmd = agent_tool_arg_value(c, "command");
            bool hp = false;
            if (kind == AGENT_VW_VERIFY && agent_vw_bash_writes(cmd)) {
                (void)agent_vw_signature(v, c, &hp);
                if (hp) kind = AGENT_VW_HELPER;
            }
        }
    }
    bool helperish = false;
    uint64_t sig = agent_vw_signature(v, c, &helperish);
    bool dup = false;
    const bool check = executed && kind == AGENT_VW_VERIFY &&
        ((!strcmp(name, "bash") && (helperish || agent_vw_check_command(agent_tool_arg_value(c, "command")))) ||
         !strcmp(name, "visit_page"));
    bool exhausted = agent_vw_step(v, &w->m, kind, sig, tpath, check, &dup);
    const char *miss = executed && !err ? agent_vw_missing_item(res) : NULL;
    if (miss && v->armed) {
        if (!v->npending) {
            size_t n = strcspn(miss, "\n");
            snprintf(v->pending, sizeof(v->pending), "%.*s", (int)(n < sizeof(v->pending) - 1 ? n : sizeof(v->pending) - 1), miss);
        }
        v->npending++;
        if (v->phase == AGENT_PH_VERIFY) {   /* work is missing: back to implementing, never finalize */
            if (exhausted) { v->finalize = false; exhausted = false; w->m.verify_exhaustions--; w->m.ph_vf--; }
            v->phase = AGENT_PH_IMPLEMENT;
            w->m.ph_vi++;
        }
    }

    if (kind == AGENT_VW_VERIFY) {
        /* A failing check is either a product bug or the agent's own probe being
         * wrong; neither resets the budget (only a target edit does), but the
         * label tells the final answer which one it is leaving open. */
        const char *ex = strstr(res, "exit_status=");   /* bash job result, see agent_bash_job_tool_result */
        const bool failed = err || (ex && atoi(ex + 12) != 0);
        if (failed && v->armed) {
            const char *cmd = agent_tool_arg_value(c, "command");
            char *lab = agent_st_clip(!strcmp(name, "bash") && cmd ? cmd : name, 80);
            char line[200];
            snprintf(line, sizeof(line), "%s %s", helperish ? "[TEST_HARNESS_FAILURE_OR_ASSUMPTION]" : "[PRODUCT_FAILURE?]", lab);
            free(lab);
            agent_st_push(st->unresolved, &st->nunresolved, xstrdup(line));
        }
        if (dup) {
            const char *cmd = agent_tool_arg_value(c, "command");
            const char *p = agent_tool_arg_value(c, "path");
            char *lab = agent_st_clip(!strcmp(name, "bash") && cmd ? cmd : p ? p : name, 80);
            char line[200];
            snprintf(line, sizeof(line), "do not repeat: %s %s", name, lab);
            free(lab);
            agent_st_push(st->dnr, &st->ndnr, xstrdup(line));
        }
    } else if (kind == AGENT_VW_HELPER && v->helper_since == 2) {
        agent_st_push(st->dnr, &st->ndnr,
            xstrdup("do not build another verification harness unless target code changes"));
    }
    agent_trace(w, "[VERIFY] step=%d kind=%s used=%d/%d no_progress=%d/%d dup=%d%s",
                v->step, kind == AGENT_VW_TARGET ? "target" : kind == AGENT_VW_HELPER ? "helper" : "check",
                v->verify_used, v->max_verify, v->no_progress, v->max_noprog, dup ? 1 : 0,
                agent_vw_stagnant(v) ? " VERIFY_STAGNANT" : "");
    if (exhausted) {
        agent_trace(w, "[VERIFY] FINALIZE_REQUIRED no target change since step %d", v->last_target_step);
        agent_buf b = {0};
        agent_buf_puts(&b, res);
        if (res[0] && res[strlen(res) - 1] != '\n') agent_buf_puts(&b, "\n");
        char *notice = agent_vw_notice(v);
        agent_buf_puts(&b, "\n");
        agent_buf_puts(&b, notice);
        free(notice);
        free(res);
        res = agent_buf_take(&b);
    }
    return res;
}

/* True when the model ignored the notice and kept calling tools: the turn ends
 * with a controller-written stop report instead of BLOCKED. */
