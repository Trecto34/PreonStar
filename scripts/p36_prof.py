#!/usr/bin/env python3
"""p36-prof: where every microsecond of a token goes, per model.

Two passes per model, both short (BC-250 rule: ctx 512-1024, few tokens):
  1. wall  - plain p36-bench, no profiling: the true prefill/decode speed.
  2. prof  - P36_VK_PROF_TIMELINE + _KERNEL + _OP: every GPU dispatch group with
             start/end timestamps, plus host record/submit/wait counters.

The prof pass timestamps each dispatch group at the compute stage, which drains
the pipe, so it serializes kernels that normally overlap.  Its gaps between
kernels are partly an artifact; "wall ms/token" from pass 1 is the truth and
"GPU busy" from pass 2 is an upper bound on kernel time.  Do not plan fusions
from gap_us alone (see karpathy/AlreadyTried.md, decode barrier audit).

Usage:
  scripts/p36_prof.py -m MODEL.gguf [-m MODEL2.gguf ...] [--ctx 1024] [-n 64]
                      [--out DIR] [--bin DIR]
Writes per model: summary.json, decode_kernels.csv, prefill_kernels.csv,
decode_anatomy.csv (every action of one token, in order); plus index.html
comparing all models.
"""
import argparse, csv, html, json, os, re, statistics, subprocess, sys, time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = os.path.join(REPO, "tests", "long_context_story_prompt.txt")


def run_bench(bin_dir, model, ctx, n, env_extra, log_path):
    env = dict(os.environ, **env_extra)
    cmd = [os.path.join(bin_dir, "p36-bench"), "--vulkan", "-m", model,
           "--prompt-file", PROMPT, "--ctx-start", str(ctx), "--ctx-max", str(ctx), "-n", str(n)]
    with open(log_path, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    text = open(log_path).read()
    if rc != 0:
        sys.exit(f"p36-bench failed rc={rc}, see {log_path}")
    m = re.search(rf"^{ctx},(\d+),([\d.]+),(\d+),([\d.]+),", text, re.M)
    if not m:
        sys.exit(f"no result line in {log_path}")
    return text, float(m.group(2)), float(m.group(4))


def read_timeline(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        seq, flush, kpath, groups, t0, t1, dur, host = line.split()
        rows.append((os.path.basename(kpath).removesuffix(".spv"), int(groups), int(t0), int(t1), int(dur)))
    return rows


def split_tokens(rows):
    """A token ends with its sampling kernel (the last top2 group of a run)."""
    segs, cur = [], []
    for i, r in enumerate(rows):
        cur.append(r)
        if r[0].startswith("top2") and (i + 1 == len(rows) or not rows[i + 1][0].startswith("top2")):
            segs.append(cur)
            cur = []
    return segs


def kernel_table(segs, tick_ns):
    per = defaultdict(lambda: [0, 0.0])  # calls, dur_ns
    for s in segs:
        for name, _, _, _, dur in s:
            per[name][0] += 1
            per[name][1] += dur
    n = len(segs)
    busy = sum(v[1] for v in per.values())
    out = [{"kernel": k, "calls_per_tok": round(c / n, 2), "us_per_tok": round(d / n / 1e3, 2),
            "us_per_call": round(d / c / 1e3, 2), "pct": round(100 * d / busy, 2)}
           for k, (c, d) in per.items()]
    return sorted(out, key=lambda r: -r["us_per_tok"])


def anatomy(segs, tick_ns):
    """Every action of one decode token, in order, averaged over tokens."""
    n = len(segs[0])
    segs = [s for s in segs if len(s) == n and [r[0] for r in s] == [r[0] for r in segs[0]]]
    out, seen = [], defaultdict(int)
    for i in range(n):
        name = segs[0][i][0]
        seen[name] += 1
        durs = [s[i][4] / 1e3 for s in segs]
        gaps = [((s[i][2] - s[i - 1][3]) * tick_ns / 1e3) if i else 0.0 for s in segs]
        out.append({"pos": i, "kernel": name, "occurrence": seen[name], "groups": segs[0][i][1],
                    "us": round(statistics.median(durs), 2), "gap_us": round(statistics.median(gaps), 2)})
    return out, len(segs)


def host_counters(text):
    """Decode-labelled host counters from P36_VK_PROF_OP output."""
    i = text.find("prof label=decode")
    m = re.search(r"dispatches=(\d+) record_ms=([\d.]+) flushes=(\d+) submit_wait_ms=([\d.]+)", text[i:]) if i >= 0 else None
    return {k: float(v) for k, v in zip(("dispatches", "record_ms", "flushes", "submit_wait_ms"), m.groups())} if m else {}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def profile(model, args):
    name = os.path.basename(model).removesuffix(".gguf")
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    print(f"[{name}] wall pass", flush=True)
    _, pf_tps, dec_tps = run_bench(args.bin, model, args.ctx, args.n, {}, os.path.join(d, "wall.log"))
    print(f"[{name}] prof pass", flush=True)
    tl = os.path.join(d, "timeline.txt")
    text, pf_tps_p, dec_tps_p = run_bench(args.bin, model, args.ctx, args.n, {
        "P36_VK_PROF_TIMELINE": tl, "P36_VK_PROF_KERNEL": "1", "P36_VK_PROF_OP": "1", "P36_VK_PROF_DECODE": "1"},
        os.path.join(d, "prof.log"))
    with open(tl) as f:
        tick_ns = float(re.search(r"tick_period_ns=([\d.]+)", f.read(512)).group(1))
    segs = split_tokens(read_timeline(tl))
    if len(segs) < 4:
        sys.exit(f"[{name}] only {len(segs)} token segments in timeline; is top2 the sampler?")
    prefill, decode = segs[0], segs[2:]  # segs[1] = first decode token (warm-up), skipped
    busy = statistics.median(sum(r[4] for r in s) / 1e3 for s in decode)
    span = statistics.median((s[-1][3] - s[0][2]) * tick_ns / 1e3 for s in decode)
    host = host_counters(text)
    toks = args.n
    summary = {
        "model": name, "ctx": args.ctx, "gen": args.n,
        "wall_prefill_tps": pf_tps, "wall_decode_tps": dec_tps,
        "wall_ms_per_tok": round(1000 / dec_tps, 3),
        "prof_decode_tps": dec_tps_p,
        "gpu_busy_ms_per_tok": round(busy / 1e3, 3),
        "gpu_span_ms_per_tok": round(span / 1e3, 3),
        "gpu_gaps_ms_per_tok": round((span - busy) / 1e3, 3),
        "groups_per_tok": len(decode[0]),
        "host_dispatches_per_tok": round(host.get("dispatches", 0) / toks, 1),
        "host_record_ms_per_tok": round(host.get("record_ms", 0) / toks, 3),
        "host_flushes_per_tok": round(host.get("flushes", 0) / toks, 2),
        "host_submit_wait_ms_per_tok": round(host.get("submit_wait_ms", 0) / toks, 3),
        "prefill_gpu_ms": round(sum(r[4] for r in prefill) / 1e6, 2),
    }
    dk = kernel_table(decode, tick_ns)
    pk = kernel_table([prefill], tick_ns)
    an, n_same = anatomy(decode, tick_ns)
    summary["anatomy_tokens"] = n_same
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
    write_csv(os.path.join(d, "decode_kernels.csv"), dk)
    write_csv(os.path.join(d, "prefill_kernels.csv"), pk)
    write_csv(os.path.join(d, "decode_anatomy.csv"), an)
    os.remove(tl)  # large; the CSVs carry everything derived from it
    print(f"[{name}] decode {dec_tps:.2f} t/s = {summary['wall_ms_per_tok']} ms/tok wall; "
          f"GPU busy {summary['gpu_busy_ms_per_tok']} ms, span {summary['gpu_span_ms_per_tok']} ms (serialized); "
          f"{summary['groups_per_tok']} groups/tok", flush=True)
    for r in dk[:8]:
        print(f"   {r['kernel']:<44} {r['us_per_tok']:>8.1f} us/tok {r['pct']:>5.1f}%  x{r['calls_per_tok']}")
    return summary, dk


def table(rows, cols):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def write_html(path, results):
    keys = [k for k in results[0][0] if k != "model"]
    comp = [{"metric": k, **{s["model"]: s[k] for s, _ in results}} for k in keys]
    parts = ["<!doctype html><meta charset=utf-8><title>p36-prof</title><style>"
             "body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
             "table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}"
             "td:first-child,th:first-child{text-align:left}</style>",
             f"<h1>p36-prof</h1><p>{time.strftime('%Y-%m-%d %H:%M')}. Wall numbers come from an unprofiled pass; "
             "GPU numbers from a serialized timestamp pass (upper bound per kernel).</p>",
             "<h2>Models</h2>", table(comp, ["metric"] + [s["model"] for s, _ in results])]
    for s, dk in results:
        parts += [f"<h2>{html.escape(s['model'])}: decode kernels per token</h2>",
                  table(dk, ["kernel", "calls_per_tok", "us_per_tok", "us_per_call", "pct"]),
                  f"<p>Every action of one token: <code>{html.escape(s['model'])}/decode_anatomy.csv</code></p>"]
    open(path, "w").write("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", action="append", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--out", default=os.path.expanduser("~/q36-ab/p36-prof"))
    ap.add_argument("--bin", default=REPO)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gov = subprocess.run(["systemctl", "is-active", "cyan-skillfish-governor-smu"], capture_output=True, text=True).stdout.strip()
    if gov != "active":
        print(f"WARNING: cyan-skillfish-governor-smu is {gov or 'unknown'}; numbers will read ~20-28% slow", file=sys.stderr)
    results = [profile(m, args) for m in args.model]
    write_html(os.path.join(args.out, "index.html"), results)
    print(f"report: {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
