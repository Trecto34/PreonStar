#!/usr/bin/env python3
"""p36-prof: where every microsecond of a token goes, per model.

Passes per model, all short (BC-250 rule: ctx 512-1024, few tokens):
  1. wall   - plain p36-bench, no profiling: the true prefill/decode speed.
  2. prof   - P36_VK_PROF_TIMELINE + _KERNEL + _OP: every GPU dispatch group
              with timestamps and bytes read, plus host record/submit/wait.
  3. ablate - optional (--ablate-top N, default 6): compiles empty / loads-only
              / math-only variants of the top N instrumented decode shaders in a
              temp dir (production vulkan/*.spv untouched) and runs 3 short
              timeline passes to classify each kernel as memory / math / fixed.

The prof pass timestamps each dispatch group at the compute stage, which drains
the pipe, so it serializes kernels that normally overlap.  Its gaps between
kernels are partly an artifact; "wall ms/token" from pass 1 is the truth and
"GPU busy" from pass 2 is an upper bound on kernel time.  Do not plan fusions
from gap_us alone (see karpathy/AlreadyTried.md, decode barrier audit).

Usage:
  scripts/p36_prof.py -m MODEL.gguf [-m MODEL2.gguf ...] [--ctx 1024] [-n 64]
                      [--peak-bw 437.5] [--ablate-top 6] [--out DIR] [--bin DIR]
Writes per model: summary.json, decode_kernels.csv, prefill_kernels.csv,
decode_anatomy.csv (every action of one token, in order); plus index.html
comparing all models.
"""
import argparse, csv, html, json, os, re, shlex, shutil, statistics, subprocess, sys, tempfile, time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = os.path.join(REPO, "tests", "long_context_story_prompt.txt")


def run_bench(bin_dir, model, ctx, n, env_extra, log_path, cwd=None):
    env = dict(os.environ, **env_extra)
    cmd = [os.path.join(bin_dir, "p36-bench"), "--vulkan", "-m", model,
           "--prompt-file", PROMPT, "--ctx-start", str(ctx), "--ctx-max", str(ctx), "-n", str(n)]
    with open(log_path, "w") as f:
        rc = subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
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
        parts = line.split()
        if len(parts) < 8:
            continue
        kname = os.path.basename(parts[2]).removesuffix(".spv")
        nbytes = int(parts[8]) if len(parts) > 8 else 0
        rows.append((kname, int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6]), nbytes))
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


def kernel_table(segs, peak_bw):
    per = defaultdict(lambda: [0, 0.0, 0.0])  # calls, dur_ns, bytes
    for s in segs:
        for name, _, _, _, dur, nbytes in s:
            per[name][0] += 1
            per[name][1] += dur
            per[name][2] += nbytes
    n = len(segs)
    busy = sum(v[1] for v in per.values())
    out = []
    for k, (c, d, b) in per.items():
        gbs = (b / d) if d > 0 else 0.0
        out.append({
            "kernel": k,
            "calls_per_tok": round(c / n, 2),
            "us_per_tok": round(d / n / 1e3, 2),
            "us_per_call": round(d / c / 1e3, 2),
            "pct": round(100 * d / busy, 2) if busy > 0 else 0.0,
            "MB_per_tok": round(b / n / 1e6, 2),
            "GB_s": round(gbs, 1),
            "pct_bw": round(100 * gbs / peak_bw, 1) if peak_bw > 0 else 0.0,
        })
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
        nbytes = statistics.median(s[i][5] for s in segs)
        med_us = statistics.median(durs)
        gbs = (nbytes / (med_us * 1e3)) if med_us > 0 else 0.0
        out.append({
            "pos": i, "kernel": name, "occurrence": seen[name], "groups": segs[0][i][1],
            "us": round(med_us, 2), "gap_us": round(statistics.median(gaps), 2),
            "KB": round(nbytes / 1e3, 1), "GB_s": round(gbs, 1),
        })
    return out, len(segs)


def host_counters(text):
    """Decode-labelled host counters from P36_VK_PROF_OP output."""
    i = text.find("prof label=decode")
    m = re.search(r"dispatches=(\d+) record_ms=([\d.]+) flushes=(\d+) submit_wait_ms=([\d.]+)", text[i:]) if i >= 0 else None
    return {k: float(v) for k, v in zip(("dispatches", "record_ms", "flushes", "submit_wait_ms"), m.groups())} if m else {}


def parse_shader_rules():
    """Extract (src_comp, flags) for every vulkan/*.spv rule in Makefile."""
    rules = {}
    lines = open(os.path.join(REPO, "Makefile")).read().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^vulkan/([^:\s]+)\.spv:\s+vulkan/([^\s]+\.comp)", line)
        if m and i + 1 < len(lines) and "$(GLSLC)" in lines[i + 1]:
            toks = shlex.split(lines[i + 1].strip())
            flags = [t for t in toks if t.startswith("-D") or t.startswith("--target-env=")]
            rules[m.group(1)] = (os.path.join(REPO, "vulkan", m.group(2)), flags)
    return rules


def classify_bound(full_us, empty_us, loads_us, math_us):
    if full_us <= 0:
        return ""
    if empty_us / full_us >= 0.40:
        return "fixed"
    l_hi = loads_us >= 0.85 * full_us
    m_hi = math_us >= 0.85 * full_us
    if l_hi and not m_hi:
        return "memory"
    if m_hi and not l_hi:
        return "math"
    return "mem+math"


def run_ablation(model, name, dk, args, out_dir, rules):
    """Run batched empty / loads / math ablation for top-N instrumented kernels."""
    targets = []
    for r in dk:
        if len(targets) >= args.ablate_top:
            break
        k = r["kernel"]
        if k not in rules:
            continue
        src, flags = rules[k]
        txt = open(src).read()
        if "P36_ABLATE_EMPTY" in txt and "P36_ABLATE_LOADS" in txt and "P36_ABLATE_MATH" in txt:
            targets.append((k, src, flags))
    if not targets:
        return {}

    modes = [("empty", "-DP36_ABLATE_EMPTY"), ("loads", "-DP36_ABLATE_LOADS"), ("math", "-DP36_ABLATE_MATH")]
    ab_ctx = min(args.ctx, 512)
    ab_n = min(args.n, 32)
    glslc = os.path.join(REPO, "glslc")
    vk_src = os.path.join(REPO, "vulkan")
    per_mode = {}

    tmp = tempfile.mkdtemp(prefix="p36_ablate_")
    try:
        for mode, define in modes:
            root = os.path.join(tmp, mode)
            vk_dir = os.path.join(root, "vulkan")
            os.makedirs(vk_dir, exist_ok=True)
            swap = {k for k, _, _ in targets}
            for fn in os.listdir(vk_src):
                if fn.endswith(".spv") and fn.removesuffix(".spv") not in swap:
                    os.symlink(os.path.join(vk_src, fn), os.path.join(vk_dir, fn))
            for k, src, flags in targets:
                out_spv = os.path.join(vk_dir, k + ".spv")
                cmd = [glslc, *flags, define, "-o", out_spv, src]
                rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if rc.returncode != 0:
                    sys.exit(f"ablate compile failed for {k} ({mode}):\n{rc.stderr}")
            print(f"[{name}] ablate {mode} ({len(targets)} kernels, ctx={ab_ctx}, n={ab_n})", flush=True)
            tl = os.path.join(tmp, f"{mode}.timeline")
            run_bench(args.bin, model, ab_ctx, ab_n, {
                "P36_SHADER_DIR": root,
                "P36_VK_PROF_TIMELINE": tl, "P36_VK_PROF_KERNEL": "1", "P36_VK_PROF_DECODE": "1"},
                os.path.join(out_dir, f"ablate_{mode}.log"))
            segs = split_tokens(read_timeline(tl))
            dec = segs[2:] if len(segs) > 2 else segs
            kt = {r["kernel"]: r["us_per_tok"] for r in kernel_table(dec, args.peak_bw)}
            per_mode[mode] = kt
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = {}
    for k, _, _ in targets:
        out[k] = {m: per_mode[m].get(k, 0.0) for m, _ in modes}
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def profile(model, args, rules):
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
    tot_mb = statistics.median(sum(r[5] for r in s) / 1e6 for s in decode)
    wall_ms = 1000.0 / dec_tps
    eff_gbs = tot_mb / wall_ms if wall_ms > 0 else 0.0
    host = host_counters(text)
    toks = args.n
    summary = {
        "model": name, "ctx": args.ctx, "gen": args.n,
        "wall_prefill_tps": pf_tps, "wall_decode_tps": dec_tps,
        "wall_ms_per_tok": round(wall_ms, 3),
        "decode_MB_per_tok": round(tot_mb, 1),
        "effective_GB_s": round(eff_gbs, 1),
        "pct_bw_ceiling": round(100 * eff_gbs / args.peak_bw, 1) if args.peak_bw > 0 else 0.0,
        "peak_bw_GB_s": args.peak_bw,
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
    dk = kernel_table(decode, args.peak_bw)
    pk = kernel_table([prefill], args.peak_bw)
    an, n_same = anatomy(decode, tick_ns)
    summary["anatomy_tokens"] = n_same

    ab = run_ablation(model, name, dk, args, d, rules) if args.ablate_top > 0 else {}
    for r in dk:
        a = ab.get(r["kernel"])
        if a:
            net_us = max(r["us_per_tok"] - a["empty"], 0.01)
            r["empty_us"] = round(a["empty"], 1)
            r["loads_us"] = round(a["loads"], 1)
            r["math_us"] = round(a["math"], 1)
            r["net_GB_s"] = round((r["MB_per_tok"] * 1e3) / net_us, 1)
            r["bound_by"] = classify_bound(r["us_per_tok"], a["empty"], a["loads"], a["math"])
        else:
            r["empty_us"] = r["loads_us"] = r["math_us"] = r["net_GB_s"] = r["bound_by"] = ""

    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
    write_csv(os.path.join(d, "decode_kernels.csv"), dk)
    write_csv(os.path.join(d, "prefill_kernels.csv"), pk)
    write_csv(os.path.join(d, "decode_anatomy.csv"), an)
    os.remove(tl)  # large; the CSVs carry everything derived from it
    print(f"[{name}] decode {dec_tps:.2f} t/s = {summary['wall_ms_per_tok']} ms/tok wall; "
          f"{summary['decode_MB_per_tok']} MB/tok = {summary['effective_GB_s']} GB/s ({summary['pct_bw_ceiling']}% of {args.peak_bw}); "
          f"GPU busy {summary['gpu_busy_ms_per_tok']} ms; {summary['groups_per_tok']} groups/tok", flush=True)
    for r in dk[:10]:
        ab_s = f"  [{r['bound_by']:<8} empty={r['empty_us']} loads={r['loads_us']} math={r['math_us']} net={r['net_GB_s']} GB/s]" if r["bound_by"] else ""
        print(f"   {r['kernel']:<38} {r['us_per_tok']:>7.1f} us {r['pct']:>5.1f}% "
              f"{r['MB_per_tok']:>6.1f} MB {r['GB_s']:>5.1f} GB/s ({r['pct_bw']:>4.1f}%){ab_s}")
    return summary, dk


def table(rows, cols):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def write_html(path, results):
    keys = [k for k in results[0][0] if k != "model"]
    comp = [{"metric": k, **{s["model"]: s[k] for s, _ in results}} for k in keys]
    dk_cols = ["kernel", "calls_per_tok", "us_per_tok", "us_per_call", "pct",
               "MB_per_tok", "GB_s", "pct_bw", "empty_us", "loads_us", "math_us", "net_GB_s", "bound_by"]
    parts = ["<!doctype html><meta charset=utf-8><title>p36-prof</title><style>"
             "body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
             "table{border-collapse:collapse;margin:8px 0 24px}td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}"
             "td:first-child,th:first-child,td:last-child,th:last-child{text-align:left}</style>",
             f"<h1>p36-prof</h1><p>{time.strftime('%Y-%m-%d %H:%M')}. Wall numbers come from an unprofiled pass; "
             "GPU numbers from a serialized timestamp pass (upper bound per kernel). "
             "Ablation runs empty / loads-only / math-only variants of the top instrumented shaders.</p>",
             "<h2>Models</h2>", table(comp, ["metric"] + [s["model"] for s, _ in results])]
    for s, dk in results:
        parts += [f"<h2>{html.escape(s['model'])}: decode kernels per token</h2>",
                  table(dk, dk_cols),
                  f"<p>Every action of one token: <code>{html.escape(s['model'])}/decode_anatomy.csv</code></p>"]
    open(path, "w").write("".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", action="append", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--peak-bw", type=float, default=437.5, help="Measured hardware sequential bandwidth in GB/s")
    ap.add_argument("--ablate-top", type=int, default=6, help="Top N instrumented decode kernels to ablate (0 to skip)")
    ap.add_argument("--out", default=os.path.expanduser("~/q36-ab/p36-prof"))
    ap.add_argument("--bin", default=REPO)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    gov = subprocess.run(["systemctl", "is-active", "cyan-skillfish-governor-smu"], capture_output=True, text=True).stdout.strip()
    if gov != "active":
        print(f"WARNING: cyan-skillfish-governor-smu is {gov or 'unknown'}; numbers will read ~20-28% slow", file=sys.stderr)
    rules = parse_shader_rules()
    results = [profile(m, args, rules) for m in args.model]
    write_html(os.path.join(args.out, "index.html"), results)
    print(f"report: {os.path.join(args.out, 'index.html')}")


if __name__ == "__main__":
    main()
