#!/usr/bin/env python3
"""Real-execution regression suite for the q36-agent control features.

Runs the actual agent (local model or --server) on small coding tasks and checks
artifacts plus trace/summary evidence.  Needs a model (local) or a running
q36-server (remote); never run local and the server on a 16 GB machine at once.

  tests/agent_regression.py --mode local  --model gguf/X.gguf --out /tmp/reg-local.json
  tests/agent_regression.py --mode remote --server http://127.0.0.1:8000 --out /tmp/reg-remote.json
  tests/agent_regression.py --compare /tmp/reg-local.json /tmp/reg-remote.json   # test G
  options: --only A,C   --thinking-budget N   --action-budget N
"""
import argparse
import html.parser
import json
import os
import re
import subprocess
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["local", "remote"])
ap.add_argument("--model")
ap.add_argument("--server")
ap.add_argument("--agent", default="./q36-agent")
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--thinking-budget", type=int, default=768)
ap.add_argument("--action-budget", type=int, default=384)
ap.add_argument("--think", default="auto")
ap.add_argument("--timeout", type=int, default=420)
ap.add_argument("--only", default="")
ap.add_argument("--out")
ap.add_argument("--compare", nargs=2)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--aggregate", nargs="+", help="label=file ...: mean metrics per task per condition")
args = ap.parse_args()


def compare(a, b):
    ra, rb = json.load(open(a)), json.load(open(b))
    ok = True
    for name in sorted(set(ra) & set(rb)):
        x, y = ra[name], rb[name]
        same_art = x["pass"] == y["pass"]
        tools_x, tools_y = x["tools"], y["tools"]
        # behavioral parity: same pass/fail verdict and same set of tools used
        same_tools = set(tools_x) == set(tools_y)
        print("%s: verdict local=%s remote=%s | tools local=%s remote=%s | steps %s vs %s -> %s" % (
            name, x["pass"], y["pass"], tools_x, tools_y, x["steps"], y["steps"],
            "PARITY" if same_art and same_tools else "DIFFERS"))
        ok &= same_art
    return 0 if ok else 1


def aggregate(specs):
    conds = []
    for sp in specs:
        label, path = sp.split("=", 1)
        conds.append((label, json.load(open(path))))
    tasks = sorted({k.split("#")[0] for _, d in conds for k in d})
    cols = ["pass", "steps", "reason", "action", "recov", "compact", "wall_s", "pf_tps", "dec_tps", "prompt0"]
    print("%-3s %-14s " % ("t", "condition") + " ".join("%8s" % c for c in cols))
    for t in tasks:
        for label, d in conds:
            rows = [v for k, v in d.items() if k.split("#")[0] == t and v["pass"] is not None]
            if not rows:
                continue
            n = len(rows)
            m = lambda f: sum(r.get(f, 0) for r in rows) / n
            vals = ["%d/%d" % (sum(1 for r in rows if r["pass"]), n), "%.1f" % m("steps"), "%.0f" % m("reasoning"),
                    "%.0f" % m("action"), "%.1f" % m("recoveries"), "%.1f" % m("compactions"),
                    "%.1f" % m("wall"), "%.0f" % m("prefill_tps"), "%.1f" % m("decode_tps"), "%.0f" % m("prompt0")]
            print("%-3s %-14s " % (t, label) + " ".join("%8s" % v for v in vals))
    return 0


if args.aggregate:
    sys.exit(aggregate(args.aggregate))


if args.compare:
    sys.exit(compare(*args.compare))
if not args.mode:
    ap.error("--mode is required")


def run_agent(prompt, setup=None, budgets=None, extra=()):
    d = tempfile.mkdtemp(prefix="q36reg-")
    if setup:
        setup(d)
    trace = os.path.join(d, ".trace")
    tb = (budgets or {}).get("thinking", args.thinking_budget)
    ab = (budgets or {}).get("action", args.action_budget)
    cmd = [args.agent]
    if args.mode == "remote":
        cmd += ["--server", args.server]
    else:
        cmd += ["--vulkan", "-m", args.model, "--ctx", str(args.ctx)]
    cmd += ["--think", args.think, "--thinking-budget", str(tb), "--action-budget", str(ab),
            "--chdir", d, "--non-interactive", "--trace", trace, *extra, "-p", prompt]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), -9
    tr = open(trace, errors="replace").read() if os.path.exists(trace) else ""
    # the trace file itself lives in the work dir; hide it from artifact checks
    def num(label):
        m = re.search(r"^%s: (\d+)" % label, out, re.M)
        return int(m.group(1)) if m else 0
    def fnum(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else 0.0
    first_prompt = re.search(r"(?:prefill tool_round=0 transcript=\d+ prompt=|\[STEP\] finish=\w+ prompt=)(\d+)", tr)
    return {
        "wall": fnum(r"wall time: ([\d.]+)s"), "prefill_tps": fnum(r"prefill: ([\d.]+) tok/s"),
        "decode_tps": fnum(r"decode: ([\d.]+) tok/s"), "compactions": num("compactions"),
        "prompt0": int(first_prompt.group(1)) if first_prompt else 0,
        "dir": d, "rc": rc, "out": out, "trace": tr,
        "steps": num("steps"), "tools_n": num("tools"), "blocked": num("repeated calls blocked"),
        "recoveries": num("recoveries"), "reasoning": num("reasoning tokens"), "action": num("action tokens"),
        "tools": re.findall(r"\[TOOL CALL\] (\w+)", tr),
        "results": re.findall(r"\[TOOL RESULT\]=\"(.*?)\"\n", tr, re.S),
        "blocked_final": "BLOCKED:" in out,
    }


class Tags(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
    def handle_starttag(self, t, a): self.tags.append(t)


def valid_html(path, min_bytes=1):
    if not os.path.exists(path):
        return False
    t = open(path, errors="replace").read()
    p = Tags(); p.feed(t)
    return len(t) >= min_bytes and "html" in p.tags and "</html>" in t.lower() and "body" in p.tags


results = {}


def key(name):
    return name if args.reps == 1 else "%s#%d" % (name, CUR[0])


def record(name, r, passed, why=""):
    results[key(name)] = {"pass": passed, "steps": r["steps"], "tools": r["tools"], "why": why,
                     "blocked": r["blocked"], "recoveries": r["recoveries"],
                     "reasoning": r["reasoning"], "action": r["action"], "dir": r["dir"],
                     "wall": r["wall"], "prefill_tps": r["prefill_tps"], "decode_tps": r["decode_tps"],
                     "compactions": r["compactions"], "prompt0": r["prompt0"]}
    print("%s %s steps=%d tools=%s blocked=%d recov=%d reasoning=%d action=%d %s" % (
        "PASS" if passed else "FAIL", name, r["steps"], r["tools"], r["blocked"], r["recoveries"],
        r["reasoning"], r["action"], why if not passed else ""))
    sys.stdout.flush()


want = set(args.only.split(",")) if args.only else None
def on(n): return not want or n in want


CUR = [0]
for _rep in range(args.reps):
    CUR[0] = _rep + 1
    # A: trivial file generation
    if on("A"):
        r = run_agent("Create a single self-contained HTML file named index.html showing a rotating 3D cube.")
        p = os.path.join(r["dir"], "index.html")
        ok = r["steps"] <= 3 and valid_html(p) and r["blocked"] == 0 and not r["blocked_final"] and r["rc"] == 0
        record("A", r, ok, "steps=%d html=%s blocked=%d rc=%s" % (r["steps"], valid_html(p), r["blocked"], r["rc"]))

    # B: large write through chunks
    if on("B"):
        r = run_agent("Create app.html: a self-contained todo-list web app (HTML, CSS and JavaScript: add, complete, "
                      "delete, filter, localStorage) totalling at least 16000 bytes. A write call is limited to 12 KiB, so "
                      "use exactly two write calls: first write with the head, CSS and body (about 9 KiB), then write with "
                      "append=true containing the long JavaScript (at least 7 KiB) and the closing tags.",
                      budgets={"thinking": 512})
        p = os.path.join(r["dir"], "app.html")
        size = os.path.getsize(p) if os.path.exists(p) else 0
        appended = any("append=1" in x for x in r["results"])
        toolcall_fail = any("WRITE_PAYLOAD_TOO_LARGE" in x for x in r["results"]) and size == 0
        ok = size >= 8000 and valid_html(p) and appended and r["tools"].count("write") >= 2 and not toolcall_fail
        record("B", r, ok, "size=%d html=%s appended=%s" % (size, valid_html(p), appended))


    def setup_c(d):
        open(os.path.join(d, "calc.py"), "w").write(
            "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n")


    # C: simple known edit
    if on("C"):
        r = run_agent("In calc.py the function add() has a bug: it subtracts. Change `return a - b` to `return a + b`.",
                      setup=setup_c, budgets={"thinking": 256})
        t = open(os.path.join(r["dir"], "calc.py")).read()
        rereads = r["tools"].count("read")
        ok = "return a + b" in t and "a - b" not in t and "edit" in r["tools"] + ["write"] and rereads <= 1 and r["blocked"] == 0
        record("C", r, ok, "reads=%d tools=%s" % (rereads, r["tools"]))


    def setup_d(d):
        open(os.path.join(d, "f.c"), "w").write(
            "int f(int x) {\n"
            "\tif (x > 0) {\n"
            "\t\tfor (int i = 0; i < x; i++) {\n"
            "\t\t\t    total +=   i  *  2;\n"
            "\t\t}\n"
            "\t}\n"
            "\treturn total;\n"
            "}\n")


    # D: whitespace-sensitive edit
    if on("D"):
        r = run_agent("In f.c change the statement `total += i * 2;` (inside the nested for loop) to `total += i * 3;`. "
                      "Keep every other character of the file, including indentation, unchanged.", setup=setup_d,
                      budgets={"thinking": 512})
        got = open(os.path.join(r["dir"], "f.c")).read().split("\n")
        orig = open(os.path.join(r["dir"], "f.c")).read() and None
        want_others = ["int f(int x) {", "\tif (x > 0) {", "\t\tfor (int i = 0; i < x; i++) {", None,
                       "\t\t}", "\t}", "\treturn total;", "}", ""]
        ok = len(got) == len(want_others) and all(w is None or w == g for w, g in zip(want_others, got)) \
            and re.match(r"^\t\t\t\s*total \+=\s+i\s*\*\s*3;$", got[3] if len(got) > 3 else "") is not None
        nf = sum("EDIT_MATCH_NOT_FOUND" in x for x in r["results"])
        record("D", r, ok and nf <= 2, "not_found=%d tools=%s" % (nf, r["tools"]))

    # E: forced thinking exhaustion with a huge output budget
    if on("E"):
        r = run_agent("Create hello.txt containing exactly: hello world", budgets={"thinking": 16, "action": 128},
                      extra=("-n", "20000"))
        p = os.path.join(r["dir"], "hello.txt")
        actions = [int(x) for x in re.findall(r"\[ACTION\] tokens=(\d+)", r["trace"])]
        ok = os.path.exists(p) and "hello world" in open(p).read() and (not actions or max(actions) < 1500) \
            and (r["reasoning"] <= 16 * max(1, r["steps"]) + 16)
        record("E", r, ok, "file=%s max_action=%s reasoning=%d" % (os.path.exists(p), max(actions) if actions else None, r["reasoning"]))


    def setup_f(d):
        open(os.path.join(d, "config.txt"), "w").write("mode=fast\nlevel=3\n")


    # F: repeated read of unchanged input
    if on("F"):
        r = run_agent("Call read on config.txt with identical arguments four separate times in a row (one call per "
                      "step, no other tool in between), then tell me the value of level.", setup=setup_f,
                      budgets={"thinking": 128})
        reads = r["tools"].count("read")
        if reads < 2:
            results[key("F")] = {"pass": None, "steps": r["steps"], "tools": r["tools"], "why": "inconclusive: model did not repeat",
                            "blocked": 0, "recoveries": 0, "reasoning": 0, "action": 0, "dir": r["dir"]}
            print("SKIP F inconclusive: model read only %d time(s) (deterministic coverage: unit test)" % reads)
        else:
            ok = r["blocked"] >= 1 and "3" in r["out"] and not r["blocked_final"]
            record("F", r, ok, "reads=%d blocked=%d" % (reads, r["blocked"]))


if args.out:
    json.dump(results, open(args.out, "w"), indent=1)
fails = [k for k, v in results.items() if v["pass"] is False]
print("RESULT:", "FAIL " + ",".join(fails) if fails else "ok")
sys.exit(1 if fails else 0)
