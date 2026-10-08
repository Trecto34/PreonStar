"""Offline checks for the harness. No GPU, no model server, no network. eval.py and bench.py run against fake_server.py.

  python3 test_offline.py [--fixture-source DIR]

--fixture-source: directory with guard_known.jsonl and guard_held.jsonl (default: the cache written by fetch_fixtures).
Runs: golden regression (prompt text and read-once layout re-derived from common.py, compared with golden/*.json),
ECE/percentile arithmetic, eval.py and bench.py against fake_server.py, eval.py against a closed port (failure path),
and a re-run of fetch_fixtures.py --source-dir that must reproduce fixtures/guard_lost_sample.jsonl byte for byte.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as c  # noqa: E402
import eval as ev  # noqa: E402
import fake_server  # noqa: E402

PY = sys.executable
OUT = os.path.join(c.CACHE, "offline_test")


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def golden_regression():
    n = 0
    for path in sorted(glob.glob(os.path.join(HERE, "golden", "*.json"))):
        g = json.load(open(path, encoding="utf-8"))
        if "mode" not in g:  # labels.json and the a/b/c.request.json inputs of gen_golden.py are not golden cases
            continue
        req = g["request"]
        questions = [(k, v) for k, v in req["questions"].items()]
        assert c.build_request(req["state"], questions, req.get("read_once", False)) == req, g["name"]
        if g["mode"] == "read_once":
            pieces = c.readonce_pieces(req["state"], [q for _, q in questions])
            check(f"golden {g['name']}: read-once pieces re-derived", pieces == g["read_once_pieces"])
            n += 1
        else:
            for step, (qn, q) in zip(g["steps"], questions):
                ok = c.chat_prompt(req["state"], q) == step["prompt_text"] and c.user_text(req["state"], q) == step["user_json"]
                check(f"golden {g['name']} {qn}: prompt text re-derived", ok)
                n += 1
    check(f"golden regression covered {n} prompts", n >= 20)


def arithmetic():
    tbl_ece, rows = ev.ece_table([1.0] * 10, [True] * 10)
    check("ECE of a perfectly confident, always-right set is 0", tbl_ece == 0.0 and len(rows) == 15)
    e, _ = ev.ece_table([0.65] * 10, [True] * 5 + [False] * 5)
    check("ECE: conf 0.65 with accuracy 0.5 gives 0.15", abs(e - 0.15) < 1e-12)
    check("percentile p50 of [1,2,3,4] is 2.5", c.percentile([1, 2, 3, 4], 50) == 2.5)
    check("percentile p95 of 1..100 is 95.05", abs(c.percentile(list(range(1, 101)), 95) - 95.05) < 1e-9)
    check("percentile of empty list is None", c.percentile([], 50) is None)


def run(cmd):
    cmd = [os.path.join(HERE, cmd[0])] + cmd[1:]  # the scripts live next to this file, not in the caller's cwd
    r = subprocess.run([PY] + cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"command failed: {cmd}")
    return r.stdout


def with_fake_server(fn):
    srv = fake_server.serve(0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        return fn(f"http://127.0.0.1:{srv.server_address[1]}")
    finally:
        srv.shutdown()


def eval_and_bench(url):
    out = os.path.join(OUT, "eval_fake")
    run(["eval.py", "--url", url, "--limit", "30", "--out-dir", out])
    res = json.load(open(os.path.join(out, "results.json")))
    lines = [json.loads(x) for x in open(os.path.join(out, "responses.jsonl"), encoding="utf-8")]
    check("eval: 30 requests sent and answered", res["counts"] == {"sent": 30, "answered": 30, "failed": 0,
                                                                  "contract_violations": 0})
    check("eval: responses.jsonl has 30 lines with raw text and parsed JSON",
          len(lines) == 30 and all(x["raw"] and x["response"] for x in lines))
    check("eval: request body hash recorded", all(len(x["request_sha256"]) == 64 for x in lines))
    check("eval: groups add up", res["metrics"]["destructive"]["n"] + res["metrics"]["safe"]["n"] == 30)
    check("eval: reliability table has 15 bins", len(res["reliability"]) == 15)
    check("eval: markdown written", os.path.getsize(os.path.join(out, "results.md")) > 200)

    bout = os.path.join(OUT, "bench_fake")
    run(["bench.py", "--url", url, "--sizes", "128,256", "--questions", "1,3", "--reps", "2",
         "--token-mode", "server", "--sample-interval", "0.2", "--out-dir", bout])
    b = json.load(open(os.path.join(bout, "results.json")))
    check("bench: 2 sizes x 2 question counts x 2 read_once = 8 configs", len(b["configs"]) == 8)
    check("bench: every config has 2 OK timed requests",
          all(cfg["ok"] == 2 and cfg["errors"] == 0 for cfg in b["configs"]))
    check("bench: server-usage sizing of the state reached the 256 target within 10%",
          abs(b["sizing"]["256"]["state_tokens"] - 256) <= 0.10 * 256)
    check("bench: samples.jsonl written", os.path.getsize(os.path.join(bout, "samples.jsonl")) > 0)


def failure_path():
    out = os.path.join(OUT, "eval_closed")
    run(["eval.py", "--url", "http://127.0.0.1:9", "--limit", "2", "--timeout", "3", "--out-dir", out])
    res = json.load(open(os.path.join(out, "results.json")))
    check("eval: unreachable server is counted as failed, not a crash",
          res["counts"]["failed"] == 2 and res["counts"]["answered"] == 0 and res["latency_s"]["p50"] is None)


def fixture_reproduce(source):
    tmp = os.path.join(OUT, "sample_again.jsonl")
    run(["fetch_fixtures.py", "--source-dir", source, "--out", tmp])
    same = open(tmp, "rb").read() == open(os.path.join(HERE, "fixtures", "guard_lost_sample.jsonl"), "rb").read()
    check("fetch_fixtures: re-running from the same source reproduces the committed sample", same)


GIT_FIX = os.path.join(HERE, "fixtures", "git_choice_sample.jsonl")


def git_arithmetic():
    import eval_git as eg
    # two choice answers, both right, confidence 0.9 and 0.6: ECE = |1 - 0.75| with the 0.6 row in its own bin
    items = [{"gold": "a", "pred": "a", "conf": 0.9, "p_gold": 0.9, "qtype": "stash", "split": "known"},
             {"gold": "b", "pred": "b", "conf": 0.6, "p_gold": 0.6, "qtype": "stash", "split": "held"}]
    m = eg.group_metrics(items)
    check("git: accuracy 1.0 and mean confidence 0.75 on two right answers", m["accuracy"] == 1.0 and abs(m["mean_confidence"] - 0.75) < 1e-12)
    check("git: ECE of two right answers at 0.9 and 0.6 is (0.1 + 0.4) / 2 = 0.25",
          abs(m["ece"] - 0.25) < 1e-12)
    check("git: Brier (gold term) is mean of (1 - P(gold))^2 = (0.01 + 0.16) / 2",
          abs(m["brier"] - 0.085) < 1e-12)
    a = {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.7}
    check("git: a well-formed choice answer has no contract problem", eg.contract_violations(a, ["a", "b"]) == [])
    bad = dict(a, choice="a")
    check("git: a choice that is not the largest probability is flagged", eg.contract_violations(bad, ["a", "b"]) != [])


def git_fixture_checks():
    rows = [json.loads(x) for x in open(GIT_FIX, encoding="utf-8")]
    check("git fixture: at most 400 rows and under 3 MB", len(rows) <= 400 and os.path.getsize(GIT_FIX) < 3 * 1024 * 1024)
    counts = {}
    for r in rows:
        counts[(r["split"], r["qtype"])] = counts.get((r["split"], r["qtype"]), 0) + 1
    check("git fixture: balanced, 66 rows in each of the 6 (split, qtype) cells", len(counts) == 6 and set(counts.values()) == {66})
    ok = True
    for r in rows:
        req = r["request"]
        q = req["questions"]["choice"]
        labels = list(q["criteria"].keys())
        ok = ok and set(req) == {"state", "questions"} and q["type"] == "choice" and r["gold"] in labels \
            and r["expected_letter"] == "ABCDEFG"[labels.index(r["gold"])] \
            and "State after" not in req["state"] and "State after" not in json.dumps(req) \
            and "Commands, in order:" in req["state"]
    check("git fixture: request bodies are choice questions, gold is a criteria key, state has no after-state", ok)


def git_eval_fake(url):
    out = os.path.join(OUT, "eval_git_fake")
    run(["eval_git.py", "--url", url, "--out-dir", out])
    res = json.load(open(os.path.join(out, "results.json")))
    lines = [json.loads(x) for x in open(os.path.join(out, "responses.jsonl"), encoding="utf-8")]
    n = len(open(GIT_FIX, encoding="utf-8").readlines())
    check("git eval (fake): every request sent and answered", res["counts"] == {"sent": n, "answered": n, "failed": 0,
                                                                              "contract_violations": 0})
    check("git eval (fake): responses.jsonl has one line per row with raw text",
          len(lines) == n and all(x["raw"] and x["response"] for x in lines))
    g = res["metrics"]
    check("git eval (fake): qtype and split groups add up to overall",
          sum(g[q]["n"] for q in ("stash", "content", "branch")) == n and g["known"]["n"] + g["held"]["n"] == n)
    check("git eval (fake): 6 qtype x split groups of 66", all(g[f"{q} / {s}"]["n"] == 66
          for q in ("stash", "content", "branch") for s in ("known", "held")))
    check("git eval (fake): reliability table has 15 bins", len(res["reliability"]) == 15)
    check("git eval (fake): markdown written", os.path.getsize(os.path.join(out, "results.md")) > 200)


def git_failure_path():
    out = os.path.join(OUT, "eval_git_closed")
    run(["eval_git.py", "--url", "http://127.0.0.1:9", "--limit", "2", "--timeout", "3", "--out-dir", out])
    res = json.load(open(os.path.join(out, "results.json")))
    check("git eval: unreachable server is counted as failed, not a crash",
          res["counts"]["failed"] == 2 and res["counts"]["answered"] == 0)


def git_fixture_reproduce(source):
    tmp = os.path.join(OUT, "git_choice_sample_again.jsonl")
    run(["fetch_fixtures_git.py", "--source-dir", source, "--out", tmp])
    same = open(tmp, "rb").read() == open(GIT_FIX, "rb").read()
    check("fetch_fixtures_git: re-running from the same source reproduces the committed sample", same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-source", default=os.path.join(c.CACHE, "ekbasis-data-raw"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("FAKE SERVER TESTS: responses come from fake_server.py, not from Ekbasis or q36-server.")
    golden_regression()
    arithmetic()
    with_fake_server(eval_and_bench)
    failure_path()
    fixture_reproduce(a.fixture_source)
    git_arithmetic()
    git_fixture_checks()
    with_fake_server(git_eval_fake)
    git_failure_path()
    git_fixture_reproduce(a.fixture_source)
    print("ALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    main()
