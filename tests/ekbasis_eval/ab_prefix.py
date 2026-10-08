#!/usr/bin/env python3
"""A/B harness for the shared-prefix path of POST /v1/systemone.

Run once against a server started normally and once with Q36_SYSTEMONE_PREFIX_REUSE=0, same model, then compare:

  python3 -I ab_prefix.py run --url http://127.0.0.1:8123 --out reuse_on.json
  python3 -I ab_prefix.py run --url http://127.0.0.1:8123 --out reuse_off.json
  python3 -I ab_prefix.py compare reuse_on.json reuse_off.json

Requests are deterministic (a fixed nonce line per request keeps every request cold for the server, and identical
between the two runs). `compare` demands bit-identical answers (probabilities, choices, usage) and prints the latency
of both runs side by side. A speed-up with different answers is a failure, not a result.
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402  (state_text, questions_for)


def post(url, body, timeout=900):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw), time.perf_counter() - t


def requests_set():
    out = []
    n = 0
    lines_for = {128: 5, 256: 11, 512: 21, 1024: 41}
    for size, lines in lines_for.items():
        for nq in (2, 3, 5):
            for ro in (False, True):
                n += 1
                state = f"[ab {n}]\n" + bench.state_text(lines)
                qs = {k: v for k, v in bench.questions_for(nq)}
                out.append((f"s{size}-q{nq}-ro{int(ro)}", {"state": state, "questions": qs, "read_once": ro}))
    # mixed question types sharing one state
    n += 1
    state = f"[ab {n}]\n" + bench.state_text(21)
    out.append(("mixed-types", {"state": state, "read_once": False, "questions": {
        "lost": {"type": "noul", "instructions": "Does any file need a backup?"},
        "pick": {"type": "choice", "instructions": "Which file is oldest?",
                 "criteria": {"file_0001.txt": "first", "file_0002.txt": "second", "file_0003.txt": "third"}},
        "risk": {"type": "score", "instructions": "Rate the risk.", "criteria": {"0": "none", "1": "low", "2": "high"}}}}))
    # unicode + a 160-option question + short state (prefix shorter than one chunk: reuse must be a no-op)
    n += 1
    out.append(("short-state", {"state": f"[ab {n}] tiny", "read_once": False, "questions": {
        "a": {"type": "noul", "instructions": "A?"}, "b": {"type": "noul", "instructions": "B?"}}}))
    n += 1
    out.append(("unicode", {"state": f"[ab {n}] café ☃ " + bench.state_text(8), "read_once": False,
                            "questions": {"a": {"type": "noul", "instructions": "Você tem certeza?"},
                                          "b": {"type": "noul", "instructions": "¿Seguro?"}}}))
    return out


def run(a):
    res = {}
    for name, body in requests_set():
        ans, dt = post(a.url, body)
        res[name] = {"answers": ans["answers"], "usage": ans["usage"], "latency_s": dt}
        print(f"{name:20s} {dt:7.2f}s  input_tokens={ans['usage']['input_tokens']}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def compare(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    bad = 0
    print(f"{'request':20s} {'A s':>8s} {'B s':>8s} {'B/A':>6s}  identical")
    for k in x:
        same = x[k]["answers"] == y[k]["answers"] and x[k]["usage"] == y[k]["usage"]
        bad += not same
        print(f"{k:20s} {x[k]['latency_s']:8.2f} {y[k]['latency_s']:8.2f} {y[k]['latency_s']/x[k]['latency_s']:6.2f}  {'yes' if same else 'NO'}")
        if not same:
            for q in x[k]["answers"]:
                if x[k]["answers"][q] != y[k]["answers"][q]:
                    print("   differs:", q, json.dumps(x[k]["answers"][q])[:200], "|", json.dumps(y[k]["answers"][q])[:200])
    print("ALL BIT-IDENTICAL" if not bad else f"{bad} MISMATCHES")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--url", default="http://127.0.0.1:8123"); r.add_argument("--out", required=True)
    c = sp.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
