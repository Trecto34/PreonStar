#!/usr/bin/env python3
"""Concurrency / cross-request contamination check for POST /v1/systemone (stdlib only, needs a running server).

  python3 -I concurrency_check.py --url http://127.0.0.1:8123 [--rounds 3]

1. sequential reference answers for a set of distinct systemone requests and one greedy chat completion;
2. the same requests fired concurrently (threads), mixed with chat completions, for several rounds;
3. a chat request, then systemone calls on other content, then the same chat again (warm-checkpoint reuse).
Every answer must be bit-identical to its sequential reference. Exit 1 on any difference or HTTP error.
"""
import argparse, json, os, sys, threading, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_prefix as ab  # noqa: E402


def post_json(url, path, body, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


CHAT_PROMPTS = ["Name three colours. Answer in one short line.", "What is 17 + 25? Answer with the number only."]


def chat(url, k):
    d = post_json(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": CHAT_PROMPTS[k % 2]}],
                                                  "temperature": 0, "max_tokens": 12,
                                                  "chat_template_kwargs": {"enable_thinking": False}})
    return d["choices"][0]["message"]["content"]


def leaves(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{path}/{k}")
    else:
        yield path, x


def max_dev(a, b):
    """(max |dp| over numeric leaves, number of non-numeric leaves that differ e.g. the chosen label)."""
    da, db = dict(leaves(a)), dict(leaves(b))
    num = [abs(da[k] - db[k]) for k in da if k in db and isinstance(da[k], float) and isinstance(db[k], float)]
    other = sum(1 for k in da if k not in db or (not isinstance(da[k], float) and da[k] != db[k]))
    return (max(num) if num else 0.0), other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0, help="allowed max |dp| vs the sequential reference (0 = bit-identical)")
    ap.add_argument("--url", default="http://127.0.0.1:8123")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    url = a.url
    sysreq = [(n, b) for n, b in ab.requests_set() if n in ("s256-q3-ro0", "s512-q3-ro0", "s256-q3-ro1", "mixed-types",
                                                           "unicode", "short-state")]
    ref = {n: post_json(url, "/v1/systemone", b)["answers"] for n, b in sysreq}
    ref_chat = [chat(url, k) for k in range(2)]
    print("reference chat:", ref_chat)
    bad = []

    for rnd in range(a.rounds):
        out, errs = {}, []

        def sys_worker(n, b):
            try:
                out[n] = post_json(url, "/v1/systemone", b)["answers"]
            except Exception as e:  # noqa: BLE001
                errs.append((n, repr(e)))

        def chat_worker(k):
            try:
                out[f"chat{k}"] = chat(url, k)
            except Exception as e:  # noqa: BLE001
                errs.append((f"chat{k}", repr(e)))

        th = [threading.Thread(target=sys_worker, args=nb) for nb in sysreq]
        th += [threading.Thread(target=chat_worker, args=(k,)) for k in range(2)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        for n, _ in sysreq:
            if out.get(n) != ref[n]:
                dev, other = max_dev(out[n], ref[n]) if n in out else (float("inf"), 1)
                print(f"  round {rnd} {n}: max|dp|={dev:.3g}, non-numeric diffs={other}")
                if dev > a.tol or other:
                    bad.append((rnd, n, dev, other))
        for k in range(2):
            if out.get(f"chat{k}") != ref_chat[k]:
                bad.append((rnd, f"chat{k}", out.get(f"chat{k}"), ref_chat[k]))
        bad += [(rnd,) + e for e in errs]
        print(f"round {rnd}: {'ok' if not [x for x in bad if x[0] == rnd] else 'DIFFERENCES'}", flush=True)

    # warm-checkpoint contamination: chat, systemone on other content, chat again
    c1 = chat(url, 0)
    for n, b in sysreq[:3]:
        post_json(url, "/v1/systemone", b)
    c2 = chat(url, 0)
    print("chat before/after systemone:", c1 == c2, "|", c1)
    if c1 != c2 or c1 != ref_chat[0]:
        bad.append(("contamination", c1, c2))
    print("CONCURRENCY OK" if not bad else f"FAILED: {bad}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
