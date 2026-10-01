#!/usr/bin/env python3
"""Short plain-generation benchmark for q36-server (no agent features): TTFT, prefill and decode tok/s.
Usage: tests/bench_server_plain.py URL [reps]"""
import json, random, sys, time, urllib.request

url = sys.argv[1] + "/v1/chat/completions"
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
filler = " ".join("alpha beta gamma delta epsilon zeta eta theta iota kappa".split() * 30)
rows = []
for i in range(reps):
    body = {"messages": [{"role": "user", "content": "[run %d-%d] %s\nSummarize the words above in one paragraph." % (i, random.randrange(10**9), filler)}],
            "max_tokens": 160, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None
    for line in urllib.request.urlopen(req, timeout=300):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        j = json.loads(line[5:])
        if j.get("usage"): usage = j["usage"]
        if first is None and any(c.get("delta", {}).get("content") for c in j.get("choices", [])):
            first = time.time()
    t1 = time.time()
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    rows.append((first - t0, pt / (first - t0), (ct - 1) / (t1 - first)))
    print("rep %d: ttft=%.3fs prompt=%d prefill=%.0f tok/s decode=%.1f tok/s" % (i, rows[-1][0], pt, rows[-1][1], rows[-1][2]))
n = len(rows)
print("MEAN ttft=%.3fs prefill=%.0f tok/s decode=%.1f tok/s" % tuple(sum(r[k] for r in rows) / n for k in range(3)))
