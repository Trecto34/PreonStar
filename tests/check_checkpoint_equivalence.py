#!/usr/bin/env python3
"""Greedy replies (reasoning + content + tool calls) for several fresh sessions sharing one system+tools prefix.
Run against a server, save with --out; run again against another server and --compare to compare.
  tests/check_checkpoint_equivalence.py URL --out cold1.json [--prime]
  tests/check_checkpoint_equivalence.py --compare a.json b.json
--prime sends one throw-away session first so every measured session is a checkpoint restore (if enabled)."""
import json
import sys
import urllib.request

sys.argv_backup = list(sys.argv)
if "--compare" in sys.argv:
    a, b = [json.load(open(p)) for p in sys.argv[sys.argv.index("--compare") + 1:][:2]]
    bad = [k for k in a if a[k] != b.get(k)]
    for k in bad:
        print("DIFF", k, "\n  a:", json.dumps(a[k])[:300], "\n  b:", json.dumps(b[k])[:300])
    print("IDENTICAL" if not bad else "DIFFERENT", "(%d sessions)" % len(a))
    sys.exit(1 if bad else 0)

URL = sys.argv[1] + "/v1/chat/completions"
OUT = sys.argv[sys.argv.index("--out") + 1]
names = ["read", "write", "edit", "bash", "search", "list", "more", "bash_status", "bash_stop", "visit_page"]
TOOLS = [{"type": "function", "function": {
    "name": n, "description": "Tool %s. Use it carefully and only when needed for the task at hand." % n,
    "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path for %s." % n},
                                                    "text": {"type": "string", "description": "Text for %s." % n}},
                   "required": ["path"]}}} for n in names]
SYSTEM = "You are a coding agent running in a local workspace. " + " ".join(
    "Rule %d: work carefully, verify results, keep edits minimal and explain failures briefly." % i for i in range(40))
TASKS = ["Create hello.txt containing hi.", "List the files in src.", "Fix the typo in README.md.",
         "Run the tests and summarize.", "Search for TODO comments in the repository.", "Show me the first lines of main.c."]


def session(user, thinking):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], "tools": TOOLS,
            "max_tokens": 120, "temperature": 0, "chat_template_kwargs": {"enable_thinking": thinking}}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    m = r["choices"][0]["message"]
    return {"content": m.get("content"), "reasoning": m.get("reasoning_content"),
            "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
            "finish": r["choices"][0]["finish_reason"], "cached": r["usage"]["prompt_tokens_details"]["cached_tokens"],
            "tokens": r["usage"]["completion_tokens"]}


if "--prime" in sys.argv:
    session("prime the cache", False)
res = {}
for thinking in (False, True):
    for t in TASKS:
        r = session(t, thinking)
        res["%s|think=%s" % (t, thinking)] = {k: v for k, v in r.items() if k != "cached"}
        print("think=%-5s cached=%-5d tokens=%-3d calls=%s %s" % (thinking, r["cached"], r["tokens"],
              [c[0] for c in r["calls"]], t))
json.dump(res, open(OUT, "w"), indent=1)
