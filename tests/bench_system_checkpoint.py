#!/usr/bin/env python3
"""New-session cost with/without the system+tools KV checkpoint (--kv-system-checkpoint).

Each "session" is a fresh conversation (different user message) that shares a ~1600-token system prompt
with tools.  Prints cached tokens, server prefill_ms, TTFT, decode tok/s and the checkpoint metrics.
Usage: tests/bench_system_checkpoint.py URL [--variants] [--dump-text] [--rules=N]   (40 rules ~ 1600 tokens, like the agent)
  --variants   also run: different system prompt, changed tool schema (both must miss)
  --dump-text  print each reply (for cross-server equivalence checks)"""
import json
import sys
import time
import urllib.request

URL = sys.argv[1] + "/v1/chat/completions"
TOK = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--tokens=")), 24))
RULES = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--rules=")), 40))
VARIANTS = "--variants" in sys.argv
DUMP = "--dump-text" in sys.argv


def tools(extra=""):
    names = ["read", "write", "edit", "bash", "search", "list", "more", "bash_status", "bash_stop", "visit_page"]
    return [{"type": "function", "function": {
        "name": n, "description": "Tool %s. %s Use it carefully and only when needed for the task at hand." % (n, extra),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path argument for %s." % n},
            "text": {"type": "string", "description": "Text argument for %s." % n}}, "required": ["path"]}}} for n in names]


def system(word="carefully"):
    rules = " ".join("Rule %d: work %s, verify results, keep edits minimal and explain failures briefly." % (i, word)
                     for i in range(RULES))
    return "You are a coding agent running in a local workspace. " + rules


def session(user, sys_word="carefully", tool_extra=""):
    body = {"messages": [{"role": "system", "content": system(sys_word)}, {"role": "user", "content": user}],
            "tools": tools(tool_extra), "max_tokens": TOK, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; usage = None; text = ""
    for line in urllib.request.urlopen(req, timeout=600):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        j = json.loads(line[5:])
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices", []):
            d = ch.get("delta", {})
            piece = d.get("content") or ""
            text += piece
            if first is None and (piece or d.get("tool_calls")):
                first = time.time()
    t1 = time.time()
    u, tm = usage, usage.get("timings", {})
    sc = tm.get("system_checkpoint")
    return {"prompt": u["prompt_tokens"], "cached": u["prompt_tokens_details"]["cached_tokens"],
            "prefill_ms": tm.get("prefill_ms", 0), "ttft": (first or t1) - t0,
            "decode": (u["completion_tokens"] - 1) / max(t1 - (first or t0), 1e-6), "sys": sc, "text": text}


def show(label, r):
    sc = r["sys"]
    extra = "" if not sc else " | ckpt hit=%s tokens=%d restore=%.0fms build=%.0fms" % (
        sc["hit"], sc["tokens"], sc["restore_ms"], sc["build_ms"])
    print("%-34s prompt=%d cached=%d prefill=%.0fms ttft=%.3fs decode=%.1f tok/s%s" % (
        label, r["prompt"], r["cached"], r["prefill_ms"], r["ttft"], r["decode"], extra))
    if DUMP:
        print("   text:", json.dumps(r["text"]))


tasks = ["Create hello.txt containing hi.", "List the files in src and explain each.", "Fix the typo in README.md.",
         "Run the tests and summarize failures."]
show("session 1 (first, cold)", session(tasks[0]))
for i, t in enumerate(tasks[1:], 2):
    show("session %d (same system+tools)" % i, session(t))
if VARIANTS:
    show("different system prompt", session(tasks[0], sys_word="quickly"))
    show("changed tool schema", session(tasks[0], tool_extra="Extra note."))
    show("original again (should hit)", session(tasks[1]))
