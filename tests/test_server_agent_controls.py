#!/usr/bin/env python3
"""Live checks for the agent-control extensions of /v1/chat/completions.

Usage: tests/test_server_agent_controls.py [http://host:port]
Needs a running q36-server with a thinking-capable model.
"""
import json
import sys
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000") + "/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "write", "description": "Create a file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}]


def chat(**body):
    body.setdefault("temperature", 0)
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (" " + detail if detail and not cond else ""))
    if not cond:
        check.failed = True


check.failed = False

# 1. tool call ends decode at the call; usage carries reasoning + timings.
r = chat(messages=[{"role": "user", "content": "Create hello.txt containing the word hi."}],
         tools=TOOLS, max_tokens=2000, stop_after_tool_call=True, thinking_budget=256)
ch = r["choices"][0]
u = r["usage"]
check("tool_calls finish", ch["finish_reason"] == "tool_calls", str(ch)[:300])
check("tool call parsed", bool(ch["message"].get("tool_calls")))
check("usage reasoning_tokens", "reasoning_tokens" in u.get("completion_tokens_details", {}), str(u))
check("usage timings", u.get("timings", {}).get("decode_ms", 0) > 0, str(u))

# 1b. the flag is an execution boundary: two requested calls -> only the first is emitted.
two = [{"role": "user", "content": "Call write twice in one reply: a.txt with 'a' and b.txt with 'b'."}]
n_free = len(chat(messages=two, tools=TOOLS, max_tokens=1500,
                  chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"].get("tool_calls") or [])
n_stop = len(chat(messages=two, tools=TOOLS, max_tokens=1500, stop_after_tool_call=True,
                  chat_template_kwargs={"enable_thinking": False})["choices"][0]["message"].get("tool_calls") or [])
check("stop_after_tool_call boundary", n_stop == 1 and n_free >= n_stop, "free=%d stop=%d" % (n_free, n_stop))

# 2. hard thinking budget: reasoning cannot run past the cap even with a huge output budget.
r = chat(messages=[{"role": "user", "content": "Plan in detail how to build a compiler. Think carefully."}],
         max_tokens=1500, thinking_budget=32)
rt = r["usage"]["completion_tokens_details"]["reasoning_tokens"]
check("hard thinking budget", rt <= 34, "reasoning_tokens=%d" % rt)

# 3. action budget: a reasoning-style monologue after (no) thinking is aborted early.
r = chat(messages=[{"role": "user", "content":
         "Output exactly 60 lines, each alternating 'Wait, let me reconsider.' and "
         "'Actually, let me read it again.' Nothing else."}],
         max_tokens=1500, action_budget=16, chat_template_kwargs={"enable_thinking": False})
ch = r["choices"][0]
check("action_budget finish", ch["finish_reason"] == "action_budget",
      "%s tokens=%d" % (ch["finish_reason"], r["usage"]["completion_tokens"]))
check("aborted early", r["usage"]["completion_tokens"] < 400)

# 4. defaults untouched: no extension fields -> normal stop/length semantics.
r = chat(messages=[{"role": "user", "content": "Say hi."}], max_tokens=300,
         chat_template_kwargs={"enable_thinking": False})
check("default finish", r["choices"][0]["finish_reason"] in ("stop", "length"))

sys.exit(1 if check.failed else 0)
