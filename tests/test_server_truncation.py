#!/usr/bin/env python3
"""Live truncation sweep for /v1/chat/completions on a real model (temperature 0).

The same tool-calling generation is cut at many max_tokens values (inside reasoning, around </think>,
before the call, inside the name, inside the arguments, exactly complete, past the end), streaming and
not.  Invariants checked at every cut: no delimiter or tool markup in content/reasoning, tool_calls only
with finish_reason=tool_calls and equal to the uncut call, truncated calls flagged
incomplete_tool_call and never turned into content.

Usage: tests/test_server_truncation.py URL"""
import json
import sys
import urllib.request

URL = sys.argv[1] + "/v1/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "write", "description": "Create a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}]
MSG = [{"role": "user", "content": "Create hello.txt containing the word hi."}]
BAD = ["</think>", "<think>", "<tool_call>", "</tool_call>", "<function=", "<parameter="]
failed = []


def check(name, cond, detail=""):
    if not cond:
        failed.append(name)
        print("FAIL", name, detail)


def post(body):
    return urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=300)


def call(max_tokens, stream, **extra):
    body = {"messages": MSG, "tools": TOOLS, "temperature": 0, "max_tokens": max_tokens, **extra}
    if not stream:
        r = json.load(post(body))
        ch = r["choices"][0]
        m = ch["message"]
        return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
                "finish": ch["finish_reason"], "incomplete": bool(ch.get("incomplete_tool_call")),
                "tokens": r["usage"]["completion_tokens"]}
    body["stream"] = True
    out = {"content": "", "reasoning": "", "finish": None, "incomplete": False, "tokens": 0}
    calls = {}
    for line in post(body):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        for ch in json.loads(line[5:]).get("choices", []):
            d = ch.get("delta", {})
            out["content"] += d.get("content") or ""
            out["reasoning"] += d.get("reasoning_content") or ""
            for t in d.get("tool_calls") or []:
                c = calls.setdefault(t.get("index", 0), ["", ""])
                c[0] += (t.get("function") or {}).get("name") or ""
                c[1] += (t.get("function") or {}).get("arguments") or ""
            out["finish"] = ch.get("finish_reason") or out["finish"]
            out["incomplete"] |= bool(ch.get("incomplete_tool_call"))
    out["calls"] = [tuple(v) for _, v in sorted(calls.items())]
    return out


full = call(4000, False, stop_after_tool_call=True)
check("uncut run is a tool call", full["finish"] == "tool_calls" and len(full["calls"]) == 1, str(full))
N = full["tokens"]
want = full["calls"][0]
print("uncut generation: %d tokens, reasoning=%d chars, call=%s" % (N, len(full["reasoning"]), want[0]))
seen_incomplete = seen_inside_reasoning = seen_before_call = False
cuts = sorted(set(list(range(1, N + 4, max(1, N // 36))) + [N - 2, N - 1, N, N + 1]))
for stream in (False, True):
    got_call = False
    for m in cuts:
        if m < 1:
            continue
        r = call(m, stream, stop_after_tool_call=True)
        tag = "%s max_tokens=%d" % ("stream" if stream else "plain", m)
        for field in ("content", "reasoning"):
            check(tag + " no markup in " + field, not any(b in r[field] for b in BAD), repr(r[field][-80:]))
        complete = bool(r["calls"]) and r["calls"][0] == want and r["finish"] == "tool_calls"
        if complete:
            got_call = True
            check(tag + " complete call is not flagged", not r["incomplete"])
            check(tag + " no content after a call", r["content"].strip() == "", repr(r["content"]))
        elif r["calls"]:
            # streaming may have sent call fragments before the cut; the final chunk must say so
            check(tag + " fragments only when streaming", stream)
            check(tag + " fragment => length + incomplete_tool_call", r["finish"] == "length" and r["incomplete"],
                  "%s %s" % (r["finish"], r["incomplete"]))
        else:
            check(tag + " truncated => length", r["finish"] == "length", r["finish"])
            check(tag + " truncated => empty content", r["content"].strip() == "", repr(r["content"][:80]))
        check(tag + " monotonic (call stays once complete)", not (got_call and not complete))
        if r["incomplete"]:
            seen_incomplete = True
            check(tag + " incomplete flag => never a complete call", not complete)
        if not r["calls"] and r["reasoning"] and not r["incomplete"]:
            seen_inside_reasoning = True
        if not r["calls"] and not r["incomplete"] and not r["reasoning"] == "" and m > N // 2:
            seen_before_call = True
    check("%s: the sweep reached a complete call" % ("stream" if stream else "plain"), got_call)
check("sweep saw a flagged truncated tool call", seen_incomplete)
check("sweep saw a cut inside reasoning", seen_inside_reasoning)

# action_budget abort with tools: still no markup
r = call(400, False, action_budget=4, chat_template_kwargs={"enable_thinking": False})
check("action_budget abort has no markup", not any(b in r["content"] + r["reasoning"] for b in BAD), repr(r["content"]))

# tool-less chat is unchanged: reasoning split + content, no incomplete flag
b = {"messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}], "temperature": 0, "max_tokens": 300}
ch = json.load(post(b))["choices"][0]
check("plain chat unchanged", ch["finish_reason"] in ("stop", "length") and "incomplete_tool_call" not in ch
      and "</think>" not in ch["message"]["content"], str(ch)[:200])
print("RESULT:", ("FAIL " + ", ".join(sorted(set(failed))[:6])) if failed else "ok (%d cuts x 2 modes)" % len(cuts))
sys.exit(1 if failed else 0)
