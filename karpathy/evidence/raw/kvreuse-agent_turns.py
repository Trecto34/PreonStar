#!/usr/bin/env python3
"""Simulate an agent client: long stable system prompt + tools, conversation grows each turn.
Usage: agent_turns.py <turns> [resume_file]. Prints wall time per request."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8011/v1/chat/completions"
story = open("/home/server/q36-opt-27b/tests/long_context_story_prompt.txt").read()
SYSTEM = "You are a coding agent. Follow the rules below.\n\n" + story[:24000]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
state_file = sys.argv[2] if len(sys.argv) > 2 else None
msgs = json.load(open(state_file)) if state_file and len(sys.argv) > 3 else [{"role": "system", "content": SYSTEM}]

for turn in range(int(sys.argv[1])):
    msgs.append({"role": "user", "content": f"Turn {turn}: summarize paragraph {turn + 1} of the rules in one sentence."})
    body = json.dumps({"model": "q36", "messages": msgs, "tools": TOOLS, "max_tokens": 24, "temperature": 0}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(URL, body, {"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t0
    m = r["choices"][0]["message"]
    msgs.append({"role": "assistant", "content": m.get("content") or ""})
    print(f"turn {turn}: {dt:6.2f}s prompt_tokens={r.get('usage', {}).get('prompt_tokens')}", flush=True)
if state_file:
    json.dump(msgs, open(state_file, "w"))
