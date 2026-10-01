#!/usr/bin/env python3
"""q36-agent --server against a scripted fake server that returns truncated generations.

Checks: a discarded (incomplete) tool call or empty truncated output is never appended to the
transcript, recovery goes through the bounded recovery machinery (BLOCKED, no endless retry), and a
genuinely truncated *content* answer is still accepted as the final reply.
Usage: tests/test_agent_remote_truncation.py [path/to/q36-agent]   (needs no model)"""
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

AGENT = sys.argv[1] if len(sys.argv) > 1 else "./q36-agent"


def sse(delta, finish=None, extra=None):
    ch = {"index": 0, "delta": delta, "finish_reason": finish}
    ch.update(extra or {})
    return "data: " + json.dumps({"choices": [ch]}) + "\n\n"


def reply(kind):
    usage = "data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 20,
        "completion_tokens_details": {"reasoning_tokens": 20}}}) + "\n\n"
    if kind == "incomplete_tool":
        body = sse({"reasoning_content": "plan the write"}) + sse({}, "length", {"incomplete_tool_call": True})
    elif kind == "cut_empty":
        body = sse({"reasoning_content": "still thinking"}) + sse({}, "length")
    elif kind == "cut_content":
        body = sse({"content": "The answer so far is 42 and"}) + sse({}, "length")
    else:
        body = sse({"content": "Done."}) + sse({}, "stop")
    return body + usage + "data: [DONE]\n\n"


def run(script):
    """script: list of reply kinds; the last one repeats."""
    reqs = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"data": [{"id": "fake", "context_length": 8192}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        def do_POST(self):
            reqs.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            kind = script[min(len(reqs) - 1, len(script) - 1)]
            b = reply(kind).encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close"); self.end_headers()
            self.wfile.write(b)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp(prefix="q36trunc-")
    p = subprocess.run([AGENT, "--server", "http://127.0.0.1:%d" % srv.server_port, "--think",
                        "--chdir", d, "--non-interactive", "-p", "write hello.txt"],
                       capture_output=True, text=True, timeout=60)
    srv.shutdown()
    return p, reqs


failed = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond:
        failed.append(name)


def roles(req): return [m["role"] for m in req["messages"]]


# 1. always incomplete: bounded retries, BLOCKED, nothing poisonous in any transcript
p, reqs = run(["incomplete_tool"])
check("incomplete: bounded (4 requests)", len(reqs) == 4, "requests=%d" % len(reqs))
check("incomplete: ends BLOCKED, clean exit", "BLOCKED:" in p.stdout + p.stderr and p.returncode == 0, (p.stdout + p.stderr)[-200:])
check("incomplete: no assistant message ever appended", all("assistant" not in roles(r) for r in reqs))
check("incomplete: recovery message is the tool-cut instruction",
      "cut off" in reqs[1]["messages"][-1]["content"] and reqs[1]["messages"][-1]["role"] == "user")
check("incomplete: retries run without thinking", reqs[1]["chat_template_kwargs"]["enable_thinking"] is False)

# 2. recover then succeed
p, reqs = run(["incomplete_tool", "ok"])
check("recover: 2 requests then final", len(reqs) == 2 and "Done." in p.stdout, "requests=%d" % len(reqs))
check("recover: partial never replayed", roles(reqs[1]) == ["system", "user", "user"], str(roles(reqs[1])))
check("recover: not blocked", "BLOCKED" not in p.stdout)

# 3. output cut before any content or call: also a failed generation
p, reqs = run(["cut_empty", "ok"])
check("cut_empty: recovered", len(reqs) == 2 and roles(reqs[1]) == ["system", "user", "user"] and
      "ACTION REQUIRED" in reqs[1]["messages"][-1]["content"], str([roles(r) for r in reqs]))

# 4. a truncated but real content answer is a normal final reply (no recovery)
p, reqs = run(["cut_content"])
check("cut_content: single request, kept as final", len(reqs) == 1 and "The answer so far is 42" in p.stdout,
      "requests=%d" % len(reqs))
check("cut_content: truncation reported", "limit" in p.stdout.lower())

sys.exit(1 if failed else 0)
