#!/usr/bin/env python3
"""q36-agent --server against a scripted fake server: the verification-convergence watchdog.

The scripted "model" fixes a file once and then keeps checking its own work, the failure seen in a
real run (157 steps, ended BLOCKED).  Checks, all in remote mode and without a model:
  - after the budget the next tool result carries VERIFICATION_BUDGET_EXHAUSTED and a model that
    obeys it finishes normally (no BLOCKED);
  - a model that ignores it is refused twice, then the controller ends the turn itself with a stop
    report built from tracked facts, still not BLOCKED, and the whole session stays tiny;
  - nothing is limited before the first edit (read-only investigation is free);
  - 0 disables the budget.
Usage: tests/test_agent_verify_watchdog.py [path/to/q36-agent]"""
import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

AGENT = sys.argv[1] if len(sys.argv) > 1 else "./q36-agent"
NOTICE = "VERIFICATION_BUDGET_EXHAUSTED"


def sse(delta, finish=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"


def reply(action):
    usage = "data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": 100, "completion_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": 2}}}) + "\n\n"
    if action[0] == "tool":
        _, name, args, n = action
        tc = {"tool_calls": [{"index": 0, "id": "call_%d" % n,
                              "function": {"name": name, "arguments": json.dumps(args)}}]}
        body = sse(tc, "tool_calls")
    else:
        body = sse({"content": action[1]}) + sse({}, "stop")
    return body + usage + "data: [DONE]\n\n"


def run(policy, extra=()):
    """policy(reqs) -> ("tool", name, args, n) | ("final", text); reqs = request bodies so far."""
    reqs = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            b = json.dumps({"data": [{"id": "fake", "context_length": 32768}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)
        def do_POST(self):
            reqs.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            b = reply(policy(reqs)).encode()
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close"); self.end_headers()
            self.wfile.write(b)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    d = tempfile.mkdtemp(prefix="q36vw-")
    p = subprocess.run([AGENT, "--server", "http://127.0.0.1:%d" % srv.server_port,
                        "--chdir", d, "--non-interactive", "--max-verify-steps", "3", *extra,
                        "-p", "fix main.html and verify"],
                       capture_output=True, text=True, timeout=90)
    srv.shutdown()
    return p, reqs


failed = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond:
        failed.append(name)


def last_tool_text(req):
    tools = [m for m in req["messages"] if m["role"] == "tool"]
    return tools[-1]["content"] if tools else ""


def checker(obey, limit=20):
    """Edit once, then keep checking; optionally answer when told the budget is spent.
    limit only keeps a broken controller from looping the test forever."""
    def policy(reqs):
        n = len(reqs)
        if n > limit:
            return ("final", "test limit reached")
        if obey and NOTICE in last_tool_text(reqs[-1]):
            return ("final", "Fixed main.html. Confirmed working: the page loads. Unresolved: none.")
        if n == 1:
            return ("tool", "write", {"path": "main.html", "content": "<html>ok</html>\n"}, n)
        return ("tool", "bash", {"command": "echo check%s" % chr(96 + n)}, n)
    return policy


# 1. a model that obeys the notice finishes normally
p, reqs = run(checker(obey=True))
out = p.stdout + p.stderr
check("obey: ends with the model's own answer", "Confirmed working" in p.stdout and p.returncode == 0, out[-200:])
check("obey: 5 requests (edit, 3 checks, answer)", len(reqs) == 5, "requests=%d" % len(reqs))
check("obey: notice rides on the call that spent the budget", NOTICE in last_tool_text(reqs[4]))
check("obey: notice names the last target change", "No target files changed since step 1." in last_tool_text(reqs[4]))
check("obey: asks for working/failing/inconclusive", "confirmed failing" in last_tool_text(reqs[4]))
check("obey: not BLOCKED and no forced stop", "BLOCKED" not in out and "Stopping:" not in out)
check("obey: summary counts the exhaustion", "verification budget exhaustions: 1" in out and "forced finalizations: 0" in out, out[-400:])
check("obey: normal prompts carry no watchdog text", all(NOTICE not in json.dumps(r["messages"][:2]) for r in reqs))

# 2. a model that ignores it is refused twice, then the controller finalizes
p, reqs = run(checker(obey=False))
out = p.stdout + p.stderr
check("ignore: bounded (6 requests, not dozens)", len(reqs) == 6, "requests=%d" % len(reqs))
check("ignore: the call after the notice is refused with it again",
      len(reqs) == 6 and NOTICE in last_tool_text(reqs[5]))
check("ignore: controller stop report, not BLOCKED", "Stopping:" in p.stdout and "BLOCKED" not in out and p.returncode == 0, out[-300:])
check("ignore: report is not a success claim", "not a claim that every check passed" in p.stdout)
check("ignore: summary counts the forced finalization", "forced finalizations: 1" in out and "target mutations: 1" in out, out[-400:])

# 3. read-only investigation before the first edit is never limited
def investigate(reqs):
    n = len(reqs)
    if n <= 12:
        return ("tool", "bash", {"command": "echo look%s" % chr(96 + n)}, n)
    return ("final", "Looked around.")


p, reqs = run(investigate)
check("unarmed: 12 checks then answer, nothing refused", len(reqs) == 13 and "Looked around." in p.stdout and
      all(NOTICE not in json.dumps(r) for r in reqs), "requests=%d" % len(reqs))

# 4. 0 turns the budget off
p, reqs = run(checker(obey=False, limit=9), extra=("--max-verify-steps", "0", "--max-no-progress-steps", "0"))
check("off: never refused over 10 requests", len(reqs) == 10 and all(NOTICE not in json.dumps(r) for r in reqs),
      "requests=%d" % len(reqs))

sys.exit(1 if failed else 0)
