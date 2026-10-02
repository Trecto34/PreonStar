#!/usr/bin/env python3
"""q36-agent --server against a scripted fake server: the workspace jail and the implementation ceiling.

Replays the real pathology: the agent works in a project directory, runs `find / -name index.html`,
finds an unrelated project and starts editing it.  Checks (remote mode, no model):
  - the shell cannot see the unrelated project and the file tools refuse it, so it is never modified;
  - work on the real target continues, and the root survives compaction in the task state;
  - --yolo opts out explicitly (with a warning); --unsafe-filesystem only opens the file tools;
  - mutation thrashing ends at --max-implementation-steps with an INCOMPLETE report, never VERIFY.
Usage: tests/test_agent_workspace_jail.py [path/to/q36-agent]"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

AGENT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "./q36-agent")
LIMIT = "IMPLEMENTATION_STEP_LIMIT_REACHED"


def sse(delta, finish=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"


def reply(action):
    usage = "data: " + json.dumps({"choices": [], "usage": {
        "prompt_tokens": action[4] if len(action) > 4 else 100, "completion_tokens": 10,
        "completion_tokens_details": {"reasoning_tokens": 2}}}) + "\n\n"
    if action[0] == "tool":
        tc = {"tool_calls": [{"index": 0, "id": "call_%d" % action[3],
                              "function": {"name": action[1], "arguments": json.dumps(action[2])}}]}
        body = sse(tc, "tool_calls")
    else:
        body = sse({"content": action[1]}) + sse({}, "stop")
    return body + usage + "data: [DONE]\n\n"


def run(policy, ws, extra=(), launch=None):
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
    p = subprocess.run([AGENT, "--server", "http://127.0.0.1:%d" % srv.server_port,
                        *(["--chdir", ws] if launch is None else []), "--non-interactive", *extra,
                        "-p", "implement index.html"],
                       capture_output=True, text=True, timeout=90, cwd=launch)
    srv.shutdown()
    return p, reqs


failed = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  " + detail))
    if not cond:
        failed.append(name)


def tool_results(reqs):
    return [m["content"] for m in reqs[-1]["messages"] if m["role"] == "tool"]


def fixture():
    ws = os.path.realpath(tempfile.mkdtemp(prefix="q36html-"))
    other = os.path.realpath(tempfile.mkdtemp(prefix="q36other-"))
    with open(os.path.join(other, "index.html"), "w") as f:
        f.write("UNRELATED-PROJECT\n")
    return ws, other


def read_other(other):
    return open(os.path.join(other, "index.html")).read()


def pathology(ws, other):
    """find / for another index.html, then try to adopt it, then finish the real target."""
    oidx = os.path.join(other, "index.html")
    script = [
        ("tool", "bash", {"command": "find / -name index.html 2>/dev/null; cat %s 2>&1" % oidx}, 1),
        ("tool", "read", {"path": oidx}, 2),
        ("tool", "write", {"path": oidx, "content": "CORRUPTED\n"}, 3),
        ("tool", "write", {"path": "index.html", "content": "<html>real target</html>\n"}, 4),
    ]
    def policy(reqs):
        n = len(reqs)
        return script[n - 1] if n <= len(script) else ("final", "done")
    return policy


# 1. the exact pathology, jailed
ws, other = fixture()
p, reqs = run(pathology(ws, other), ws)
out = p.stdout + p.stderr
res = tool_results(reqs)
check("jail: the shell cannot see the unrelated project", len(res) == 4 and other not in res[0] and "UNRELATED" not in res[0], str(res[:1])[-300:])
check("jail: read outside is WORKSPACE_ESCAPE_BLOCKED", "WORKSPACE_ESCAPE_BLOCKED" in res[1] and "workspace_root: " + ws in res[1], res[1])
check("jail: write outside is blocked and the file is untouched", "WORKSPACE_ESCAPE_BLOCKED" in res[2] and read_other(other) == "UNRELATED-PROJECT\n")
check("jail: the real target is still written", open(os.path.join(ws, "index.html")).read() == "<html>real target</html>\n")
check("jail: ends normally", p.returncode == 0 and "BLOCKED:" not in out, out[-300:])

# 2. --yolo: explicit opt-out, with a warning
ws, other = fixture()
p, reqs = run(pathology(ws, other), ws, extra=("--yolo",))
res = tool_results(reqs)
check("yolo: warns at startup", "WARNING --yolo" in p.stderr, p.stderr[-300:])
check("yolo: outside read and write are allowed", "UNRELATED-PROJECT" in res[1] and read_other(other) == "CORRUPTED\n", str(res[1:3]))

# 3. --unsafe-filesystem: file tools only, the shell stays confined
ws, other = fixture()
p, reqs = run(pathology(ws, other), ws, extra=("--unsafe-filesystem",))
res = tool_results(reqs)
check("unsafe-fs: file tools reach outside, the shell does not",
      "UNRELATED" not in res[0] and "UNRELATED-PROJECT" in res[1] and read_other(other) == "CORRUPTED\n" and "WARNING --unsafe-filesystem" in p.stderr)

# 4. --workspace PATH names the root and the launch directory is irrelevant
ws, other = fixture()
here = os.path.realpath(tempfile.mkdtemp(prefix="q36launch-"))
p, reqs = run(pathology(ws, other), ws, extra=("--workspace", ws), launch=here)
check("workspace flag: the root is PATH, not the launch directory",
      "WORKSPACE_ESCAPE_BLOCKED" in tool_results(reqs)[1] and "workspace_root: " + ws in tool_results(reqs)[1] and
      os.path.exists(os.path.join(ws, "index.html")) and not os.path.exists(os.path.join(here, "index.html")))

# 5. root survives compaction: the compacted state message carries WORKSPACE_ROOT
ws, other = fixture()
def compacting(reqs):
    n = len(reqs)
    script = [("tool", "write", {"path": "index.html", "content": "<html>a</html>\n"}, 1),
              ("tool", "edit", {"path": "index.html", "old": "a", "new": "b"}, 2, 30000),
              ("tool", "read", {"path": "index.html"}, 3),
              ("tool", "read", {"path": "index.html"}, 4),
              ("tool", "list", {"path": "."}, 5)]
    return script[n - 1] if n <= len(script) else ("final", "done")
p, reqs = run(compacting, ws)
out = p.stdout + p.stderr
check("compaction: happened", "compactions: 1" in out, out[-400:])
check("compaction: WORKSPACE_ROOT is in the carried state", "WORKSPACE_ROOT: " + ws in json.dumps(reqs[-1]["messages"]), "")

# 6. mutation thrash: never VERIFY, stopped by --max-implementation-steps, INCOMPLETE report
ws, other = fixture()
def thrash(reqs):
    n = len(reqs)
    if LIMIT in (tool_results(reqs)[-1] if tool_results(reqs) else ""):
        return ("tool", "read", {"path": "index.html"}, n)       # ignores the notice
    return ("tool", "write", {"path": "index.html", "content": "<html>try %d</html>\n" % n}, n)
p, reqs = run(thrash, ws, extra=("--max-implementation-steps", "5"))
out = p.stdout + p.stderr
check("thrash: implementation limit hit once", "implementation step limit hits: 1" in out and "implementation steps: 5" in out, out[-500:])
check("thrash: never entered VERIFY", "IMPLEMENT->VERIFY 0" in out and "verification steps: 0" in out, out[-500:])
check("thrash: INCOMPLETE report, not BLOCKED, bounded",
      "INCOMPLETE" in p.stdout and "index.html" in p.stdout and "BLOCKED:" not in out and len(reqs) <= 9, "requests=%d %s" % (len(reqs), out[-300:]))
check("thrash: summary has the new counters", "implementation steps:" in out and "implementation step limit hits:" in out)

sys.exit(1 if failed else 0)
