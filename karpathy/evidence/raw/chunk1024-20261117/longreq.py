"""Long-context server check: wait for p36-server, send a ~22k-token prompt,
then the same prompt plus a short tail (disk/live KV reuse). Prints timings."""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8123"
text = open(sys.argv[1], encoding="utf-8").read()[: int(sys.argv[2])]

def get(path):
    with urllib.request.urlopen(URL + path, timeout=5) as r:
        return r.read()

def chat(content):
    body = json.dumps({"model": "p36", "temperature": 0, "max_tokens": 16, "stream": False,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    return time.time() - t, out

for i in range(600):
    try:
        get("/v1/models"); break
    except Exception:
        time.sleep(1)
else:
    sys.exit("server never came up")

for label, content in (("cold", text + "\n\nSummarize the story in one sentence."),
                       ("reuse", text + "\n\nName the main character.")):
    dt, out = chat(content)
    u = out.get("usage", {})
    msg = out["choices"][0]["message"].get("content", "")
    print(f"{label}: {dt:.1f} s  prompt_tokens={u.get('prompt_tokens')}  "
          f"prefill_tps~{(u.get('prompt_tokens') or 0) / dt:.0f}  reply={msg[:60]!r}")
