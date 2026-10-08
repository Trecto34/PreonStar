"""FAKE server for offline tests of eval.py and bench.py. It is NOT Ekbasis and NOT q36-server: answers are a fixed
hash-based guess, so accuracy from it means nothing. It only speaks the POST /v1/systemone shape from SPEC.md.

  python3 fake_server.py --port 8799        # serves until interrupted
"""
import argparse
import hashlib
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "FAKE-not-ekbasis (offline test server)"


def answer(name, q, state):
    h = int(hashlib.sha256((name + json.dumps(state)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if q.get("type") in ("noul", "boolean"):
        p = 0.05 + 0.9 * h
        return {"type": q["type"], "noul": p, "probability": p, "value": p >= 0.5, "confidence": max(p, 1 - p)}
    labels = list((q.get("criteria") or {}).keys()) or ["0", "1"]
    probs = {lab: 1.0 / len(labels) for lab in labels}
    return {"type": q.get("type"), "choice": labels[0], "probabilities": probs, "confidence": 1.0 / len(labels)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/systemone" or "state" not in body or not isinstance(body.get("questions"), dict):
            return self._send(400, {"error": {"message": "FAKE: bad request"}})
        qs = body["questions"]
        ans = {k: answer(k, v, body["state"]) for k, v in qs.items()}
        n_tok = len(json.dumps(body["state"])) // 4 + 60 * len(qs) + 40
        time.sleep(0.002 * len(qs))
        self._send(200, {"model": MODEL, "answers": ans, "usage": {"input_tokens": n_tok, "output_tokens": 0}})

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port=0):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    print(f"FAKE server on 127.0.0.1:{a.port}")
    serve(a.port).serve_forever()
