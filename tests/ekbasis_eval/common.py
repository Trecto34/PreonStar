"""Shared helpers for the Ekbasis POST /v1/systemone harness (stdlib only).

Prompt text is built from decision_core.py (copied unchanged from caiovicentino1/Ekbasis-27B, MIT; see its header), so
the bytes match the Python reference the model was trained with. The chat wrapper is the literal ChatML string from
SPEC.md; build_golden.py checks it against the model's chat_template.jinja.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decision_core as dc  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8123"
# Cache for downloads (dataset raw files, tokenizer.json). Never inside the repository.
CACHE = os.environ.get("EKBASIS_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "q36-ekbasis-eval"))
# calib.json is the identity (b=0, w=0), so T = exp(0) = 1. A single named constant, as in SPEC.md.
TEMPERATURE = 1.0
ENDPOINT = "/v1/systemone"


HF = "https://huggingface.co"
MODEL_REPO = "caiovicentino1/Ekbasis-27B"
MAX_DOWNLOAD = 50 * 1024 * 1024  # SPEC: nothing over 50 MB is downloaded


def hf_tree(repo, dataset=False, sub=""):
    """Entries of a repo folder from the HF tree API (dataset or model)."""
    api = f"{HF}/api/{'datasets' if dataset else 'models'}/{repo}/tree/main" + (f"/{sub}" if sub else "")
    with urllib.request.urlopen(api, timeout=60) as r:
        return {e["path"]: e for e in json.loads(r.read()) if e.get("type") == "file"}


def hf_get(repo, rel, entry, dest_dir, dataset=False):
    """Download repo/rel into dest_dir unless already there and verified against the tree entry (LFS sha256 or git
    blob sha1, and size). Returns the local path."""
    size = (entry.get("lfs") or {}).get("size", entry["size"])
    if size > MAX_DOWNLOAD:
        raise SystemExit(f"{rel}: {size} bytes is over the 50 MB limit")
    path = os.path.join(dest_dir, os.path.basename(rel))

    def ok(data):
        if len(data) != size:
            return False
        if entry.get("lfs"):
            return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == entry["oid"]

    if os.path.exists(path):
        with open(path, "rb") as f:
            if ok(f.read()):
                return path
    base = f"{HF}/datasets/{repo}" if dataset else f"{HF}/{repo}"
    with urllib.request.urlopen(f"{base}/resolve/main/{rel}", timeout=300) as r:
        data = r.read(MAX_DOWNLOAD + 1)
    if not ok(data):
        raise SystemExit(f"{rel}: downloaded bytes do not match the tree entry (size or hash)")
    os.makedirs(dest_dir, exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return path


def model_file(name):
    """tokenizer.json or chat_template.jinja from the model repo, cached under CACHE/model."""
    dest = os.path.join(CACHE, "model")
    return hf_get(MODEL_REPO, name, hf_tree(MODEL_REPO)[name], dest)


def check_style():
    # The server's prompt is the semif style; decision_core reads PROMPT_STYLE from the environment at import time.
    if dc.STYLE != "semif":
        raise SystemExit(f"decision_core.STYLE is {dc.STYLE!r}; unset PROMPT_STYLE (SPEC.md requires semif)")


check_style()
LABELS = dc.LABELS  # 588 labels: A..Z, then the two-letter codes that are one token each


def options(question):
    """(label, description) pairs, as the server derives them (criteria order preserved)."""
    return dc.options_of(question)


def user_text(state, question):
    """The user turn: the JSON {"evidence", "criterion", "options"} from decision_core.render_user."""
    return dc.render_user(state, question, options(question))


def chat_prompt(state, question):
    """One question as its own prompt (enable_thinking=False): system, user, and the empty think block."""
    return ("<|im_start|>system\n" + dc.SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\n" + user_text(state, question) + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def readonce_pieces(state, questions):
    """Read-once layout (letter_adapter.py _read_once_ids): question 0 as a one-question prompt, then for each further
    question "?" (the placeholder answer to the previous one), a new user turn with render_followup, and the assistant
    turn opened. The pieces are tokenised separately and concatenated, as the reference does."""
    pieces = [chat_prompt(state, questions[0])]
    for q in questions[1:]:
        followup = dc.render_followup(q, options(q))
        pieces.append("?<|im_end|>\n<|im_start|>user\n" + followup +
                      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return pieces


def build_request(state, questions, read_once=False):
    """questions: a list of (name, question) pairs, in order. Returns the POST body."""
    body = {"state": state, "questions": {name: q for name, q in questions}}
    if read_once:
        body["read_once"] = True
    return body


def post_systemone(url, body, timeout=600.0):
    """POST one request. Returns (http_status, raw_text, seconds). Network errors raise OSError."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + ENDPOINT, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    return status, raw.decode("utf-8", "replace"), time.perf_counter() - t0


def percentile(values, q):
    """Linear interpolation between order statistics; q in [0, 100]. Empty input gives None."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)
