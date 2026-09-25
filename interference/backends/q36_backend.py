"""
Q36 LLM Backend for Interference Search.

Connects to either:
1. A running q36-server instance via HTTP (/v1/chat/completions)
2. Direct invocation of the q36 binary via subprocess with explicit sampling flags
"""
import glob
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

from interference.backends.base import LLMBackend, CandidateResult

DEFAULT_MODEL_DIRS = [
    ("/home/server/models/llm", "SSD NVMe (Rápido)"),
    ("/home/server/models", "SSD NVMe (Rápido)"),
    ("/mnt/external/models/llm", "HDD Externo"),
    ("/mnt/external/models", "HDD Externo"),
]


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:6]


def list_available_models() -> List[Dict[str, Any]]:
    """Discovers all loadable GGUF models on disk."""
    models = []
    seen_paths = set()

    for folder, storage_label in DEFAULT_MODEL_DIRS:
        if not os.path.exists(folder):
            continue
        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path) and entry.endswith(".gguf") and full_path not in seen_paths:
                seen_paths.add(full_path)
                size_gb = os.path.getsize(full_path) / (1024 ** 3)
                models.append({
                    "name": entry,
                    "path": full_path,
                    "size_gb": size_gb,
                    "storage": storage_label,
                })

    models.sort(key=lambda m: (0 if "SSD" in m["storage"] else 1, m["name"]))
    return models


def resolve_model_path(query: str) -> Optional[str]:
    """Resolves a model by index, filename, substring or absolute path."""
    if not query:
        return None

    if os.path.exists(query) and os.path.isfile(query):
        return os.path.abspath(query)

    models = list_available_models()
    if not models:
        return None

    if query.isdigit():
        idx = int(query) - 1
        if 0 <= idx < len(models):
            return models[idx]["path"]

    for m in models:
        if m["name"].lower() == query.lower():
            return m["path"]

    q = query.lower()
    matches = [m for m in models if q in m["name"].lower()]
    if matches:
        matches.sort(key=lambda m: (0 if "SSD" in m["storage"] else 1))
        return matches[0]["path"]

    return None


class Q36Backend(LLMBackend):
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        q36_binary_path: str = "/home/server/q36-interference/q36",
        model_path: Optional[str] = None,
        use_vulkan: bool = True,
        threads: int = 16,
        generation_timeout: int = 600,
    ):
        self.server_url = server_url.rstrip("/")
        self.q36_binary_path = q36_binary_path
        self.generation_timeout = generation_timeout

        if model_path:
            resolved = resolve_model_path(model_path)
            self.model_path = resolved if resolved else model_path
        else:
            avail = list_available_models()
            self.model_path = avail[0]["path"] if avail else "/home/server/models/llm/Ternary-Bonsai-2-27B-PQ2_0.gguf"

        self.use_vulkan = use_vulkan
        self.threads = threads
        self._server_alive: Optional[bool] = None

    def set_model(self, model_query: str) -> Tuple[bool, str]:
        resolved = resolve_model_path(model_query)
        if resolved:
            self.model_path = resolved
            return True, resolved
        return False, f"Modelo '{model_query}' não encontrado."

    def _check_server(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.server_url}/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        return False

    def generate_candidates(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.8,
        seeds: Optional[List[int]] = None,
        max_tokens: int = 2048,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[CandidateResult], int]:
        if not prompts:
            return [], 0

        seeds = seeds or [42 + i for i in range(len(prompts))]

        # Try server first if available
        if self._server_alive is None:
            self._server_alive = self._check_server()

        if self._server_alive:
            try:
                return self._generate_via_server(prompts, temperature, top_p, seeds, max_tokens, stop_tokens)
            except Exception:
                self._server_alive = False

        return self._generate_via_cli(prompts, temperature, top_p, seeds, max_tokens, stop_tokens)

    def _generate_via_server(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        seeds: List[int],
        max_tokens: int,
        stop_tokens: Optional[List[str]],
    ) -> Tuple[List[CandidateResult], int]:
        results: List[CandidateResult] = []
        total_tokens = 0
        endpoint = f"{self.server_url}/v1/chat/completions"

        for p, seed in zip(prompts, seeds):
            payload = {
                "model": os.path.basename(self.model_path),
                "messages": [{"role": "user", "content": p}],
                "temperature": temperature,
                "top_p": top_p,
                "seed": seed,
                "max_tokens": max_tokens,
            }
            if stop_tokens:
                payload["stop"] = stop_tokens

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.generation_timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    choice = body["choices"][0]
                    text = choice["message"]["content"]
                    usage = body.get("usage", {})
                    tokens = usage.get("completion_tokens", len(text.split()))
                    total_tokens += tokens
                    results.append(CandidateResult(
                        text=text,
                        raw_hash=short_hash(text),
                        tokens=tokens,
                        is_error=False,
                    ))
            except urllib.error.URLError as e:
                if "timed out" in str(e).lower():
                    results.append(CandidateResult(
                        text=None,
                        raw_hash="none00",
                        tokens=0,
                        is_error=True,
                        error_reason="backend_timeout",
                    ))
                else:
                    results.append(CandidateResult(
                        text=None,
                        raw_hash="none00",
                        tokens=0,
                        is_error=True,
                        error_reason="backend_error",
                    ))

        return results, total_tokens

    def _generate_via_cli(
        self,
        prompts: List[str],
        temperature: float,
        top_p: float,
        seeds: List[int],
        max_tokens: int,
        stop_tokens: Optional[List[str]],
    ) -> Tuple[List[CandidateResult], int]:
        results: List[CandidateResult] = []
        total_tokens = 0

        for p, seed in zip(prompts, seeds):
            cmd = [
                self.q36_binary_path,
                "-m", self.model_path,
                "-t", str(self.threads),
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--top-p", str(top_p),
                "--seed", str(seed),
                "-p", p,
            ]
            if self.use_vulkan:
                cmd.append("--vulkan")
            else:
                cmd.append("--cpu")

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.generation_timeout,
                    check=False,
                )
                if proc.returncode != 0:
                    results.append(CandidateResult(
                        text=None,
                        raw_hash="none00",
                        tokens=0,
                        is_error=True,
                        error_reason=f"backend_error_exit_{proc.returncode}",
                    ))
                    continue

                output = proc.stdout.strip()
                tokens = max(1, len(output.split()))
                total_tokens += tokens
                results.append(CandidateResult(
                    text=output,
                    raw_hash=short_hash(output),
                    tokens=tokens,
                    is_error=False,
                ))
            except subprocess.TimeoutExpired:
                # Crucial fix: never inject timeout as code or write to disk
                results.append(CandidateResult(
                    text=None,
                    raw_hash="none00",
                    tokens=0,
                    is_error=True,
                    error_reason="backend_timeout",
                ))
            except Exception as e:
                results.append(CandidateResult(
                    text=None,
                    raw_hash="none00",
                    tokens=0,
                    is_error=True,
                    error_reason="backend_error",
                ))

        return results, total_tokens
