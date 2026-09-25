"""
Q36 LLM Backend for Interference Search.

Connects to either:
1. A running q36-server instance via HTTP (/v1/chat/completions)
2. Direct invocation of the q36 binary via subprocess
"""
import json
import os
import subprocess
import time
from typing import List, Optional, Tuple
import urllib.request
import urllib.error

from interference.backends.base import LLMBackend


class Q36Backend(LLMBackend):
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        q36_binary_path: str = "/home/server/q36-interference/q36",
        model_path: str = "/home/server/models/llm/Ternary-Bonsai-2-27B-PQ2_0.gguf",
        use_vulkan: bool = True,
        threads: int = 16,
    ):
        self.server_url = server_url.rstrip("/")
        self.q36_binary_path = q36_binary_path
        self.model_path = model_path
        self.use_vulkan = use_vulkan
        self.threads = threads
        self._server_alive: Optional[bool] = None

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
        max_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[str], int]:
        if not prompts:
            return [], 0

        # Try server first if available
        if self._server_alive is None:
            self._server_alive = self._check_server()

        if self._server_alive:
            try:
                return self._generate_via_server(prompts, temperature, max_tokens, stop_tokens)
            except Exception as e:
                # Fallback to binary execution
                self._server_alive = False

        return self._generate_via_cli(prompts, temperature, max_tokens, stop_tokens)

    def _generate_via_server(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        stop_tokens: Optional[List[str]],
    ) -> Tuple[List[str], int]:
        results = []
        total_tokens = 0
        endpoint = f"{self.server_url}/v1/chat/completions"

        for p in prompts:
            payload = {
                "model": os.path.basename(self.model_path),
                "messages": [{"role": "user", "content": p}],
                "temperature": temperature,
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
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choice = body["choices"][0]
                text = choice["message"]["content"]
                results.append(text)
                usage = body.get("usage", {})
                total_tokens += usage.get("completion_tokens", len(text.split()))

        return results, total_tokens

    def _generate_via_cli(
        self,
        prompts: List[str],
        temperature: float,
        max_tokens: int,
        stop_tokens: Optional[List[str]],
    ) -> Tuple[List[str], int]:
        results = []
        total_tokens = 0

        for p in prompts:
            cmd = [
                self.q36_binary_path,
                "-m", self.model_path,
                "-t", str(self.threads),
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
                    timeout=180,
                    check=False,
                )
                output = proc.stdout.strip()
                # Basic token estimate if not reported
                tokens = max(1, len(output.split()))
                results.append(output)
                total_tokens += tokens
            except Exception as e:
                results.append(f"# Error invoking q36: {e}")
                total_tokens += 10

        return results, total_tokens
