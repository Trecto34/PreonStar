"""
Mock LLM Backend for deterministic testing and benchmarking.
"""
from typing import Callable, List, Optional, Tuple
from interference.backends.base import LLMBackend


class MockBackend(LLMBackend):
    def __init__(self, response_generator: Optional[Callable[[str], List[str]]] = None):
        self.response_generator = response_generator or (lambda p: ["def solution(): return 42"])

    def generate_candidates(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[str], int]:
        results = []
        tokens = 0
        for p in prompts:
            cands = self.response_generator(p)
            for c in cands:
                results.append(c)
                tokens += len(c.split())
        return results, tokens
