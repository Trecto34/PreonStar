"""
Mock LLM Backend for deterministic testing and benchmarking.
"""
import hashlib
from typing import Callable, List, Optional, Tuple
from interference.backends.base import LLMBackend, CandidateResult


def short_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:6]


class MockBackend(LLMBackend):
    def __init__(self, response_generator: Optional[Callable[[str], List[str]]] = None):
        self.response_generator = response_generator or (lambda p: ["def solution(): return 42"])
        self.recorded_seeds: List[Optional[int]] = []
        self.recorded_temps: List[float] = []
        self.recorded_top_ps: List[float] = []

    def generate_candidates(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.8,
        seeds: Optional[List[int]] = None,
        max_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[CandidateResult], int]:
        results = []
        total_tokens = 0
        seeds = seeds or [None] * len(prompts)

        for p, s in zip(prompts, seeds):
            self.recorded_seeds.append(s)
            self.recorded_temps.append(temperature)
            self.recorded_top_ps.append(top_p)

            cands = self.response_generator(p)
            for c in cands:
                if c is None or c.startswith("TIMEOUT"):
                    results.append(CandidateResult(
                        text=None,
                        raw_hash="none00",
                        tokens=0,
                        is_error=True,
                        error_reason="backend_timeout",
                    ))
                elif c.startswith("ERROR"):
                    results.append(CandidateResult(
                        text=None,
                        raw_hash="none00",
                        tokens=0,
                        is_error=True,
                        error_reason="backend_error",
                    ))
                else:
                    tok_count = max(1, len(c.split()))
                    total_tokens += tok_count
                    results.append(CandidateResult(
                        text=c,
                        raw_hash=short_hash(c),
                        tokens=tok_count,
                        is_error=False,
                    ))

        return results, total_tokens
