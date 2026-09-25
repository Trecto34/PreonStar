"""
Base interface for LLM backends in Interference Search.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CandidateResult:
    text: Optional[str]
    raw_hash: str
    tokens: int
    is_error: bool = False
    error_reason: Optional[str] = None  # e.g. "backend_timeout", "backend_error"


class LLMBackend(ABC):
    @abstractmethod
    def generate_candidates(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.8,
        seeds: Optional[List[int]] = None,
        max_tokens: int = 512,
        stop_tokens: Optional[List[str]] = None,
    ) -> Tuple[List[CandidateResult], int]:
        """
        Generate candidate outputs for a list of prompts.
        Returns:
            (candidate_results, total_tokens_spent)
        """
        pass
