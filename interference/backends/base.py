"""
Base interface for LLM backends in Interference Search.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple


class LLMBackend(ABC):
    @abstractmethod
    def generate_candidates(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop_tokens: List[str] = None,
    ) -> Tuple[List[str], int]:
        """
        Generate candidate outputs for a list of prompts.
        Returns:
            (texts, tokens_spent)
        """
        pass
