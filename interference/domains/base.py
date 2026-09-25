"""
Base domain contract for Interference Search.

A domain supplies:
- start(problem): Initial state.
- propose(states, problem): Expands candidate child states and reports generation cost.
- key(state): Hashable key representing the behavioral/semantic situation. Identical keys merge.
- judge(states, problem): Evaluates states (higher = better). If None, pooled parent votes are used.
- is_goal(state, problem): Checks if state satisfies problem goal.
- is_dead(state, problem): Checks if state reached a dead end.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class Domain(ABC):
    @abstractmethod
    def start(self, problem: Any) -> Any:
        pass

    @abstractmethod
    def propose(self, states: List[Any], problem: Any) -> Tuple[List[List[Any]], float]:
        pass

    @abstractmethod
    def key(self, state: Any) -> Any:
        pass

    @abstractmethod
    def judge(self, states: List[Any], problem: Any) -> Tuple[Optional[List[float]], float]:
        pass

    @abstractmethod
    def is_goal(self, state: Any, problem: Any) -> bool:
        pass

    @abstractmethod
    def is_dead(self, state: Any, problem: Any) -> bool:
        pass
