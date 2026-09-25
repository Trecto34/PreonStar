"""
Countdown Domain for Interference Search.

From Bad Theory Labs:
Given a list of numbers and a target, find an arithmetic expression using each number once.
State = Remaining numbers and accumulated history.
Merge = Branches with the same multiset of remaining numbers land on the same situation.
"""
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from interference.domains.base import Domain


@dataclass(frozen=True)
class CountdownState:
    numbers: Tuple[int, ...]
    exprs: Tuple[str, ...]

    def __post_init__(self):
        # Sort numbers so equivalent sets have identical hash
        pass


class CountdownDomain(Domain):
    def __init__(self, max_branch: int = 6):
        self.max_branch = max_branch

    def start(self, problem: Tuple[List[int], int]) -> CountdownState:
        nums, target = problem
        sorted_nums = tuple(sorted(nums))
        exprs = tuple(str(x) for x in sorted_nums)
        return CountdownState(numbers=sorted_nums, exprs=exprs)

    def propose(self, states: List[CountdownState], problem: Tuple[List[int], int]) -> Tuple[List[List[CountdownState]], float]:
        _, target = problem
        children_lists = []
        for s in states:
            kids = []
            n = len(s.numbers)
            if n < 2:
                children_lists.append([])
                continue

            for i in range(n):
                for j in range(i + 1, n):
                    a, b = s.numbers[i], s.numbers[j]
                    ea, eb = s.exprs[i], s.exprs[j]
                    rem_nums = tuple(s.numbers[k] for k in range(n) if k != i and k != j)
                    rem_exprs = tuple(s.exprs[k] for k in range(n) if k != i and k != j)

                    ops = [
                        (a + b, f"({ea} + {eb})"),
                        (a * b, f"({ea} * {eb})"),
                    ]
                    if a >= b:
                        ops.append((a - b, f"({ea} - {eb})"))
                    else:
                        ops.append((b - a, f"({eb} - {ea})"))

                    if b != 0 and a % b == 0:
                        ops.append((a // b, f"({ea} / {eb})"))
                    elif a != 0 and b % a == 0:
                        ops.append((b // a, f"({eb} / {ea})"))

                    for res_val, res_expr in ops:
                        new_nums = tuple(sorted(rem_nums + (res_val,)))
                        new_exprs = rem_exprs + (res_expr,)
                        kids.append(CountdownState(numbers=new_nums, exprs=new_exprs))

            children_lists.append(kids[:self.max_branch])

        return children_lists, float(len(states))

    def key(self, state: CountdownState) -> Any:
        return state.numbers

    def judge(self, states: List[CountdownState], problem: Tuple[List[int], int]) -> Tuple[Optional[List[float]], float]:
        _, target = problem
        # Distance to target as heuristic
        scores = []
        for s in states:
            min_dist = min(abs(x - target) for x in s.numbers)
            scores.append(1000.0 / (1.0 + min_dist))
        return scores, 0.0

    def is_goal(self, state: CountdownState, problem: Tuple[List[int], int]) -> bool:
        _, target = problem
        return target in state.numbers

    def is_dead(self, state: CountdownState, problem: Tuple[List[int], int]) -> bool:
        return len(state.numbers) < 2 and problem[1] not in state.numbers
