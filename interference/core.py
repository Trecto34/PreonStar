"""
Interference Search Core Engine.

Maintains explicit state frontiers. Each round:
1. Every live state expands concurrently.
2. The environment executes proposed moves/actions.
3. Constructive Interference (Merging):
   States landing on identical semantic merge keys collapse into one,
   and reciprocal rank votes (1 / (rank + 1)) from all parent paths pool together.
4. Destructive Interference (Cancellation):
   Branches hitting invalid states or dead ends are dropped immediately.
5. Selection:
   A judge (e.g. test assertions or scoring model) ranks candidate states,
   and the top `width` states advance together.
"""
from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Result:
    solved: bool
    solution: Optional[Any] = None
    cost: float = 0.0
    rounds: int = 0
    expanded: int = 0
    merged_away: int = 0
    dead_ends: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)


def search(
    domain,
    problem: Any,
    budget: float,
    width: int = 4,
    merge: bool = True,
    restarts: bool = True,
) -> Result:
    """
    Executes Interference Search over a problem within a given budget.
    """
    live = [domain.start(problem)]
    expanded_keys: Set[Any] = set()
    res = Result(solved=False)

    while res.cost < budget:
        if not live:
            if not restarts:
                break
            live = [domain.start(problem)]

        res.rounds += 1
        proposals, pcost = domain.propose(live, problem)
        res.cost += pcost
        res.expanded += len(live)

        for s in live:
            expanded_keys.add(domain.key(s))

        pool: Dict[Any, List[Any]] = {}  # key -> [state, pooled_vote]
        raw_candidates = 0

        for kids in proposals:
            for rank, candidate in enumerate(kids):
                raw_candidates += 1
                if domain.is_goal(candidate, problem):
                    res.solved = True
                    res.solution = candidate
                    res.trace.append({
                        "round": res.rounds,
                        "raw": raw_candidates,
                        "pool": len(pool) + 1,
                        "kept": 1,
                        "goal_reached": True,
                    })
                    return res

                if domain.is_dead(candidate, problem):
                    res.dead_ends += 1
                    continue

                k = domain.key(candidate) if merge else (domain.key(candidate), raw_candidates)
                if merge and k in expanded_keys:
                    continue

                vote_weight = 1.0 / (rank + 1)
                if k in pool:
                    pool[k][1] += vote_weight  # Constructive interference: votes from multiple paths combine
                else:
                    pool[k] = [candidate, vote_weight]

        res.merged_away += (raw_candidates - len(pool) - res.dead_ends)
        if not pool:
            live = []
            res.trace.append({
                "round": res.rounds,
                "raw": raw_candidates,
                "pool": 0,
                "kept": 0,
                "dead_ends": res.dead_ends,
            })
            continue

        candidates = [entry[0] for entry in pool.values()]
        pooled_votes = [entry[1] for entry in pool.values()]

        scores, jcost = domain.judge(candidates, problem)
        res.cost += jcost

        if scores is None:
            scores = pooled_votes

        # Rank descending
        ranked_indices = sorted(range(len(candidates)), key=lambda i: -scores[i])
        live = [candidates[i] for i in ranked_indices[:width]]

        res.trace.append({
            "round": res.rounds,
            "raw": raw_candidates,
            "pool": len(pool),
            "kept": len(live),
            "best_score": scores[ranked_indices[0]] if ranked_indices else 0,
        })

    return res


def linear_search(
    domain,
    problem: Any,
    budget: float,
    temp: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Result:
    """
    Baseline single-chain search (Chain of Thought / greedy / sampled single line).
    """
    rng = rng or random.Random(0)
    res = Result(solved=False)

    while res.cost < budget:
        s = domain.start(problem)
        while res.cost < budget:
            res.rounds += 1
            res.expanded += 1
            (kids,), pcost = domain.propose([s], problem)
            res.cost += pcost

            for c in kids:
                if domain.is_goal(c, problem):
                    res.solved = True
                    res.solution = c
                    return res

            kids = [c for c in kids if not domain.is_dead(c, problem)]
            if not kids:
                break

            scores, jcost = domain.judge(kids, problem)
            res.cost += jcost

            if scores is None:
                scores = [1.0 / (i + 1) for i in range(len(kids))]

            weights = [math.exp(math.log(max(x, 1e-6)) / temp) for x in scores]
            s = rng.choices(kids, weights=weights)[0]

    return res
