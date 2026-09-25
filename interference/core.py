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
    verbose: bool = False,
) -> Result:
    """
    Executes Interference Search over a problem within a given budget.
    """
    live = [domain.start(problem)]
    expanded_keys: Set[Any] = set()
    res = Result(solved=False)

    if verbose:
        print("\n\033[1;34m[Interference Search: Iniciando busca]\033[0m")
        print(f"  • Budget máximo: {budget} | Largura (width): {width} | Merging: {merge}")

    while res.cost < budget:
        if not live:
            if not restarts:
                if verbose:
                    print("\033[33m  [!] Nenhum estado vivo restante e restarts desativados. Encerrando.\033[0m")
                break
            if verbose:
                print("\033[33m  [!] Fronteira vazia, reiniciando do estado inicial...\033[0m")
            live = [domain.start(problem)]

        res.rounds += 1
        if verbose:
            print(f"\n\033[1;36m━━━ Rodada {res.rounds} ━━━\033[0m")
            print(f"  • Expandindo {len(live)} estado(s) vivo(s)... (Custo acumulado: {res.cost:.1f})")

        proposals, pcost = domain.propose(live, problem)
        res.cost += pcost
        res.expanded += len(live)

        for s in live:
            expanded_keys.add(domain.key(s))

        pool: Dict[Any, List[Any]] = {}  # key -> [state, pooled_vote]
        raw_candidates = 0
        dead_ends_this_round = 0

        for p_idx, kids in enumerate(proposals):
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
                    if verbose:
                        print(f"  \033[1;32m🎯 [OBJETIVO ALCANÇADO] Candidato {raw_candidates} satisfez 100% dos requisitos!\033[0m")
                        print(f"     Rodadas: {res.rounds} | Custo total: {res.cost:.1f} | Estados fundidos no total: {res.merged_away}")
                    return res

                if domain.is_dead(candidate, problem):
                    res.dead_ends += 1
                    dead_ends_this_round += 1
                    if verbose:
                        print(f"  \033[31m  ✗ [Poda Destrutiva] Candidato {raw_candidates} descartado (estado morto/erro irrecuperável)\033[0m")
                    continue

                k = domain.key(candidate) if merge else (domain.key(candidate), raw_candidates)
                if merge and k in expanded_keys:
                    if verbose:
                        print(f"  \033[90m  ~ [Já Visitado] Candidato {raw_candidates} redundante com chave já expandida.\033[0m")
                    continue

                vote_weight = 1.0 / (rank + 1)
                if k in pool:
                    pool[k][1] += vote_weight  # Constructive interference
                    if verbose:
                        print(f"  \033[35m  ⚡ [Fusão Construtiva] Candidato {raw_candidates} colapsou na chave existente! Novo peso acumulado: {pool[k][1]:.3f}\033[0m")
                else:
                    pool[k] = [candidate, vote_weight]

        merged_this_round = (raw_candidates - len(pool) - dead_ends_this_round)
        res.merged_away += max(0, merged_this_round)

        if verbose:
            print(f"  📊 Síntese da Rodada {res.rounds}:")
            print(f"     - Candidatos brutos gerados: {raw_candidates}")
            print(f"     - Podados por erro/morte: {dead_ends_this_round}")
            print(f"     - Estados únicos no pool: {len(pool)} ({merged_this_round} fundidos por equivalência)")

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

        if verbose:
            print(f"  🏆 Selecionados top {len(live)} para avançar:")
            for rank_pos, idx in enumerate(ranked_indices[:width]):
                print(f"     #{rank_pos + 1}: Score = {scores[idx]:.2f} (Voto pooled = {pooled_votes[idx]:.3f})")

        res.trace.append({
            "round": res.rounds,
            "raw": raw_candidates,
            "pool": len(pool),
            "kept": len(live),
            "best_score": scores[ranked_indices[0]] if ranked_indices else 0,
        })

    if verbose and not res.solved:
        print(f"\n\033[33m[Busca Encerrada] Limite de budget atingido ({res.cost:.1f}/{budget}).\033[0m")

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
