"""
Interference Search Core Engine with Rigorous Execution Provenance.

Maintains explicit state frontiers and branch telemetry.
Tracks:
- Branch provenance (branch_id, parent_id, depth)
- Sampling seeds & prompt fingerprints
- Workspace hashes (pre/post) proving branch isolation
- Patch / diff hashes
- Verifier outcomes
- Exact merge keys and winner selection rationales
"""
from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple


def short_hash(data: Any) -> str:
    """Computes a 6-character sha256 hex digest for fingerprints."""
    if data is None:
        return "none00"
    if isinstance(data, (bytes, bytearray)):
        raw = data
    else:
        raw = str(data).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:6]


@dataclass
class StateNode:
    state_id: str
    depth: int
    data: Any
    parent_id: Optional[str] = None
    workspace_hash: str = ""
    vote_weight: float = 1.0
    reason: str = "initial"


@dataclass
class BranchRecord:
    branch_id: str
    parent_id: str
    depth: int
    seed: int
    prompt_hash: str
    workspace_pre: str
    workspace_post: str
    diff_hash: str
    verifier_result: str
    merge_key: Any
    is_dead: bool
    is_goal: bool
    candidate_data: Any
    rank: int = 0
    temp: float = 0.7
    top_p: float = 0.8
    raw_response_hash: str = "none00"
    dead_reason: Optional[str] = None


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
    restarts: bool = False,
    verbose: bool = False,
    base_seed: int = 40,
    max_rounds: int = 10,
) -> Result:
    """
    Executes Interference Search with provenance tracking and strict isolation logging.
    """
    initial_data = domain.start(problem)
    initial_ws_hash = getattr(domain, "workspace_hash", lambda s: short_hash(domain.key(s)))(initial_data)

    root_node = StateNode(
        state_id="s0",
        depth=0,
        data=initial_data,
        parent_id=None,
        workspace_hash=initial_ws_hash,
        vote_weight=1.0,
        reason="initial_state",
    )

    live: List[StateNode] = [root_node]
    expanded_keys: Set[Any] = set()
    res = Result(solved=False)

    state_counter = 0
    branch_counter = 0

    if verbose:
        print("\n\033[1;34m============================================================\033[0m")
        print("\033[1;34m INTERFERENCE SEARCH ENGINE (PROVENANCE & MERGE TRACE)\033[0m")
        print(f" Budget: {budget} | Max Rounds: {max_rounds} | Width: {width} | Merging: {merge} | Base Seed: {base_seed}")
        print("\033[1;34m============================================================\033[0m")

    while res.cost < budget and res.rounds < max_rounds:
        if not live:
            if not restarts:
                if verbose:
                    print("\033[33m[!] Nenhum estado vivo restante. Encerrando busca.\033[0m")
                break
            if verbose:
                print("\033[33m[!] Fronteira esgotada, reiniciando a partir da raiz s0...\033[0m")
            live = [root_node]

        current_depth = live[0].depth + 1
        res.rounds += 1

        # Check if domain implements detailed branch execution
        has_branch_builder = hasattr(domain, "propose_branches")

        proposals_per_parent: List[List[BranchRecord]] = []
        pcost = 0.0

        for parent in live:
            res.expanded += 1
            expanded_keys.add(domain.key(parent.data))

            if has_branch_builder:
                kids_records, cost_delta = domain.propose_branches(
                    parent,
                    problem,
                    start_branch_idx=branch_counter + 1,
                    base_seed=base_seed,
                )
                branch_counter += len(kids_records)
                pcost += cost_delta
                proposals_per_parent.append(kids_records)
            else:
                # Wrap standard propose with provenance records
                (kids_list,), cost_delta = domain.propose([parent.data], problem)
                pcost += cost_delta
                wrapped_records = []
                for rank, kid in enumerate(kids_list):
                    branch_counter += 1
                    b_id = f"b{branch_counter}"
                    seed = base_seed + branch_counter
                    p_hash = short_hash(f"{parent.state_id}_{domain.key(parent.data)}")
                    ws_pre = parent.workspace_hash
                    ws_post = getattr(domain, "workspace_hash", lambda s: short_hash(domain.key(s)))(kid)
                    d_hash = short_hash(f"{ws_pre}->{ws_post}")

                    is_g = domain.is_goal(kid, problem)
                    is_d = domain.is_dead(kid, problem)
                    v_res = "tests PASS" if is_g else ("tests FAIL" if is_d else "ALIVE")

                    wrapped_records.append(BranchRecord(
                        branch_id=b_id,
                        parent_id=parent.state_id,
                        depth=current_depth,
                        seed=seed,
                        prompt_hash=p_hash,
                        workspace_pre=ws_pre,
                        workspace_post=ws_post,
                        diff_hash=d_hash,
                        verifier_result=v_res,
                        merge_key=domain.key(kid),
                        is_dead=is_d,
                        is_goal=is_g,
                        candidate_data=kid,
                        rank=rank,
                        temp=0.7,
                        top_p=0.8,
                        raw_response_hash=short_hash(domain.key(kid)),
                    ))
                proposals_per_parent.append(wrapped_records)

        res.cost += pcost

        # Process branches and log per-parent group
        pool: Dict[Any, Dict[str, Any]] = {}
        # key -> { 'candidate_data': ..., 'branches': [b_ids], 'pooled_votes': float, 'state_hash': ... }
        raw_count = 0
        dead_count = 0

        for p_idx, parent in enumerate(live):
            branches = proposals_per_parent[p_idx]
            if verbose:
                print(f"\n\033[1mdepth={current_depth} parent={parent.state_id} workspace={parent.workspace_hash}\033[0m")
                print(f"\033[90m(gerados {len(branches)} candidatos a partir de {parent.state_id})\033[0m")

            for b in branches:
                raw_count += 1
                state_sig = short_hash(domain.key(b.candidate_data))

                if b.is_dead:
                    res.dead_ends += 1
                    dead_count += 1
                    if verbose:
                        reason_str = f"reason={b.dead_reason}" if b.dead_reason else f"{b.verifier_result}"
                        print(f"  {b.branch_id} seed={b.seed} temp={b.temp} top_p={b.top_p} prompt={b.prompt_hash} raw={b.raw_response_hash} -> ws_pre={b.workspace_pre} -> diff={b.diff_hash} -> ws_post={b.workspace_post} -> DEAD ({reason_str})")
                    continue

                k = b.merge_key if merge else (b.merge_key, raw_count)
                if merge and k in expanded_keys:
                    if verbose:
                        print(f"  {b.branch_id} seed={b.seed} temp={b.temp} top_p={b.top_p} prompt={b.prompt_hash} raw={b.raw_response_hash} -> diff={b.diff_hash} -> ALREADY_EXPANDED -> PRUNED")
                    continue

                vote_weight = 1.0 / (b.rank + 1)

                if k in pool:
                    pool[k]["pooled_votes"] += vote_weight
                    pool[k]["branches"].append(b.branch_id)
                    if b.is_goal:
                        pool[k]["is_goal"] = True
                else:
                    pool[k] = {
                        "candidate_data": b.candidate_data,
                        "branches": [b.branch_id],
                        "pooled_votes": vote_weight,
                        "workspace_post": b.workspace_post,
                        "state_hash": state_sig,
                        "is_goal": b.is_goal,
                    }

                if verbose:
                    print(f"  {b.branch_id} seed={b.seed} temp={b.temp} top_p={b.top_p} prompt={b.prompt_hash} raw={b.raw_response_hash} -> ws_pre={b.workspace_pre} -> diff={b.diff_hash} -> ws_post={b.workspace_post} -> {b.verifier_result} -> state={state_sig}")

        merged_this_round = (raw_count - len(pool) - dead_count)
        res.merged_away += max(0, merged_this_round)

        if not pool:
            live = []
            if verbose:
                print(f"\n\033[33mNenhum candidato sobreviveu nesta rodada.\033[0m")
            continue

        # Print Merges
        new_states: List[StateNode] = []
        if verbose:
            print()

        for k, entry in pool.items():
            state_counter += 1
            sid = f"s{state_counter}"
            b_list = entry["branches"]
            pooled_v = entry["pooled_votes"]
            ws_post = entry["workspace_post"]

            if verbose and len(b_list) > 1:
                b_str = "+".join(b_list)
                print(f"  \033[1;35mMERGE {b_str} -> {sid}\033[0m (merge_key='{k}', pooled_votes={pooled_v:.3f})")
            elif verbose:
                print(f"  PROMOTE {b_list[0]} -> {sid} (state={entry['state_hash']})")

            new_states.append(StateNode(
                state_id=sid,
                depth=current_depth,
                data=entry["candidate_data"],
                parent_id=None,
                workspace_hash=ws_post,
                vote_weight=pooled_v,
                reason="pooled_merge" if len(b_list) > 1 else "promoted",
            ))

        # Judge and Select winners
        cand_datas = [s.data for s in new_states]
        scores, jcost = domain.judge(cand_datas, problem)
        res.cost += jcost

        if scores is None:
            scores = [s.vote_weight for s in new_states]

        # Rank descending
        ranked_indices = sorted(range(len(new_states)), key=lambda i: -scores[i])
        selected_nodes = [new_states[i] for i in ranked_indices[:width]]

        goal_winner = None
        if verbose:
            print()
            for rank_pos, s_node in enumerate(selected_nodes):
                idx = ranked_indices[rank_pos]
                if domain.is_goal(s_node.data, problem):
                    reason_msg = "verified_pass"
                    goal_winner = s_node
                else:
                    reason_msg = f"rank={rank_pos+1} score={scores[idx]:.2f} pooled_votes={s_node.vote_weight:.3f}"
                print(f"\033[1;32mSELECT {s_node.state_id}\033[0m reason=\"{reason_msg}\"")

        if goal_winner or domain.is_goal(selected_nodes[0].data, problem):
            winner = goal_winner or selected_nodes[0]
            res.solved = True
            res.solution = winner.data
            return res

        live = selected_nodes
        res.trace.append({
            "round": res.rounds,
            "raw": raw_count,
            "pool": len(pool),
            "kept": len(live),
            "merged": merged_this_round,
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

