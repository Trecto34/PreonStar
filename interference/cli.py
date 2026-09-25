"""
CLI Visualizer for Interference Search on q36.
"""
import argparse
import sys
import time

from interference.core import search, linear_search
from interference.domains.countdown_domain import CountdownDomain
from interference.domains.code_domain import CodeDomain
from interference.backends.q36_backend import Q36Backend
from interference.backends.mock_backend import MockBackend


def run_countdown_demo():
    print("=" * 60)
    print(" INTERFERENCE SEARCH - COUNTDOWN BENCHMARK")
    print("=" * 60)

    # Hard problem from Bad Theory Labs paper
    problem = ([24, 98, 19, 3], 361)
    nums, target = problem
    print(f"Problem: Use {nums} to reach target {target}\n")

    domain = CountdownDomain(max_branch=12)

    print("Executing Interference Search (Width = 6, Merging enabled)...")
    start_t = time.perf_counter()
    res = search(domain, problem, budget=200, width=6, merge=True)
    elapsed = time.perf_counter() - start_t

    print(f"\nStatus: {'SOLVED' if res.solved else 'FAILED'}")
    print(f"Rounds: {res.rounds}")
    print(f"States Expanded: {res.expanded}")
    print(f"States Merged Away (Constructive Interference): {res.merged_away}")
    print(f"Dead Ends Cancelled (Destructive Interference): {res.dead_ends}")
    print(f"Time: {elapsed * 1000:.2f} ms")

    if res.solved and res.solution:
        for val, expr in zip(res.solution.numbers, res.solution.exprs):
            if val == target:
                print(f"Solution: {expr} = {target}")

    print("\nSearch Trace by Round:")
    for step in res.trace:
        print(f"  Round {step['round']}: {step.get('raw', 0)} raw candidates -> {step.get('pool', 0)} merged pool -> {step.get('kept', 0)} kept for next round")


def run_code_demo(backend_type="mock"):
    print("=" * 60)
    print(" INTERFERENCE SEARCH - CODE REASONING")
    print("=" * 60)

    problem = {
        "prompt": "Write a function `fibonacci(n)` that returns the n-th Fibonacci number (0-indexed, where fib(0)=0, fib(1)=1).",
        "test_list": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(5) == 5",
            "assert fibonacci(10) == 55",
        ],
    }

    if backend_type == "q36":
        print("Using Q36 Vulkan Backend with local Ternary-Bonsai...")
        backend = Q36Backend()
    else:
        print("Using Deterministic Mock Backend for rapid test...")
        # Simulates candidate generation with diverse paths
        step_counter = [0]
        def mock_gen(p):
            step_counter[0] += 1
            if step_counter[0] <= 2:
                # First round proposals (one buggy, one working)
                return [
                    "```python\ndef fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
                    "```python\ndef fibonacci(n):\n    return n * 2\n```",
                    "```python\ndef fibonacci(n):\n    # Alternative syntax\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
                ]
            return [
                "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```"
            ]
        backend = MockBackend(response_generator=mock_gen)

    domain = CodeDomain(backend=backend, branch_factor=3, fresh_branches=3)

    print(f"Task: {problem['prompt']}")
    print(f"Tests: {len(problem['test_list'])} assertions\n")

    res = search(domain, problem, budget=2000, width=3, merge=True)

    print(f"\nStatus: {'SOLVED' if res.solved else 'FAILED'}")
    print(f"Rounds: {res.rounds}")
    print(f"Total Cost (Tokens): {res.cost}")
    print(f"States Merged Away: {res.merged_away}")

    if res.solved and res.solution:
        print("\nGenerated Solution:")
        print("-" * 40)
        print(res.solution.src)
        print("-" * 40)
        print(f"Passed Tests: {res.solution.passed}/{len(problem['test_list'])}")


def main():
    parser = argparse.ArgumentParser(description="Interference Search for Q36")
    parser.add_argument("--domain", choices=["countdown", "code"], default="countdown")
    parser.add_argument("--backend", choices=["mock", "q36"], default="mock")
    args = parser.parse_args()

    if args.domain == "countdown":
        run_countdown_demo()
    else:
        run_code_demo(backend_type=args.backend)


if __name__ == "__main__":
    main()
