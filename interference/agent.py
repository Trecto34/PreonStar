"""
Autonomous Coding Agent powered by Interference Search.

Workflow:
1. Takes a target source file and a verification command (e.g. "pytest", "python3 test.py", "make test").
2. In each round, generates K candidate fixes/implementations in parallel using Q36.
3. Executes the verification command in the project environment for each candidate.
4. Constructive Interference: Candidates that yield identical test outcomes collapse into one.
5. Destructive Interference: Candidates that introduce compilation/syntax crashes are dropped.
6. The best candidate states (highest pass rate / fewest errors) advance together.
7. Commits the working code to disk once all checks pass!
"""
import argparse
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from interference.core import Result
from interference.domains.base import Domain
from interference.backends.q36_backend import Q36Backend
from interference.backends.mock_backend import MockBackend


def run_command(cmd: str, cwd: str, timeout: int = 30) -> Tuple[int, str, str]:
    """Runs a shell command and captures return code and output."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "Execution timed out"
    except Exception as e:
        return 1, "", str(e)


def extract_code_block(text: str) -> str:
    """Extracts code block from model response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    return text.strip()


class FileRepairDomain(Domain):
    def __init__(
        self,
        backend,
        target_file: str,
        test_cmd: str,
        cwd: str,
        task_instruction: str,
        k: int = 3,
    ):
        self.backend = backend
        self.target_file = target_file
        self.test_cmd = test_cmd
        self.cwd = cwd
        self.task_instruction = task_instruction
        self.k = k

        # Cache original content to restore on cancellation
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            self.original_content = f.read()

    def start(self, problem: Any) -> Dict[str, Any]:
        ret, stdout, stderr = run_command(self.test_cmd, self.cwd)
        return {
            "content": self.original_content,
            "returncode": ret,
            "output": (stdout + "\n" + stderr).strip(),
            "passed": (ret == 0),
        }

    def _make_prompt(self, state: Dict[str, Any]) -> str:
        return (
            f"You are an expert programming agent. Your task is:\n"
            f"{self.task_instruction}\n\n"
            f"Target file: {os.path.basename(self.target_file)}\n\n"
            f"Current file contents:\n"
            f"```\n{state['content']}\n```\n\n"
            f"Verification command output ({self.test_cmd}):\n"
            f"```\n{state['output'][:2000]}\n```\n\n"
            f"Provide the complete updated file content that fixes the issues and makes the verification command pass.\n"
            f"Reply ONLY with the new file contents inside a single ``` block."
        )

    def propose(self, states: List[Dict[str, Any]], problem: Any) -> Tuple[List[List[Dict[str, Any]]], float]:
        prompts = []
        state_indices = []

        for i, s in enumerate(states):
            p = self._make_prompt(s)
            for _ in range(self.k):
                prompts.append(p)
                state_indices.append(i)

        raw_responses, token_cost = self.backend.generate_candidates(prompts, max_tokens=2048)

        child_groups: List[List[Dict[str, Any]]] = [[] for _ in range(len(states))]

        # Temporarily write each candidate to the file, test it, and restore
        for resp, s_idx in zip(raw_responses, state_indices):
            code = extract_code_block(resp)
            if not code or code == states[s_idx]["content"]:
                continue

            # Apply candidate to disk
            with open(self.target_file, "w", encoding="utf-8") as f:
                f.write(code)

            ret, stdout, stderr = run_command(self.test_cmd, self.cwd)
            combined_out = (stdout + "\n" + stderr).strip()

            candidate_state = {
                "content": code,
                "returncode": ret,
                "output": combined_out,
                "passed": (ret == 0),
            }
            child_groups[s_idx].append(candidate_state)

        # Restore original file during search
        with open(self.target_file, "w", encoding="utf-8") as f:
            f.write(self.original_content)

        return child_groups, float(token_cost)

    def key(self, state: Dict[str, Any]) -> Any:
        # Merge key: return code + last 3 lines of output (behavioral error signature)
        tail_lines = "\n".join(state["output"].splitlines()[-4:])
        return (state["returncode"], tail_lines)

    def judge(self, states: List[Dict[str, Any]], problem: Any) -> Tuple[Optional[List[float]], float]:
        # Score higher if return code is 0, or shorter error message / fewer test failures
        scores = []
        for s in states:
            if s["passed"]:
                scores.append(100.0)
            else:
                # Basic heuristic: non-empty output, fewer error traces
                err_penalty = len(s["output"].splitlines())
                scores.append(max(1.0, 50.0 - err_penalty))
        return scores, 0.0

    def is_goal(self, state: Dict[str, Any], problem: Any) -> bool:
        return state["passed"]

    def is_dead(self, state: Dict[str, Any], problem: Any) -> bool:
        return False


def run_agent_task(
    target_file: str,
    test_cmd: str,
    task_instruction: str,
    backend_choice: str = "q36",
    model_choice: str = None,
    max_rounds: int = 5,
    width: int = 3,
    verbose: bool = True,
):
    print("=" * 65)
    print(" Q36 INTERFERENCE SEARCH CODING AGENT")
    print("=" * 65)
    print(f"Target File : {target_file}")
    print(f"Verify Cmd  : {test_cmd}")
    print(f"Task        : {task_instruction}")
    print(f"Backend     : {backend_choice}")

    if backend_choice == "q36":
        backend = Q36Backend(model_path=model_choice)
        print(f"Model       : {os.path.basename(backend.model_path)}")
    else:
        # Simple test mock
        backend = MockBackend(lambda p: ["```python\ndef solve(): return True\n```"])
        print("Model       : Mock")

    print(f"Verbose     : {'Ativado' if verbose else 'Desativado'}")
    print("=" * 65)

    abs_path = os.path.abspath(target_file)
    cwd = os.path.dirname(abs_path) or "."

    domain = FileRepairDomain(
        backend=backend,
        target_file=abs_path,
        test_cmd=test_cmd,
        cwd=cwd,
        task_instruction=task_instruction,
        k=3,
    )

    initial_state = domain.start(None)
    if initial_state["passed"]:
        print("\nVerification command already passes! Nothing to fix.")
        return

    print(f"\nInitial State: FAILING (Exit code: {initial_state['returncode']})")
    print(f"Output snippet:\n{initial_state['output'][:300]}\n")
    print("Starting multi-branch search with constructive merging...")

    from interference.core import search
    res = search(domain, problem=None, budget=10000, width=width, merge=True, verbose=verbose)

    if res.solved and res.solution:
        print("\n" + "=" * 65)
        print(" SUCCESS! Found verified solution that passes all checks.")
        print(f" Rounds: {res.rounds} | States Merged Away: {res.merged_away} | Dead Ends: {res.dead_ends}")
        print("=" * 65)

        # Write solution permanently
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(res.solution["content"])
        print(f"Saved verified fix to {abs_path}")
    else:
        print("\n" + "=" * 65)
        print(" Failed to find a complete solution within budget.")
        print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Q36 Interference Coding Agent")
    parser.add_argument("--file", required=True, help="Target file to modify/fix")
    parser.add_argument("--test", required=True, help="Verification command (e.g. 'pytest', 'python3 test.py')")
    parser.add_argument("--task", required=True, help="Description of the task or bug to fix")
    parser.add_argument("--backend", choices=["q36", "mock"], default="q36", help="LLM backend")
    parser.add_argument("--model", "-m", default=None, help="Name, number or path of the model to load")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable live verbose trace of search")
    parser.add_argument("--quiet", action="store_false", dest="verbose", help="Disable verbose trace")
    args = parser.parse_args()

    run_agent_task(
        target_file=args.file,
        test_cmd=args.test,
        task_instruction=args.task,
        backend_choice=args.backend,
        model_choice=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
