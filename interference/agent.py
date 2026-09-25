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

    def workspace_hash(self, state: Dict[str, Any]) -> str:
        from interference.core import short_hash
        return short_hash(state["content"])

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

    def propose_branches(
        self,
        parent_node,
        problem: Any,
        start_branch_idx: int,
        base_seed: int,
    ) -> Tuple[List[Any], float]:
        import difflib
        from interference.core import BranchRecord, short_hash

        parent_state = parent_node.data
        prompt = self._make_prompt(parent_state)
        p_hash = short_hash(prompt)
        ws_pre = self.workspace_hash(parent_state)

        # Prepare branch seeds
        prompts = [prompt] * self.k
        raw_responses, token_cost = self.backend.generate_candidates(prompts, max_tokens=2048)

        branch_records = []
        for i, resp in enumerate(raw_responses):
            b_num = start_branch_idx + i
            b_id = f"b{b_num}"
            seed = base_seed + b_num

            code = extract_code_block(resp)
            if not code:
                # Empty response -> Dead
                branch_records.append(BranchRecord(
                    branch_id=b_id,
                    parent_id=parent_node.state_id,
                    depth=parent_node.depth + 1,
                    seed=seed,
                    prompt_hash=p_hash,
                    workspace_pre=ws_pre,
                    workspace_post=ws_pre,
                    diff_hash="empty0",
                    verifier_result="EMPTY_RESPONSE",
                    merge_key=("empty",),
                    is_dead=True,
                    is_goal=False,
                    candidate_data=parent_state,
                    rank=i,
                ))
                continue

            # Compute Unified Diff
            diff_lines = list(difflib.unified_diff(
                parent_state["content"].splitlines(keepends=True),
                code.splitlines(keepends=True),
                fromfile=f"a/{os.path.basename(self.target_file)}",
                tofile=f"b/{os.path.basename(self.target_file)}",
            ))
            diff_text = "".join(diff_lines)
            d_hash = short_hash(diff_text if diff_text else "nodiff")

            # Apply candidate to disk and verify workspace isolation
            with open(self.target_file, "w", encoding="utf-8") as f:
                f.write(code)

            ws_post = short_hash(code)
            ret, stdout, stderr = run_command(self.test_cmd, self.cwd)
            combined_out = (stdout + "\n" + stderr).strip()

            # Restore original file immediately to ensure strict branch isolation
            with open(self.target_file, "w", encoding="utf-8") as f:
                f.write(self.original_content)

            is_g = (ret == 0)
            is_d = (ret != 0 and ("SyntaxError" in combined_out or "IndentationError" in combined_out))
            v_res = "tests PASS" if is_g else f"tests FAIL (exit={ret})"

            verifier_fingerprint = short_hash(f"{ret}:{combined_out}")
            m_key = f"{ws_post}:{verifier_fingerprint}"

            candidate_state = {
                "content": code,
                "returncode": ret,
                "output": combined_out,
                "passed": is_g,
            }

            branch_records.append(BranchRecord(
                branch_id=b_id,
                parent_id=parent_node.state_id,
                depth=parent_node.depth + 1,
                seed=seed,
                prompt_hash=p_hash,
                workspace_pre=ws_pre,
                workspace_post=ws_post,
                diff_hash=d_hash,
                verifier_result=v_res,
                merge_key=m_key,
                is_dead=is_d,
                is_goal=is_g,
                candidate_data=candidate_state,
                rank=i,
            ))

        return branch_records, float(token_cost)

    def propose(self, states: List[Dict[str, Any]], problem: Any) -> Tuple[List[List[Dict[str, Any]]], float]:
        # Fallback if called directly
        return [], 0.0

    def key(self, state: Dict[str, Any]) -> Any:
        from interference.core import short_hash
        ws_hash = self.workspace_hash(state)
        verifier_fingerprint = short_hash(f"{state['returncode']}:{state['output']}")
        return f"{ws_hash}:{verifier_fingerprint}"

    def judge(self, states: List[Dict[str, Any]], problem: Any) -> Tuple[Optional[List[float]], float]:
        scores = []
        for s in states:
            if s["passed"]:
                scores.append(100.0)
            else:
                err_lines = len(s["output"].splitlines())
                scores.append(max(1.0, 50.0 - err_lines))
        return scores, 0.0

    def is_goal(self, state: Dict[str, Any], problem: Any) -> bool:
        return state["passed"]

    def is_dead(self, state: Dict[str, Any], problem: Any) -> bool:
        return ("SyntaxError" in state["output"] or "IndentationError" in state["output"])


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
