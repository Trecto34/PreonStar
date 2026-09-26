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
    # Handle unclosed code block (e.g. if truncated before closing backticks)
    unclosed = re.search(r"```(?:\w+)?\n(.*)$", text, re.DOTALL)
    if unclosed:
        return unclosed.group(1).strip()
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
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 2048,
    ):
        self.backend = backend
        self.target_file = target_file
        self.test_cmd = test_cmd
        self.cwd = cwd
        self.task_instruction = task_instruction
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # Cache original content to restore on cancellation
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            self.original_content = f.read()

    def start(self, problem: Any) -> Dict[str, Any]:
        ret, stdout, stderr = run_command(self.test_cmd, self.cwd)
        combined = (stdout + "\n" + stderr).strip()
        checks_match = re.search(r"CHECKS:\s*(\d+)/(\d+)", combined)
        checks_passed = int(checks_match.group(1)) if checks_match else None
        checks_total = int(checks_match.group(2)) if checks_match else None
        return {
            "content": self.original_content,
            "returncode": ret,
            "output": combined,
            "passed": (ret == 0),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
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

        # Bug 2: Generate distinct seeds for each branch and pass to backend
        branch_seeds = [base_seed + start_branch_idx + i for i in range(self.k)]
        prompts = [prompt] * self.k
        raw_responses, token_cost = self.backend.generate_candidates(
            prompts,
            temperature=self.temperature,
            top_p=self.top_p,
            seeds=branch_seeds,
            max_tokens=self.max_tokens,
        )

        branch_records = []
        for i, resp in enumerate(raw_responses):
            b_num = start_branch_idx + i
            b_id = f"b{b_num}"
            seed = branch_seeds[i]

            # Bug 1: Backend timeout or error must NEVER become code or modify workspace
            if resp.is_error:
                branch_records.append(BranchRecord(
                    branch_id=b_id,
                    parent_id=parent_node.state_id,
                    depth=parent_node.depth + 1,
                    seed=seed,
                    temp=self.temperature,
                    top_p=self.top_p,
                    prompt_hash=p_hash,
                    raw_response_hash=resp.raw_hash,
                    workspace_pre=ws_pre,
                    workspace_post=ws_pre,
                    diff_hash="nodiff",
                    verifier_result=f"DEAD reason={resp.error_reason}",
                    dead_reason=resp.error_reason,
                    merge_key=("dead", resp.error_reason, b_id),
                    is_dead=True,
                    is_goal=False,
                    candidate_data=parent_state,
                    rank=i,
                ))
                continue

            # Bug 4: Log raw_response_hash before extracting code
            raw_hash = resp.raw_hash
            raw_text = resp.text or ""
            code = extract_code_block(raw_text)
            if not code:
                # Empty response -> Dead
                branch_records.append(BranchRecord(
                    branch_id=b_id,
                    parent_id=parent_node.state_id,
                    depth=parent_node.depth + 1,
                    seed=seed,
                    temp=self.temperature,
                    top_p=self.top_p,
                    prompt_hash=p_hash,
                    raw_response_hash=raw_hash,
                    workspace_pre=ws_pre,
                    workspace_post=ws_pre,
                    diff_hash="nodiff",
                    verifier_result="DEAD reason=empty_response",
                    dead_reason="empty_response",
                    merge_key=("empty", b_id),
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

            # Apply candidate to disk and verify under try-finally for workspace isolation
            try:
                with open(self.target_file, "w", encoding="utf-8") as f:
                    f.write(code)

                ws_post = short_hash(code)
                ret, stdout, stderr = run_command(self.test_cmd, self.cwd)
                combined_out = (stdout + "\n" + stderr).strip()
            finally:
                # Restore original file immediately to ensure strict branch isolation
                with open(self.target_file, "w", encoding="utf-8") as f:
                    f.write(self.original_content)

            is_g = (ret == 0)
            is_d = (ret != 0 and ("SyntaxError" in combined_out or "IndentationError" in combined_out))
            v_res = "tests PASS" if is_g else f"tests FAIL (exit={ret})"

            checks_match = re.search(r"CHECKS:\s*(\d+)/(\d+)", combined_out)
            checks_passed = int(checks_match.group(1)) if checks_match else None
            checks_total = int(checks_match.group(2)) if checks_match else None
            checks_str = f"{checks_passed}/{checks_total} checks" if (checks_passed is not None and checks_total is not None) else None

            verifier_fingerprint = short_hash(f"{ret}:{combined_out}")
            m_key = f"{ws_post}:{verifier_fingerprint}"

            candidate_state = {
                "content": code,
                "returncode": ret,
                "output": combined_out,
                "passed": is_g,
                "checks_passed": checks_passed,
                "checks_total": checks_total,
            }

            branch_records.append(BranchRecord(
                branch_id=b_id,
                parent_id=parent_node.state_id,
                depth=parent_node.depth + 1,
                seed=seed,
                temp=self.temperature,
                top_p=self.top_p,
                prompt_hash=p_hash,
                raw_response_hash=raw_hash,
                workspace_pre=ws_pre,
                workspace_post=ws_post,
                diff_hash=d_hash,
                verifier_result=v_res,
                dead_reason="syntax_error" if is_d else None,
                checks=checks_str,
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
            if s.get("passed", False):
                scores.append(100.0)
            else:
                passed = s.get("checks_passed")
                total = s.get("checks_total")
                if passed is not None and total and total > 0:
                    # Granular score based on percentage of checks passed, scaled to 0-99
                    scores.append(round((passed / total) * 99.0, 2))
                else:
                    err_lines = len(s.get("output", "").splitlines())
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
    generation_timeout: int = 600,
    temperature: float = 0.7,
    top_p: float = 0.8,
    max_tokens: int = 2048,
):
    print("=" * 65)
    print(" Q36 INTERFERENCE SEARCH CODING AGENT")
    print("=" * 65)
    print(f"Target File : {target_file}")
    print(f"Verify Cmd  : {test_cmd}")
    print(f"Task        : {task_instruction}")
    print(f"Backend     : {backend_choice}")

    if backend_choice == "q36":
        backend = Q36Backend(
            model_path=model_choice,
            generation_timeout=generation_timeout,
        )
        print(f"Model       : {os.path.basename(backend.model_path)}")
        print(f"Gen Timeout : {generation_timeout}s")
    else:
        # Simple test mock
        backend = MockBackend(lambda p: ["```python\ndef solve(): return True\n```"])
        print("Model       : Mock")

    print(f"Sampling    : temp={temperature} | top_p={top_p} | max_tokens={max_tokens}")
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
        k=width,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    initial_state = domain.start(None)
    if initial_state["passed"]:
        print("\nVerification command already passes! Nothing to fix.")
        return

    print(f"\nInitial State: FAILING (Exit code: {initial_state['returncode']})")
    print(f"Output snippet:\n{initial_state['output'][:300]}\n")
    print("Starting multi-branch search with constructive merging...")

    from interference.core import search
    res = search(domain, problem=None, budget=10000, width=width, merge=True, verbose=verbose, max_rounds=max_rounds)

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
    parser.add_argument("--generation-timeout", type=int, default=600, help="Timeout in seconds for backend generation per candidate")
    parser.add_argument("--temp", "--temperature", type=float, default=0.7, dest="temperature", help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, dest="top_p", help="Sampling top-p")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per candidate generation")
    parser.add_argument("--width", "-w", type=int, default=3, help="Search beam width / number of branches")
    parser.add_argument("--max-rounds", "-r", type=int, default=5, help="Maximum search rounds")
    args = parser.parse_args()

    run_agent_task(
        target_file=args.file,
        test_cmd=args.test,
        task_instruction=args.task,
        backend_choice=args.backend,
        model_choice=args.model,
        max_rounds=args.max_rounds,
        width=args.width,
        verbose=args.verbose,
        generation_timeout=args.generation_timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
