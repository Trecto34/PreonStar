"""
Code Synthesis & Repair Domain for Interference Search.

States represent generated code plus execution behavior on test cases.
Two candidate programs with identical test behaviors (same passes, fails, and outputs)
share the same merge key and collapse into a single state, pooling their parent path weights!
"""
import ast
from dataclasses import dataclass
import json
import re
import signal
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from interference.domains.base import Domain
from interference.backends.base import LLMBackend


SANDBOX_RUNNER = r'''
import json, sys, signal, ast

src, tests = json.loads(sys.stdin.read())
out = []

def handler(*a):
    raise TimeoutError("timeout")

signal.signal(signal.SIGALRM, handler)
g = {}

try:
    signal.alarm(3)
    exec(src, g)
    signal.alarm(0)
except BaseException as e:
    err_msg = f"{type(e).__name__}: {str(e)[:120]}"
    print(json.dumps([["error", err_msg]] * len(tests)))
    sys.exit(0)

for t in tests:
    try:
        node = ast.parse(t).body[0]
        signal.alarm(2)
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 \
                and isinstance(node.test.ops[0], ast.Eq):
            left = eval(compile(ast.Expression(node.test.left), "<t>", "eval"), g)
            right = eval(compile(ast.Expression(node.test.comparators[0]), "<t>", "eval"), g)
            ok = (left == right)
            out.append(["pass" if ok else "fail", repr(left)[:120]])
        else:
            exec(t, g)
            out.append(["pass", ""])
        signal.alarm(0)
    except AssertionError:
        signal.alarm(0)
        out.append(["fail", "assertion"])
    except BaseException as e:
        signal.alarm(0)
        out.append(["error", f"{type(e).__name__}: {str(e)[:120]}"])

print(json.dumps(out))
'''


def run_tests_sandbox(code: str, tests: List[str], timeout: int = 5) -> List[Tuple[str, str]]:
    """Runs code against unit tests in a separate process with alarms."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", SANDBOX_RUNNER],
            input=json.dumps([code, tests]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        lines = proc.stdout.strip().splitlines()
        if not lines:
            return [("error", proc.stderr.strip()[:120] or "no output")] * len(tests)
        return [tuple(x) for x in json.loads(lines[-1])]
    except subprocess.TimeoutExpired:
        return [("error", "timeout")] * len(tests)
    except Exception as e:
        return [("error", f"runner: {type(e).__name__}")] * len(tests)


def extract_python_code(text: str) -> str:
    """Extracts python code blocks or function definitions from LLM output."""
    # Strip thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[0].strip()
    match = re.search(r"(def\s+.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class CodeState:
    src: Optional[str] = None
    behaviour: Optional[List[Tuple[str, str]]] = None
    passed: int = 0

    def __post_init__(self):
        if self.behaviour:
            self.passed = sum(1 for status, _ in self.behaviour if status == "pass")


class CodeDomain(Domain):
    def __init__(
        self,
        backend: LLMBackend,
        branch_factor: int = 3,
        fresh_branches: int = 4,
        max_tokens: int = 384,
        temperature: float = 0.7,
    ):
        self.backend = backend
        self.k = branch_factor
        self.fresh_k = fresh_branches
        self.max_tokens = max_tokens
        self.temperature = temperature

    def start(self, problem: Dict[str, Any]) -> CodeState:
        return CodeState(src=None, behaviour=None)

    def _fresh_prompt(self, problem: Dict[str, Any]) -> str:
        prompt_text = problem.get("prompt", "")
        tests = problem.get("test_list", [])
        first_test = tests[0] if tests else ""
        return (
            f"Write a Python function to solve this problem:\n"
            f"{prompt_text}\n"
            f"Example test that must pass:\n"
            f"{first_test}\n"
            f"Reply with ONLY valid Python code inside a single ```python block."
        )

    def _revise_prompt(self, problem: Dict[str, Any], state: CodeState) -> str:
        prompt_text = problem.get("prompt", "")
        tests = problem.get("test_list", [])
        feedback_lines = []
        if state.behaviour:
            for test, (status, detail) in zip(tests, state.behaviour):
                if status == "pass":
                    feedback_lines.append(f"PASS: {test}")
                elif status == "fail":
                    feedback_lines.append(f"FAIL: {test} (returned {detail})")
                else:
                    feedback_lines.append(f"ERROR: {test} ({detail})")
        feedback = "\n".join(feedback_lines)

        return (
            f"Fix the Python code for the problem below:\n{prompt_text}\n\n"
            f"Current code:\n```python\n{state.src}\n```\n\n"
            f"Test execution results:\n{feedback}\n\n"
            f"Correct the code so ALL tests pass. Reply with ONLY the corrected code inside a ```python block."
        )

    def propose(self, states: List[CodeState], problem: Dict[str, Any]) -> Tuple[List[List[CodeState]], float]:
        prompts = []
        state_map = []
        tests = problem.get("test_list", [])

        for i, s in enumerate(states):
            count = self.fresh_k if s.src is None else self.k
            p = self._fresh_prompt(problem) if s.src is None else self._revise_prompt(problem, s)
            for _ in range(count):
                prompts.append(p)
                state_map.append(i)

        raw_texts, tokens_cost = self.backend.generate_candidates(
            prompts,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        child_groups: List[List[CodeState]] = [[] for _ in range(len(states))]
        for cand, state_idx in zip(raw_texts, state_map):
            if hasattr(cand, "is_error") and cand.is_error:
                continue
            text = cand.text if hasattr(cand, "text") else cand
            if not text:
                continue
            code = extract_python_code(text)
            behaviour = run_tests_sandbox(code, tests)
            child_groups[state_idx].append(CodeState(src=code, behaviour=behaviour))

        return child_groups, float(tokens_cost)

    def key(self, state: CodeState) -> Any:
        """Hashable behavioral signature."""
        if state.behaviour is None:
            return None
        return tuple(state.behaviour)

    def judge(self, states: List[CodeState], problem: Dict[str, Any]) -> Tuple[Optional[List[float]], float]:
        """Judges by the number of passed tests."""
        scores = [float(s.passed) for s in states]
        return scores, 0.0

    def is_goal(self, state: CodeState, problem: Dict[str, Any]) -> bool:
        tests = problem.get("test_list", [])
        return state.behaviour is not None and state.passed == len(tests) and len(tests) > 0

    def is_dead(self, state: CodeState, problem: Dict[str, Any]) -> bool:
        if state.behaviour is None:
            return False
        # If all tests threw syntax/load errors, mark dead
        all_errors = all(status == "error" and "SyntaxError" in detail for status, detail in state.behaviour)
        return all_errors
