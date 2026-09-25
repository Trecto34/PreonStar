"""
Unit tests for Interference Search core and domains.
"""
import unittest

from interference.core import search, linear_search
from interference.domains.countdown_domain import CountdownDomain
from interference.domains.code_domain import CodeDomain, run_tests_sandbox, extract_python_code
from interference.backends.mock_backend import MockBackend


class TestInterferenceSearch(unittest.TestCase):
    def test_countdown_exact_solve(self):
        # [24, 98, 19, 3] -> 361 from Bad Theory Labs paper
        domain = CountdownDomain(max_branch=12)
        problem = ([24, 98, 19, 3], 361)
        res = search(domain, problem, budget=200, width=6, merge=True)

        self.assertTrue(res.solved)
        self.assertIsNotNone(res.solution)
        self.assertIn(361, res.solution.numbers)
        # Verify constructive interference actually merged states
        self.assertGreater(res.merged_away, 0)
        self.assertGreater(res.rounds, 0)

    def test_sandbox_runner_pass_and_fail(self):
        code = "def add(a, b): return a + b"
        tests = [
            "assert add(2, 3) == 5",
            "assert add(0, 0) == 0",
            "assert add(2, 2) == 5",  # Should fail
        ]
        results = run_tests_sandbox(code, tests)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], "pass")
        self.assertEqual(results[1][0], "pass")
        self.assertEqual(results[2][0], "fail")

    def test_sandbox_syntax_error(self):
        code = "def broken(: return 1"
        tests = ["assert 1 == 1"]
        results = run_tests_sandbox(code, tests)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "error")
        self.assertIn("SyntaxError", results[0][1])

    def test_code_domain_solve_with_mock(self):
        # Round 1: candidate fails
        # Round 2: candidate passes
        responses = [
            "```python\ndef is_even(n):\n    return False\n```",
            "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        ]
        backend = MockBackend(response_generator=lambda p: responses)
        domain = CodeDomain(backend=backend, branch_factor=2, fresh_branches=2)

        problem = {
            "prompt": "Return True if n is even, False otherwise",
            "test_list": [
                "assert is_even(2) == True",
                "assert is_even(3) == False",
            ],
        }

        res = search(domain, problem, budget=1000, width=2, merge=True)
        self.assertTrue(res.solved)
        self.assertEqual(res.solution.passed, 2)
        self.assertIn("n % 2 == 0", res.solution.src)


if __name__ == "__main__":
    unittest.main()
