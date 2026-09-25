"""
Unit tests proving exact merge key logic:
- Same workspace + same verifier => MERGE
- Different workspace + same verifier (PASS) => DO NOT MERGE (stay separate)
"""
import unittest

from interference.core import search, StateNode, BranchRecord
from interference.agent import FileRepairDomain
from interference.backends.mock_backend import MockBackend


class TestMergeLogic(unittest.TestCase):
    def test_different_workspace_same_pass_no_merge(self):
        """
        Two branches produce different code/workspaces, but both pass tests.
        They MUST NOT be merged, remaining separate candidate states!
        """
        target_file = "/tmp/test_no_merge.py"
        with open(target_file, "w") as f:
            f.write("def solve(): return 0\n")

        # Branch 1: Implementation A (returns 42) -> PASS
        # Branch 2: Implementation B (returns 21 * 2) -> PASS
        candidates = [
            "def solve():\n    return 42\n",
            "def solve():\n    val = 21\n    return val * 2\n",
        ]
        backend = MockBackend(response_generator=lambda p: [candidates.pop(0) if candidates else "pass"])

        domain = FileRepairDomain(
            backend=backend,
            target_file=target_file,
            test_cmd="python3 -c \"import sys; sys.path.insert(0, '/tmp'); from test_no_merge import solve; assert solve() == 42\"",
            cwd="/tmp",
            task_instruction="Make solve return 42",
            k=2,
        )

        res = search(domain, None, budget=1000, width=5, merge=True, verbose=False)

        # Since workspaces are different, merged_away must be 0!
        self.assertEqual(res.merged_away, 0, "Branches with different workspaces must NOT merge!")
        self.assertTrue(res.solved)

    def test_same_workspace_same_verifier_merges(self):
        """
        Two branches produce the exact same workspace code and pass tests.
        They MUST MERGE into a single state pool!
        """
        target_file = "/tmp/test_yes_merge.py"
        with open(target_file, "w") as f:
            f.write("def solve(): return 0\n")

        # Both branches produce the EXACT SAME final code
        identical_code = "def solve():\n    return 42\n"
        candidates = [
            identical_code,
            identical_code,
        ]
        backend = MockBackend(response_generator=lambda p: [candidates.pop(0) if candidates else "pass"])

        domain = FileRepairDomain(
            backend=backend,
            target_file=target_file,
            test_cmd="python3 -c \"import sys; sys.path.insert(0, '/tmp'); from test_yes_merge import solve; assert solve() == 42\"",
            cwd="/tmp",
            task_instruction="Make solve return 42",
            k=2,
        )

        res = search(domain, None, budget=1000, width=5, merge=True, verbose=False)

        # Because both arrived at the identical workspace + same PASS verifier, one must be merged away!
        self.assertEqual(res.merged_away, 1, "Branches with identical workspace and verifier MUST merge!")
        self.assertTrue(res.solved)


if __name__ == "__main__":
    unittest.main()
