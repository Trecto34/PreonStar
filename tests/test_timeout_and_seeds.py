"""
Unit tests proving:
1. Backend timeout/error never modifies target file or workspace, and marks branch DEAD with reason=backend_timeout.
2. Distinct seeds and sampling parameters (temp, top_p, max_tokens) reach the backend sampler.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import subprocess

from interference.core import search, StateNode
from interference.agent import FileRepairDomain
from interference.backends.mock_backend import MockBackend
from interference.backends.q36_backend import Q36Backend


class TestTimeoutAndSeeds(unittest.TestCase):
    def test_timeout_does_not_modify_workspace(self):
        """
        When the backend times out, the target file on disk MUST NOT be modified,
        no error strings must be written to disk, and the branch must be marked DEAD
        with reason=backend_timeout.
        """
        target_file = "/tmp/test_timeout_workspace.py"
        original_code = "def solve():\n    return 'original'\n"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(original_code)

        # MockBackend configured to simulate backend timeout
        backend = MockBackend(response_generator=lambda p: ["TIMEOUT"])

        domain = FileRepairDomain(
            backend=backend,
            target_file=target_file,
            test_cmd="python3 -c 'exit(1)'",
            cwd="/tmp",
            task_instruction="Fix something",
            k=2,
        )

        parent_node = StateNode(
            state_id="s0",
            depth=0,
            data=domain.start(None),
            parent_id=None,
            workspace_hash=domain.workspace_hash(domain.start(None)),
        )

        branch_records, cost = domain.propose_branches(
            parent_node=parent_node,
            problem=None,
            start_branch_idx=1,
            base_seed=40,
        )

        # 1. Target file content must remain completely unchanged
        with open(target_file, "r", encoding="utf-8") as f:
            current_content = f.read()
        self.assertEqual(current_content, original_code, "Target file was modified on backend timeout!")

        # 2. Branch records must be marked DEAD with backend_timeout
        self.assertEqual(len(branch_records), 2)
        for b in branch_records:
            self.assertTrue(b.is_dead, f"Branch {b.branch_id} should be dead on timeout")
            self.assertEqual(b.dead_reason, "backend_timeout")
            self.assertEqual(b.workspace_pre, b.workspace_post, "workspace_pre must equal workspace_post")
            self.assertEqual(b.diff_hash, "nodiff")
            self.assertEqual(b.raw_response_hash, "none00")

    def test_seeds_and_sampling_reach_backend(self):
        """
        Verify that distinct seeds (base_seed + branch_idx) and custom temperature/top_p
        are accurately propagated to the backend for every branch.
        """
        target_file = "/tmp/test_seed_propagation.py"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("def solve(): return 0\n")

        backend = MockBackend(response_generator=lambda p: ["def solve(): return 1\n"])

        domain = FileRepairDomain(
            backend=backend,
            target_file=target_file,
            test_cmd="python3 -c 'exit(0)'",
            cwd="/tmp",
            task_instruction="Fix solve",
            k=3,
            temperature=0.85,
            top_p=0.92,
            max_tokens=1024,
        )

        parent_node = StateNode(
            state_id="s0",
            depth=0,
            data=domain.start(None),
            parent_id=None,
            workspace_hash=domain.workspace_hash(domain.start(None)),
        )

        base_seed = 100
        start_branch_idx = 1
        branch_records, _ = domain.propose_branches(
            parent_node=parent_node,
            problem=None,
            start_branch_idx=start_branch_idx,
            base_seed=base_seed,
        )

        # Check recorded seeds in backend
        expected_seeds = [101, 102, 103]
        self.assertEqual(backend.recorded_seeds, expected_seeds)
        self.assertEqual(backend.recorded_temps, [0.85, 0.85, 0.85])
        self.assertEqual(backend.recorded_top_ps, [0.92, 0.92, 0.92])

        # Check branch records telemetry
        for i, b in enumerate(branch_records):
            self.assertEqual(b.seed, expected_seeds[i])
            self.assertEqual(b.temp, 0.85)
            self.assertEqual(b.top_p, 0.92)

    def test_q36_cli_command_args_and_timeout(self):
        """
        Verify that Q36Backend CLI invocations include --seed, --temp, --top-p, -n,
        and honor the configured generation_timeout.
        """
        backend = Q36Backend(
            model_path="/fake/model.gguf",
            q36_binary_path="/fake/q36",
            generation_timeout=450,
        )

        with patch("subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = "```python\ndef solve(): pass\n```"
            mock_run.return_value = mock_proc

            prompts = ["prompt1", "prompt2"]
            seeds = [77, 88]
            results, _ = backend.generate_candidates(
                prompts=prompts,
                temperature=0.6,
                top_p=0.9,
                seeds=seeds,
                max_tokens=500,
            )

            self.assertEqual(mock_run.call_count, 2)

            # First call inspection
            first_cmd = mock_run.call_args_list[0][0][0]
            first_timeout = mock_run.call_args_list[0][1]["timeout"]

            self.assertEqual(first_timeout, 450)
            self.assertIn("--seed", first_cmd)
            seed_idx = first_cmd.index("--seed")
            self.assertEqual(first_cmd[seed_idx + 1], "77")

            self.assertIn("--temp", first_cmd)
            temp_idx = first_cmd.index("--temp")
            self.assertEqual(first_cmd[temp_idx + 1], "0.6")

            self.assertIn("--top-p", first_cmd)
            topp_idx = first_cmd.index("--top-p")
            self.assertEqual(first_cmd[topp_idx + 1], "0.9")

            self.assertIn("-n", first_cmd)
            n_idx = first_cmd.index("-n")
            self.assertEqual(first_cmd[n_idx + 1], "500")

    def test_granular_checks_and_scoring(self):
        """
        Verify that granular verifier output (CHECKS: X/Y passed) is parsed into BranchRecord
        and directly translates into proportional scoring in judge().
        """
        target_file = "/tmp/test_granular_checks.py"
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("def solve(): pass\n")

        # Mock candidates (1 per prompt)
        candidates = [
            "def solve(): return 1\n",
            "def solve(): return 2\n",
            "def solve(): return 3\n",
        ]
        backend = MockBackend(response_generator=lambda p: [candidates.pop(0)])

        domain = FileRepairDomain(
            backend=backend,
            target_file=target_file,
            test_cmd="echo 'dummy'",
            cwd="/tmp",
            task_instruction="Implement solve",
            k=3,
        )

        parent_node = StateNode(
            state_id="s0",
            depth=0,
            data=domain.start(None),
            parent_id=None,
            workspace_hash="abc",
        )

        # Mock run_command to return different granular checks per candidate
        # b1: 7/10, b2: 5/10, b3: 8/10
        cmd_responses = [
            (1, "[PASS] c1\nCHECKS: 7/10 passed", ""),
            (1, "[PASS] c2\nCHECKS: 5/10 passed", ""),
            (1, "[PASS] c3\nCHECKS: 8/10 passed", ""),
        ]
        with patch("interference.agent.run_command", side_effect=cmd_responses):
            branch_records, _ = domain.propose_branches(
                parent_node=parent_node,
                problem=None,
                start_branch_idx=1,
                base_seed=40,
            )

        self.assertEqual(len(branch_records), 3)
        self.assertEqual(branch_records[0].checks, "7/10 checks")
        self.assertEqual(branch_records[1].checks, "5/10 checks")
        self.assertEqual(branch_records[2].checks, "8/10 checks")

        # Check scoring in judge
        states = [b.candidate_data for b in branch_records]
        scores, _ = domain.judge(states, None)
        self.assertEqual(scores, [69.3, 49.5, 79.2])
        # Branch 3 (8/10) must have the highest score
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
