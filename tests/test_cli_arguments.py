#!/usr/bin/env python3
"""Fail-closed CLI regression coverage for every supported q36 entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVALID = "--definitely-invalid-flag"


@dataclass(frozen=True)
class CliCase:
    name: str
    program: tuple[str, ...]
    help_args: tuple[str, ...]
    equals_args: tuple[str, ...] | None
    missing_args: tuple[str, ...]
    typo_args: tuple[str, ...]
    foreign_args: tuple[str, ...]
    boolean_value_args: tuple[str, ...]


def run(args: tuple[str, ...], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )


def require_failure(case: str, result: subprocess.CompletedProcess[str], needle: str | None = None) -> None:
    if result.returncode == 0:
        raise AssertionError(f"{case}: unexpectedly exited 0\nstdout={result.stdout}\nstderr={result.stderr}")
    if needle and needle.lower() not in result.stderr.lower():
        raise AssertionError(
            f"{case}: stderr does not contain {needle!r}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )


def test_compiled_entrypoints() -> int:
    cases = (
        CliCase("q36", ("./q36",), ("--help",), ("--model=fake", "--help"),
                ("--model",), ("--modle",), ("--host", "127.0.0.1"),
                ("--quality=true",)),
        CliCase("q36-agent", ("./q36-agent",), ("--help",),
                ("--thinking-budget=32", "--help"), ("--thinking-budget",),
                ("--thinking-buget",), ("--host", "127.0.0.1"),
                ("--non-interactive=true",)),
        CliCase("q36-server", ("./q36-server",), ("--help",),
                ("--port=9000", "--help"), ("--port",), ("--prot",),
                ("--thinking-budget", "32"), ("--cors=true",)),
        CliCase("q36-bench", ("./q36-bench",), ("--help",),
                ("--gen-tokens=1", "--help"), ("--gen-tokens",),
                ("--gen-token",), ("--port", "9000"), ("--quality=true",)),
        CliCase("q36-eval", ("./q36-eval",), ("--help",),
                ("--questions=1", "--help"), ("--questions",),
                ("--question",), ("--cors",), ("--list-cases=true",)),
        CliCase("q36_test", ("./q36_test",), ("--help",),
                ("--model=fake", "--list"), ("--model",), ("--modle",),
                ("--host", "127.0.0.1"), ("--all=true",)),
        CliCase("q36_agent_test", ("./q36_agent_test",), ("--help",), None,
                ("--terminal-fixtures",), ("--terminal-fixturs",),
                ("--host", "127.0.0.1"), ("--terminal-driver=true",)),
        CliCase("qwen36-quantize", ("./gguf-tools/qwen36-quantize",), ("--help",),
                ("--threads=1", "--help"), ("--threads",), ("--threds",),
                ("--port", "9000"), ("--dry-run=true",)),
        CliCase("score_openrouter", ("./gguf-tools/quality-testing/score_openrouter",),
                ("--help",), ("model", "manifest", "out", "--threads=1", "--help"),
                ("model", "manifest", "out", "--threads"),
                ("model", "manifest", "out", "--threds"),
                ("model", "manifest", "out", "--cors"),
                ("model", "manifest", "out", "--quality=true")),
    )

    checks = 0
    for cli in cases:
        help_result = run(cli.program + cli.help_args)
        if help_result.returncode != 0 or "usage" not in (help_result.stdout + help_result.stderr).lower():
            raise AssertionError(f"{cli.name}: --help failed: {help_result}")
        checks += 1

        require_failure(
            f"{cli.name} invalid option after help",
            run(cli.program + cli.help_args + (INVALID,)),
        )
        checks += 1
        if cli.name in ("q36-agent", "q36-eval"):
            require_failure(
                f"{cli.name} unknown help topic",
                run(cli.program + ("--help", "definitely-not-a-topic")),
                "unknown help topic",
            )
            checks += 1

        if cli.equals_args is not None:
            equals_result = run(cli.program + cli.equals_args)
            if equals_result.returncode != 0:
                raise AssertionError(f"{cli.name}: --option=value failed: {equals_result}")
            checks += 1

        require_failure(
            f"{cli.name} missing value before another option",
            run(cli.program + cli.missing_args + ("--help",)),
        )
        checks += 1

        for label, args, needle in (
            ("unknown long", (INVALID,), "unknown"),
            ("unknown short", ("-Z",), "unknown"),
            ("typo", cli.typo_args, "unknown"),
            ("foreign option", cli.foreign_args, None),
            ("missing value", cli.missing_args, None),
            ("value on boolean", cli.boolean_value_args, None),
        ):
            result = run(cli.program + args)
            require_failure(f"{cli.name} {label}", result, needle)
            checks += 1
    return checks


def test_script_entrypoints() -> int:
    checks = 0
    direct = (
        ("test_prompt_prefix", ("./tests/test_prompt_prefix", INVALID), "unknown option"),
        ("test_quality_api", ("./tests/test_quality_api", INVALID), "unknown option"),
        ("test_sampling", ("./tests/test_sampling", INVALID), "unknown option"),
        ("test_topk8", ("./tests/test_topk8", INVALID), "unknown option"),
        ("download_model", ("./download_model.sh", INVALID), "unknown"),
        ("setup_bc250_env", ("./scripts/setup_bc250_env.sh", INVALID), "unknown option"),
        ("bench_soak", ("./scripts/bench_soak.sh", INVALID), "unknown option"),
        ("build_direction", (sys.executable, "dir-steering/tools/build_direction.py",
                             "--good-file", "good", "--bad-file", "bad", INVALID),
         "unrecognized arguments"),
        ("run_sweep", (sys.executable, "dir-steering/tools/run_sweep.py",
                       "--direction", "direction", "--prompts", "prompts", INVALID),
         "unrecognized arguments"),
        ("build_active_ledger", (sys.executable, "karpathy/tools/build_active_ledger.py",
                                 "source", "output", INVALID),
         "unrecognized arguments"),
    )
    for name, args, needle in direct:
        require_failure(name, run(args), needle)
        checks += 1

    with tempfile.NamedTemporaryFile(prefix="q36-cli-model-", dir=ROOT) as model:
        model.write(b"not a real model\n")
        model.flush()
        env = os.environ.copy()
        env["Q36_SWIFT_MODEL"] = model.name
        for script in ("run_swift.sh", "run_swift_iq3_xxs.sh", "run_swift_iq4_xs.sh"):
            require_failure(script, run((f"./{script}", INVALID), env), "unknown option")
            checks += 1

    require_failure(
        "download_model missing --token value",
        run(("./download_model.sh", "q2-imatrix", "--token")),
        "missing value",
    )
    checks += 1
    return checks


def main() -> int:
    checks = test_compiled_entrypoints() + test_script_entrypoints()
    print(f"CLI argument tests: {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
