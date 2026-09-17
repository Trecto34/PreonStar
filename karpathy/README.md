# Karpathy loops for the Swift Qwen3.8 target

This is the repo-local bilevel autoresearch setup for the BC-250 checkout.

- **Inner loop:** one isolated experiment worktree and one candidate change.
- **Outer loop:** serialized preflight, inner-loop execution, postflight, and
  periodic process audits.
- **Compatibility gate:** every candidate checks Swift IQ3_XXS and the local
  Qwen3.5/Qwen3.6 MoE reference (`qwen35moe`) in separate GPU processes.

The authoritative loop logic, persistent state machine, and experiment branch
manager remain in `~/Karpathy`. The scripts here are target wrappers plus the
Swift/Qwen compatibility gate and target guidance.

## Commands

```sh
# One isolated inner experiment (shared Karpathy logic)
./karpathy/run_inner.sh

# Infinite outer campaign, owned by ~/Karpathy (Ctrl-C stops its process group)
./karpathy/run_outer.sh

# Compatibility-only check
./karpathy/compat_gate.sh
```

The shared orchestrator never starts concurrent `q36`, `q36-bench`, or
`q36_test` jobs. The default context and generation lengths are deliberately
short enough for this 16 GB UMA machine. Override target/model paths with
`KARPATHY_TARGET_DIR`, `KARPATHY_SWIFT_MODEL`, and `KARPATHY_QWEN35_MODEL`.
Ctrl-C is handled by the shared orchestrator, which terminates the active gate,
agent, and child process group before returning.
The meta-agent provider watchdog defaults to 180 seconds per provider and can
be shortened with `KARPATHY_META_TIMEOUT=120`.
If a persisted cadence lands on meta before you want it, use
`KARPATHY_SKIP_META=1 ./karpathy/run_outer.sh` for that invocation.

## DeepSeek V4.1 Flash for both loops

The repository contains a project-local OpenCode provider named
`deepseek-direct`. DeepSeek's current API model name is `deepseek-flash`; the
provider label in the config identifies it as V4.1 Flash. The key is never
stored in this repository. The outer loop validates the key before launching
the compatibility gate, so a missing credential cannot start GPU work.

For a shell session:

```sh
read -rsp 'DeepSeek API key: ' DEEPSEEK_API_KEY; echo
export DEEPSEEK_API_KEY KARPATHY_PROVIDER=deepseek
./karpathy/run_outer.sh
```

For a protected key file, set `DEEPSEEK_API_KEY_FILE` instead of exporting the
key directly. Keep that file owner-readable only (`chmod 600`):

```sh
export DEEPSEEK_API_KEY_FILE="$HOME/.config/deepseek/api-key"
export KARPATHY_PROVIDER=deepseek
./karpathy/run_outer.sh
```

Both the inner worker and outer meta audit then use
`deepseek-direct/deepseek-flash`. DeepSeek mode is provider-exclusive by
default, so an API failure stops the pass rather than silently spending quota
with another provider. If needed, explicitly opt into the old fallback ladder
with `KARPATHY_DEEPSEEK_FALLBACK=1` and `INNER_ALLOW_FALLBACK=1`. Use
`KARPATHY_DEEPSEEK_MODEL` to override the model route, or
`KARPATHY_USE_DEEPSEEK=1` as an equivalent switch.
