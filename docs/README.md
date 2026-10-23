# Documentation

Start with the [README](../README.md) for what the project is and how to get it
running. Everything else lives here.

## Using it

| document | what it covers |
|---|---|
| [configuration.md](configuration.md) | every CLI, server and agent flag; KV cache options; migrating from q36 |
| [server.md](server.md) | HTTP server, OpenAI and Anthropic endpoints, SSE, tool calls, client setup |
| [agent.md](agent.md) | the native coding agent, its tools, budgets and session model |
| [models.md](models.md) | supported model families, quants, vision support, where weights come from |
| [streaming.md](streaming.md) | running models that do not fit in memory |

## Running it well

| document | what it covers |
|---|---|
| [backends.md](backends.md) | Vulkan, Metal and CPU requirements, capability detection, device policy |
| [benchmarks.md](benchmarks.md) | how to measure throughput, the A/B protocol, result tables |
| [troubleshooting.md](troubleshooting.md) | steering, debugging switches, common failure modes |
| [hardware/bc250.md](hardware/bc250.md) | AMD BC-250 reference setup: RADV, UMA split, governor, optional unlocks |

## Working on it

| document | what it covers |
|---|---|
| [architecture.md](architecture.md) | how the engine is put together and why |
| [testing.md](testing.md) | test targets and the CI gate |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | correctness and speed regression rules for contributors |
| [../QA_BEFORE_RELEASES.md](../QA_BEFORE_RELEASES.md) | the release test matrix |
| [../karpathy/CAMPAIGN.md](../karpathy/CAMPAIGN.md) | the measurement campaign: gates, noise floors, exact commands |

## Hardware support

The engine is not tied to one board. Anything with a working Vulkan 1.1 driver and
the required shader features runs; the BC-250 is the board the fast path was tuned
on and is documented as a **reference configuration**, not a requirement.

- Generic Vulkan: `make` (runtime capability detection).
- BC-250 tuned: `make vulkan-bc250` (same code, asserts the device at startup).
- Apple Silicon: `make metal`.

See [backends.md](backends.md) for the capability list and
[hardware/bc250.md](hardware/bc250.md) for the board-specific steps.
