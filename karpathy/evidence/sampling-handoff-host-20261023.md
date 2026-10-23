# Sampling handoff: host-side breakdown on MiniPlus (2026-10-23)

Model: KAT-Coder-V2.5-Dev.APEX-I-MiniPlus-V2.1.gguf, --vulkan -n 64 --ctx 512.
Two probes, both env-gated so they cost nothing in production:

- `P36_HOST_PROF=1` prints, per token, the four phases of the token end
  (`hprof` lines from core_15_forward_tokens_cpu.inc).
- `P36_VK_PROF_KERNEL=1` now also splits the record path into
  desc/cmd/book per dispatch and adds `vk_end_cb`, `vk_queue_submit`,
  `vk_wait_fence` rows.

## Steady-state token (median of 18 decode steps)

    caller_gap 0.57 ms   argmax + emit + loop
    forward   16.8  ms   record 1317 dispatches + eager submits + ring-slot waits
    last_h     0.004 ms
    top2       0.002 ms
    read       2.50  ms   fence wait for the token's queue + 8-byte D2H
    ---------------------------------------------------------------
    total     19.9  ms   (measured 20.7 ms at 48.3 t/s)

Inside `forward`, per dispatch (P36_VK_PROF_KERNEL, includes its own overhead):

    desc  2.67 us   kernel_init, 6 descriptor buffer infos, vkAllocateDescriptorSets,
                    vkUpdateDescriptorSets
    cmd   0.84 us   begin_batch, hazard barrier, binds, push constants, vkCmdDispatch
    book  0.06 us   hazard sets, last_use_seq, write_gen, op counter

    => ~5.3 ms of the 16.8 ms is recording; the other ~11.5 ms is
       vkQueueSubmit + vk_wait_fence backpressure inside eager flushes
       (submit_eager: 406 flushes per profile window).

## GPU side of the same window

    span 176.2 ms, dispatch time 147.5 ms, idle 28.7 ms (16.3%)

    idle 12.1 ms in 15 boundary gaps top2 -> copy_rows (1.4-2.0 ms each, 1.52 ms median)
    idle 16.6 ms spread over 10144 transitions = 1.6 us per dispatch launch bubble
    per token: 18.4 ms GPU work + 3.6 ms idle

## What this rules out

- **Sampler**: 0.57 ms/token for argmax, emit and loop overhead. Not the stall.
- **Recording**: 5.3 ms of CPU, fully overlapped with GPU execution — the CPU
  runs ahead because recording is cheaper than the GPU work it feeds.
- **Queue submit**: vk_end_cb 0.085 ms and vk_queue_submit 0.83 ms per token.
- **Ring depth**: P36_VK_CB_RING 4 -> 8 made it *worse* (idle 28.7 -> 31.6 ms,
  boundary gaps 12.1 -> 14.9 ms) and was reverted. The GPU is not starved by
  a shallow command-buffer ring.

## What it leaves

The token end costs ~1.5 ms of GPU idle after `top2` finishes: the next kernel
(copy_rows, the embedding row fetch) cannot be recorded until the sampled id
exists. Of that, 0.86 ms is host work inside the next forward before its first
dispatch and the rest is the fence-to-submit round trip. Everything else on the
GPU side is 1.6 us launch bubbles across 1317 dispatches/token — a dispatch
count problem, not a host problem.
