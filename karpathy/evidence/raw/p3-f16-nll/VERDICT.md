# P3 — prefill f16 accumulation and f16 DeltaNet state: teacher-forced NLL check

2026-09-24, audit §P3. Measurement only; **no change landed**.

## Method

`q36-bench --vulkan --gen-tokens 0 --dump-frontier-logits-dir DIR` walks the
token stream of `tests/long_context_story_prompt.txt` one frontier at a time:
each frontier teacher-forces one more token and dumps the full-vocab logits.
`p3stats.py` (this directory) computes the NLL of the *actual* next token
(`lse - logit[t]`) per frontier and the paired signed difference between two
arms, plus top-1 agreement.  `frontier_nll.py` only prints means; the paired
statistics are what decide this item, because a change that helps some
frontiers and hurts others must not average out to "no effect".

Teacher forcing is exact here: the tokens in `tokens.json` are produced by the
same tokenizer and are identical for both arms, and `--gen-tokens 0` means the
model never samples, so the two arms see the same input sequence.

Cost: one frontier = one incremental prefill of `step-incr` tokens.  The whole
sweep is 25-33 frontiers per arm, 38-380 s per arm.

## Arm A — prefill f16 accumulator (chunk 256, MMQ `f16vec2 sum[16]`) vs decode-path reference (chunk 1, f32)

| window | n | mean NLL chunk256 / chunk1 | dNLL mean | dNLL median | MAD | mean\|d\| | worse / better | top-1 same |
|---|---|---|---|---|---|---|---|---|
| ctx 512..2048 (step 64) | 25 | 8.6713 / 8.7357 | +0.0644 | -0.0000 | 0.2753 | 0.9698 | 10 / 14 | 25/25 |
| ctx 2048..4096 (step 64) | 33 | 13.1309 / 12.5390 | -0.5919 | +0.0005 | 0.4474 | 1.7924 | 17 / 15 | 32/33 |

Determinism control (chunk 256 run twice, same settings, different process):
`mean_nll=8.6713` both, `dNLL mean=med=MAD=mean|d|=0.0000`, `top1_same=25/25`.
So the whole spread above is caused by the numeric path difference, not by run
to run variation.

Reading: at both windows the **median** dNLL is 0.0000 and the sign split is
~50/50 (10/14 and 17/15).  The mean is dominated by a handful of frontiers
where the distribution is nearly flat (frontier 832 +6.53, 1408 -3.00,
1664 +2.92 at 512..2048; NLL ~13 nats at 2048..4096).  Top-1 is preserved on
25/25 and 32/33.  There is no systematic NLL penalty for accumulating the
prefill dot product in `f16vec2` over K = 17408.

Honest limitation: chunk 256 and chunk 1 are *different kernels* (MMQ +
`predequant_b16` vs `dense_iq3_xxs_decode_r4`), so this comparison does not
isolate the accumulation dtype alone; it bounds the total "different prefill
path" error, and the measured effect is zero at the median.  Isolating the
dtype would need a diagnostic MMQ build with `vec2 sum[16]` — not built,
because there is no signal to chase.

## Arm B — f16 DeltaNet recurrent state (`Q36_VK_RECURRENT_STATE_F16=0` disables)

| window | n | mean NLL f16 / f32 | dNLL mean | median | MAD | mean\|d\| | worse / better | top-1 same |
|---|---|---|---|---|---|---|---|---|
| ctx 2048..8192 (step 2048) | 4 | 16.1012 / 11.6592 | -4.4420 | -1.1968 | 1.6813 | 4.7640 | 2 / 2 | 3/4 |
| ctx 8192..16384 (step 256) | 33 | 15.0033 / 15.3388 | +0.3355 | -0.0053 | 1.3570 | 3.0220 | 14 / 19 | 30/33 |
| ctx 16384..32768 (step 16384) | 2 | 13.1553 / 11.7054 | -1.4499 | -1.4499 | 1.4502 | 1.4502 | 1 / 1 | 2/2 |

The 33-point window is the only one with a useful sample size: mean +0.34 nats
(the *f16* state arm is the lower one), median -0.005, sign split 14/19, top-1
30/33.  No systematic penalty; the scatter is large because these are long
contexts where the next-token distributions are flat (mean NLL 15 nats, i.e.
top-1 is a near-tie decision) and the two arms differ in reassociation.

## Non-finite check

Every frontier logit dump in every arm above (all 11 directories, 219 files,
each a full 248,320-logit vocab) was scanned (`nonfinite.py`): **zero** NaN or
+-inf values, largest `|logit|` 33.30.  (The per-frontier `max|dlogit|` between
arms is a different quantity and reached 28.3 at ctx 16384 on the state-dtype
arm.)

## Verdict

**No change.**  Neither the f16 prefill accumulator nor the f16 recurrent state
shows a measurable NLL penalty at the sample sizes that matter, so the audit's
conditional ("if prefill NLL is measurably worse, add a per-256-K-block f32
flush") does not fire.  `reconsider_if`: a task where the same prompt is scored
with a *paired* per-token criterion (e.g. exact-match rate over many short
completions) instead of a frontier NLL, or hardware where f16 accumulation has
a different rounding path (GFX11+ `v_dot` with f16 accumulate) — then re-run
arm A at n >= 64 frontiers and look at the sign split again.

## Reproduce

```
cd /home/server/q36
# tokens.json: the prompt's token ids, one JSON list, from tests/long_context_story_prompt.txt
./q36-bench --vulkan -m /home/server/q36/gguf/Swift-Qwen3.8-27B-IQ3_XXS.gguf \
    --prompt-file tests/long_context_story_prompt.txt \
    --ctx-start 2048 --ctx-max 4096 --step-incr 64 --ctx-alloc 4224 \
    --gen-tokens 0 --prefill-chunk 256 --dump-frontier-logits-dir /tmp/p3i
./q36-bench ... --prefill-chunk 1 --dump-frontier-logits-dir /tmp/p3j
python3 p3stats.py tokens.json /tmp/p3i /tmp/p3j --dump
Q36_VK_RECURRENT_STATE_F16=0 ./q36-bench ...   # f32 state arm
```

`run4.s` / `run5.s` in this directory are the exact sweep scripts; their output
is `run4.out` / `run5.out`.
