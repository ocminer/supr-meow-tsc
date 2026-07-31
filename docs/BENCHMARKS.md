# supr-meow-tsc — measured performance

All numbers are **end-to-end** (`[prof-e2e]`), i.e. what the pool actually
receives, not loop-internal timings. Mining Qwen3-8B (the TSC mainnet model,
bf16 GGUF) unless stated. Every figure was produced against the live pool with
share verdicts checked, not in a synthetic harness.

> **Measure with `[prof-e2e]`, never `[prof]`.** The older `[prof]` line times
> only the generation loop; it once read 90 w/s while the miner was delivering
> 19. The gap was per-share proof files being written to disk and proof
> serialisation on the mining thread.

## Summary

| GPU | VRAM | best config | **w/s (8B)** | notes |
|---|---|---|---|---|
| **H100 80GB HBM3** | 80 GB | `--slots 256 --groups 12`, single-buffered | **~38** | llama `n_seq_max` caps slots at 256; groups flat 12–48 |
| **RTX 5090** | 32 GB | `--slots 128 --groups 12`, single-buffered | **19.5** | VRAM-limited before the seq cap |

Both are selected **automatically** — the miner reads compute capability and
VRAM at startup and applies the matching profile (`src/tuning.cpp`). Explicit
`--slots` / `--groups` always win over auto-tuning.

## Why the optimum is hardware-specific (and counter-intuitive)

8B decode streams all ~15.3 GB of weights every step, so at low batch it is
**memory-bandwidth-bound** and throughput is *sequences per weight-read*. That
single fact explains most of the table, and it points in opposite directions on
the two cards:

**RTX 5090 — double-buffering LOSES.** Two alternating contexts pay the weight
read twice to hide a ~2.4 ms CPU sampler tail, and 32 GB forces 64 slots per
context instead of 128.

| config | w/s | VRAM |
|---|---|---|
| 64 × 2 (double) | 12.8 | 25.9 GB |
| 96 × 2 (double) | 17.0 | 31.1 GB |
| **128 × 1 (single)** | **19.5** | **25.5 GB** |
| 192 × 1 (single) | 17.4 | 30.5 GB |
| 128 × 2 (double) | OOM | — |

192 slots *regresses*: KV read traffic grows with batch (~147 KB/token/seq)
while weight traffic is fixed. Lowering `--ctx` to free KV did not rescue it
(192 @ ctx336 = 17.4, identical), so it is traffic, not capacity.

**H100 80GB — double-buffering is NEUTRAL, and VRAM is not the limit at all.**
llama.cpp rejects `n_seq_max > 256`, so a single context cannot exceed 256
slots regardless of free memory (512 slots fails to initialise).

| config | w/s | VRAM |
|---|---|---|
| 128 × 1 | 28.9 | 25.5 GB |
| 192 × 1 | 33.8 | 30.6 GB |
| **256 × 1 (single)** | **37.6–38.0** | **35.5 GB** |
| 256 × 2 (double) | 37.4 | 56.3 GB |
| 512 × 1 | fails | `n_seq_max must be <= 256` |

Slots scale cleanly to the cap (28.9 → 33.8 → 37.6), so the H100 is limited by
llama's 256-sequence ceiling, not by the card. **Groups barely matter** on this
box (80 cores): 12 → 37.8, 24 → 37.6, 32 → 37.5, 48 → 37.3 — a slight decline
from thread contention, so the lowest value that keeps up is best.

Double-buffering costs 21 GB more for no gain, so it is off.

## Where the time goes (RTX 5090, 128 slots, 8B)

| phase | ms/step | share |
|---|---|---|
| decode (8B forward) | ~20.9 | **82%** |
| sample (GPU sort + CPU tail) | 3.4 | 13% |
| decode-issue (async submit) | 1.3 | 5% |

A *perfect* sampler would give +15%. The realistic floor is ~12–14 ms/step
(memory 8.5 ms; compute ~10 ms at batch 128 on a 5090), so ~1.8× remains
inside llama.cpp's decode efficiency rather than in miner code.

Notably the H100 shows a **similar ~26 ms step time** at twice the batch — it
converts its bandwidth advantage into more sequences per step rather than
faster steps.

## Model size matters more than anything else

| model | GPU | w/s |
|---|---|---|
| Qwen3-0.6B | RTX 5090 | ~51 |
| Qwen3-8B | RTX 5090 | 19.5 |
| Qwen3-8B | H100 80GB | ~38 |

A 13× larger model costs ~2.6× throughput, because decode is bandwidth-bound
rather than FLOP-bound at these batch sizes.

## Optimisation history (RTX 5090, Qwen3-0.6B testnet)

| change | w/s |
|---|---|
| baseline (CPU sampler) | 0.27 |
| fused CUB sort + stats on GPU | 1.08 |
| batched multi-stream windows | 1.84 |
| pinned D2H, flat combined-key sort, bf16 wire | 12.8 |
| AVX2 narrowing + groups tuning | 17.7 |
| survivor-only sampler tail | 19.3 |
| removed per-token log I/O + param re-parse | 19.9 |
| survivor fast-path guard fix + per-branch resize | 30.5 |
| per-group sort pipelining | 35.3 |
| dead-store removal + fused snap + groups retune | 43.5 |
| **GPU-resident logits** | **~90 (loop) / ~51 true** |

The last row is also where the measurement discipline came from: the loop-only
number was 90 while true end-to-end was ~19 until the between-batch costs were
found and removed.

## Reproducing

```bash
# auto-tuned (recommended)
supr-meow-tsc -o stratum+tcp://POOL:PORT -u WALLET.WORKER --model model.gguf

# pin explicitly to reproduce a table row
supr-meow-tsc ... --slots 128 --groups 12          # 5090
supr-meow-tsc ... --slots 256 --groups 12          # H100
MEOW_DOUBLE_BUFFER=1 supr-meow-tsc ...             # force double-buffering
```

Read `[prof-e2e]` from the log and average the last few lines once the rate has
settled (the first batch after start includes model load).
