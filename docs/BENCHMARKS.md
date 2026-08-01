# supr-meow-tsc — measured performance

All numbers are **end-to-end** (`[prof-e2e]`), i.e. what the pool actually
receives, not loop-internal timings. Mining Qwen3-8B (the TSC mainnet model,
bf16 GGUF) unless stated. Every figure was produced against the live pool with
share verdicts checked, not in a synthetic harness.

> **Measure with `[prof-e2e]`, never `[prof]`.** The older `[prof]` line times
> only the generation loop; it once read 90 w/s while the miner was delivering
> 19. The gap was per-share proof files being written to disk and proof
> serialisation on the mining thread.

## Summary — throughput and cost per GPU

Spot prices are what these cards cost on the rental market on 2026-08-01; the
last column is the number that actually matters when you are renting.

| GPU | VRAM | best config | **w/s** | €/GPU/h | **w/s per €/h** |
|---|---|---|---|---|---|
| **RTX A6000** | 48 GB | `--slots 361 --groups 12` | 9.5 | **0.187** | **50.8** ★ |
| **RTX PRO 6000 Blackwell** | 96 GB | `--slots 512 --groups 12` | 25.6 | 0.579 | **44.2** |
| **H100 80GB HBM3** | 80 GB | `--slots 512 --groups 12` | 40.5 | 0.996 | 40.7 |
| **A100 80GB SXM** | 80 GB | `--slots 512 --groups 12` | 20.7 | 0.549 | 37.7 |
| **H200 141GB HBM3e** | 141 GB | `--slots 512 --groups 12` | 40.5 | 1.226 | 33.0 |
| **RTX 5090** | 32 GB | `--slots 128 --groups 12` | 19.5 | (owned) | — |

All single-buffered. Every config is selected **automatically** — the miner
reads compute capability and VRAM at startup and applies the matching profile
(`src/tuning.cpp`); explicit `--slots`/`--groups` always win.

**The cheapest card is the best buy.** An A6000 returns 50.8 w/s per €/h
against the H200's 33.0 — the H200 costs 6.6× more per GPU-hour for 4.3× the
throughput. Per instance: 8× A6000 ≈ 76 w/s for €1.50/h, versus 2× H200 ≈ 80
w/s for €2.45/h.


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
| 256 × 1 | 37.6–38.0 | 35.5 GB | ← stock llama ceiling |
| 256 × 2 (double) | 37.4 | 56.3 GB |
| 384 × 1 | 38.1 | 46.5 GB |
| **512 × 1 (single)** | **40.5** | **56.3 GB** | ← with the patch |

Stock llama.cpp hard-codes `#define LLAMA_MAX_SEQ 256`, and 512 slots fails
outright with `n_seq_max must be <= 256`. `tools/llama-max-seq-512.patch`
raises it; the Docker build applies it. **Verified correct: 0 rejects at 384
and 512** — this matters because the constant sizes `std::bitset<LLAMA_MAX_SEQ>`
and per-batch arrays, so a regression would corrupt transcripts rather than
crash. Gains are real but diminishing (+7.7% from 256 to 512 for +21 GB), and
768 will not fit alongside the model.

Slots scale cleanly to the cap (28.9 → 33.8 → 37.6), so the H100 is limited by
llama's 256-sequence ceiling, not by the card. **Groups barely matter** on this
box (80 cores): 12 → 37.8, 24 → 37.6, 32 → 37.5, 48 → 37.3 — a slight decline
from thread contention, so the lowest value that keeps up is best.

Double-buffering costs 21 GB more for no gain, so it is off.

## The H200 result: bandwidth stops being the limit

The H200 has 4.8 TB/s against the H100's 3.35 and 141 GB against 80. It
delivers **exactly the same 40.5 w/s**. Everything that spends the extra
memory made it worse, each measured against a *concurrent* control on the
other card of the same box:

| config (H200, 8B) | w/s |
|---|---|
| **512 slots, single-buffered** | **40.4 / 40.5 / 39.9** (three runs) |
| 512 slots, `MEOW_UBATCH=4096` | 40.9 (noise) |
| 512 × 2 double-buffered (98 GB) | 36.6 |
| 1024 slots (`LLAMA_MAX_SEQ=1024`) | 29.6 |
| 768 slots (`LLAMA_MAX_SEQ=1024`) | 24.0 |
| 512 slots on a `LLAMA_MAX_SEQ=1024` build | 35.9 |

Two conclusions worth keeping:

**Raising `LLAMA_MAX_SEQ` is not free.** The constant sizes
`std::bitset<LLAMA_MAX_SEQ>` and per-batch arrays regardless of the slots you
actually use, so a 1024 build costs **11%** even when run at 512 slots
(35.9 vs 40.4). Any gain from more slots has to beat that handicap first, and
none did.

**Above the H100 the workload is no longer memory-bandwidth-bound.** Clocks
sat pinned at 1980/1980 MHz drawing 476 W of a 700 W limit, so the card was
neither throttling nor compute-saturated — the ceiling is elsewhere (CPU
sampler tail, per-window prompt eval at ~1.07 s of a 12.6 s batch, and llama's
decode efficiency). **Practical consequence: buying more HBM bandwidth than an
H100 buys nothing for this miner.** A 5090 at 1.79 TB/s does 19.5 and scales
with bandwidth; an H200 at 4.8 does not.

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

## Two regimes, and which one a card is in decides everything

Every card falls into one of two regimes, and they behave in **opposite**
directions — which is why one global "best slots" number does not exist.

**VRAM-bound (A6000, 5090, PRO 6000 below its cap).** More slots always help,
right up to the memory ceiling, with no KV-traffic penalty:

| slots | A6000 (48 GB) | PRO 6000 (96 GB) |
|---|---|---|
| 128 | 7.8 | 18.7 |
| 192 | 8.3 | 21.6 |
| 256 | 8.8 / 8.8 | 23.9 / 23.8 |
| 320 | 9.0 | 24.3 |
| 384 | OOM | 25.5 |
| 361 | **9.5** (ceiling) | — |
| 512 | — | **25.6** (llama seq cap) |

The PRO 6000 is the cleanest demonstration: it is a 5090 with 3× the VRAM, and
unlocking slots takes it from the 5090's 18.7 to **25.6, +37%**. The 5090's
`--slots 128` was never an optimum, only a memory limit.

On the A6000, 361 is a hard ceiling — 400 slots OOMs, and trading context for
slots does not rescue it (420 and 460 at ctx 336 also OOM), because compute
buffers grow with batch alongside the KV cache.

**Bandwidth-bound (A100, H100, H200).** Slots stop helping, and past the
optimum they *hurt*, because KV read traffic grows with batch while weight
traffic is fixed. The A100 is flat from 256 to 512 (19.9 → 20.7); the H200
regresses hard above 512 (768 → 24.0, 1024 → 29.6).

**Groups barely matter in either regime** — even on a 10-CPU A6000, 5 / 8 / 12
groups gave 8.7 / 8.8 / 8.8. Use the lowest value that keeps `slots/groups`
under the GPU sort's 64-stream limit.

## Where bandwidth stops paying (measured, not modelled)

Throughput tracks memory bandwidth **up to about an H100, and not beyond**:

| GPU | bandwidth | w/s | w/s per TB/s |
|---|---|---|---|
| RTX A6000 | 0.77 TB/s | 9.5 | 12.3 |
| RTX 5090 | 1.79 TB/s | 19.5 | 10.9 |
| RTX PRO 6000 | 1.79 TB/s | 25.6 | 14.3 |
| A100 80GB | 2.04 TB/s | 20.7 | 10.1 |
| H100 80GB | 3.35 TB/s | 40.5 | 12.1 |
| **H200 141GB** | **4.8 TB/s** | **40.5** | **8.4** |

The H200 has 43% more bandwidth than the H100 and returns exactly the same
number. Buying bandwidth above an H100 buys nothing here, so **B200/B300/GB300
class parts (~8 TB/s) should not be expected to exceed ~40 w/s either.**

The shape of each card's slot curve says the same thing from the other side:
the A100 is flat from 256 to 512 slots (19.9 → 20.7, i.e. saturated almost
immediately), while the H100 climbs steeply over the identical range
(28.9 → 40.5). The A100 also ran pinned at its 400 W limit; the H200 idled at
476 W of 700 W.

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
