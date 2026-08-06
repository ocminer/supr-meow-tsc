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
| **L40S** | 48 GB | `--slots 403 --groups 12` | 15.3 | 0.599 | **25.5** |
| **RTX 6000 Ada** | 48 GB | `--slots 403 --groups 12` | 12.5 | 0.319 | **39.2** |
| **H100 80GB HBM3** | 80 GB | `--slots 512 --groups 12` | 40.5 | 0.996 | 40.7 |
| **A100 80GB SXM** | 80 GB | `--slots 512 --groups 12` | 20.7 | 0.549 | 37.7 |
| **H200 141GB HBM3e** | 141 GB | `--slots 512 --groups 12` | 40.5 | 1.226 | 33.0 |
| **B200** | 183 GB | `--slots 512 --groups 12` | **56.2** | 1.873 | 30.0 |
| **RTX 5090** | 32 GB | `--slots 128 --groups 12` | 19.5 | (owned) | — |

All single-buffered. Every config is selected **automatically** — the miner
reads compute capability and VRAM at startup and applies the matching profile
(`src/tuning.cpp`); explicit `--slots`/`--groups` always win.

**The cheapest card is the best buy.** An A6000 returns 50.8 w/s per €/h
against the H200's 33.0 — the H200 costs 6.6× more per GPU-hour for 4.3× the
throughput. Per instance: 8× A6000 ≈ 76 w/s for €1.50/h, versus 2× H200 ≈ 80
w/s for €2.45/h.

**Fastest and best-value are different questions.** The B200 is the quickest
card measured — 5.8× an A6000 — and the worst buy on the list, because it also
costs 10× as much per GPU-hour. Rent B200s when you want maximum throughput
per box (~450 w/s from a single 8-GPU instance); rent A6000s when you want
throughput per euro.

### Stale shares are part of the price

Throughput alone overstates cheap cards. A window batch takes `slots / rate`
seconds and everything still in flight when the job changes is discarded, so
**stale rate scales with batch latency**:

| GPU | batch latency | stale | w/s per €/h (raw → stale-adjusted) |
|---|---|---|---|
| H200 | 512/40.5 ≈ 12 s | 1.6% | 33.0 → 32.5 |
| RTX PRO 6000 | 512/25.6 ≈ 20 s | 2.1–3.7% | 44.2 → 44.0 |
| RTX A6000 | 361/9.5 ≈ 38 s | **8.2%** | 50.8 → **46.1** |

The A6000 still wins, but by ~5% rather than ~12%. Cutting its slots to shorten
the batch does not help: 256 slots gives 8.8 w/s at ~29 s and ~6.3% stale, for
8.25 effective against 361's 8.63.

`[prof-e2e]` measures generation and structurally cannot see any of this —
cross-check the pool-side `shares: N accepted, M rejected, S stale` line before
ranking hardware.

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
outright with `n_seq_max must be <= 256`. The Docker build raises it
via the `LLAMA_MAX_SEQ` build argument. **Verified correct: 0 rejects at 384
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

## The B200 result: compiling *for* the card makes it slower

Measured 2026-08-03 on an 8× B200 box (sm_100, 183 GB, 240 cores, €14.98/h).

**56.2 w/s — the fastest card tested, and the first to break the H100/H200
plateau** (+39%). It is also the worst value on the list at 30.0 w/s per €/h.

### Do not compile sm_100 into the image

The obvious move on new silicon is to add its arch to `CUDA_ARCHS`. Here that
is a **15% regression**, and the miner is faster when the driver JIT-compiles
the sm_90 PTX forward to sm_100 instead:

| arm (all 512 slots) | per-card w/s |
|---|---|
| **JIT from sm_90 PTX** | **56.2** (55.0 / 56.0 / 56.1 / 57.9) |
| native sm_100 SASS | 49.0 (47.9 / 48.1 / 50.0 / 50.1) |

Run as a *concurrent* A/B — GPU0–3 on one image, GPU4–7 on the other, same box,
same moment — because the first comparison put the arms 40 minutes apart and
could not separate the image from drift. Four samples per arm, no overlap.
Reproduced three ways (mixed-slot sweep, all-512 run, concurrent A/B).
ggml-cuda's Blackwell-datacenter kernel selection is simply worse for this
workload than its Hopper path retargeted by `ptxas`. Re-measure before adding
sm_100 back; a future ggml release may flip it.

> **A missing arch is not a blocker.** The build embeds PTX as well as SASS, so
> any newer NVIDIA architecture runs by forward JIT. Beware the check itself:
> `cuobjdump --list-ptx` names entries `*.sm_90.ptx`, **not** `compute_90` —
> grepping for the latter reports "no PTX" on a binary full of it, which is
> exactly how an 8× B200 box was briefly written off as unusable while it was
> already mining.

### 512 slots is still the cap, despite 183 GB

The slot curve is still climbing at 512 and only 56 of 183 GB is used, so it is
tempting to raise `LLAMA_MAX_SEQ`. It loses, the same way it lost on the H200:

| config | per-card w/s |
|---|---|
| 512 slots | 47.3 |
| 768 slots (`LLAMA_MAX_SEQ=1024`) | 40.2 |

Both arms ran the native image concurrently, so the sm_100 penalty is common to
both and cancels; what is left is the slot gain minus the 11% `LLAMA_MAX_SEQ`
tax, and it is **15% net worse**. The underlying curve at 512 (192→39.9,
256→43.4, 320→45.2, 384→46.0, 448→46.4, 512→48.8) rises but is flattening.

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

## What the ~40 w/s ceiling actually is: the CPU tail

Profiling an H100 at 512 slots shows the card **idle 22% of the time**:

| phase | ms/step | share |
|---|---|---|
| decode (GPU) | ~34.9 | 67% |
| sampler tail (CPU) | 11.0 | 21% |
| decode-issue (CPU) | 6.0 | 12% |

The 5090 splits 82/13/5 — the H100 decodes so much faster that a roughly fixed
CPU cost dominates. **That, not memory bandwidth, is the ceiling**, and it
explains the H200 result: more bandwidth cannot fill a gap the CPU is causing.
Efficiency per TB/s says the same thing from the other direction — the PRO 6000
manages 14.3 w/s per TB/s against the H100's 12.1, because it is not waiting.

**Attacking the CPU tail works.** On a 176-CPU H100 box, 4 GPUs per arm:

| | mean w/s | range | sampler tail |
|---|---|---|---|
| default (`groups 12`) | 37.42 | 36.9–38.2 | 11.1–11.3 ms |
| **`MEOW_GROUPS=24` + `MEOW_UBATCH=4096`** | **39.75** | 38.8–40.5 | 8.8–9.1 ms |

**+6.2%**, ranges barely overlapping. The two settings attack different parts
of the step and compose: more sampler groups shorten the tail (11.5 → 9.2 ms
alone), while a larger ubatch raises utilisation (76% → 99% alone).

**The size of the gain tracks GPU idle, not the card.** The same settings on a
PRO 6000 — which already runs at 88–100% utilisation — gained only **+1.3%**
(25.77 → 26.10, overlapping ranges) against the H100's +6.2%. The sampler tail
shrank by the same ~1 ms on both, but that is 1 ms of a 52 ms step on the H100
versus a 77 ms step on the PRO 6000. **Check `nvidia-smi
--query-gpu=utilization.gpu` before applying any CPU-side fix**: well below
100% means there is idle to reclaim; near 100% means the card is saturated and
the tail is already hidden behind decode.

**`MEOW_UBATCH=4096` also costs VRAM.** On an RTX 6000 Ada at 403 slots — 47.8
GB of 49.1, about 1.3 GB spare — enabling it **OOMs outright**. It is safe on
the H100 and PRO 6000 only because 512 slots leave 24–40 GB free. Never enable
it on a card already near its slot ceiling.

**But the best group count depends on the HOST, not the GPU.** An 80-CPU H100
measured groups flat (12 → 37.8, 24 → 37.6, 48 → 37.3); the 176-CPU box gains
clearly from 24. So this is deliberately **not** baked into the per-GPU
profile — set `MEOW_GROUPS=24` yourself when the host has ≥128 cores.

### Loading the model multiple times does not help

A frequently suggested idea, measured and rejected on three cards. Two llama
contexts sharing **one** model load (strictly better than two copies, which
would waste 15.3 GB):

| card | two contexts | one context |
|---|---|---|
| H100 | 320×2 → 31.4, 256×2 → 30.9 | 512×1 → 37.4 |
| H200 | 512×2 → 36.6 | 512×1 → 39.9 |
| RTX 5090 | 64×2 → 12.8 | 128×1 → 19.5 |

Weights are not the bottleneck, so a second copy buys nothing and costs the KV
budget that slots actually need.

## Where bandwidth stops paying (measured, not modelled)

| GPU | bandwidth | w/s | w/s per TB/s |
|---|---|---|---|
| RTX A6000 | 0.77 TB/s | 9.5 | 12.3 |
| RTX 5090 | 1.79 TB/s | 19.5 | 10.9 |
| RTX PRO 6000 | 1.79 TB/s | 25.6 | 14.3 |
| A100 80GB | 2.04 TB/s | 20.7 | 10.1 |
| H100 80GB | 3.35 TB/s | 40.5 | 12.1 |
| **H200 141GB** | **4.8 TB/s** | **40.5** | **8.4** |
| **B200** | **~8 TB/s** | **56.2** | **7.0** |

Bandwidth stops paying **within** a generation, not across generations. The
H200 has 43% more bandwidth than the H100 and returns exactly the same number
— on Hopper, 512 slots is already past the point where more bandwidth helps,
and the binding constraint is llama's sequence cap plus the CPU sampler tail.

> **An earlier revision of this document predicted from that plateau that
> "B200/B300/GB300 class parts should not be expected to exceed ~40 w/s". The
> B200 measured 56.2 w/s — 39% above the H100 and H200.** The plateau was a
> property of Hopper at this batch size, not a ceiling of the workload. Marginal
> return per TB/s does keep falling (12.1 → 8.4 → 7.0), so bandwidth is still
> not what you are buying; Blackwell simply moved the other limits. Do not
> extrapolate a plateau across an architecture change — measure it.

The shape of each card's slot curve says the same thing from the other side:
the A100 is flat from 256 to 512 slots (19.9 → 20.7, i.e. saturated almost
immediately), while the H100 climbs steeply over the identical range
(28.9 → 40.5). The A100 also ran pinned at its 400 W limit; the H200 idled at
476 W of 700 W.

## Splitting one model across several GPUs (`--split-model`)

For cards that cannot hold the 15.3 GB model alone, one model can be spread
over several GPUs and mined as a single worker. It aggregates VRAM; it does
**not** add throughput.

| config | w/s |
|---|---|
| 1× RTX 5090, device logits | 19.5 |
| 1× RTX 5090, **host** logits | 7.4 |
| 2× RTX 5090 split | 12.3 |
| 2× RTX 5080 split, host logits | 2.7 |
| **2× RTX 5080 split, device logits** | **9.0** |

**The split scales; the logits path was the cost.** 2× 5090 split gives 12.3
against 7.4 for a single card on the same footing — the cards do add up. The
apparent disaster was that split mode fell back to llama's host logits copy,
worth **2.6×** on its own. Recovering GPU-resident logits under a split turned
2× 5080 from 2.7 into **9.0 w/s**.

Two things had to be fixed to get there, the second only visible after the
first: the sampler must bind to the card that actually **owns** the logits
(under a layer split that is wherever the last layer lives, not the first card
and not reliably the last one either — ask CUDA), and its scratch buffers must
be **dropped and reallocated** on that card, or kernels write across devices
and fault.

Sizing differs too: the sampler scratch (~0.6 GB) lands entirely on the output
card instead of being spread like model and KV, so split mode reserves 2 GiB
of the aggregate. Without that, 167 slots died on the scratch allocation while
140 ran clean.

**Slot count on a split is left conservative on purpose.** Measured on 2x RTX
5080 (16 GB each), with 138 re-measured last as a drift control and reading
identically both times:

| slots | w/s | VRAM used (of 15872 MiB/card) | headroom |
|---|---|---|---|
| **138 (auto)** | **7.2** | 13188 / 14772 | 1100 MiB |
| 150 | 7.6 | 13652 / 15236 | 636 MiB |
| 160 | 7.9 | 14026 / 15642 | **230 MiB** |
| 172 | fails to load | — | — |

160 slots is ~10% faster and leaves 230 MiB on the second card. The auto-tuner
stays at 138 anyway: an OOM here is fatal at startup, not a slowdown, and 230
MiB is inside the noise of a driver version, an attached display, or a slightly
different board. The 2 GiB reserve is a SAFETY parameter calibrated across
split shapes, and retuning it from one rig is how per-architecture tunings have
gone wrong here before.

If your rig is headless and you watch it, `--slots 150` is a reasonable opt-in
for ~+5%; `--slots 160` for ~+10% if you are willing to test it.

**Neither way of attacking the pipeline bubble works.** A layer split runs one
card at a time, so utilisation sits at 31-64% (~45%) drawing ~135 W of a ~360 W
part. Both fixes lost, on the same rig:

| config | w/s |
|---|---|
| **138 slots, layer split** | **7.2** |
| double-buffered, 69 slots x2 | 2.8 |
| `--split-rows` | ~0 (1 window vs 31) |

Double-buffering looked like the one case where it should pay — filling a real
structural idle instead of paying a second weight read to hide a small sampler
tail. It lost 2.6x: the two contexts do not overlap across devices, they
serialise, and per-window overhead doubles. Groups are flat (6 -> 7.3,
8 -> 7.3, 12 -> 7.2), so the 8-core CPU is not the constraint either, even
though the sampler shows `wait(sort+tail)=10.9 ms/step`.

**Known limits.** Cards should be identical and the aggregate must genuinely
exceed model + KV: 2× 8 GB (15.4 GB total) correctly refuses a 15.3 GB model,
matching the pool's 4×8 GB minimum. Mixed architectures additionally need a
multi-arch build, or the odd card out dies with "no kernel image is available".

**Watch the ordinals:** the miner uses CUDA ordering, which is *not*
`nvidia-smi`'s PCI order — on one test rig `nvidia-smi` showed 0,1 as the
3070s while the miner saw 0,1 as the 5080s.

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
