# Hardware compatibility

Short version: **you need an NVIDIA GPU of the Ampere generation or newer, and
at least 24 GB of VRAM on one card — or several smaller cards that together
have more than ~20 GB.**

Two independent requirements, and both are hard:

1. **bf16.** The chain fixes the compute precision at bf16 and the verifier
   checks it. Cards without bf16 hardware cannot produce a valid proof at all —
   this is not a speed problem, it is a "will never work" problem.
2. **The model must fit.** Qwen3-8B in bf16 is **15.3 GiB** and has to be in
   VRAM, plus roughly 4 GB of compute buffers and ~57 MB per concurrent slot.

---

## Will my card work?

| generation | examples | bf16 | verdict |
|---|---|---|---|
| Blackwell (sm_120, sm_100) | RTX 5090, 5080, RTX PRO 6000, B200 | yes | **works** |
| Ada (sm_89) | RTX 4090, 4080, RTX 6000 Ada, L40S | yes | **works** |
| Hopper (sm_90) | H100, H200 | yes | **works** |
| Ampere (sm_80, sm_86) | A100, A6000, A40, RTX 3090/3080/3070 | yes | **works if VRAM suffices** |
| **Turing (sm_75)** | **RTX 2080 Ti, 2070, Titan RTX, T4** | **no** | **NEVER works — see below** |
| Pascal and older | GTX 1080 Ti, P40, Tesla P100 | no | never works |
| AMD, Intel, Apple | any | — | not supported |

### Turing (RTX 20-series) — read this before trying

A 2080 Ti has 11 GB and no bf16. The miner will **start** on it, because the
release packages still contain Turing GPU code, and it may even appear to run —
but every proof it produces is invalid and the pool will reject all of it. There
is no configuration, no flag and no future version that changes this: the
precision is fixed by the chain, not by us.

If you have Turing cards, they cannot mine TSC. That is the whole answer.

---

## VRAM: how much do you actually need?

Per GPU, to hold the model alone:

```
15.3 GiB  model (bf16 Qwen3-8B)
 ~4   GiB  compute buffers
 ~57  MB   per concurrent slot (at the default 384 context)
```

| card VRAM | single card | notes |
|---|---|---|
| 32 GB+ | **yes** | comfortable; 5090 runs 128 slots |
| 24 GB | **yes** | the practical minimum — 3090, 4090, A5000 |
| 20 GB | marginal | fits, but very few slots and low throughput |
| 16 GB | **no** | model + buffers do not fit — use `--split-model` |
| 8–12 GB | **no** | single card impossible — see splitting |

**A note on the most common question:** an RTX 3070 / 3070 Ti (8 GB) or 3080
(10–12 GB) **cannot mine on its own.** The card is fine — Ampere has bf16 —
but the model does not fit. You need either a 24 GB card, or several of these
cards split together.

---

## Splitting one model across several GPUs

`--split-model` spreads a single model over several cards and mines them as one
worker. It **aggregates VRAM; it does not add throughput** — two cards are not
twice as fast, they just let the model fit at all.

```bash
supr-meow-tsc -o stratum+tcp://pool:3310 -u <address> \
              -d 0,1 --split-model --model Qwen3-8B-9c925d64-bf16.gguf
```

| configuration | total VRAM | works? |
|---|---|---|
| 2 × 16 GB (e.g. 2× RTX 5080) | 32 GB | **yes** — measured |
| 4 × 8 GB (e.g. 4× RTX 3070) | 32 GB | **yes** |
| 2 × 12 GB | 24 GB | yes, tight |
| **2 × 8 GB** | **16 GB** | **no** — refuses to start, correctly |

The 2 × 8 GB case is rejected on purpose: 16 GB total minus the model's 15.3 GiB
leaves nothing for the KV cache, so it could never mine. The miner tells you
instead of failing later.

Cards in a split should be the **same model**. Mixing generations works only if
your build contains code for both, and the slowest card sets the pace.

---

## Measured throughput

Every number below is end-to-end against a live pool, not a synthetic
benchmark. Full curves and the reasoning are in [BENCHMARKS.md](BENCHMARKS.md).

| GPU | VRAM | w/s |
|---|---|---|
| B200 | 183 GB | 56.2 |
| H100 / H200 | 80 / 141 GB | 40.5 |
| RTX PRO 6000 Blackwell | 96 GB | 25.6 |
| A100 | 80 GB | 20.7 |
| RTX 5090 | 32 GB | ~18.6 |
| RTX 6000 Ada | 48 GB | 12.5 |
| RTX A6000 / A40 | 48 GB | 9.5 |
| 2 × RTX 5080 (split) | 2 × 16 GB | 7.5 |

Cards not listed have no measured profile yet — the miner sizes them from VRAM
automatically and they work fine, you just do not get a hand-tuned config.

**Rates are PoI/s (proofs per second), not hashes.** Single or double digits is
normal and correct for proof-of-inference.

---

## Driver and OS

| requirement | version |
|---|---|
| NVIDIA driver | **R570 or newer** |
| CUDA toolkit | not needed — the runtime ships in the package |
| glibc | 2.35+ (Ubuntu 22.04 and newer) |
| architecture | x86_64 |

The packages bundle everything except `libcuda.so.1` and `libnvidia-ml.so.1`,
which must come from your installed driver.

---

## Mining platforms

| platform | package |
|---|---|
| HiveOS | `supr-meow-tsc-<v>-hiveos.tar.gz` — use as the Installation URL |
| MMPOS | `supr-meow-tsc-<v>-mmpos.tar.gz` |
| SimpleMining (SMOS) | `supr-meow-tsc-<v>-smos.tar.gz` |
| Any Linux rig | `supr-meow-tsc-<v>-linux-x86_64.tar.gz` |
| Docker / cloud (Vast, Clore, OctaSpace) | `ocminersupr/supr-meow-tsc:latest` |

Windows is not supported.

---

## Disk and network

The model is **16.4 GB** and is downloaded once, then cached. It is not
included in any package. Each rig needs:

- ~20 GB free disk for the model and the miner
- a working internet connection for the one-time download (fetched over 8
  parallel ranges; typically a few minutes)

The download is checksum-verified. If the hash does not match, the miner
**refuses to mine** rather than producing rejected shares.

---

## Quick answers

**"Can I mine with my 3070 Ti?"** Not alone — 8 GB is too small for a 15.3 GiB
model. Four of them with `--split-model` will work.

**"Can I mine with my 2080 Ti?"** No. Turing has no bf16 and the chain requires
it. It is not a performance question; the proofs would be invalid.

**"Can I mine with my 3090?"** Yes — 24 GB, Ampere, bf16. Works on its own.

**"Do two GPUs mine twice as fast?"** Two *separate* miners on two cards, yes.
`--split-model` across two cards, no — that only makes the model fit.

**"Is my card too slow to be worth it?"** Check [BENCHMARKS.md](BENCHMARKS.md)
for w/s per euro. The cheapest card measured is also the best value per euro,
so "slow" is not the same as "not worth running".
