# Hardware compatibility

Short version: **you need an NVIDIA GPU of the Ampere generation or newer —
either one card with at least 24 GB of VRAM (llama.cpp miner, fastest), or two
matched cards of ~16 GB each via the [vLLM backend](../vllm-miner/)** (pipeline-
parallel; validated end-to-end against the chain's full verification).
The llama.cpp `--split-model` flag remains disabled — multi-GPU is served by
the vLLM backend instead (see "Splitting one model across several GPUs" below).

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

A 2080 Ti has 11 GB and no bf16. **From 0.3.5 the miner refuses to start on
it** and tells you why:

```
error: this GPU cannot mine TensorCash — it has no bf16 support.

         GPU 0  NVIDIA GeForce RTX 2080 Ti  (Turing (sm_75))
```

Earlier versions would start on Turing and appear to run while every share was
rejected, which looked like a miner bug rather than unusable hardware.

There is no configuration, no flag and no future version that changes this: the
precision is fixed by the chain, not by us. If you have Turing cards, they
cannot mine TSC. That is the whole answer.

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
| 16 GB | no alone | **two matched 16 GB cards mine via the [vLLM backend](../vllm-miner/)** (~3.6 w/s/pair) |
| 8–12 GB | **no** | too small even paired — per-card share of the model plus buffers does not fit |

**A note on the most common question:** an RTX 3070 / 3070 Ti (8 GB) or 3080
(10–12 GB) **still cannot mine, even paired** — pipeline-parallel puts roughly
half the model (~7.7 GiB) plus buffers and KV cache on each card, which does
not fit in 8–12 GB. The smallest validated multi-GPU config is **two matched
~16 GB cards** (tested: 2x RTX 5080) via the [vLLM backend](../vllm-miner/).

---

## Splitting one model across several GPUs

**Use the [vLLM backend](../vllm-miner/) for multi-GPU rigs.** It splits the
model across two matched cards with pipeline parallelism and produces proofs
that pass the chain's full verification (validated at 71 audited / 0 rejected).
Expect ~3.6 w/s per 2x16 GB pair on PCIe; NVLink rigs auto-select
tensor-parallel instead.

**The llama.cpp `--split-model` flag remains disabled and refuses to start.**

It produced proofs that were *accepted as shares* (0% reject) but **failed the
chain's model replay**, so any block such a miner won was **orphaned**. That is
not theoretical — it cost a real block. A split miner cannot detect this itself:
its own logs look perfectly healthy, which is exactly what makes it dangerous.

**The cause is now known, and it is upstream.** Under a layer split, llama.cpp's
forward pass itself returns wrong logits when two conditions hold together:

1. the context is **reused for a second window** — the first window after a fresh
   context is always correct — and
2. the **slot count is high** (a large KV cache).

The corruption is per-stream, spreads to more streams as slots rise, and is
deterministic. Judged against a CPU reference, 1–32 slots were clean, 64 was
marginal, 96 was mixed, and 112+ were corrupt from the second window onward —
while production ran far above that. It is not the sampler, not the proof
format, and not this miner's split configuration; disabling is containment
until the upstream KV/compute path is fixed.

**What this means for you:** on single cards that hold the model (>= 24 GB),
run one llama.cpp miner per GPU — the fastest path. On a pair of matched
~16 GB cards, run the vLLM backend. Do not attempt llama `--split-model`.

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
| L40S | 48 GB | 15.3 |
| RTX 6000 Ada | 48 GB | 12.5 |
| RTX A6000 / A40 | 48 GB | 9.5 |
| 2 × RTX 5080 ([vLLM backend](../vllm-miner/), PP=2) | 2 × 16 GB | 3.6 per pair |

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
| HiveOS | `supr-meow-tsc-<v>.tar.gz` — use as the Installation URL |
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

**"Can I mine with my 3070 Ti?"** No — 8 GB is too small even paired: each
card in a pipeline-parallel pair needs ~7.7 GiB of weights plus buffers and
KV. The smallest working multi-GPU config is two matched ~16 GB cards.

**"Can I mine with my 2080 Ti?"** No. Turing has no bf16 and the chain requires
it. It is not a performance question; the proofs would be invalid.

**"Can I mine with my 3090?"** Yes — 24 GB, Ampere, bf16. Works on its own.

**"Do two GPUs mine twice as fast?"** Two *separate* llama.cpp miners on two
>=24 GB cards, yes. Two ~16 GB cards combine via the vLLM backend at ~3.6 w/s
for the pair — slower than one big card, but mining where none was possible.

**"Is my card too slow to be worth it?"** Check [BENCHMARKS.md](BENCHMARKS.md)
for w/s per euro. The cheapest card measured is also the best value per euro,
so "slow" is not the same as "not worth running".
