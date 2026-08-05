# supr-meow-tsc

A self-contained GPU miner for **TensorCash (TSC)** — one binary, no Docker, no
Python, no service install. Built for real rigs: drop it in a folder, point it
at a pool, run it.

**Mining.** The full pipeline is in: embedded inference engine
(llama.cpp/CUDA), GPU-offloaded proof-of-inference sampler with GPU-resident
logits, chiavdf VDF, TSC Stratum client with failover, local JSON stats API,
and a HiveOS integration package (`docs/hiveos/`).

---

## Important: TensorCash is not a hash coin

TSC is **proof-of-inference**. A "share" is a cryptographic transcript proving a
GPU ran a real forward pass of a chain-registered LLM (currently `Qwen3-8B` on
mainnet, `Qwen3-0.6B` on testnet). The consequences for a miner:

- **The model must be on disk and in VRAM.** ~1.5 GB for the testnet 0.6B model,
  ~16 GB for the mainnet 8B one. A 24 GB card is the practical mainnet minimum.
- **Rates are PoI/s** (proofs per second), not hashes. Expect single or double
  digits, not megahashes. This is normal.
- **bf16 is mandatory** — the proof declares its compute precision and the
  network's verifier checks it against the model checkpoint. Quantised weights
  would be faster and every proof would be rejected.
- Pools speak **TSC Stratum** (`stratum+tcp://`, line-delimited JSON-RPC —
  spec in `docs/STRATUM-TSC.md`). The upstream WebSocket broker dialect is a
  different protocol; `ws://` and `wss://` URLs are **rejected** with an
explanation. Always use `stratum+tcp://` (or `stratum+ssl://`).

## Usage

```
supr-meow-tsc -o <pool> -u <wallet>[.<worker>] [-p x] [options]
```

| flag | meaning |
|---|---|
| `-o, --pool` | Pool URL — `stratum+tcp://host:port`; repeat for failover |
| `-u, --user` | TSC address, optionally `.workername` (`tc1…`, testnet `tct1…`) |
| `-p, --pass` | Accepted and ignored (TSC pools do not use it) |
| `-d, --devices` | `-d 0` or `-d 0,1`; **omit to mine on every GPU** |
| `--model` | Path to the chain-registered model (GGUF) |
| `--slots` / `--groups` | Concurrent windows per GPU / sampler threads per GPU |
| `--api-bind` | Local JSON stats API, default `127.0.0.1:21550` (`off` disables) — HiveOS reads this |
| `--list-devices` | Print detected GPUs and exit |
| `--dry-run` | Set up and show telemetry without mining |
| `--log-interval` | Seconds between full status reports (default 30) |

Tuning (per selected device; omit any flag to leave the card alone):

| flag | meaning |
|---|---|
| `--cclock <mhz>` | Core clock **offset**, may be negative |
| `--mclock <mhz>` | Memory clock **offset** |
| `--lock-core <mhz>` | Lock core to an absolute clock |
| `--pl <watts>` | Power limit |
| `--fan <percent>` | Fixed fan speed (omit → driver's own curve) |

Tuning needs privileges the driver grants only to root or a persistence-mode
session. If a setting is refused the miner warns and **keeps mining** — a fan
that cannot be set is not a reason to stop earning. On exit, fans are handed
back to the driver and locked clocks are released, so a crashed miner never
leaves a card pinned.

### Examples

```bash
# what is in this rig?
supr-meow-tsc --list-devices

# mine on every GPU (slots/groups are auto-tuned per card — leave them unset)
supr-meow-tsc -o stratum+tcp://tsc.suprnova.cc:3310 -u tc1qexample.rig01 -p x \
  --model Qwen3-8B-9c925d64-bf16.gguf

# only GPU 0 and 1, power limited
supr-meow-tsc -o stratum+tcp://tsc.suprnova.cc:3310 -u tc1qexample -d 0,1 --pl 400 \
  --model Qwen3-8B-9c925d64-bf16.gguf

# cards too small to hold the model alone (2x12GB, 4x8GB): split it across them
supr-meow-tsc -o stratum+tcp://tsc.suprnova.cc:3310 -u tc1qexample -d 0,1 \
  --split-model --model Qwen3-8B-9c925d64-bf16.gguf

# verify device setup and watch telemetry without mining
supr-meow-tsc --dry-run --log-interval 5
```

Sample telemetry:

```
  GPU temp   fan    power  core      mem       util   vram
  0   39C    30%    67W    2572MHz   13801MHz  3%     31.2 GB
  1   48C    0%     71W    2400MHz   13801MHz  0%     1.0 GB
```

## Performance

Measured end-to-end against the live pool, not in a synthetic harness. Full
curves, the cost-per-GPU-hour table and the reasoning behind each setting are
in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**; operational notes are in
**[docs/OPERATIONS.md](docs/OPERATIONS.md)**.

| GPU | best config | w/s | €/GPU/h | w/s per €/h |
|---|---|---|---|---|
| RTX A6000 48GB | 361 slots | 9.5 | 0.187 | **50.8** |
| RTX PRO 6000 96GB | 512 slots | 25.6 | 0.579 | 44.2 |
| H100 80GB | 512 slots | 40.5 | 0.996 | 40.7 |
| RTX 6000 Ada 48GB | 403 slots | 12.5 | 0.319 | 39.2 |
| A100 80GB | 512 slots | 20.7 | 0.549 | 37.7 |
| H200 141GB | 512 slots | 40.5 | 1.226 | 33.0 |
| B200 183GB | 512 slots | **56.2** | 1.873 | 30.0 |
| RTX 5090 32GB | 128 slots | 19.5 | — | — |

All are selected **automatically** — the miner reads compute capability and
VRAM at startup and applies the matching profile. Three results worth knowing
before buying hardware:

- **The cheapest card is the best buy per euro.** An A6000 returns 50.8 w/s per
  €/h against the B200's 30.0, despite being 5.8× slower — it is also 10×
  cheaper per GPU-hour.
- **Fastest ≠ best value.** The B200 is the quickest card measured and the worst
  buy on the list. Pick by which one you actually need.
- **Bandwidth stops paying within a generation, not across one.** The H200 has
  43% more bandwidth than the H100 and returns the identical number; the B200
  then beat both by 39%. Marginal return per TB/s keeps falling, so bandwidth is
  not the thing you are buying — but a plateau seen on one architecture says
  nothing about the next.

Stale shares are part of the price and `[prof-e2e]` cannot see them: slow cards
hold a batch longer, so more of it is discarded when the job changes (A6000
8.2% stale versus 1.6% on an H200). Stale-adjusted, the A6000's lead over the
PRO 6000 narrows from ~12% to ~5%. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Status

| area | state |
|---|---|
| CLI, help, flight-sheet-friendly args | **done** |
| Multi-GPU enumeration + `-d` selection | **done** |
| Telemetry: temp, fan, power, core/mem MHz, util, VRAM | **done** |
| Tuning: core/mem offsets, locked clocks, power limit, fan | **done** |
| Clean restore of fans/clocks on exit | **done** |
| Inference engine (libllama + CUDA, GPU-resident logits) | **done** |
| Proof-of-inference sampler (fused CUB sort/stats on GPU) | **done** |
| VDF (chiavdf) | **done** |
| Pool client (TSC Stratum, reconnect + failover) | **done** |
| Local JSON stats API + HiveOS package (`docs/hiveos/`) | **done** |
| v3 Argon2id admission grinding | optional band, not mined |

## Download

Prebuilt packages for each release are on the
[GitHub releases page](https://github.com/ocminer/supr-meow-tsc/releases) —
pick the one matching your rig:

| package | for |
|---|---|
| `supr-meow-tsc-<v>-linux-x86_64.tar.gz` | any Linux rig — unpack and run |
| `supr-meow-tsc-<v>-hiveos.tar.gz` | HiveOS custom miner (Installation URL) |
| `supr-meow-tsc-<v>-mmpos.tar.gz` | MMPOS custom miner |
| `supr-meow-tsc-<v>-smos.tar.gz` | SimpleMining (SMOS) custom miner |

All are built against Ubuntu 22.04 (glibc 2.35), so they run on 22.04 and
newer — a 24.04 build fails on older rigs with `GLIBC_2.38 not found`.
The 16.4 GB model is **not** bundled; each package fetches it once and verifies
`MODEL_SHA256` before mining.

### Docker

```bash
docker pull ocminersupr/supr-meow-tsc:latest

docker run -d --name meow0 --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /path/to/models:/models \
  -e POOL_URL=stratum+tcp://tsc.suprnova.cc:3310 \
  -e WALLET=tc1q… -e WORKER=rig01 \
  -e MODEL_URL=https://huggingface.co/ocminer/Qwen3-8B-9c925d64-bf16-GGUF/resolve/main/Qwen3-8B-9c925d64-bf16.gguf \
  -e MODEL_SHA256=fef5847c18f860086007cec0b08960f206c13c5cba69e3ed6292ee1e02ed7e44 \
  ocminersupr/supr-meow-tsc:latest
```

Tags: `:latest` tracks the newest release, or pin a version like `:0.3.4`.
One container per GPU — set `NVIDIA_VISIBLE_DEVICES` and a distinct `WORKER`
for each. `SSH_PUBKEY` (or `SSH_PASSWORD`) starts an sshd for platforms like
OctaSpace that give you no other shell; it comes up **before** the model
download, so you can log in while that runs.

## Build

Needs a CUDA toolkit (13.x tested) and an NVIDIA driver with NVML.

**Do not clone with `--recursive`.** `vendor/tensorcash` contains nested
submodules this project does not use, and at least one of them is unreachable,
so a recursive clone aborts. Initialise only the two that are needed:

```bash
git clone https://github.com/ocminer/supr-meow-tsc
cd supr-meow-tsc
git submodule update --init vendor/llama.cpp vendor/tensorcash
```

`vendor/llama.cpp` is our own mirror of the TensorCash llama.cpp fork — the
upstream disappeared mid-2026 and the pinned commit exists nowhere else. The
GPU-resident-logits change is applied at build time from
`tools/llama-gpu-logits.patch`, so the mirror stays a clean copy of the fork.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/supr-meow-tsc --list-devices
```

`libstdc++`/`libgcc` are linked statically by default so the binary copies
between rigs cleanly; CUDA and NVML stay dynamic because they must match the
installed driver.

## Credits and licence

This miner is **MIT** licensed — see [LICENSE](LICENSE). The GPU offload for
the sampler (`pow_gpu.cu`), the per-GPU tuning profiles, the model splitting
and the per-phase profiler are original work from the suprnova TSC pool
project.

It builds on third-party components that keep their own terms, listed in full
in [NOTICE](NOTICE):

- **llama.cpp** (MIT) — the inference engine, via
  [`ocminer/llama.cpp`](https://github.com/ocminer/llama.cpp), our mirror of
  the TensorCash fork after the original upstream became unreachable. Modified
  at build time by `tools/llama-gpu-logits.patch` for GPU-resident logits.
- **TensorCash pow-utils** (Apache-2.0) — the proof-of-inference sampler and
  proof serialisation, from
  [git.tensorcash.org](https://git.tensorcash.org/tensorcash/tensorcash).
- **chiavdf** (Apache-2.0) — the verifiable delay function.
- **nlohmann/json** (MIT).

Those terms travel with any binary built from this repository.
