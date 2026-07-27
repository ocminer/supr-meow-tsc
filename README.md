# supr-meow-tsc

A self-contained GPU miner for **TensorCash (TSC)** — one binary, no Docker, no
Python, no service install. Built for real rigs: drop it in a folder, point it
at a pool, run it.

**Version 0.1.0 — foundation.** Device selection, tuning and telemetry are
complete and usable. **It does not mine yet.** The inference and
proof-of-inference pipeline is being integrated; the binary tells you so and
exits rather than printing a hashrate it is not producing. See *Status* below.

---

## Important: TensorCash is not a hash coin, and has no stratum

TSC is **proof-of-inference**. A "share" is a cryptographic transcript proving a
GPU ran a real forward pass of a chain-registered LLM (currently `Qwen3-8B` on
mainnet, `Qwen3-0.6B` on testnet). The consequences for a miner:

- **No `stratum+tcp://`.** Pools speak a WebSocket broker protocol
  (`HELLO` / `ACK` / `MINE_REQUEST` / `MINE_SHARE` / `MINE_RESULT`). Use
  `ws://` or `wss://`. Passing a `stratum+tcp://` URL is rejected with an
  explanation rather than a confusing connection error.
- **The model must be on disk and in VRAM.** ~1.5 GB for the testnet 0.6B model,
  ~16 GB for the mainnet 8B one. A 24 GB card is the practical mainnet minimum.
- **Rates are PoI/s** (proofs per second), not hashes. Expect single or double
  digits, not megahashes. This is normal.
- **bf16 is mandatory** — the proof declares its compute precision and the
  network's verifier checks it against the model checkpoint. Quantised weights
  would be faster and every proof would be rejected.

## Usage

```
supr-meow-tsc -o <pool> -u <wallet>[.<worker>] [-p x] [options]
```

| flag | meaning |
|---|---|
| `-o, --pool` | Pool URL — `ws://` or `wss://` |
| `-u, --user` | TSC address, optionally `.workername` (`tc1…`, testnet `tct1…`) |
| `-p, --pass` | Accepted and ignored (TSC pools do not use it) |
| `-d, --devices` | `-d 0` or `-d 0,1`; **omit to mine on every GPU** |
| `--model` | Path to the chain-registered model (GGUF) |
| `--list-devices` | Print detected GPUs and exit |
| `--dry-run` | Set up and show telemetry without mining |
| `--log-interval` | Seconds between status lines (default 30) |

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

# mine on every GPU
supr-meow-tsc -o wss://tsc.suprnova.cc/ws -u tc1qexample.rig01 -p x

# only GPU 0 and 1, power limited
supr-meow-tsc -o wss://tsc.suprnova.cc/ws -u tc1qexample -d 0,1 --pl 400

# verify device setup and watch telemetry without mining
supr-meow-tsc --dry-run --log-interval 5
```

Sample telemetry:

```
  GPU temp   fan    power  core      mem       util   vram
  0   39C    30%    67W    2572MHz   13801MHz  3%     31.2 GB
  1   48C    0%     71W    2400MHz   13801MHz  0%     1.0 GB
```

## Status

| area | state |
|---|---|
| CLI, help, flight-sheet-friendly args | **done** |
| Multi-GPU enumeration + `-d` selection | **done** |
| Telemetry: temp, fan, power, core/mem MHz, util, VRAM | **done** |
| Tuning: core/mem offsets, locked clocks, power limit, fan | **done** |
| Clean restore of fans/clocks on exit | **done** |
| Inference engine (libllama + CUDA) | in progress |
| Proof-of-inference sampler (+ CUDA offload) | in progress |
| VDF, v3 Argon2id admission | in progress |
| Pool client (WebSocket broker protocol) | in progress |

The mining pieces already exist and run today as a two-process stack
(`llama-server` + a Python broker proxy). This project folds them into one
binary; that work is tracked in the suprnova TSC notes.

## Build

Needs a CUDA toolkit (13.x tested) and an NVIDIA driver with NVML.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/supr-meow-tsc --list-devices
```

`libstdc++`/`libgcc` are linked statically by default so the binary copies
between rigs cleanly; CUDA and NVML stay dynamic because they must match the
installed driver.

## Credits and licence

The proof-of-inference sampler and the inference engine derive from
**TensorCash's fork of `llama.cpp`** (`github.com/tensorcash/llama.cpp`, MIT),
itself a fork of `ggml-org/llama.cpp` (MIT). The GPU offload for the sampler
(`pow_gpu.cu`) and the per-phase profiler are original work from the suprnova
TSC pool project. MIT, in keeping with upstream.
