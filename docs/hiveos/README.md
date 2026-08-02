# HiveOS custom miner package

`supr-meow-tsc-<version>.tar.gz` — install it as a **Custom** miner in HiveOS.

Built with `tools/build-hiveos-package.sh`; see `docker/Dockerfile.hiveos`.

## Why it is its own build

HiveOS is Ubuntu 22.04 (glibc 2.35). The normal Docker image is built on 24.04
(glibc 2.39), and glibc is forward- but not backward-compatible, so that binary
fails on a rig with `GLIBC_2.38 not found`. This package is compiled against
22.04 and verified to need no symbol newer than **GLIBC_2.34**.

It is also built for **consumer** architectures — sm_75 (2080 Ti), sm_86
(3090), sm_89 (4090), sm_120 (5090) — rather than the datacentre set.

## What is inside, and what the rig supplies

The tarball carries the miner, the llama/ggml libraries, the CUDA runtime
(`cudart`, `cublas`, `cublasLt`) and the handful of system libraries HiveOS
does not ship — `libzmq`, `libargon2`, `libgmp` and their dependencies. On a
bare 22.04 the only unresolved libraries are `libcuda.so.1` and
`libnvidia-ml.so.1`, which come from the rig's NVIDIA driver.

`h-run.sh` appends the package's `lib/` to the **end** of `LD_LIBRARY_PATH`, so
anything the rig already provides keeps winning; ours only fill gaps. The CUDA
stub `libcuda.so` is deliberately not shipped — it would shadow the driver.

**The model is not included.** It is 16.4 GB, so it is fetched once on first
start and cached in `MODEL_DIR` (default `/hive-config/models`). Budget disk
accordingly: the model alone needs more space than many rig SSDs have free.

## Flight sheet

| field | value |
|---|---|
| Miner | Custom |
| Installation URL | the release asset URL for this tarball |
| Hash algorithm | `tsc-poi` (label only) |
| Wallet and worker template | `%WAL%.%WORKER_NAME%` |
| Pool URL | `stratum+tcp://host:port` |

**Extra config arguments** — one `KEY=VALUE` per line:

```
MODEL_URL=https://your-host/Qwen3-8B-<commit>-bf16.gguf
MODEL_SHA256=<sha256 of that file>
```

`MODEL_SHA256` is strongly recommended: it turns a truncated download into a
clean startup failure instead of a stream of rejected shares.

Optional: `MODEL_PATH` (model already on the rig), `MODEL_DIR`, `DEVICES`,
`API_PORT` (default 21550), `CTX`.

**Rigs of small cards:** set `SPLIT_MODEL=1` to spread one model over every
selected GPU and mine them as a single worker. This is for cards that cannot
hold the 15.3 GB model alone — 2×12 GB, 4×8 GB, 8×6 GB. It aggregates VRAM, it
does not add throughput, and the cards should be identical. Measured 9.0 w/s on
2× RTX 5080. Do **not** also run one instance per card; they would fight over
the same GPUs.

Note `DEVICES` takes **CUDA ordinals**, which are not `nvidia-smi`'s PCI order —
run the miner with `--list-devices` once to see the mapping.

**Leave `SLOTS` and `MEOW_GROUPS` unset** — the miner reads compute capability
and VRAM at startup and applies a measured per-GPU profile. **Never use the
name `GROUPS`**: it is a bash built-in array, so it always arrives as `0`.

## Reading the stats

HiveOS only understands `khs`, and this miner's unit is proof-of-inference
**windows per second**, not hashes. w/s is therefore reported directly in the
`khs` field: a rig displaying "26 kH/s" is doing 26 windows/s. The honest
figure also appears in the stats JSON.

`h-stats.sh` reads the miner's own JSON API on `127.0.0.1:$API_PORT` and maps
it to HiveOS's schema. If the miner is down or still loading the model, it
reports zero rather than stale numbers.

## First start takes a while

Model download (16.4 GB, parallelised across 8 ranges) then load, so expect
several minutes before the first share. `MODEL_SHA256` is checked before the
miner runs. Watch the miner log for `auto-tuned for <card>` and then
`share accepted`.
