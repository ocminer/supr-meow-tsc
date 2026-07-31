# supr-meow-tsc — Docker deployment

Slim image: the miner, its libraries and the CUDA runtime. **No model inside** —
the GGUF is fetched at container start from `MODEL_URL` and cached in `/models`.

## Quick start

```bash
docker load -i supr-meow-tsc-slim.tar        # if using the shipped tarball
docker run -d --name meow --gpus all --restart unless-stopped \
  -v meow-models:/models \
  -e POOL_URL="stratum+tcp://YOUR_POOL:PORT" \
  -e WALLET="YOUR_PAYOUT_ADDRESS" \
  -e WORKER="rig01" \
  -e MODEL_URL="https://YOUR_HOST/Qwen3-8B-<commit>-bf16.gguf" \
  -e MODEL_SHA256="<sha256>" \
  supr-meow-tsc:slim
```

Or `docker compose up -d` with the bundled `docker-compose.yml`.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `POOL_URL` | — | **required**, `stratum+tcp://host:port` |
| `WALLET` | — | **required**, payout address |
| `WORKER` | — | appended as `WALLET.WORKER` |
| `PASSWORD` | `x` | ignored by TSC pools, accepted for flight sheets |
| `MODEL_URL` | — | GGUF to download on first start (cached) |
| `MODEL_SHA256` | — | integrity check; **mining is refused on mismatch** |
| `MODEL_PATH` | — | use a model already in the container/volume instead |
| `MODEL_DIR` | `/models` | cache location — mount a volume here |
| `DEVICES` | all | `0` or `0,1` |
| `SLOTS` | `128` | concurrent windows per GPU |
| `GROUPS` | `12` | sampler threads per GPU |
| `CTX` | engine default | KV tokens per slot; must exceed window+64 (>320) |
| `API_BIND` | `off` | `0.0.0.0:21550` to expose the JSON stats API |
| `DOUBLE_BUFFER` | `0` | keep 0 for large models (see below) |
| `PROMPT_STYLE` | `1` | keep 1 (see below) |
| `EXTRA_ARGS` | — | raw extra flags |

## Why the defaults are what they are (measured, RTX 5090 + Qwen3-8B)

- **`SLOTS=128`, `DOUBLE_BUFFER=0`** — 8B decode is memory-bandwidth bound
  (~15 GB of weights streamed per step). Throughput is *sequences per weight
  read*, so one big batch beats two alternating ones: 128×1 measured 19.5 w/s
  vs 12.8 for 64×2. Above 128 it plateaus (KV traffic grows with batch).
  **Smaller cards need a lower `SLOTS`** — KV is ~57 MB/slot for 8B at the
  default ctx, so 128 slots needs ~26 GB. On a 24 GB card start near 96.
- **`PROMPT_STYLE=1`** — the chain's reuse-entropy guard rejects transcripts
  whose continuations are too *predictable*. A factual prompt on an 8B model
  measured **3.9% honest rejects**; the open-ended prompt measured **0 in 952**
  over 4 hours (baseline would predict 37).

## Sizing

| GPU | VRAM | suggested `SLOTS` (8B) |
|---|---|---|
| RTX 5090 / A100 40GB+ | 32 GB+ | 128 |
| RTX 4090 / 3090 | 24 GB | 96 |
| A10 / L4 | 24 GB | 96 |

Model needs ~15.3 GB resident; the rest is KV cache and compute buffers.

## Notes

- The image is multi-arch: `sm_80 86 89 90 120` (A100, A10/3090, 4090, H100, 5090).
- Requires the NVIDIA container runtime on the host (`--gpus all`).
- `libcuda.so.1` / `libnvidia-ml.so.1` are intentionally absent from the image;
  the container runtime injects them from the host driver.
- The miner is a stratum client only — it needs outbound TCP to the pool, and
  outbound HTTPS once for the model download.
