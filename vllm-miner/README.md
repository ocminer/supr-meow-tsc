# supr-meow-tsc — vLLM backend (sub-24 GB multi-GPU tier)

The TensorCash miner is **two engines, one stratum layer**:

- **llama.cpp** (the main `supr-meow-tsc` binary) — fastest on a single GPU with
  ≥24 GB, plus CPU/Apple/edge and all packaged platforms. Use it there.
- **vLLM** (this backend) — for **two matched ~16 GB cards** (e.g. 2× RTX 5080)
  that llama.cpp cannot serve, and for 24 GB+ single-GPU vLLM. ~12.7 w/s per
  16 GB pair, chain-verified.

## Run it (one command)

```bash
docker run -d --restart unless-stopped --gpus all --shm-size=4g \
  -v meow-vllm-models:/models \
  -e WALLET=tc1qYOUR_ADDRESS \
  -e WORKER=rig1 \
  -e POOL=stratum+tcp://tsc.suprnova.cc:3310 \
  ocminersupr/supr-meow-tsc-vllm:latest
```

First boot fetches Qwen3-8B and captures CUDA graphs (10–30 min); the model and
compile cache persist in the named volume across restarts.

**One image, every supported GPU.** The vLLM 0.19 wheel ships native kernels for
Ampere → Blackwell (incl. RTX 5080/5090, sm_120), so no per-arch build.

## Common options (all `-e`)

| var | default | meaning |
|---|---|---|
| `WALLET` | — (required) | your `tc1q…` payout address |
| `POOL` | — (required) | `stratum+tcp://host:port` |
| `WORKER` | `vllmrig0` | worker name shown on the pool |
| `GPUS` | `all` | `all`, `0`, or `2,3` (a 16 GB pair) |
| `PARALLEL` | `auto` | `auto`\|`pp`\|`tp`\|`single` — auto picks PP on PCIe, TP on NVLink |
| `MAX_NUM_SEQS` | `256` | batch (audited high-batch default) |
| `GPU_MEM_UTIL` | `0.84` | measured-safe VRAM fraction |

## What's baked in

The image installs the stock `vllm==0.19.0` wheel and overlays our audited
proof-of-inference path (compact top-50 CDF, topk-opt snapshot, the `-inf`
statistics fix, a pre-submit B_cred gate, and a pinned model revision). Every
change is validated against the pool's full-tier GPU verification before ship.

## Build from source

```bash
docker build -t supr-meow-tsc-vllm vllm-miner/
```

`config/miner.env` documents every knob; `docs/DEPLOYMENT.md` covers wheel vs
source and multi-GPU layouts.
