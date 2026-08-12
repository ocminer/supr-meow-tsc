# vLLM multi-GPU miner — deployment story

The engine boundary is now measured and chain-verified in both directions:

| tier | engine | numbers |
|---|---|---|
| single GPU ≥24 GB | **llama.cpp** | 19.5 w/s — unchanged, fleet untouched |
| sub-24 GB multi-GPU (PCIe) | **vLLM PP=2** | ~3.63 w/s per 2×16 GB pair, honest (was "cannot mine at all") |
| NVLink rigs | vLLM TP (candidate) | untested at scale |

This package serves the **sub-24 GB multi-GPU** tier that llama.cpp cannot: an
8B bf16 model that doesn't fit one small card, split across a pair.

## What's in the box
- `engine/topk_topp_sampler.py` — the PoW sampler with the **-inf fix** (sha
  `50f2fdda…`). vLLM masks a few in-vocab tokens to -inf; unfixed, those poison
  `logsumexp_stats[4:5]` and every proof REDs. This slice is mandatory.
- `bridge/vllm-stratum-bridge.py` — hardened stratum bridge: reconnect
  supervisor, B_cred≥70 pre-submit gate, model-id byte-stamp, clean-jobs
  handling, and llama-format stats (`share accepted`, `prof-e2e … windows/s`)
  so existing fleet tooling parses it.
- `scripts/run-miner.sh` + `config/miner.env` — one-command, config-driven.
- `scripts/gpu_parallel_select.py` — PP-vs-TP auto-select by interconnect.

## The three deployable paths (by GPU arch)

**1. Ada / Ampere (cc 8.0–8.9: 3090, 4090, A-series) — the wheel path (easy).**
The stock `vllm==0.19.0` wheel runs natively. Install it into a venv, then
overlay this package's fixed sampler over the wheel's copy:
```
uv venv --python 3.10 && . .venv/bin/activate
uv pip install vllm==0.19.0 --torch-backend=auto
# overlay the fix (path may vary by site-packages layout):
cp engine/topk_topp_sampler.py $SITE/vllm/v1/sample/ops/topk_topp_sampler.py
```
No multi-hour compile. Point `config/miner.env: TC_VLLM` at the wheel's tree.

**2. Blackwell (cc 12.x: 5080, 5090) — source build (slow, one-time).**
No stock wheel yet targets sm_120; build vLLM 0.19 from source with
`TORCH_CUDA_ARCH_LIST=12.0`, torch 2.10+cu129, CUDA 12.9 nvcc (see
`vllm-build-info.txt`). The fixed sampler lives in the editable tree. ~hours to
build once; the resulting env is reusable.

**3. Any arch — per-arch Docker / proot (portable).**
A precompiled per-arch image can run the vLLM engine under proot (userspace,
no root/Docker required) — verified end-to-end. If we publish images, build one
per arch (Ada/Ampere wheel;
Blackwell source) and bake the fixed sampler in. The proot native path is the
fallback for rented/locked containers (≥24 GB there; splitting needs matched
local cards).

## Run
```
# edit config/miner.env: WALLET, GPUS (the pair), POOL_HOST, WORKER_SUFFIX
bash scripts/run-miner.sh
```
Auto-selects PP (PCIe) or TP (NVLink), starts engine + bridge, submits to the
pool. Logs: `engine.log`, `bridge.log`; `last-submitted-proof.bin` for byte
checks (`strings last-submitted-proof.bin | grep 'Qwen/Qwen3-8B@'`; decode
should show all 6 `logsumexp_stats` finite).

## Requirements
- Matched GPUs for a split (tensor-parallel needs identical cards; pipeline
  tolerates a weak PCIe link and is auto-chosen there).
- bf16 no-quant (consensus: the proof commits to exact weights).
- Offline HF cache seeded with the model at the pinned commit.

## Before shipping
Run one higher-volume pool confirmation round **against this packaged artifact**
(the thing users run), not the dev setup — the last gate.
