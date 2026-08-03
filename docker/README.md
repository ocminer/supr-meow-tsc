# supr-meow-tsc — Docker deployment

Slim image: the miner, its libraries and the CUDA runtime. **No model inside** —
the GGUF is fetched at container start from `MODEL_URL` and cached in `/models`.

## Quick start

```bash
docker pull ocminersupr/supr-meow-tsc:latest
docker run -d --name meow --gpus all --restart unless-stopped \
  -v meow-models:/models \
  -e POOL_URL="stratum+tcp://YOUR_POOL:PORT" \
  -e WALLET="YOUR_PAYOUT_ADDRESS" \
  -e WORKER="rig01" \
  -e MODEL_URL="https://YOUR_HOST/Qwen3-8B-<commit>-bf16.gguf" \
  -e MODEL_SHA256="<sha256>" \
  ocminersupr/supr-meow-tsc:latest
```

**Use `:latest` or a pinned version tag.** Releases before 0.2.1 were built
against stock llama.cpp, which hard-caps `n_seq_max` at 256 — auto-tuning
selects 512 slots on an H100 and such a build fails to create a context.

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
| `SLOTS` | auto | concurrent windows per GPU |
| `MEOW_GROUPS` | auto | sampler threads per GPU. **Not `GROUPS`** — bash overwrites that name with its built-in group-id array, so it always arrives as `0`. |
| `CTX` | engine default | KV tokens per slot; must exceed window+64 (>320) |
| `API_BIND` | `off` | `0.0.0.0:21550` to expose the JSON stats API |
| `DOUBLE_BUFFER` | `0` | keep 0 for large models (see below) |
| `PROMPT_STYLE` | `1` | keep 1 (see below) |
| `SPLIT_MODEL` | `0` | `1` = one model across all `DEVICES`, mined as a single worker. For cards too small to hold it alone (2×12, 4×8, 8×6 GB). Aggregates VRAM, does not add throughput; cards should be identical. |
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
| H100 80GB | 80 GB | 512 |
| RTX 5090 | 32 GB | 128 |
| A100 40GB | 40 GB | ~192 |
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

## OctaSpace

Rigs are started **from an image**, not by SSHing to a machine: push to Docker
Hub → add the image under **My Applications** → select it when renting.

Two consequences that shape this image:

1. **SSH exists only because this image ships `openssh-server`** — it is not a
   platform feature. sshd starts if you set `SSH_PUBKEY` (an authorized_keys
   line), `SSH_PASSWORD` (a root password), or both; set neither and no daemon
   runs at all. Without one of them a rented rig is a black box.

   Prefer the key: a password on a port the whole internet can reach will be
   guessed at continuously, so if you use `SSH_PASSWORD` make it long and
   random. The daemon starts **before** the 15 GB model download, because a rig
   that cannot fetch the model is exactly when you need a shell.
2. **Monitoring is the platform's "View logs"** on container stdout, so the
   miner logs there, `--no-color` is the default, and the first lines report
   the detected GPU, the auto-tuned config, the model and the pool.

Everything else is env vars — there is no config file to place and no shell
step required for a normal run.

| variable | purpose |
|---|---|
| `SSH_PUBKEY` | enables sshd with this key (preferred; off by default) |
| `SSH_PASSWORD` | enables sshd with root password login (use a long random one) |
| `SSH_PORT` | sshd port, default 22 |

## Model hosting

The image ships **without** a model — every rig fetches it once from
`MODEL_URL` into the `/models` volume and reuses it thereafter.

For TSC mainnet the model is `Qwen/Qwen3-8B` at the **chain-registered
commit** `9c925d64d72725edaf899c6cb9c377fd0709d9c5`, converted to a bf16 GGUF
with llama.cpp's `convert_hf_to_gguf.py`:

```bash
python3 -c "from huggingface_hub import snapshot_download as d; \
  d(repo_id='Qwen/Qwen3-8B', revision='9c925d64d72725edaf899c6cb9c377fd0709d9c5', \
    local_dir='qwen3-8b', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])"
python3 llama.cpp/convert_hf_to_gguf.py qwen3-8b \
  --outfile Qwen3-8B-9c925d64-bf16.gguf --outtype bf16
```

Result: 16,388,043,744 bytes, sha256
`fef5847c18f860086007cec0b08960f206c13c5cba69e3ed6292ee1e02ed7e44`.

**Always set `MODEL_SHA256`** — the entrypoint refuses to mine on a mismatch,
which turns a truncated download or a swapped file into a clean startup
failure instead of a stream of rejected shares.

Host it anywhere that serves large files with **range requests** — that part
matters: without `Accept-Ranges: bytes`, a rig whose download drops at 14 GB
starts again from zero. Your own web server or a public Hugging Face repo both
work (Qwen3 is Apache-2.0, so redistributing a converted GGUF is fine; credit
the source commit).

A ready-to-use copy is on Hugging Face (preferred — it has the bandwidth for a
fleet pulling in parallel, and resumes cleanly):

```
MODEL_URL=https://huggingface.co/ocminer/Qwen3-8B-9c925d64-bf16-GGUF/resolve/main/Qwen3-8B-9c925d64-bf16.gguf
MODEL_SHA256=fef5847c18f860086007cec0b08960f206c13c5cba69e3ed6292ee1e02ed7e44
```

Mirror: `https://www.suprnova.cc/models/Qwen3-8B-9c925d64-bf16.gguf` (same
bytes, same checksum).

Worth verifying any mirror before a fleet depends on it — `curl -sI` should
report `content-length: 16388043744` and `accept-ranges: bytes`, and hashing
the first and last megabyte against a known-good copy catches a truncated
upload that a size check alone would miss.

**Do not use a random third-party GGUF.** The registered commit is not
HuggingFace `main`, so a generic "Qwen3-8B GGUF" is likely built from a
different revision — and the chain verifies what your proofs claim.
