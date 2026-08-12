# TensorCash vLLM multi-GPU miner (round-5 GREEN build)

The vLLM backend for the **sub-24 GB multi-GPU** tier that llama.cpp can't serve:
an 8B bf16 model split across a matched pair of small cards, producing
chain-verified (full-tier GREEN) PoW proofs. Joined to the existing stratum
layer, so it mines the same pools as the llama miner.

Status: **cleared** — round-5 pool audit, PP=2 cell 10/10 GREEN + single-GPU
25/25 GREEN + llama control 24/24 GREEN, 59 audited / 0 RED.

## Quickstart
```
# 1. install the engine for your GPU arch (see docs/DEPLOYMENT.md)
# 2. edit config/miner.env: WALLET, GPUS (the pair), POOL_HOST, WORKER_SUFFIX
# 3. run
bash scripts/run-miner.sh
```

## Layout
- `config/miner.env`   — all operator knobs (model/commit pin, GPUs, parallel
                         mode, pool, B_cred floor). Code fixes are baked in.
- `scripts/run-miner.sh` — config-driven launcher: PP/TP auto-select, engine +
                         bridge, supervised.
- `scripts/gpu_parallel_select.py` — interconnect detect (PCIe→PP, NVLink→TP).
- `engine/topk_topp_sampler.py` — PoW sampler WITH the mandatory -inf fix
                         (sha 50f2fdda…). Overlay onto the vLLM 0.19 tree.
- `bridge/vllm-stratum-bridge.py` — hardened stratum bridge.
- `docs/DEPLOYMENT.md`  — per-arch install (wheel / source / docker).
- `docs/COMPATIBILITY.md` — hardware pointer (full matrix in the repo-level docs).

## The fixes baked in (the R1–R5 arc)
1. **model-id pin** — proof stamps `Qwen/Qwen3-8B@<40-hex commit>`; unpinned →
   `@unknown` → verifier hangs (R1).
2. **B_cred ≥70 pre-submit gate** — drop low-entropy windows before submit.
3. **socket-timeout reader fix** — no more reader death dropping the pool conn.
4. **padded-vocab -inf fix (THE one)** — vLLM masks ~2 in-vocab tokens to -inf,
   poisoning `logsumexp_stats[4:5]` → infinite Mahalanobis → p=0 every step →
   RED (every engine and topology we tested). Replace -inf with the per-row
   finite min before the PoW snapshot. This closed the entire arc.

## Verify a proof (offline)
```
strings last-submitted-proof.bin | grep 'Qwen/Qwen3-8B@'    # commit, not 'unknown'
# decode: all 6 logsumexp_stats finite at every step (no -inf)
```

## Lessons (recorded)
- **Decode the artifact before theorizing about the system.** Four hypotheses
  died to reasoning; the truth was in the raw proof bytes.
- A **from-step-0, p=0.0-everywhere** failure with tiny per-step errors means a
  **poisoned/degenerate stat** (-inf/NaN), not a subtle model difference. Local
  drift was never the right proxy — the -inf lived in a stat drift doesn't weight.
