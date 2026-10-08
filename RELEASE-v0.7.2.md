# supr-meow-tsc v0.7.2

A performance release. The proof format, the pre-submit check and every number
the verifier sees are unchanged from v0.7.1 (proof v4, TensorCash v1.2.2
rules). v0.7.2 produces bit-identical logits and proofs; it only gets there
faster.

## What changed

- **Fused bf16 rounding (`MEOW_BF16_FUSE`, default `2`).** To match the
  verifier, the Q8 profile rounds about 22 intermediate results per layer to
  bf16. v0.7.1 did each rounding as two separate copy kernels (f32 → bf16 →
  f32). That cost about a fifth of the GPU time. v0.7.2 runs:
  - each rounding as one vectorized kernel;
  - the norm chain (RMS norm, round, weight multiply, round) as one kernel;
  - the SwiGLU chain (round, SiLU, round, up-projection round, multiply,
    round) as one kernel after the up-projection;
  - the residual chain (round, add, round) as one kernel.

  Every rounding stays in the same place and order, with the same conversion
  and arithmetic, so the result is bit-identical. `MEOW_BF16_FUSE=1` keeps
  only the single-kernel rounding; `MEOW_BF16_FUSE=0` restores the v0.7.1
  kernels. The rounding layout itself (`MEOW_BF16_GRAPH`) is unchanged.
- `tests/bf16_fuse_exact.cpp`: a deterministic forward-pass fingerprint
  (480 sequences, greedy decoding, a hash of every full-vocabulary logit row
  per step) used to confirm the levels give identical output.

## Speed

Measured as the time the miner takes to generate one batch of windows while
mining on the pool with the default Q8 profile (`MEOW_PROFILE=1`). The release
packages of v0.7.1 and v0.7.2 ran on the same card, alternating, 2 × 180 s
each:

| Card | Slots | v0.7.1 | v0.7.2 | Gain |
|---|---|---|---|---|
| RTX 5090 (450 W cap) | 480 | 34.7 s | 27.7 s | +25.5 % |
| CMP 170HX (74 SMs) | 480 | 78.7 s | 67.6 s | +16.6 % |
| 2× RTX 3070, `--split-model` | 128 | 32.4 s | 28.7 s | +12.8 % |
| RTX 5070 Ti | 128 | 17.9 s | 15.9 s | +12.1 % |

On every card the logit fingerprint of v0.7.2 is identical to v0.7.1. All
shares in these runs were accepted (0 rejected).

## Upgrading

Replace the files of v0.7.1 with this package. Flight sheets, wallets and
command lines do not change. The pool identifies the miner as
`supr-meow-tsc/0.7.2`.
