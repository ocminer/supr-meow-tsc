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

Measured as the time the miner takes to generate one batch of windows with
the default Q8 profile (`MEOW_PROFILE=1`), v0.7.1 against v0.7.2 on the same
card, alternating runs:

| Card | Slots | v0.7.1 | v0.7.2 | Gain |
|---|---|---|---|---|
| CMP 170HX (74 SMs) | 480 | 76.5 s | 65.4 s | +17 % |

## Upgrading

Replace the files of v0.7.1 with this package. Flight sheets, wallets and
command lines do not change. The pool identifies the miner as
`supr-meow-tsc/0.7.2`.
