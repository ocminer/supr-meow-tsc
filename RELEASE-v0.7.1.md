# supr-meow-tsc v0.7.1

A quality release. The proof format is unchanged from v0.7.0 (proof v4, same
window roots, Gumbel race and commitments), so v0.7.1 works with the same
TensorCash v1.2.2 verifiers. What is new is a **pre-submit consensus check**:
the miner now checks every finished proof with the verifier's own rules and
does not submit a proof the verifier would reject.

## What changed

- **Pre-submit check (`--precheck`, default `on`).** Before a proof is
  submitted, the miner decodes it, without changing its bytes, and runs the
  TensorCash v1.2.2 window rules in the verifier's order. The check stops at
  the first rule that fails:
  - `strict`: the chosen token is present in its row and the declared
    probability matches the canonical kernel.
  - `near-pin`: proof v4 structure rule, at most 174 near-pinned positions.
  - `candidate-band`: the V90 candidate-band condition.
  - `prompt-scaffold`: the post-sunset prompt-scaffold rule.
  - `flat-region`: the post-sunset flat-region rule.
  - `credit-tier`: B_cred below B_FLOOR, or in the admission band below B_FREE.
  - `anti-parrot`: the anti-parrot hard gate.

  The rules are compiled from the unmodified upstream sources (TensorCash
  v1.2.2, commit 48d1344, Apache-2.0), vendored in `vendor/tensorcash-v1.2.2`.
  A proof that fails any rule could not be accepted at any hash. Withholding
  it costs nothing and avoids a rejected share. It does **not** add accepted
  shares, because the window was invalid either way.
- A withheld proof never reaches the pool. It is not counted as submitted,
  accepted, rejected or stale. The miner logs one line per withheld proof
  with the rule, the proof hash and the verifier's reason:

  ```
  [12:34:56] [precheck] dropped share [GPU 0] gate=candidate-band hash=<64 hex> 87ms: proof v4 candidate band rejected (bad-candidate-band): N=85 ...
  ```

  If a withheld proof also met the block target, the miner logs a loud
  `BLOCK candidate WITHHELD` line. Such a block would be consensus-invalid.
- The status line and the JSON stats API (`"precheck"`) report checked,
  dropped and unchecked proofs per rule, and the average check time.
- Each check takes about 0.1 to 0.3 s of one CPU core per share. It runs on
  the submitter threads, never on the sampler threads.

## Options

| Option | Environment | Meaning |
|---|---|---|
| `--precheck on` (default) | `MEOW_PRECHECK=on` | check, do not submit failing proofs |
| `--precheck shadow` | `MEOW_PRECHECK=shadow` | check and log, submit everything |
| `--precheck off` | `MEOW_PRECHECK=off` | submit every proof unchecked (v0.7.0 behaviour) |
| `--precheck-gates all,-credit-tier` | `MEOW_PRECHECK_GATES` | choose rules: names above, `all`, `none`, `-name` |
| | `MEOW_PRECHECK_THREADS` | check threads (default 1 to 4 by CPU count) |

The check fails open: if a proof cannot be evaluated, the miner submits it
and counts it as `unchecked`. On HiveOS, MMPOS and SimpleMining, set
`MEOW_PRECHECK=off` in the extra configuration to restore v0.7.0 behaviour.

## Windows

The native Windows build does **not** include the pre-check. The upstream
rule sources need `unsigned __int128`, which MSVC does not provide. The
Windows miner prints `precheck: unavailable in this build` and submits every
proof, exactly as v0.7.0 did. Everything else is the same as Linux.

## Downloads

| Platform | File |
|---|---|
| HiveOS | `supr-meow-tsc-0.7.1.tar.gz` — do not rename |
| MMPOS | `supr-meow-tsc-0.7.1-mmpos.tar.gz` |
| SimpleMining | `supr-meow-tsc-0.7.1-smos.tar.gz` |
| Linux x86-64 | `supr-meow-tsc-0.7.1-linux-x86_64.tar.gz` |
| Windows x64 | `supr-meow-tsc-0.7.1-windows-x86_64.zip` |
| Docker | `ocminersupr/supr-meow-tsc:0.7.1` |
| Checksums | `supr-meow-tsc-0.7.1.sha256` |

Installation, launchers, model and pool settings are unchanged from v0.7.0.
See the [v0.7.0 notes](RELEASE-v0.7.0.md) for per-platform instructions,
replacing `0.7.0` with `0.7.1`.

## Upgrade and pool identification

Stop the old miner before starting this version. The standard mining profile
(`--q8`, default F16 KV cache) submits user agent **`supr-meow-tsc/0.7.1`**.
The Q8 model is unchanged (sha256 `30f3a9df…73c7`).

## Validation

- Proof v4 root parity: 24/24 vectors, unchanged from v0.7.0.
- `meow-precheck-test` (CPU) runs upstream's golden vectors through the
  pre-check: the candidate-band mainnet windows, 49 prompt-scaffold cases,
  125 flat-region cases, and the five anti-parrot mainnet windows against
  bcore's recorded metrics. It also runs synthetic windows for every rule and
  the FlatBuffer decode path. `tools/check-vendored-precheck.sh` verifies
  that the vendored sources are byte-identical to upstream.
