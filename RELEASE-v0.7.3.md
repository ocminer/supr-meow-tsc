# supr-meow-tsc v0.7.3

A consensus-compatibility release for **TensorCash v1.2.3 (ruleset 5)**, which
activates at **mainnet block height 31,400**. The proof format and the mining
kernels are unchanged from v0.7.2 — logits and proofs are bit-identical. What
changes is which finished proofs the miner submits, so the fleet keeps
producing valid shares and blocks after the activation.

Update before height 31,400. Older versions keep mining afterwards, but they
submit near-pinned free windows that the network now rejects, so they waste
work on shares that come back RED.

## What changed

- **Pre-check rules re-vendored to TensorCash v1.2.3.** The pre-submit check
  compiles the verifier's own consensus sources, unmodified, from a pinned
  upstream tag (`vendor/tensorcash-v1.2.3/`, provenance and per-file hashes in
  its `UPSTREAM.txt`; verified by `tools/check-vendored-precheck.sh`). Only the
  pricing/admission rule files changed between v1.2.2 and v1.2.3; every other
  consensus source is byte-identical.

- **Height-aware ruleset-5 admission tier.** From ruleset 5 the network
  promotes a window that would otherwise be *free* (enough credit) to
  *admission-required* when it has **more than 128 near-pins**. This miner
  produces no admission grind, so such a window is a near-certain reject; the
  pre-check now withholds it — exactly as it already withholds the low-credit
  admission band. The decision uses each job's **own candidate height**
  (`mining.notify` height), never the current tip, so jobs issued across the
  boundary are judged correctly. Below the activation height, behaviour is
  unchanged.

- **Measurement built in.** Every mode (including `--precheck=shadow`) now
  reports, per run and in the JSON API, how many ruleset-5 windows were judged,
  how many were withheld by the new rule, and how many were priced more
  strictly than before. Run `--precheck=shadow` before the activation to see
  what ruleset 5 costs without withholding anything yet.

## Pricing

Ruleset 5 roughly doubles the *extra* cost that heavy-tail and near-pinned
windows already paid above the clean-window baseline (a clean window costs
exactly what it did before). The miner does not price its own proofs — the pool
and node do — so this needs no miner change for correctness; it only means a
few more submitted windows will fall short of the stricter target. The
shadow-mode measurement above quantifies it.

## Scope and limits

- Proof format is unchanged (proof v4). Mining kernels, flight sheets, wallets
  and command lines do not change.
- The Linux/NVIDIA builds carry the pre-check. The native Windows/MSVC build
  compiles the pre-check out (it needs 128-bit integers MSVC does not provide),
  so a Windows miner keeps submitting the newly-rejected near-pinned windows
  after height 31,400 — it still mines, it just wastes those shares. Run a
  Linux build for the ruleset-5 withholding, or update when a Windows pre-check
  ships.

## Upgrading

Replace the files of v0.7.2 with this package. The pool identifies the miner as
`supr-meow-tsc/0.7.3`; the pool operator must allow that version string before
it can connect. Use the standard Q8/F16 profile.
