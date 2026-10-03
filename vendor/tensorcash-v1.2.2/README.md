# TensorCash v1.2.2 consensus predicates (vendored, unmodified)

This directory holds the subset of TensorCash v1.2.2 `shared-utils/pow-utils/`
that the miner's pre-submit check (`src/precheck.cpp`, `--precheck`) needs to
evaluate a finished proof with the same compiled predicates the verifier runs:

| Rule | Upstream source |
|---|---|
| canonical credit kernel, near-pin count, candidate band | `credit_v4.{h,cpp}`, `candidate_band.h`, `pow_v4.{h,cpp}` |
| anti-parrot hard gate | `antiparrot_gate.h`, `pow_v3.{h,cpp}`, `term_table_r1024.*` |
| prompt scaffold, flat region | `prompt_scaffold.h`, `flat_region.{h,cpp}`, `sunset_abi.cpp` |
| correctly rounded exp/log | `coremath/` (CORE-MATH, MIT) |
| include-path shims | `bcore_compat/` |

Golden vectors from upstream are kept under `shared-utils/pow-utils/tests/vectors/`
and are exercised by the CPU test `meow-precheck-test` (`-DMEOW_BUILD_TESTS=ON`).

The files are byte-identical to upstream tag v1.2.2 (commit 48d1344); see
`UPSTREAM.txt` for the per-file sha256 list and licences, and run
`tools/check-vendored-precheck.sh` to verify. The older, locally adapted
snapshot in `vendor/tensorcash/` is unrelated to this directory and still
defines the miner's proof producer.

Upstream rule documents: `docs/candidate-band.md`, `SUNSET_ADMISSION.md`, and
bcore `doc/PROMPT_SCAFFOLD.md` / `doc/FLAT_REGION.md`.
