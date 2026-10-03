# Licensing

This repository aggregates first-party code and third-party components under
several licenses. This file is the authoritative map.

## First-party code — Apache License 2.0

The umbrella repository's own files are licensed under the **Apache License,
Version 2.0** (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). Apache-2.0 is
used deliberately for its **explicit patent grant** (§3) and its
**patent-retaliation** clause, which protect users and contributors of the
project's novel mechanisms — in particular the proof-of-inference verification
math.

This explicitly includes:

- **`services/verification-api/`** — the proof verifier and its statistical
  verification procedures.
- **`shared-utils/`** first-party code — `kyc-prover/`, `pow-utils/`,
  `fb-schemas/`, `ipfs/`, `config/`.
- The umbrella's own scripts, deployment manifests, documentation, and the
  `tips/` process documents.

### How this supports a defensive posture

Two things together support a defensive posture against a third party later
claiming the verification methods as their own:

1. **Prior art.** These methods are published openly here, dated by the
   version-control history (and, at release, by signed tags and archived source
   snapshots). Public, dated disclosure supports a prior-art defense against a
   third party obtaining a *valid* patent on the same invention afterward.
2. **Patent grant + retaliation (Apache-2.0 §3).** Each contributor grants a
   patent license over their contributions, and anyone who brings patent
   litigation alleging the Work infringes their patent loses their license.

This is a *defensive* posture: the methods stay open for anyone to use, audit,
and reimplement, while making it harder for others to fence them off or assert
them against the project or its users.

**What this is not.** Apache-2.0 grants patents only from the project's own
contributors; it does not, and cannot, immunize the project against unrelated
third-party patents. Prior art helps invalidate later patents on the same
methods but is not an absolute guarantee, and it only takes effect once the work
is genuinely public. Treat this as defensive publication plus a contributor
patent grant — not a total patent shield. This is not legal advice.

## Third-party components — their own licenses

These are git submodules or vendored code and remain under the licenses set by
their upstream authors:

| Path | Component | License |
| --- | --- | --- |
| `services/core-node/bcore` | Bitcoin-Core-derived consensus node | MIT |
| `services/miner-api/vllm-v0*` | vLLM forks | Apache-2.0 (upstream) |
| `services/miner-api/llama.cpp` | llama.cpp fork | MIT (upstream) |
| `deployments/kubernetes/vllm-stack` | vLLM production stack fork | Apache-2.0 (upstream) |
| `shared-utils/secp256k1-zkp` | secp256k1-zkp | MIT (upstream) |
| `shared-utils/liboqs` | liboqs | MIT (upstream) |
| `shared-utils/chiavdf` | chiavdf (vendored) | Apache-2.0 (upstream) |
| `website` | tensorcash.org source | see that repository |

See each component's own `LICENSE`/`COPYING`/`NOTICE`.

> Note: the consensus-side verification code lives in `bcore` (MIT, upstream
> lineage). MIT carries no patent grant; if the patent grant should also cover
> the consensus verification math, that is a separate relicensing decision for
> the bcore additions — flagged, not assumed.
