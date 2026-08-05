# TSC Stratum — an open pool protocol for TensorCash

Version **1.0-draft** · line-delimited JSON-RPC 2.0 over TCP (optionally TLS)

This document specifies a stratum-style protocol for mining TensorCash (TSC).
It is published so that **any** miner or pool may implement it; nothing in it is
specific to one operator. Reference implementation: `supr-meow-tsc`.

---

## 1. Why not classic stratum, and why not the broker protocol

TSC is **proof-of-inference**: a share is a transcript proving a GPU executed a
real forward pass of a chain-registered language model. Two consequences drive
every design choice below.

**A share is ~120 KB, not 4 bytes.** Classic stratum submits a nonce and the
pool reconstructs the block. Here the miner must ship the whole sampling
transcript — chosen tokens, top-k logits, sampling values, VDF proof. So the
protocol needs framing that tolerates large submits, and it must not assume the
pool can recompute a share from a nonce.

**The work unit is model-bound.** A job is only valid for one model at one
registered commit, and the share threshold depends on that model's on-chain
difficulty. So the protocol carries model identity explicitly, and difficulty is
expressed as a 256-bit *target* rather than a scalar.

The upstream project also defines a WebSocket "compute broker" dialect. It works,
but it presumes a Python worker process and an HTTP/WS stack inside the miner.
This protocol exists so a **single native binary** can mine with a plain TCP
socket, which is what rigs (HiveOS, SMOS, flight sheets) actually want.

## 2. Transport and framing

- TCP. One **UTF-8 JSON object per line**, terminated by `\n`. No embedded newlines.
- TLS is the same protocol on a different port. `stratum+tcp://` and
  `stratum+ssl://` URL schemes are used for symmetry with other coins.
- JSON-RPC 2.0. Requests carry `id`; notifications omit it.
- **Lines may be large** (a submit is ~160 KB base64). Implementations MUST NOT
  impose a line limit below **1 MiB**.
- Optional `deflate` compression may be negotiated at subscribe; a submit
  compresses to roughly a third. Both sides MUST support uncompressed.

## 3. Session flow

```
miner                                  pool
  │  mining.subscribe                    │
  │─────────────────────────────────────>│
  │<─────────────────────────────────────│  session id, extranonce1, capabilities
  │  mining.authorize                     │
  │─────────────────────────────────────>│
  │<─────────────────────────────────────│  true
  │<─────────────────────────────────────│  mining.set_model      (notification)
  │<─────────────────────────────────────│  mining.set_target     (notification)
  │<─────────────────────────────────────│  mining.notify         (job)
  │  mining.submit                        │
  │─────────────────────────────────────>│
  │<─────────────────────────────────────│  accepted / rejected(reason)
```

A pool MUST send `set_model` and `set_target` before the first `notify`, so a
miner never receives a job it cannot interpret.

## 4. Methods

### 4.1 `mining.subscribe` (miner → pool)

```json
{"id":1,"method":"mining.subscribe",
 "params":["supr-meow-tsc/0.1.0", null, {"compression":["deflate"],"protocol":"tsc/1.0",
                                          "prompt_seed":true,"pool_vdf":true}]}
```

`params`: `[user_agent, session_id_to_resume|null, options]`.

**Capability flags in `options`.** A miner advertises what optional job fields it
understands. Both are opt-in and a pool MUST keep working with miners that send
neither:

| flag | meaning |
|---|---|
| `prompt_seed` | miner accepts seed-derived prompts (notify field 9, §4.5) |
| `pool_vdf` | miner accepts a pool-issued VDF (notify fields 10 and 11, §4.5) |

A pool SHOULD only populate those fields for miners that advertised the matching
flag; a miner that did not advertise one MUST ignore the field if sent anyway.

Result:

```json
{"id":1,"result":{
   "session_id":"9f2c…",
   "extranonce1":"a1b2c3d4",
   "extranonce2_size":4,
   "protocol":"tsc/1.0",
   "compression":null}}
```

**Extranonce semantics differ from Bitcoin stratum.** In TSC the pool builds the
coinbase and merkle root itself when it requests a work unit from the node
(`extranonce_tag`), so `extranonce1` is *informational* — it identifies the
worker's search space and appears inside `header_prefix`. A miner MUST NOT try to
splice `extranonce1` into a coinbase; it mines the `header_prefix` it is given.
`extranonce2_size` is reserved for future use and MAY be 0.

### 4.2 `mining.authorize` (miner → pool)

```json
{"id":2,"method":"mining.authorize","params":["tc1qexample.rig01","x"]}
```

`params`: `[worker, password]`. `worker` is a TSC payout address, optionally
`.workername`. The password is ignored by pools that do not use one but MUST be
accepted. Result is `true`, or an error object (§7).

### 4.3 `mining.set_model` (pool → miner, notification)

```json
{"method":"mining.set_model","params":[{
   "name":"Qwen/Qwen3-0.6B",
   "commit":"c1899de289a04d12100db370d81485cdf75e47ca",
   "model_hash":"43339e87…",
   "difficulty":1000000,
   "precision":"bf16",
   "normalizer":1000000}]}
```

The miner MUST mine this exact `name@commit`; a proof produced with any other
model or precision is rejected by the network. `difficulty` is the model's
on-chain difficulty and `normalizer` the chain's `ModelDifficultyNormalizer`
(1,000,000). If the miner does not have the model, it SHOULD report
`unsupported_model` (§7) rather than mining something else.

A pool MAY send this at any time; a miner that cannot switch models live MAY
finish its current job first.

### 4.4 `mining.set_target` (pool → miner, notification)

```json
{"method":"mining.set_target","params":["000000ffff000000000000000000000000000000000000000000000000000000"]}
```

A 64-character hex **share target**, big-endian. A proof counts as a share when
its digest is numerically ≤ this value. This replaces `mining.set_difficulty`
because PoI difficulty spans a 256-bit range and a float scalar loses precision.

Miners MAY display difficulty as `2^256 / target`, which is the expected number
of forward passes per share. The pool is responsible for vardiff; the miner MUST
apply a new target to jobs received after it.

**Ordering:** a target applies to every job that arrives after it. A pool that
wants a target change to apply to an in-flight job MUST send a new job.

### 4.5 `mining.notify` (pool → miner, notification)

```json
{"method":"mining.notify","params":[
  "job-11238-56",                                        // job_id
  "00000020a6957b8f…",                                   // header_prefix, 152 hex chars (76 bytes)
  "7fffff00000000000000000000000000000000000000000000000000000000",  // block target
  545259519,                                             // nBits
  11238,                                                 // height
  1785165301,                                            // expires_at (unix)
  true,                                                  // clean_jobs
  "3f8a…64 hex…",                                        // 9  prompt_seed   (optional)
  "0a1b2c…hex…",                                         // 10 pool VDF      (optional)
  315000                                                 // 11 pool VDF tick (optional)
]}
```

Fields 1–7 are mandatory. Fields 9–11 are **optional extensions**; a pool that
omits them is fully conformant, and a miner MUST treat a short `params` array as
"not offered" rather than an error.

`clean_jobs=true` means the parent block changed: the miner MUST abandon
in-flight work, because a proof for a dead parent can never be accepted. When
`false` the miner MAY finish the window it is generating — inference is slow
(seconds per window) and discarding it wastes real work.

`header_prefix` is exactly 152 hex characters. Its bytes are
`version | prev_hash | merkle_root | ntime | nAdjBits`; the miner appends the
4-byte nonce derived from its sampling transcript.

**Field 9 — `prompt_seed` (optional, 64 hex chars).** Instead of the miner
choosing its own prompt, the prompt for window *w* is derived deterministically
from `(seed, w)`. This lets a pool precompute the expected step-0 output for any
window and check a submitted proof against it cheaply, without replaying the
whole transcript. A miner that advertised `prompt_seed` and receives a field
that is not exactly 64 hex characters MUST ignore it and fall back to its own
prompts.

**Fields 10 and 11 — pool-issued VDF (optional).** Field 10 is a hex VDF proof
for this job; field 11 is the tick count it was proved to. When present the
miner uses them instead of proving locally, which removes seconds of VDF work
from every window. Rules that matter for interoperability:

- A **malformed** field 10 (odd length, non-hex, empty) MUST be treated as
  absent, not as an error. The miner then proves locally and the job is simply
  not precomputable for that pool.
- If field 10 is present but field 11 is absent, the tick is unspecified and the
  miner SHOULD fall back to local proving rather than guess — an incorrectly
  assumed tick produces a proof the network rejects.
- **Tick choice is a propagation concern, not a validity one.** A proof is valid
  at any tick, but a block found with too few ticks is held under an
  announcement embargo by the chain's relay rules and can be orphaned. Pools
  SHOULD issue, and self-proving miners SHOULD use, a tick at or above the
  network's prompt-relay threshold. `supr-meow-tsc` self-proves at 315000.

### 4.6 `mining.submit` (miner → pool)

```json
{"id":7,"method":"mining.submit","params":[
  "tc1qexample.rig01",   // worker
  "job-11238-56",        // job_id
  58185448,              // nonce
  "PROF…base64…",        // proof: FlatBuffers MiningResponse, base64
  "0000…achieved hash…", // achieved digest, 64 hex, display-endian
  1234                   // vdf_tick
]}
```

Result: `{"id":7,"result":true}` on acceptance, or an error (§7).

A pool MUST verify the proof before crediting — the hash alone proves nothing
about whether inference happened. Pools SHOULD run at least the `quick` tier of
the reference verifier.

**Block-tier shares:** a proof whose digest also clears the *block* target is a
block. The miner submits it identically; the pool recognises it and submits to
the node. A block hit is also a share and MUST be credited as one.

### 4.7 `mining.set_extranonce` (pool → miner, notification, optional)

```json
{"method":"mining.set_extranonce","params":["a1b2c3d5",4]}
```

Informational, mirroring §4.1. A miner may ignore it.

## 5. Keepalive

Either side MAY send `mining.ping`; the peer replies `mining.pong`. A pool
SHOULD consider a session dead after two missed intervals. Recommended interval
30 s. Miners MUST tolerate long idle periods between jobs — TSC targets 10-minute
blocks.

## 6. What the miner computes locally

A pool supplies the *header context*; everything proving inference is the
miner's job:

- run the registered model in the declared precision (bf16 today) and record the
  sampling transcript over a 256-token window,
- compute the Wesolowski **VDF** over the parent block hash,
- for chains past **v3 activation**, solve the **Argon2id admission puzzle**
  before the window's first token,
- assemble the FlatBuffers `MiningResponse` including `pow_blob_hash`.

A pool MUST NOT expect a miner to submit anything else, and MUST NOT ask a miner
to reveal weights or prompts.

## 7. Errors

`{"id":7,"error":{"code":21,"message":"job not found"},"result":null}`

| code | meaning |
|---|---|
| 20 | other / unknown |
| 21 | job not found — stale or unknown `job_id` |
| 22 | duplicate share |
| 23 | above share target |
| 24 | unauthorized worker |
| 25 | not subscribed |
| 26 | unsupported model — miner cannot serve the requested `name@commit` |
| 27 | malformed proof — not a parseable `MiningResponse` |
| 28 | proof rejected — verification returned RED |

Stale work (code 21) is normal at block boundaries and MUST NOT be treated by a
miner as an error worth disconnecting over.

## 8. Design notes for implementers

- **Do not cancel in-flight inference on a target change.** Only a parent-block
  change (`clean_jobs=true`) invalidates work. A window costs seconds of GPU
  time; discarding it on every vardiff tick destroys throughput.
- **Grace period for stale submits.** A proof that started before a job change
  arrives late by design. Pools SHOULD accept shares against the previous job for
  ~30 s and credit them.
- **Rates are PoI/s** — proofs per second, single or double digits. Do not
  present them as hashes.
- **Precision is consensus.** bf16 is not a performance knob; a quantised model
  produces proofs the network rejects.

## 9. Licence

This specification is published for free use by any implementation, with no
restriction. Contributions and corrections welcome.
