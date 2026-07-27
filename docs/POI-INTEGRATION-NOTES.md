# Finishing the proof-of-inference path (task #9)

Build integration is done — `meow_poi` compiles the sampler, the v3 admission
puzzle and the MiningResponse serializer into the binary. What remains is
wiring, and this file records the exact API surface so the next session starts
from facts rather than re-reading upstream.

## API surface (verified against `pow_utils.h`)

```cpp
PowSamplingCoordinator coord(/*window_size=*/256, /*max_concurrency=*/1024);
coord.initialize(log_dir, proof_dir);

// per mining job, per sequence:
coord.update_pow_params_for_sequence(seq_id, {
    {"target",            "<64 hex>"},   // model-adjusted BLOCK target
    {"share_target",      "<64 hex>"},   // from mining.set_target
    {"header_prefix",     "<152 hex>"},  // from mining.notify
    {"vdf",               "<hex>"},      // Wesolowski proof over the parent hash
    {"tick",              "<u64>"},
    {"difficulty",        "<u64>"},      // model difficulty from mining.set_model
    {"model_identifier",  "name@commit"},
    {"compute_precision", "bf16"},
    {"proof_version",     "3"},
    {"request_id",        "<u64>"},
});
coord.ensure_sequences(seq_ids, prompt_mapping);

// per generated token, INSTEAD of argmax in InferenceEngine::generate_window:
auto r = coord.sample_token_complete(seq_id, logits, n_vocab,
                                     temperature, top_k, top_p, context, "bf16");
coord.record_complete_step(seq_id, r, /*is_valid=*/true);
// r carries: token_id, token_prob, u_value, digest, logsumexp_full,
//            softmax_log_z, logsumexp_stats[6], topk_logits[71], topk_indices[71]
```

The coordinator emits a serialized `MiningResponse` when a window closes and the
digest clears a threshold. Upstream routes that to ZMQ; we must intercept it and
hand it to `StratumClient::submit()` instead.

## Remaining work, in dependency order

1. **`src/poi.{h,cpp}`** — a `PoiMiner` owning one coordinator per device,
   translating a `PoolJob` + `PoolModel` + share target into the params map
   above, and exposing a per-token hook.
2. **Engine hook** — replace the argmax in `InferenceEngine::generate_window`
   with `sample_token_complete`. The signature already passes `logits`/`n_vocab`
   to `on_token`, so the change is contained. Note the sampler also needs the
   **context token vector** (the window's tokens so far), which the engine must
   now carry.
3. **VDF** — the one genuinely new dependency. `vdf` above is a Wesolowski proof
   over the parent block hash, computed locally and continuously (it is a
   sequential-by-design computation, correctly on CPU). Needs `chiavdf` +
   GMP; add `shared-utils/chiavdf` from the umbrella submodule and build its
   verifier/prover sources. **Without a valid VDF every proof is rejected**, so
   this cannot be stubbed.
4. **Submit** — intercept the serialized MiningResponse, base64 it, and call
   `StratumClient::submit(job_id, nonce, proof_b64, achieved_hex, vdf_tick)`.
5. **Validate** — a proof is worthless unless the verifier accepts it. Run the
   miner against a pool with verification enabled and require GREEN verdicts
   before claiming the path works; then an accepted block (#10).

## Constraints that must not be relaxed

- **`POW_PROOF_VERSION` must be 3** and must match what the pool's chain expects;
  a v2 proof is rejected outright on a v3-active chain.
- **bf16 only** — the proof declares its precision and the verifier checks it
  against the model checkpoint.
- **A window must always reach 256 tokens.** EOS is ignored inside a window
  (already enforced in the engine); truncation means no proof ever closes.
- **Bit-exactness applies to the sort order and top-k membership**, not to the
  sums — the verifier feeds the bucket statistics into a Mahalanobis test with a
  covariance sized for cross-GPU variation. Our CUDA sort is already bit-exact;
  the fused stats kernel relies on that tolerance.
