// =============================================================================
// poi.h — proof-of-inference mining state.
//
// Turns a stratum job into the parameters the sampler needs, drives the sampler
// per generated token, and hands finished proofs back to the caller for submit.
//
// One instance per GPU. The sampler coordinator is not thread-safe across
// devices, and each device mines its own windows anyway.
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace meow {

struct PoolJob;
struct PoolModel;

struct PoiJobParams {
    std::string header_prefix;    // 152 hex, from mining.notify
    std::string block_target;     // 64 hex
    std::string share_target;     // 64 hex, from mining.set_target
    std::string model_identifier; // "name@commit"
    uint64_t    model_difficulty = 1000000;
    uint64_t    normalizer       = 1000000;
    uint64_t    request_id       = 0;
    std::string job_id;
    bool        valid = false;
};

// A finished window that met a threshold and should be submitted.
struct PoiShare {
    std::string job_id;
    uint64_t    nonce = 0;
    std::string proof_b64;     // serialized MiningResponse
    std::string achieved_hex;  // 64 hex, display-endian
    uint64_t    vdf_tick = 0;
    bool        is_block = false;   // also cleared the chain target
};

class PoiMiner {
public:
    PoiMiner();
    ~PoiMiner();

    // window_tokens must match the chain's PoW window (256).
    bool init(int window_tokens, std::string& error);

    // Install a new job. Recomputes the VDF for the new parent when the parent
    // changed — the VDF is bound to it, so a stale one invalidates every proof.
    bool set_job(const PoiJobParams& p, std::string& error);

    // Per generated token, in place of the engine's argmax. Returns the token
    // the sampler chose, which MUST be fed back to the engine — the transcript
    // and the model's state have to agree or the proof is meaningless.
    int on_logits(int seq_id, const float* logits, int n_vocab,
                  const std::vector<int64_t>& context,
                  float temperature, int top_k, float top_p);

    // Non-null when the last on_logits() closed a window that met a threshold.
    std::unique_ptr<PoiShare> take_share();

    bool ready() const { return ready_; }
    const std::string& vdf_hex() const { return vdf_hex_; }
    uint64_t vdf_tick() const { return vdf_tick_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool        ready_ = false;
    std::string vdf_hex_;
    uint64_t    vdf_tick_ = 0;
    std::string parent_hash_hex_;   // to detect a parent change
    const char* stage_ = "";       // which coordinator call is in flight
    size_t      last_ctx_ = 0;
    size_t      prompt_len_ = 0;    // context size at the window's first token
    int         window_tokens_ = 256;      // to detect the start of a new window
};

}  // namespace meow
