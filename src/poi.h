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
    // n_streams = concurrent windows (engine slots); each stream is a
    // sampler sequence with its own params row and window state.
    // egress_port: loopback port for this instance's proof egress — must be
    // unique per PoiMiner (one coordinator+writer pair per device).
    bool init(int window_tokens, std::string& error, int n_streams = 1,
              int egress_port = 47021);

    // Install a new job. Recomputes the VDF for the new parent when the parent
    // changed — the VDF is bound to it, so a stale one invalidates every proof.
    bool set_job(const PoiJobParams& p, std::string& error);

    // Pre-sorts EVERY stream's logits in one segmented GPU launch; results
    // are consumed by the next on_logits call per stream. Call once per step
    // with all streams' logits, before the per-stream on_logits calls.
    void prepare_batch(const std::vector<const float*>& logits, int n_vocab);

    // Per generated token, in place of the engine's argmax. Returns the token
    // the sampler chose, which MUST be fed back to the engine — the transcript
    // and the model's state have to agree or the proof is meaningless.
    int on_logits(int seq_id, const float* logits, int n_vocab,
                  const std::vector<int64_t>& context,
                  float temperature, int top_k, float top_p);

    // Non-null when the last on_logits() closed a window that met a threshold.
    std::unique_ptr<PoiShare> take_share();

    bool ready() const { return ready_; }
    const std::string& job_id() const { return job_id_; }
    const std::string& vdf_hex() const { return vdf_hex_; }
    uint64_t vdf_tick() const { return vdf_tick_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool        ready_ = false;
    std::string vdf_hex_;
    uint64_t    vdf_tick_ = 0;
    std::string parent_hash_hex_;   // to detect a parent change
    std::string job_id_;            // job the current params belong to
    const char* stage_ = "";       // which coordinator call is in flight
    int         n_streams_ = 1;
    int         window_tokens_ = 256;
    uint64_t    nonce_seq_ = 0;     // pool dedup key, one per emitted share      // to detect the start of a new window
};

// A pool of PoiMiner coordinators that samples one decode step for ALL streams
// in parallel. The engine keeps ONE llama context per device and decodes the
// whole batch single-threaded (that is what serialized when we tried multiple
// contexts); only the per-token sampler tail — the actual wall once the sort
// moved to the GPU — is split across `groups` worker threads, each owning its
// own coordinator (separate scratch → no data race), egress port and nonce
// partition. Streams are assigned round-robin-contiguously to groups.
class SamplerPool {
public:
    SamplerPool();
    ~SamplerPool();

    bool init(int n_streams, int groups, int base_egress_port,
              int cuda_device, std::string& error);
    bool set_job(const PoiJobParams& p, std::string& error);   // fans out to all
    bool ready() const;
    std::string job_id() const;

    // Called once per decode step by the mining thread: fills out_tokens[s] for
    // every stream, running each group's streams on its own thread.
    bool sample_step(const std::vector<const float*>& logits, int n_vocab,
                     const std::vector<std::vector<int64_t>>& ctx,
                     std::vector<int>& out_tokens);

    // Drains one finished share across all groups (nullptr if none pending).
    std::unique_ptr<PoiShare> take_share();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace meow
