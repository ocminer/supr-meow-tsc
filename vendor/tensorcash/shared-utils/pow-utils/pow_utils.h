// SPDX-License-Identifier: Apache-2.0
#ifndef POW_UTILS_H
#define POW_UTILS_H


#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include <memory>
#include <deque>
#include <chrono>
#include <optional>
#include <functional>
#include <any>
#include <array>
#include <tuple>
#include <atomic>

#include "pow_zmq_writer.h"    // for MiningResponseSubmitter
#include <flatbuffers/flatbuffers.h>
#include "proof_generated.h"
#include "blockheader_generated.h"

// If using ggml, uncomment this:
// #include "ggml.h"

constexpr size_t POW_WINDOW_SIZE = 256;

// Utility function to get environment variable with default
inline std::string get_env_var(const std::string& name, const std::string& default_value) {
    const char* value = std::getenv(name.c_str());
    return value ? std::string(value) : default_value;
}

// Data structures
struct SequenceCache {
    std::vector<int> archive;  // full prompt + generations
    std::vector<int32_t> ring; // circular buffer of size POW_WINDOW_SIZE
    int ring_pos = 0;
    
    SequenceCache() : ring(POW_WINDOW_SIZE, 0) {}
};

struct PowState {
    // Immutable
    std::vector<uint8_t> target;        // 32 bytes - 256-bit target
    std::vector<uint8_t> h_b;          // 32 bytes - block hash (legacy)
    std::vector<uint8_t> v;            // 32 bytes - VDF
    int T;                             // tick
    std::vector<uint8_t> header_prefix; // 76 bytes - block header prefix (optional)
    
    // Rolling 256-step ring buffers
    std::vector<std::vector<std::vector<float>>> topk_logits;    // [256][B][50]
    std::vector<std::vector<std::vector<int32_t>>> topk_indices; // [256][B][50]
    std::vector<std::vector<float>> chosen_probs;                // [256][B]
    std::vector<std::vector<int64_t>> chosen_tokens;            // [256][B]
    std::vector<std::vector<bool>> attention_mask;              // [256][B]
    std::vector<int32_t> steps;                                 // [B]
    int window_pos = 0;
    
    // For probability reconstruction
    std::vector<std::vector<float>> sampling_u;                 // [256][B]
    
    // Per-sequence parameters
    std::unordered_map<int, float> temperature_by_seq;
    std::unordered_map<int, float> top_p_by_seq;
    std::unordered_map<int, int> top_k_by_seq;
    std::unordered_map<int, float> rep_penalty_by_seq;
    std::vector<std::vector<float>> softmax_normalizers;        // [256][B]
    std::unordered_map<int, SequenceCache> seq_cache;
    
    // Additional field for logsumexp stats
    std::vector<std::vector<std::vector<float>>> logsumexp_stats; // [256][B][6]
    
    PowState() {
        target.resize(32, 0);
        h_b.resize(32, 0);
        v.resize(32, 0);
        T = 0;
    }
};

// Logger class
class Logger {
private:
    std::string log_file_path;
    
public:
    Logger(const std::string& log_dir = "");
    void log(const std::string& message, const std::string& level = "INFO");
};

// Row manager for efficient buffer management
class RowManager {
private:
    int max_rows;
    std::unordered_map<int, int> seqid_to_row;
    std::deque<int> free_rows;
    std::unordered_map<int, int> allocation_order;
    int next_allocation_id = 0;
    
public:
    RowManager(int max_rows);
    std::optional<int> get_row(int seq_id);
    std::optional<int> allocate_row(int seq_id);
    std::optional<int> free_row(int seq_id);
    std::pair<std::optional<int>, std::optional<int>> get_oldest_sequence(const std::vector<int32_t>& steps);
};

// Ring buffer manager
class RingBuffers {
private:
    int window_size;
    int max_rows;
    
public:
    std::vector<std::vector<std::vector<float>>> topk_logits;    // [window_size][max_rows][70]
    std::vector<std::vector<std::vector<int32_t>>> topk_indices; // [window_size][max_rows][70]
    std::vector<std::vector<float>> chosen_probs;                // [window_size][max_rows]
    std::vector<std::vector<int64_t>> chosen_tokens;            // [window_size][max_rows]
    std::vector<std::vector<bool>> attention_mask;              // [window_size][max_rows]
    std::vector<std::vector<float>> sampling_u;                 // [window_size][max_rows]
    std::vector<std::vector<float>> softmax_normalizers;        // [window_size][max_rows]
    std::vector<int32_t> steps;                                  // [max_rows]
    std::vector<std::vector<std::vector<float>>> logsumexp_stats; // [window_size][max_rows][6]

    // v3 admission state (TIP-0003): the selected 32-byte
    // admission nonce per row + valid flag, cleared with the row. Mirrors
    // the Python RingBuffers pow_admission_nonce / pow_admission_valid.
    std::vector<std::array<uint8_t, 32>> admission_nonce;        // [max_rows]
    std::vector<uint8_t> admission_valid;                        // [max_rows]

    // v4 (proof v4): the RECORDED admission nonce R per row, emitted in the
    // proof's extra_flags. DISTINCT from admission_nonce, which under v4 holds
    // the APPENDED effective step nonce E4. In v3 the two coincide; in v4 they
    // do not — E4 = SHA256(..||R||A||..) is what every step hashes, while R is
    // what the verifier needs to recompute it. Cleared with the row.
    std::vector<std::array<uint8_t, 32>> v4_recorded_nonce;      // [max_rows]
    std::vector<uint8_t> v4_recorded_valid;                      // [max_rows]

    RingBuffers(int window_size, int max_rows);
    std::vector<int> get_positions(const std::vector<int>& rows);
    std::vector<std::vector<uint8_t>> get_window_digests(int row);
    // nullptr clears (a nonce admits exactly one window), 32 bytes stores.
    void write_admission_nonce(int row, const uint8_t* nonce32_or_null);
    void clear_row(int row);
    void clear_rows(const std::vector<int>& rows);
    void write_batch(const std::vector<int>& positions, 
                     const std::vector<int>& rows,
                     const std::unordered_map<std::string, std::any>& values_dict);
    void increment_steps(const std::vector<int>& rows);
    std::unordered_map<std::string, std::any> get_window(int row);
    std::vector<std::vector<std::vector<uint8_t>>> digest_buffer;
};

// PoW Hasher
class PowHasher {
private:
    std::vector<uint8_t> h_b;
    std::vector<uint8_t> v;
    std::vector<uint8_t> target;
    // Slice 11.4 — model-adjusted share target (broker-derived).
    // Empty when the broker isn't in share-mode for this lease, in
    // which case check_share_solution returns all-false and
    // emission behaviour matches the pre-slice-11 path.
    std::vector<uint8_t> share_target;
    std::vector<uint8_t> header_prefix;
    int tick;
    std::string ipfs_cid;
    int request_id;
    float difficulty;
    // Registered model difficulty as the exact integer chain value — the v3
    // admission target derivation (§6) is integer-exact and must not go
    // through the float above.
    int64_t difficulty_i64 = 0;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    std::string model_config_diff = "{}";
    std::string  compute_precision;
    std::string  model_identifier;
    // Proof v4: when true, every step preimage that carries the effective step
    // nonce E4 ALSO appends the 20-byte Gumbel mode suffix (pow_v4::
    // SAMPLER_MODE_SUFFIX) after it — the verifier's _get_digest binds both
    // (nonce=E4, mode=V4ConsensusSampler().mode_suffix()). Set from the
    // payload's proof_version in update_from_payload.
    bool v4_mode_ = false;

public:
    PowHasher();
    void update_from_payload(const std::unordered_map<std::string, std::string>& payload);
    // admission_nonces: optional per-entry 32-byte admission nonces (§7),
    // one pointer per context (nullptr = no nonce for that row). Empty
    // vector = legacy v2 shape for every row. Size must otherwise match
    // contexts (throws) — rows in one batch can differ (per-row admission).
    std::tuple<std::vector<int64_t>, std::vector<float>, std::vector<std::vector<uint8_t>>>
        batch_sample_tokens(const std::vector<std::vector<int64_t>>& contexts,
                           const std::vector<int32_t>& steps,
                           const std::vector<std::vector<float>>& cdfs,
                           const std::string& compute_precision,
                           const std::vector<const uint8_t*>& admission_nonces = {});
    // admission_nonce32: when non-null, the 32 raw admission-nonce bytes are
    // appended to the legacy step preimage (TIP-0003 — the ONLY
    // change to the v3 message shape). Null keeps the v2 shape byte-for-byte.
    std::tuple<int64_t, float, std::vector<uint8_t>>
        sample_token(const std::vector<int64_t>& context,
                    int32_t step,
                    const std::vector<float>& cdf,
                    const uint8_t* admission_nonce32 = nullptr);
    // §6/§9: grind this window's Argon2id admission nonce against the
    // registered-difficulty target using this hasher's header/vdf/tick/
    // precision/model_identifier state. `context` is the window-first-step
    // rolling window (msg_w carries NO nonce); prefix_tokens/prefix_pad_mask
    // are the pre-window archive bound by the prompt commitment (the eventual
    // proof's prompt_tokens). Byte-identical to pow_v3.build_admission_preimage.
    // Returns true and fills out_nonce; false when difficulty is unset or the
    // grind exhausts expected_tries * max_tries_factor (mine nonce-less).
    bool grind_admission(const std::vector<int64_t>& context,
                         int32_t step,
                         const std::vector<int64_t>& prefix_tokens,
                         const std::vector<uint8_t>& prefix_pad_mask,
                         uint64_t max_tries_factor,
                         std::array<uint8_t, 32>& out_nonce);
    // Proof v4 window root. Mirrors grind_admission's preimage construction but
    // COMPUTES (never grinds) the window-constant effective step nonce E4:
    //   msg_w0 = build_step_message(header,v,tick,0,context,precision) [no nonce]
    //   C      = prompt_commitment(prefix_tokens, prefix_pad_mask)
    //   R      = 32 fresh random bytes (free tier admits any A; R is recorded)
    //   {A,E4} = pow_v4::derive_v4_root(msg_w0, C, model_identifier, R)
    // out_e4 is APPENDED to every step of the window; out_r is RECORDED in the
    // proof. Returns false only if the model identifier is unusable.
    bool compute_v4_window_root(const std::vector<int64_t>& context,
                                const std::vector<int64_t>& prefix_tokens,
                                const std::vector<uint8_t>& prefix_pad_mask,
                                std::array<uint8_t, 32>& out_e4,
                                std::array<uint8_t, 32>& out_r);
    // The per-step digest D (= SHA-256 of the step preimage with nonce32
    // appended), WITHOUT the CDF token search — the v4 Gumbel race consumes D
    // directly as its 32-byte seed. nonce32 is E4 for v4. Byte-identical to the
    // preimage sample_token hashes.
    std::vector<uint8_t> step_digest_for(const std::vector<int64_t>& context,
                                         int32_t step,
                                         const uint8_t* nonce32);
    std::vector<bool> check_solution(const std::vector<std::vector<uint8_t>>& digests);
    // Slice 11.4 — companion gate on the model-adjusted SHARE target
    // (numerically larger / easier than ``target``). Returns
    // all-false when ``share_target`` is empty, preserving the
    // pre-slice-11 block-only behaviour.
    std::vector<bool> check_share_solution(const std::vector<std::vector<uint8_t>>& digests);
};

// Proof Writer
class ProofWriter {
private:
    std::string output_dir;
    std::function<void(const std::string&)> submit_callback;
    std::string model_identifier;
    std::string compute_precision;
    std::string model_config_diff;
    std::string sampling_params_diff;
    std::string ipfs_cid;
    // Proof schema version: 2 = legacy, >= 3 enables the v3 carrier rules
    // (TIP-0003: canonical extra_flags, no sampler-param
    // fallbacks).
    int proof_version = 2;

public:
    ProofWriter(const std::string& output_dir = "/data/pow_proofs");
    void set_callback(std::function<void(const std::string&)> callback);
    void set_model_identifier(const std::string& model_identifier);
    void set_ipfs_cid(const std::string& ipfs_cid);
    void set_model_config_diff(const std::string& model_config_diff);
    void set_sampling_params_diff(const std::string& sampling_params_diff);
    void set_compute_precision(const std::string& precision);
    void set_proof_version(int version);
    int get_proof_version() const { return proof_version; }

    // Getters for accessing metadata fields
    std::string get_model_identifier() const { return model_identifier; }
    std::string get_compute_precision() const { return compute_precision; }
    std::string get_model_config_diff() const { return model_config_diff; }
    std::string get_ipfs_cid() const { return ipfs_cid; }
    std::pair<std::vector<uint8_t>, std::unordered_map<std::string, std::any>> 
        write_proof(int seq_id, int step_num,
                   const std::unordered_map<std::string, std::any>& window_data,
                   const std::vector<uint8_t>& digest,
                   bool is_solution,
                   const std::unordered_map<std::string, std::string>& pow_params,
                   const std::unordered_map<std::string, std::any>& seq_info);
};

// Utility functions
std::vector<uint8_t> sha256_single(const std::vector<uint8_t>& msg);
std::vector<std::vector<uint8_t>> sha256_many(const std::vector<std::vector<uint8_t>>& messages);
std::vector<bool> check_hash_against_target(const std::vector<std::vector<uint8_t>>& digests,
                                           const std::vector<uint8_t>& target);
std::vector<uint8_t> hex_to_bytes(const std::string& hex_str);
std::string bytes_to_hex(const std::vector<uint8_t>& bytes);
std::vector<uint8_t> nbits_to_target(int nbits);

// Helper functions for byte conversions
std::vector<uint8_t> tok_le_bytes(const std::vector<int64_t>& tokens);
std::vector<uint8_t> u32le(uint32_t value);
std::vector<uint8_t> str_bytes(const std::string& s);
std::vector<uint8_t> build_msg(const std::vector<uint8_t>& header_prefix,
                               const std::vector<uint8_t>& v,
                               const std::vector<uint8_t>& T8,
                               const std::vector<uint8_t>& j4,
                               const std::vector<uint8_t>& ctx_bytes,
                               const std::vector<uint8_t>& precision);
float digest_to_u(const std::vector<uint8_t>& digest);

class PowSamplingCoordinator {
private:
    std::unique_ptr<Logger> logger;
    std::unique_ptr<RowManager> row_manager;  
    std::unique_ptr<RingBuffers> ring_buffers;
    std::unique_ptr<PowHasher> hasher;
    std::unique_ptr<ProofWriter> proof_writer;
    std::unique_ptr<MiningResponseSubmitter> submitter_;
    std::unordered_map<std::string,std::string> current_pow_params;
    std::unordered_map<int, std::unordered_map<std::string, std::string>> seq_pow_params_map;
    std::optional<int> active_pow_seq_id;
    std::unordered_map<std::string,std::any>    current_sequence_info;    
    std::unordered_map<int, std::unordered_map<std::string, std::any>> sequence_info_map;

    int window_size;
    int max_concurrency;
    // --- reusable buffers (preallocated) ---
    std::vector<float> logits_;     // size n_vocab
    std::vector<float> probs_;      // size n_vocab
    std::vector<float> cdf_;        // size n_vocab

    // full-vocab telemetry sort in the effective precision domain (desc)
    std::vector<std::pair<float,int32_t>> pretemp_desc_;

    // survivor mask after top-k
    std::vector<uint8_t> keep_;     // size n_vocab

    // survivors for top-p (m ≤ k)
    std::vector<std::pair<float,int32_t>> cand_;     // (logit, idx)
    std::vector<float> cand_exp_;                    // exp(logit - max)
    std::atomic<int64_t> cooldown_until_ms_{0};
    int64_t cooldown_duration_ms_ = 0;

    // v3 config (TIP-0003), read from env at initialize() —
    // mirrors the vLLM sampler: POW_PROOF_VERSION picks the emitted proof
    // schema, POW_V3_ADMISSION_MODE ('off'|'always') gates the boundary
    // grind, POW_V3_GRIND_MAX_TRIES_FACTOR bounds the per-window attempts.
    int proof_version_ = 2;
    std::string admission_mode_ = "off";
    uint64_t admission_max_tries_factor_ = 16;
    bool admission_warned_no_grinder_ = false;
    // Full prompt+generated token archive per sequence: the pre-window
    // prefix the v3 admission commitment binds, emitted as the v3 proof's
    // prompt_tokens at each window boundary.
    std::unordered_map<int, std::vector<int64_t>> seq_archive_;

    void apply_pow_params(const std::unordered_map<std::string, std::string>& pow_params);
    bool activate_sequence_params(int seq_id);
    // Window-boundary admission prep (§6): clear the row's stale nonce,
    // refresh the v3 prompt_tokens prefix, and (mode 'always') grind the
    // window's nonce BEFORE its first sampled token.
    void prepare_window_admission_(int seq_id, int row, const std::vector<int64_t>& context);
    // Startup self-test mirroring common_sampler_helper.assert_v3_ready:
    // fail initialize(), never the first window boundary, when v3 is
    // configured on a binary that cannot grind admission.
    void assert_v3_ready_();
    static int64_t now_ms_();

public:
    PowSamplingCoordinator(int window_size = 256, int max_concurrency = 1024);
    
    void initialize(const std::string& log_dir = "/tmp/llama_pow_logs", 
                   const std::string& proof_dir = "");
    
    void update_pow_params(const std::unordered_map<std::string, std::string>& pow_params);
    void update_pow_params_for_sequence(int seq_id, const std::unordered_map<std::string, std::string>& pow_params);
    
    void ensure_sequences(const std::vector<int>& seq_ids, 
                         const std::unordered_map<int, std::vector<int64_t>>& prompt_mapping);
    
    bool is_pow_sequence(int seq_id) const;

    // Complete sampling with all state tracking
    struct SamplingResult {
        int64_t token_id;
        float token_prob;
        float u_value;
        std::vector<uint8_t> digest;
        float logsumexp_full;
        float softmax_log_z;
        std::vector<float> logsumexp_stats;  // 6 elements
        std::vector<float> topk_logits;      // 71 elements  
        std::vector<int32_t> topk_indices;   // 71 elements
    };

    void ensure_capacity_(int n_vocab, int top_k);

    SamplingResult sample_token_complete(
        int seq_id,
        const float* raw_logits, 
        int n_vocab,
        float temperature,
        int top_k,
        float top_p,
        const std::vector<int64_t>& context,
        const std::string& compute_precision 
    );

    void set_prompt_tokens(int seq_id, const std::vector<int32_t>& tokens);

    void set_completion_id(int seq_id, const std::string& completion_id);

    void activate_solution_cooldown();
    bool is_cooldown_active() const;
    double cooldown_remaining_seconds() const;
    
    void record_complete_step(int seq_id, const SamplingResult& result, bool is_valid = true);
    
    void cleanup_sequence(int seq_id);

    void check_solutions(const std::vector<int>& seq_ids);

};

#endif // POW_UTILS_H
