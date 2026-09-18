// SPDX-License-Identifier: Apache-2.0
#ifndef POW_ZMQ_WRITER_H
#define POW_ZMQ_WRITER_H

#include <zmq.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <any>
#include <chrono>

// FlatBuffers includes
#include <flatbuffers/flatbuffers.h>

// Include generated FlatBuffers headers
#include "proof_generated.h"
#include "blockheader_generated.h"

#include <stdexcept>

/**
 * Thrown at MiningResponseWriter construction when POW_EGRESS_MODE
 * env config is internally inconsistent — e.g. broker mode with the
 * Core Node primary destination still set, or POW_PROXY_ENABLE
 * truthy. Operators MUST fix the env; catching this and proceeding
 * would let solutions leak from a worker that thinks it's in
 * broker mode. The egress-mode contract is documented on the
 * egress_mode_ member below.
 */
class PowEgressConfigError : public std::runtime_error {
public:
    explicit PowEgressConfigError(const std::string& msg)
        : std::runtime_error(msg) {}
};

/**
 * Structure representing a mining response to be sent via ZMQ
 */
struct MiningResponse {
    int64_t req_id;
    uint64_t nonce;
    uint32_t adjusted_bits;
    std::vector<uint8_t> pow_blob_hash;
    uint32_t difficulty;
    std::unordered_map<std::string, std::any> proof_dict;
    bool proxy_only;
    
    MiningResponse(int64_t req_id, uint64_t nonce, uint32_t adjusted_bits,
                   std::vector<uint8_t> pow_blob_hash, uint32_t difficulty,
                   std::unordered_map<std::string, std::any> proof_dict,
                   bool proxy_only = false)
        : req_id(req_id), nonce(nonce), adjusted_bits(adjusted_bits),
          pow_blob_hash(std::move(pow_blob_hash)), difficulty(difficulty),
          proof_dict(std::move(proof_dict)), proxy_only(proxy_only) {}
};

/**
 * Statistics for the ZMQ writer (copyable version)
 */
struct ZmqWriterStats {
    uint64_t messages_sent = 0;
    uint64_t messages_failed = 0; 
    uint64_t queue_size = 0;
    bool running = false;
    std::string push_host;
    int push_port;
    size_t max_queue_size;
};

/**
 * ZMQ-based writer for sending mining response proofs
 */
class MiningResponseWriter {
public:
    explicit MiningResponseWriter(size_t max_queue_size = 100);
    ~MiningResponseWriter();
    
    bool start();
    void stop();
    
    bool submit_response(int64_t req_id,
                        uint64_t nonce,
                        uint32_t adjusted_bits,
                        const std::vector<uint8_t>& pow_blob_hash,
                        uint32_t difficulty,
                        const std::unordered_map<std::string, std::any>& proof_dict,
                        bool proxy_only = false);
    
    bool is_proxy_enabled() const;
    // Slice 11.4 — whether this writer is configured to forward to a
    // broker-controlled miner-proxy (single egress, ProofCollector
    // consumer). In broker mode, sub-block share proofs go through
    // the same physical channel as block solutions; the worker
    // disambiguates on Proof.is_solution.
    bool is_broker_mode() const;
    void get_status(ZmqWriterStats &out) const;
    void set_connection(const std::string& host, int port);

private:
    // Configuration
    size_t max_queue_size_;
    // PoW writer egress topology (the egress envvar contract).
    // One of "local_miner" or "broker".
    // local_miner: primary = Core Node, optional proxy dual-publish.
    // broker:      primary = miner-proxy ProofCollector, no proxy.
    // Parsed from POW_EGRESS_MODE at construction; mode is immutable
    // for the lifetime of the writer.
    std::string egress_mode_;
    std::string push_host_;
    int push_port_;
    std::string save_dir_;

    // Proxy configuration. proxy_enable_ is forced false in broker mode
    // — the writer construction throws PowEgressConfigError if env
    // claims otherwise.
    bool proxy_enable_;
    std::string proxy_host_;
    int proxy_port_;
    bool save_to_disk_;

    // Threading
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> writer_thread_;
    
    // Queue management
    std::queue<std::unique_ptr<MiningResponse>> response_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    
    // ZMQ components
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> zmq_socket_;
    std::unique_ptr<zmq::socket_t> proxy_socket_;
    
    // Atomic statistics (internal use)
    struct {
        std::atomic<uint64_t> messages_sent{0};
        std::atomic<uint64_t> messages_failed{0}; 
        std::atomic<uint64_t> queue_size{0};
        std::atomic<bool> running{false};
        std::string push_host;
        int push_port;
        size_t max_queue_size;
    } stats_atomic_;
    
    // Private methods
    void writer_loop();
    std::vector<uint8_t> serialize_response(const MiningResponse& response);
    std::vector<uint8_t> hex_to_bytes(const std::string& hex_str);
    std::string to_string(const std::any& value);
    void save_to_disk(const std::vector<uint8_t>& data, int64_t req_id, uint64_t nonce,
                     const std::unordered_map<std::string, std::any>& proof_dict);
    
    // FlatBuffer helpers
    flatbuffers::Offset<flatbuffers::Vector<uint32_t>> 
    create_uint32_vector(flatbuffers::FlatBufferBuilder& builder, 
                        const std::vector<uint32_t>& data);
    
    flatbuffers::Offset<flatbuffers::Vector<float>> 
    create_float32_vector(flatbuffers::FlatBufferBuilder& builder, 
                         const std::vector<float>& data);
    
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> 
    create_bool_vector(flatbuffers::FlatBufferBuilder& builder, 
                      const std::vector<bool>& data);
    
    flatbuffers::Offset<proof::FloatArray>
    create_float_array(flatbuffers::FlatBufferBuilder& builder,
                      const std::vector<float>& row_data);
    
    flatbuffers::Offset<proof::UIntArray>
    create_uint_array(flatbuffers::FlatBufferBuilder& builder,
                     const std::vector<uint32_t>& row_data);
    
    // Type conversion helpers
    std::vector<uint32_t> extract_or_convert_uint32_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    std::vector<float> extract_or_convert_float_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    std::vector<bool> extract_or_convert_bool_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    std::vector<uint8_t> extract_or_convert_byte_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    std::vector<std::vector<float>> extract_or_convert_float_matrix(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    std::vector<std::vector<uint32_t>> extract_or_convert_uint32_matrix(const std::unordered_map<std::string, std::any>& dict, const std::string& key);
    
    // Scalar extractors
    template<typename T>
    T extract_numeric(const std::unordered_map<std::string, std::any>& dict, const std::string& key, T default_value);
    std::string extract_string(const std::unordered_map<std::string, std::any>& dict, const std::string& key, const std::string& default_value = "");
    bool extract_bool(const std::unordered_map<std::string, std::any>& dict, const std::string& key, bool default_value = false);
};

/**
 * High-level interface for submitting mining responses
 */
class MiningResponseSubmitter {
public:
    MiningResponseSubmitter();
    ~MiningResponseSubmitter();
    
    bool submit_proof_for_audit(int64_t req_id,
                                const std::unordered_map<std::string, std::any>& proof_dict);
    
    bool submit_solution(int64_t req_id,
                        uint64_t nonce,
                        uint32_t adjusted_bits,
                        const std::vector<uint8_t>& pow_blob_hash,
                        uint32_t difficulty,
                        const std::unordered_map<std::string, std::any>& proof_dict);

    // Slice 11.4 — sub-block share submission. In broker mode this
    // goes down the same physical channel as ``submit_solution``
    // (single egress to miner-proxy); the worker classifies on
    // ``Proof.is_solution`` (false for shares). In local_miner mode
    // there's no broker-side share consumer, so this is a no-op.
    // The caller MUST have stamped ``proof_dict["is_solution"] = false``
    // before invoking (pow_zmq_writer.cpp serialises that bit).
    bool submit_share(int64_t req_id,
                      uint64_t nonce,
                      uint32_t adjusted_bits,
                      const std::vector<uint8_t>& pow_blob_hash,
                      uint32_t difficulty,
                      const std::unordered_map<std::string, std::any>& proof_dict);

    bool is_broker_mode() const;

    ZmqWriterStats get_stats() const;
    void set_connection(const std::string& host, int port);

private:
    std::unique_ptr<MiningResponseWriter> writer_;
};

// Utility functions
std::vector<float> extract_float_vector(const std::any& value);
std::vector<int> extract_int_vector(const std::any& value);
std::vector<bool> extract_bool_vector(const std::any& value);
std::vector<std::vector<float>> extract_float_matrix(const std::any& value);
std::vector<std::vector<int>> extract_int_matrix(const std::any& value);
std::string extract_string(const std::any& value);

template<typename T>
T extract_numeric(const std::any& value);

#endif // POW_ZMQ_WRITER_H
