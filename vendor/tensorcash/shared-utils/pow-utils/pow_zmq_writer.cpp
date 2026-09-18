// SPDX-License-Identifier: Apache-2.0
#include "pow_zmq_writer.h"
#include "pow_zmq_writer_helpers.h"
#include "pow_utils.h" // for get_env_var
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <typeinfo>
#include <any>
#include <algorithm>
#include <cctype>

namespace {

// Mirrors the Python writer's _EGRESS_MODE_* constants and parsing.
// Kept as namespace-locals so they don't leak as ODR symbols.
constexpr const char* kEgressModeLocalMiner = "local_miner";
constexpr const char* kEgressModeBroker     = "broker";

// Truthy values accepted for POW_PROXY_ENABLE; identical to the
// historical parsing so a config that was truthy yesterday stays
// truthy today.
inline bool is_truthy_proxy_enable(const std::string& v) {
    return v == "1" || v == "true" || v == "True";
}

inline std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

inline bool looks_like_core_node(const std::string& host) {
    return to_lower(host).find("core-node") != std::string::npos;
}

} // namespace

// FlatBuffers generated includes
#include "proof_generated.h"
#include "blockheader_generated.h"

// namespace {
//     // File identifier for FlatBuffers (matches schema)
//     constexpr const char* FILE_IDENTIFIER = "PROF";
// }

// static std::vector<uint8_t> hex_to_bytes(const std::string &hex) {
//     std::vector<uint8_t> out;
//     out.reserve(hex.size()/2);
//     for (size_t i = 0; i < hex.size(); i += 2) {
//         uint8_t byte = static_cast<uint8_t>(
//             std::stoul(hex.substr(i,2), nullptr, 16)
//         );
//         out.push_back(byte);
//     }
//     return out;
// }

// MiningResponseWriter Implementation

MiningResponseWriter::MiningResponseWriter(size_t max_queue_size)
    : max_queue_size_(max_queue_size)
    , egress_mode_()
    , push_host_()
    , push_port_(0)
    , save_dir_()
    , proxy_enable_(false)
    , proxy_host_()
    , proxy_port_(0)
    , save_to_disk_(false)
    , running_(false)
{
    // ---------------------------------------------------------- mode
    // POW_EGRESS_MODE drives the entire downstream config. Parse and
    // validate it BEFORE touching any other env var so a misconfigured
    // node fails loudly at construction rather than silently leaking
    // solutions to Core Node from a worker that thinks it's in broker
    // mode (see the egress-mode contract in pow_zmq_writer.h).
    egress_mode_ = get_env_var("POW_EGRESS_MODE", kEgressModeLocalMiner);
    if (egress_mode_ != kEgressModeLocalMiner &&
        egress_mode_ != kEgressModeBroker) {
        throw std::invalid_argument(
            "POW_EGRESS_MODE=" + egress_mode_ +
            " is not supported; must be 'local_miner' or 'broker'");
    }

    // -------------------------------------------------- destinations
    // local_miner: primary defaults to localhost:7000 (Core Node).
    // broker:      primary defaults to 127.0.0.1:7002 (miner-proxy
    // ProofCollector). Operator env overrides win in both modes,
    // subject to the broker-mode safety nets below.
    const std::string default_host =
        (egress_mode_ == kEgressModeBroker) ? "127.0.0.1" : "localhost";
    const std::string default_port_str =
        (egress_mode_ == kEgressModeBroker) ? "7002" : "7000";
    push_host_ = get_env_var("ZMQ_PUSH_HOST", default_host);
    push_port_ = std::stoi(get_env_var("ZMQ_PUSH_PORT", default_port_str));

    save_dir_ = get_env_var("PROOF_SAVE_DIR", "/data/pow_proofs");

    // ------------------------------------------- proxy / dual-publish
    const std::string proxy_enable_raw = get_env_var("POW_PROXY_ENABLE", "0");
    const bool proxy_enable_truthy = is_truthy_proxy_enable(proxy_enable_raw);
    if (egress_mode_ == kEgressModeBroker) {
        // Safety net 1: POW_PROXY_ENABLE must be falsy. A dual-publish
        // path would let solutions reach Core Node without the
        // broker's lease being closed — exactly the leak this slice
        // exists to prevent.
        if (proxy_enable_truthy) {
            throw PowEgressConfigError(
                "POW_EGRESS_MODE=broker is incompatible with "
                "POW_PROXY_ENABLE=" + proxy_enable_raw +
                "; broker mode has no dual-publish path. Set "
                "POW_PROXY_ENABLE=false or switch to "
                "POW_EGRESS_MODE=local_miner.");
        }
        // Safety net 2: refuse if the primary host looks like a Core
        // Node service. The common misconfiguration is flipping
        // POW_EGRESS_MODE=broker but leaving ZMQ_PUSH_HOST pointing
        // at the previous Core Node DNS name.
        if (looks_like_core_node(push_host_)) {
            throw PowEgressConfigError(
                "POW_EGRESS_MODE=broker but ZMQ_PUSH_HOST=" + push_host_ +
                " looks like a Core Node destination. The primary "
                "destination in broker mode must be the miner-proxy "
                "ProofCollector. Set ZMQ_PUSH_HOST to a non-Core-Node "
                "hostname (default 127.0.0.1).");
        }
        proxy_enable_ = false;
        proxy_host_   = "";
        proxy_port_   = 0;
    } else {
        proxy_enable_ = proxy_enable_truthy;
        proxy_host_   = get_env_var("POW_PROXY_PUSH_HOST", "localhost");
        proxy_port_   = std::stoi(get_env_var("POW_PROXY_PUSH_PORT", "7002"));
    }

    const std::string save_raw = get_env_var("POW_SAVE_TO_DISK", "0");
    save_to_disk_ = (save_raw == "1" || save_raw == "true" || save_raw == "True");

    stats_atomic_.push_host = push_host_;
    stats_atomic_.push_port = push_port_;
    stats_atomic_.max_queue_size = max_queue_size_;

    if (save_to_disk_) {
        std::filesystem::create_directories(save_dir_);
    }

    std::cout << "MiningResponseWriter configured: egress_mode="
              << egress_mode_ << " primary=" << push_host_ << ":"
              << push_port_ << " proxy_enable="
              << (proxy_enable_ ? "true" : "false") << std::endl;
}

MiningResponseWriter::~MiningResponseWriter() {
    stop();
}

bool MiningResponseWriter::start() {
    if (running_.load()) {
        std::cerr << "Mining response writer already running" << std::endl;
        return false;
    }
    
    running_.store(true);
    stats_atomic_.running.store(true);
    
    writer_thread_ = std::make_unique<std::thread>(&MiningResponseWriter::writer_loop, this);
    
    std::cout << "Mining response writer started on " << push_host_ << ":" << push_port_ << std::endl;
    return true;
}

void MiningResponseWriter::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping mining response writer..." << std::endl;
    running_.store(false);
    stats_atomic_.running.store(false);
    
    // Wake up the writer thread
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
    }
    
    if (writer_thread_ && writer_thread_->joinable()) {
        writer_thread_->join();
    }
    
    std::cout << "Mining response writer stopped" << std::endl;
}

bool MiningResponseWriter::submit_response(int64_t req_id,
                                         uint64_t nonce,
                                         uint32_t adjusted_bits,
                                         const std::vector<uint8_t>& pow_blob_hash,
                                         uint32_t difficulty,
                                         const std::unordered_map<std::string, std::any>& proof_dict,
                                         bool proxy_only) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    if (response_queue_.size() >= max_queue_size_) {
        std::cerr << "Response queue full, dropping mining response" << std::endl;
        stats_atomic_.messages_failed.fetch_add(1);
        return false;
    }
    
    auto response = std::make_unique<MiningResponse>(
        req_id, nonce, adjusted_bits, pow_blob_hash, difficulty, proof_dict, proxy_only
    );
    
    response_queue_.push(std::move(response));
    stats_atomic_.queue_size.store(response_queue_.size());
    
    queue_cv_.notify_one();
    
    std::cout << "Queued mining response for req_id=" << req_id 
              << ", nonce=" << nonce << std::endl;
    return true;
}

bool MiningResponseWriter::is_proxy_enabled() const {
    return proxy_enable_;
}

bool MiningResponseWriter::is_broker_mode() const {
    return egress_mode_ == kEgressModeBroker;
}

void MiningResponseWriter::set_connection(const std::string& host, int port) {
    push_host_ = host;
    push_port_ = port;
    stats_atomic_.push_host = host;
    stats_atomic_.push_port = port;
}

void MiningResponseWriter::get_status(ZmqWriterStats &out) const {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    out.messages_sent    = stats_atomic_.messages_sent.load();
    out.messages_failed  = stats_atomic_.messages_failed.load();
    out.queue_size       = response_queue_.size();
    out.running          = stats_atomic_.running.load();
    out.push_host        = stats_atomic_.push_host;
    out.push_port        = stats_atomic_.push_port;
    out.max_queue_size   = stats_atomic_.max_queue_size;
}


void MiningResponseWriter::writer_loop() {
    std::cout << "Mining response writer thread started" << std::endl;
    
    try {
        // Create ZMQ context and socket
        zmq_context_ = std::make_unique<zmq::context_t>(1);
        zmq_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PUSH);
        
        // Set socket options (compatible with both old and new ZMQ API)
        int hwm = 1000;
        int linger = 1000;
        #ifdef ZMQ_VERSION_MAJOR
            #if ZMQ_VERSION_MAJOR >= 4
                // Modern API (ZMQ 4.x+)
                zmq_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
                zmq_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
            #else
                // Old API (ZMQ 3.x)
                zmq_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
                zmq_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
            #endif
        #else
            // Fallback for very old versions
            zmq_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
            zmq_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
        #endif
        
        std::string address = "tcp://" + push_host_ + ":" + std::to_string(push_port_);
        zmq_socket_->connect(address);
        
        std::cout << "ZMQ connected to " << address << std::endl;
        
        // Optional proxy socket for audit trail
        if (proxy_enable_) {
            proxy_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PUSH);
            #ifdef ZMQ_VERSION_MAJOR
                #if ZMQ_VERSION_MAJOR >= 4
                    proxy_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
                    proxy_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
                #else
                    proxy_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
                    proxy_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
                #endif
            #else
                proxy_socket_->setsockopt(ZMQ_SNDHWM, &hwm, sizeof(hwm));
                proxy_socket_->setsockopt(ZMQ_LINGER, &linger, sizeof(linger));
            #endif
            
            std::string proxy_address = "tcp://" + proxy_host_ + ":" + std::to_string(proxy_port_);
            proxy_socket_->connect(proxy_address);
            
            std::cout << "Proxy ZMQ connected to " << proxy_address << " for audit trail" << std::endl;
        }
        
        while (running_.load()) {
            std::unique_ptr<MiningResponse> response;
            
            // Get response from queue
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                if (queue_cv_.wait_for(lock, std::chrono::seconds(1), 
                    [this] { return !response_queue_.empty() || !running_.load(); })) {
                    
                    if (!response_queue_.empty()) {
                        response = std::move(response_queue_.front());
                        response_queue_.pop();
                        stats_atomic_.queue_size.store(response_queue_.size());
                    }
                }
            }
            
            if (!response) {
                continue;
            }
            
            try {
                // Serialize and send
                auto fb_data = serialize_response(*response);

                // Routing depends on egress mode:
                //   broker      -> single destination (primary IS the
                //                  ProofCollector); proxy_only is a
                //                  no-op effect because there is no
                //                  separate audit channel.
                //   local_miner -> preserve legacy split: solutions go
                //                  to primary (Core Node) and optionally
                //                  also to proxy; audit (proxy_only)
                //                  goes ONLY to proxy.
                if (egress_mode_ == kEgressModeBroker) {
                    zmq::message_t message(fb_data.size());
                    std::memcpy(message.data(), fb_data.data(), fb_data.size());
                    zmq_socket_->send(message, zmq::send_flags::dontwait);

                    if (save_to_disk_ && !response->proxy_only) {
                        save_to_disk(fb_data, response->req_id, response->nonce, response->proof_dict);
                    }
                } else if (response->proxy_only) {
                    // Send ONLY to proxy for audit (non-solutions)
                    if (proxy_socket_) {
                        zmq::message_t message(fb_data.size());
                        std::memcpy(message.data(), fb_data.data(), fb_data.size());
                        proxy_socket_->send(message, zmq::send_flags::dontwait);
                        std::cout << "Sent proof to proxy only for audit: req_id=" << response->req_id << std::endl;
                    }
                } else {
                    // This is a SOLUTION - send to core-node
                    zmq::message_t message(fb_data.size());
                    std::memcpy(message.data(), fb_data.data(), fb_data.size());
                    zmq_socket_->send(message, zmq::send_flags::dontwait);

                    // Also send to proxy if configured (dual-publish for solutions)
                    if (proxy_socket_) {
                        zmq::message_t pmsg(fb_data.size());
                        std::memcpy(pmsg.data(), fb_data.data(), fb_data.size());
                        proxy_socket_->send(pmsg, zmq::send_flags::dontwait);
                    }

                    // Save solutions to disk (not audit proofs)
                    if (save_to_disk_) {
                        save_to_disk(fb_data, response->req_id, response->nonce, response->proof_dict);
                    }
                }
                
                stats_atomic_.messages_sent.fetch_add(1);
                std::cout << "Sent mining response: req_id=" << response->req_id
                         << ", nonce=" << response->nonce << std::endl;
                
            } catch (const zmq::error_t& e) {
                std::cerr << "ZMQ send error: " << e.what() << std::endl;
                stats_atomic_.messages_failed.fetch_add(1);
            } catch (const std::exception& e) {
                std::cerr << "Error sending mining response: " << e.what() << std::endl;
                stats_atomic_.messages_failed.fetch_add(1);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal writer error: " << e.what() << std::endl;
    }
    
    // Clean up resources
    zmq_socket_.reset();
    proxy_socket_.reset();
    zmq_context_.reset();
    
    std::cout << "Mining response writer thread stopped" << std::endl;
}

std::vector<uint8_t> MiningResponseWriter::serialize_response(
    const MiningResponse &response
) {
    // Option 1: Use libproofpack for consistent serialization
    // This would be cleaner but requires including libproofpack.h
    // For now, keep the existing implementation with safe extractors
    
    // 1) FlatBuffer builder
    flatbuffers::FlatBufferBuilder builder(4096);
    
    // Extract completion_id if available
    std::string completion_id_str;
    if (response.proof_dict.count("completion_id")) {
        try {
            completion_id_str = std::any_cast<std::string>(response.proof_dict.at("completion_id"));
        } catch (...) {
            completion_id_str = "";
        }
    }
    
    // IMPORTANT: Create all string offsets FIRST before any vectors
    // This is a FlatBuffer requirement to avoid nested construction assertions
    // Use safe extraction for strings with defaults
    std::string model_id = "unknown";
    std::string compute_prec = "fp32";
    std::string ipfs_cid = "";
    std::string extra_flags = "";
    
    try {
        if (response.proof_dict.count("model_identifier")) {
            model_id = std::any_cast<std::string>(response.proof_dict.at("model_identifier"));
        }
    } catch (...) {}
    
    try {
        if (response.proof_dict.count("compute_precision")) {
            compute_prec = std::any_cast<std::string>(response.proof_dict.at("compute_precision"));
        }
    } catch (...) {}
    
    try {
        if (response.proof_dict.count("ipfs_cid")) {
            ipfs_cid = std::any_cast<std::string>(response.proof_dict.at("ipfs_cid"));
        }
    } catch (...) {}
    
    try {
        if (response.proof_dict.count("model_config_diff")) {
            extra_flags = std::any_cast<std::string>(response.proof_dict.at("model_config_diff"));
        }
    } catch (...) {}
    
    auto model_id_offset = builder.CreateString(model_id);
    auto compute_prec_offset = builder.CreateString(compute_prec);
    auto ipfs_offset = builder.CreateString(ipfs_cid);
    auto extra_flags_offset = builder.CreateString(extra_flags);

    // 2) Convert fields to raw bytes (handles both hex strings and raw bytes)
    // Use safe extraction that handles both string and vector<uint8_t> types
    auto target_bytes      = extract_or_convert_bytes(response.proof_dict.at("target"));
    auto vdf_bytes         = extract_or_convert_bytes(response.proof_dict.at("vdf"));
    auto hash_bytes        = extract_or_convert_bytes(response.proof_dict.at("hash"));
    auto block_hash_bytes  = extract_or_convert_bytes(response.proof_dict.at("block_hash"));
    auto header_pref_bytes = extract_or_convert_bytes(response.proof_dict.at("header_prefix"));

    // 3) Create FlatBuffers vectors for those
    auto target_vec      = builder.CreateVector(target_bytes);
    auto vdf_vec         = builder.CreateVector(vdf_bytes);
    auto hash_vec        = builder.CreateVector(hash_bytes);
    auto block_hash_vec  = builder.CreateVector(block_hash_bytes);
    auto header_pref_vec = builder.CreateVector(header_pref_bytes);

    // 4) Grab 1D numeric arrays from proof_dict (with type flexibility)
    auto chosen_tokens      = extract_vector_flexible<int32_t>(response.proof_dict.at("chosen_tokens"));
    auto chosen_probs       = extract_vector_flexible<float>(response.proof_dict.at("chosen_probs"));
    auto sampling_u         = extract_vector_flexible<float>(response.proof_dict.at("sampling_u"));
    auto softmax_norm       = extract_vector_flexible<float>(response.proof_dict.at("softmax_normalizers"));
    auto prompt_tokens      = extract_vector_flexible<int32_t>(response.proof_dict.at("prompt_tokens"));
    
    // Handle both pad_mask and attention_mask names
    std::vector<bool> attention_mask;
    if (response.proof_dict.count("pad_mask")) {
        attention_mask = extract_vector_flexible<bool>(response.proof_dict.at("pad_mask"));
    } else if (response.proof_dict.count("attention_mask")) {
        attention_mask = extract_vector_flexible<bool>(response.proof_dict.at("attention_mask"));
    } else {
        // Default empty mask
        attention_mask = std::vector<bool>();
    }

    // Convert int32_t to uint32_t for FlatBuffers
    std::vector<uint32_t> chosen_tokens_u32(chosen_tokens.begin(), chosen_tokens.end());
    std::vector<uint32_t> prompt_tokens_u32(prompt_tokens.begin(), prompt_tokens.end());

    auto chosen_tokens_vec   = builder.CreateVector(chosen_tokens_u32);
    auto chosen_probs_vec    = builder.CreateVector(chosen_probs);
    auto sampling_u_vec      = builder.CreateVector(sampling_u);
    auto softmax_norm_vec    = builder.CreateVector(softmax_norm);
    auto prompt_tokens_vec   = builder.CreateVector(prompt_tokens_u32);
    // FlatBuffers wants uint8_t for bool arrays:
    std::vector<uint8_t> mask_bytes(attention_mask.begin(), attention_mask.end());
    auto pad_mask_vec        = builder.CreateVector(mask_bytes);

    auto make_float_array = [&](const std::vector<float> &row) {
        auto values_vec = builder.CreateVector(row);
        return proof::CreateFloatArray(builder, values_vec);
    };
    auto make_uint_array = [&](const std::vector<int32_t> &row) {
        std::vector<uint32_t> tmp(row.begin(), row.end());
        auto values_vec = builder.CreateVector(tmp);
        return proof::CreateUIntArray(builder, values_vec);
    };

    auto topk_logits_mat = extract_matrix_flexible<float>(response.proof_dict.at("topk_logits"));
    std::vector<flatbuffers::Offset<proof::FloatArray>> topk_logits_offsets;
    topk_logits_offsets.reserve(topk_logits_mat.size());
    for (auto &r : topk_logits_mat) topk_logits_offsets.push_back(make_float_array(r));
    auto topk_logits_vec = builder.CreateVector(topk_logits_offsets);

    auto topk_idx_mat = extract_matrix_flexible<int32_t>(response.proof_dict.at("topk_indices"));
    std::vector<flatbuffers::Offset<proof::UIntArray>> topk_idx_offsets;
    topk_idx_offsets.reserve(topk_idx_mat.size());
    for (auto &r : topk_idx_mat) topk_idx_offsets.push_back(make_uint_array(r));
    auto topk_idx_vec = builder.CreateVector(topk_idx_offsets);

    auto lse_mat = extract_matrix_flexible<float>(response.proof_dict.at("logsumexp_stats"));
    std::vector<flatbuffers::Offset<proof::FloatArray>> lse_offsets;
    lse_offsets.reserve(lse_mat.size());
    for (auto &r : lse_mat) lse_offsets.push_back(make_float_array(r));
    auto logsumexp_vec = builder.CreateVector(lse_offsets);

    // 6) Build the Proof table
    proof::ProofBuilder pb(builder);
    // Version comes from the proof dict (stamped from
    // ProofProcessor::proof_version_) — never hardcoded, so version=3 proofs
    // (TIP-0003) serialize with the right wire version. Missing /
    // malformed entry keeps the legacy value 2.
    uint32_t proof_version_wire = 2;
    if (response.proof_dict.count("version")) {
        proof_version_wire = extract_numeric_flexible<uint32_t>(
            response.proof_dict.at("version"), 2);
    }
    pb.add_version(proof_version_wire);
    pb.add_tick(extract_numeric_flexible<int64_t>(response.proof_dict.at("tick")));
    pb.add_timestamp(extract_numeric_flexible<int64_t>(response.proof_dict.at("timestamp")));
    // Slice 11.4 — read is_solution from the proof_dict instead of
    // hardcoding true. The dict carries a C++ bool stamped by
    // proof_processor.cpp (assemble_proof_dict, line ~261). Sub-
    // block share emissions go through with is_solution=false; the
    // worker-side _on_solution_received classifier reads this bit
    // and routes the FlatBuffer to MineShare instead of MineResult.
    bool is_solution_flag = true;  // safe default (block path)
    try {
        const auto& v = response.proof_dict.at("is_solution");
        if (v.type() == typeid(bool)) {
            is_solution_flag = std::any_cast<bool>(v);
        } else {
            // Tolerate int / numeric encoding from older callers.
            is_solution_flag = extract_numeric_flexible<int>(v, 1) != 0;
        }
    } catch (const std::out_of_range&) {
        // No is_solution key — keep legacy default (treat as block
        // solution) so a misconfig never silently drops a block hit.
    } catch (...) {
        // Bad cast — same fallback rationale as above.
    }
    pb.add_is_solution(is_solution_flag);
    pb.add_model_identifier(model_id_offset);
    pb.add_compute_precision(compute_prec_offset);
    pb.add_ipfs_cid(ipfs_offset);
    pb.add_extra_flags(extra_flags_offset);
    pb.add_temperature(extract_numeric_flexible<float>(response.proof_dict.at("temperature")));
    pb.add_top_p(extract_numeric_flexible<float>(response.proof_dict.at("top_p")));
    pb.add_top_k(extract_numeric_flexible<uint32_t>(response.proof_dict.at("top_k")));
    pb.add_repetition_penalty(extract_numeric_flexible<float>(response.proof_dict.at("repetition_penalty")));
    pb.add_target(target_vec);
    pb.add_vdf(vdf_vec);
    pb.add_hash(hash_vec);
    pb.add_block_hash(block_hash_vec);
    pb.add_header_prefix(header_pref_vec);
    pb.add_chosen_tokens(chosen_tokens_vec);
    pb.add_chosen_probs(chosen_probs_vec);
    pb.add_sampling_u(sampling_u_vec);
    pb.add_softmax_normalizers(softmax_norm_vec);
    pb.add_prompt_tokens(prompt_tokens_vec);
    pb.add_pad_mask(pad_mask_vec);
    pb.add_topk_logits(topk_logits_vec);
    pb.add_topk_indices(topk_idx_vec);
    pb.add_logsumexp_stats(logsumexp_vec);
    auto proof_off = pb.Finish();

    // 7) Build the MiningResponse table
    auto pow_blob_hash_vec = builder.CreateVector(response.pow_blob_hash);
    auto completion_id_off = builder.CreateString(completion_id_str);
    
    proof::MiningResponseBuilder rb(builder);
    rb.add_req_id(response.req_id);
    rb.add_nonce(response.nonce);
    rb.add_adjusted_bits(response.adjusted_bits);
    rb.add_pow_blob_hash(pow_blob_hash_vec);
    rb.add_difficulty(response.difficulty);
    rb.add_pow_blob(proof_off);
    rb.add_completion_id(completion_id_off);
    auto resp_off = rb.Finish();

    // 8) Finish and return raw bytes
    builder.Finish(resp_off, "PROF");
    uint8_t *buf = builder.GetBufferPointer();
    size_t   sz  = builder.GetSize();
    return std::vector<uint8_t>(buf, buf + sz);
}

std::string MiningResponseWriter::to_string(const std::any& value) {
    try {
        if (value.type() == typeid(std::string)) {
            return std::any_cast<std::string>(value);
        } else if (value.type() == typeid(int)) {
            return std::to_string(std::any_cast<int>(value));
        } else if (value.type() == typeid(float)) {
            return std::to_string(std::any_cast<float>(value));
        } else if (value.type() == typeid(double)) {
            return std::to_string(std::any_cast<double>(value));
        } else if (value.type() == typeid(bool)) {
            return std::any_cast<bool>(value) ? "true" : "false";
        }
    } catch (const std::bad_any_cast&) {
        // Fall through to default
    }
    return "unknown";
}

void MiningResponseWriter::save_to_disk(const std::vector<uint8_t>& data,
                                      int64_t req_id, uint64_t nonce,
                                      const std::unordered_map<std::string, std::any>& proof_dict) {
    try {
        // Extract hash from proof_dict for unique filename
        std::string hash_suffix = "";
        try {
            auto hash_bytes = extract_or_convert_bytes(proof_dict.at("hash"));
            if (hash_bytes.size() >= 4) {
                // Convert first 4 bytes to hex string (8 hex chars)
                std::stringstream ss;
                ss << std::hex << std::setfill('0');
                for (size_t i = 0; i < 4 && i < hash_bytes.size(); ++i) {
                    ss << std::setw(2) << static_cast<unsigned>(hash_bytes[i]);
                }
                hash_suffix = "_" + ss.str();
            }
        } catch (...) {
            // If hash extraction fails, use timestamp for uniqueness
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()).count();
            hash_suffix = "_" + std::to_string(timestamp);
        }

        std::string filename = std::to_string(req_id) + "_" +
                              std::to_string(nonce) +
                              hash_suffix + ".bin";
        std::string filepath = save_dir_ + "/" + filename;

        std::ofstream file(filepath, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size());
            std::cout << "Saved solution to: " << filename << std::endl;
        }
    } catch (...) {
        // Silent failure for disk operations
    }
}

// FlatBuffers helper methods implementation
flatbuffers::Offset<flatbuffers::Vector<uint32_t>> 
MiningResponseWriter::create_uint32_vector(flatbuffers::FlatBufferBuilder& builder, 
                                         const std::vector<uint32_t>& data) {
    return builder.CreateVector(data);
}

flatbuffers::Offset<flatbuffers::Vector<float>> 
MiningResponseWriter::create_float32_vector(flatbuffers::FlatBufferBuilder& builder, 
                                          const std::vector<float>& data) {
    return builder.CreateVector(data);
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>> 
MiningResponseWriter::create_bool_vector(flatbuffers::FlatBufferBuilder& builder, 
                                        const std::vector<bool>& data) {
    std::vector<uint8_t> byte_data;
    byte_data.reserve(data.size());
    for (bool b : data) {
        byte_data.push_back(b ? 1 : 0);
    }
    return builder.CreateVector(byte_data);
}

flatbuffers::Offset<proof::FloatArray>
MiningResponseWriter::create_float_array(flatbuffers::FlatBufferBuilder& builder,
                                        const std::vector<float>& row_data) {
    auto values_vec = builder.CreateVector(row_data);
    return proof::CreateFloatArray(builder, values_vec);
}

flatbuffers::Offset<proof::UIntArray>
MiningResponseWriter::create_uint_array(flatbuffers::FlatBufferBuilder& builder,
                                       const std::vector<uint32_t>& row_data) {
    auto values_vec = builder.CreateVector(row_data);
    return proof::CreateUIntArray(builder, values_vec);
}

// Helper methods for extracting data from std::any with type conversion
std::vector<uint32_t> MiningResponseWriter::extract_or_convert_uint32_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key) {
    auto it = dict.find(key);
    if (it == dict.end()) {
        return {};
    }
    
    try {
        // Try direct cast first
        return std::any_cast<std::vector<uint32_t>>(it->second);
    } catch (const std::bad_any_cast&) {
        try {
            // Try int vector
            auto int_vec = std::any_cast<std::vector<int>>(it->second);
            std::vector<uint32_t> result;
            result.reserve(int_vec.size());
            for (int val : int_vec) {
                result.push_back(static_cast<uint32_t>(val));
            }
            return result;
        } catch (const std::bad_any_cast&) {
            try {
                // Try float vector
                auto float_vec = std::any_cast<std::vector<float>>(it->second);
                std::vector<uint32_t> result;
                result.reserve(float_vec.size());
                for (float val : float_vec) {
                    result.push_back(static_cast<uint32_t>(val));
                }
                return result;
            } catch (const std::bad_any_cast&) {
                return {};
            }
        }
    }
}

std::vector<float> MiningResponseWriter::extract_or_convert_float_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key) {
    auto it = dict.find(key);
    if (it == dict.end()) {
        return {};
    }
    
    try {
        return std::any_cast<std::vector<float>>(it->second);
    } catch (const std::bad_any_cast&) {
        try {
            auto double_vec = std::any_cast<std::vector<double>>(it->second);
            std::vector<float> result;
            result.reserve(double_vec.size());
            for (double val : double_vec) {
                result.push_back(static_cast<float>(val));
            }
            return result;
        } catch (const std::bad_any_cast&) {
            try {
                auto int_vec = std::any_cast<std::vector<int>>(it->second);
                std::vector<float> result;
                result.reserve(int_vec.size());
                for (int val : int_vec) {
                    result.push_back(static_cast<float>(val));
                }
                return result;
            } catch (const std::bad_any_cast&) {
                return {};
            }
        }
    }
}

std::vector<bool> MiningResponseWriter::extract_or_convert_bool_vector(const std::unordered_map<std::string, std::any>& dict, const std::string& key) {
    auto it = dict.find(key);
    if (it == dict.end()) {
        return {};
    }
    
    try {
        return std::any_cast<std::vector<bool>>(it->second);
    } catch (const std::bad_any_cast&) {
        try {
            auto int_vec = std::any_cast<std::vector<int>>(it->second);
            std::vector<bool> result;
            result.reserve(int_vec.size());
            for (int val : int_vec) {
                result.push_back(val != 0);
            }
            return result;
        } catch (const std::bad_any_cast&) {
            return {};
        }
    }
}

std::vector<std::vector<float>> MiningResponseWriter::extract_or_convert_float_matrix(const std::unordered_map<std::string, std::any>& dict, const std::string& key) {
    auto it = dict.find(key);
    if (it == dict.end()) {
        return {};
    }
    
    try {
        return std::any_cast<std::vector<std::vector<float>>>(it->second);
    } catch (const std::bad_any_cast&) {
        return {};
    }
}

std::vector<std::vector<uint32_t>> MiningResponseWriter::extract_or_convert_uint32_matrix(const std::unordered_map<std::string, std::any>& dict, const std::string& key) {
    auto it = dict.find(key);
    if (it == dict.end()) {
        return {};
    }
    
    try {
        return std::any_cast<std::vector<std::vector<uint32_t>>>(it->second);
    } catch (const std::bad_any_cast&) {
        try {
            auto int_matrix = std::any_cast<std::vector<std::vector<int>>>(it->second);
            std::vector<std::vector<uint32_t>> result;
            result.reserve(int_matrix.size());
            for (const auto& row : int_matrix) {
                std::vector<uint32_t> converted_row;
                converted_row.reserve(row.size());
                for (int val : row) {
                    converted_row.push_back(static_cast<uint32_t>(val));
                }
                result.push_back(std::move(converted_row));
            }
            return result;
        } catch (const std::bad_any_cast&) {
            return {};
        }
    }
}

// MiningResponseSubmitter Implementation

MiningResponseSubmitter::MiningResponseSubmitter() {
    writer_ = std::make_unique<MiningResponseWriter>();
    writer_->start();
}

MiningResponseSubmitter::~MiningResponseSubmitter() {
    if (writer_) {
        writer_->stop();
    }
}

bool MiningResponseSubmitter::submit_proof_for_audit(int64_t req_id,
                                                    const std::unordered_map<std::string, std::any>& proof_dict) {
    // Broker mode: the single destination IS the miner-proxy
    // ProofCollector. Send via the primary socket (proxy_only=false so
    // the broker-mode router at pow_zmq_writer.cpp uses zmq_socket_).
    // What keeps audit proofs out of mining is the proof_purpose=audit
    // marker in extra_flags, which the ProofCollector branches on
    // before its mining filters — NOT topology. Mirrors the Python
    // writer's submit_proof_for_audit.
    if (writer_->is_broker_mode()) {
        return writer_->submit_response(req_id, 0, 0,
                                      std::vector<uint8_t>{}, 0, proof_dict, false);
    }

    // local_miner: only send to the proxy channel if enabled.
    if (!writer_->is_proxy_enabled()) {
        return true;  // No-op if proxy not enabled
    }

    // Submit proof only to proxy (not to core-node)
    // Use dummy values for fields only needed for solutions
    return writer_->submit_response(req_id, 0, 0,
                                  std::vector<uint8_t>{}, 0, proof_dict, true);  // proxy_only=true
}

bool MiningResponseSubmitter::submit_solution(int64_t req_id,
                                            uint64_t nonce,
                                            uint32_t adjusted_bits,
                                            const std::vector<uint8_t>& pow_blob_hash,
                                            uint32_t difficulty,
                                            const std::unordered_map<std::string, std::any>& proof_dict) {
    return writer_->submit_response(req_id, nonce, adjusted_bits,
                                  pow_blob_hash, difficulty, proof_dict, false);  // proxy_only=false
}

bool MiningResponseSubmitter::submit_share(int64_t req_id,
                                          uint64_t nonce,
                                          uint32_t adjusted_bits,
                                          const std::vector<uint8_t>& pow_blob_hash,
                                          uint32_t difficulty,
                                          const std::unordered_map<std::string, std::any>& proof_dict) {
    // Slice 11.4 — share emission has a consumer only in broker
    // mode (the broker's miner-proxy ProofCollector). In
    // local_miner mode the upstream is a Core Node which won't
    // accept sub-block proofs, so we silently no-op to avoid
    // injecting invalid traffic.
    if (!writer_->is_broker_mode()) {
        return true;
    }
    // proxy_only=false: in broker mode the writer routes to the
    // single egress (miner-proxy) regardless of the flag. The
    // worker disambiguates block-vs-share on Proof.is_solution,
    // which the serializer reads from proof_dict (see
    // pow_zmq_writer.cpp around line 537).
    return writer_->submit_response(req_id, nonce, adjusted_bits,
                                  pow_blob_hash, difficulty, proof_dict, false);
}

bool MiningResponseSubmitter::is_broker_mode() const {
    return writer_->is_broker_mode();
}

ZmqWriterStats MiningResponseSubmitter::get_stats() const {
    ZmqWriterStats stats;
    writer_->get_status(stats);
    return stats;
}

void MiningResponseSubmitter::set_connection(const std::string& host, int port) {
    writer_->set_connection(host, port);
}

// Utility Functions Implementation

std::vector<float> extract_float_vector(const std::any& value) {
    try {
        return std::any_cast<std::vector<float>>(value);
    } catch (const std::bad_any_cast&) {
        // Try to convert from other numeric types
        try {
            auto int_vec = std::any_cast<std::vector<int>>(value);
            std::vector<float> float_vec;
            for (int i : int_vec) {
                float_vec.push_back(static_cast<float>(i));
            }
            return float_vec;
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Cannot convert value to vector<float>");
        }
    }
}

std::vector<int> extract_int_vector(const std::any& value) {
    try {
        return std::any_cast<std::vector<int>>(value);
    } catch (const std::bad_any_cast&) {
        // Try to convert from other types
        try {
            auto float_vec = std::any_cast<std::vector<float>>(value);
            std::vector<int> int_vec;
            for (float f : float_vec) {
                int_vec.push_back(static_cast<int>(f));
            }
            return int_vec;
        } catch (const std::bad_any_cast&) {
            throw std::runtime_error("Cannot convert value to vector<int>");
        }
    }
}

std::vector<bool> extract_bool_vector(const std::any& value) {
    try {
        return std::any_cast<std::vector<bool>>(value);
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot convert value to vector<bool>");
    }
}

std::vector<std::vector<float>> extract_float_matrix(const std::any& value) {
    try {
        return std::any_cast<std::vector<std::vector<float>>>(value);
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot convert value to vector<vector<float>>");
    }
}

std::vector<std::vector<int>> extract_int_matrix(const std::any& value) {
    try {
        return std::any_cast<std::vector<std::vector<int>>>(value);
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot convert value to vector<vector<int>>");
    }
}

std::string extract_string(const std::any& value) {
    try {
        return std::any_cast<std::string>(value);
    } catch (const std::bad_any_cast&) {
        throw std::runtime_error("Cannot convert value to string");
    }
}

template<typename T>
T extract_numeric(const std::any& value) {
    try {
        return std::any_cast<T>(value);
    } catch (const std::bad_any_cast&) {
        // Try common numeric conversions
        try {
            if constexpr (std::is_same_v<T, float>) {
                if (value.type() == typeid(double)) {
                    return static_cast<T>(std::any_cast<double>(value));
                } else if (value.type() == typeid(int)) {
                    return static_cast<T>(std::any_cast<int>(value));
                }
            } else if constexpr (std::is_same_v<T, double>) {
                if (value.type() == typeid(float)) {
                    return static_cast<T>(std::any_cast<float>(value));
                } else if (value.type() == typeid(int)) {
                    return static_cast<T>(std::any_cast<int>(value));
                }
            } else if constexpr (std::is_same_v<T, int>) {
                if (value.type() == typeid(float)) {
                    return static_cast<T>(std::any_cast<float>(value));
                } else if (value.type() == typeid(double)) {
                    return static_cast<T>(std::any_cast<double>(value));
                }
            }
        } catch (const std::bad_any_cast&) {
            // Fall through
        }
        throw std::runtime_error("Cannot convert value to numeric type");
    }
}

// Explicit template instantiations
template int extract_numeric<int>(const std::any& value);
template float extract_numeric<float>(const std::any& value);
template double extract_numeric<double>(const std::any& value);
