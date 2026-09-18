// SPDX-License-Identifier: Apache-2.0
#include "pow_utils.h"
#include "pow_v3.h"
#include "meow_pow_v4.h"       // proof-v4 window-root derivation (E4)
#include "gumbel_sampler.h"    // proof-v4 chain-bound Gumbel race draw
#include <openssl/sha.h>
#include <openssl/rand.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <filesystem>
#include <ctime>
#include <cctype>
#include <cstdint>
#include <vector>
#include <tuple>
#include <any>
#include <functional>
#include <climits>
#include <openssl/evp.h>
#include "proof_generated.h"
#include "blockheader_generated.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "any_map_dump.h"

namespace fs = std::filesystem;

// Efficient and precise FP16 conversion functions
inline uint16_t fp32_to_fp16_precise(float value) {
    union { float f; uint32_t u; } conv = {value};
    uint32_t f32 = conv.u;
    
    uint32_t sign = (f32 >> 31) & 0x1;
    uint32_t exp = (f32 >> 23) & 0xFF;
    uint32_t frac = f32 & 0x7FFFFF;
    
    // Special cases: inf/nan
    if (exp == 0xFF) {
        return (sign << 15) | 0x7C00 | (frac ? 0x200 : 0);
    }
    
    // Adjust exponent for FP16 bias (127 -> 15)
    int32_t new_exp = int32_t(exp) - 127 + 15;
    
    // Overflow to infinity
    if (new_exp >= 31) {
        return (sign << 15) | 0x7C00;
    }
    
    // Underflow handling
    if (new_exp <= 0) {
        // Complete underflow to zero
        if (new_exp < -10) {
            return sign << 15;
        }
        
        // Denormal case: shift mantissa right
        uint32_t mantissa = (frac | 0x800000) >> (1 - new_exp);
        
        // Round to nearest even
        uint32_t sticky_bit = (mantissa & 0x1FFF) ? 1 : 0;
        uint32_t guard_bit = (mantissa >> 12) & 1;
        
        mantissa >>= 13;
        if (guard_bit && (sticky_bit || (mantissa & 1))) {
            mantissa++;
        }
        
        return (sign << 15) | (mantissa & 0x3FF);
    }
    
    // Normal case
    uint32_t mantissa = frac >> 13;
    
    // Round to nearest even
    uint32_t round_bit = (frac >> 12) & 1;
    uint32_t sticky_bit = (frac & 0xFFF) ? 1 : 0;
    
    if (round_bit && (sticky_bit || (mantissa & 1))) {
        mantissa++;
        if (mantissa > 0x3FF) {
            mantissa = 0;
            new_exp++;
            if (new_exp >= 31) {
                return (sign << 15) | 0x7C00; // overflow to inf
            }
        }
    }
    
    return (sign << 15) | (new_exp << 10) | (mantissa & 0x3FF);
}

inline float fp16_to_fp32_precise(uint16_t fp16_bits) {
    uint32_t sign = (fp16_bits >> 15) & 0x1;
    uint32_t exp = (fp16_bits >> 10) & 0x1F;
    uint32_t frac = fp16_bits & 0x3FF;
    
    uint32_t f32_bits;
    
    if (exp == 0) {
        if (frac == 0) {
            // Zero
            f32_bits = sign << 31;
        } else {
            // Denormal - convert to normal
            // Find leading bit position
            uint32_t shift = 0;
            uint32_t tmp = frac;
            while ((tmp & 0x400) == 0) {
                tmp <<= 1;
                shift++;
            }
            
            uint32_t new_exp = 127 - 15 + 1 - shift;
            uint32_t new_frac = (tmp & 0x3FF) << 13;
            f32_bits = (sign << 31) | (new_exp << 23) | new_frac;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        f32_bits = (sign << 31) | 0x7F800000 | (frac << 13);
    } else {
        // Normal
        uint32_t new_exp = exp - 15 + 127;
        uint32_t new_frac = frac << 13;
        f32_bits = (sign << 31) | (new_exp << 23) | new_frac;
    }
    
    union { uint32_t u; float f; } result = {f32_bits};
    return result.f;
}

// Vectorized FP16 snapping function
inline void snap_logits_to_fp16_inplace(float* logits, int n_vocab) {
    for (int i = 0; i < n_vocab; ++i) {
        uint16_t fp16_val = fp32_to_fp16_precise(logits[i]);
        logits[i] = fp16_to_fp32_precise(fp16_val);
    }
}

// Updated precision snapping function
inline void snap_logits_to_precision_inplace(float* logits, int n_vocab, const std::string& precision) {
    if (precision == "bf16") {
        uint32_t* data = reinterpret_cast<uint32_t*>(logits);
        for (int i = 0; i < n_vocab; ++i) {
            uint32_t x = data[i];
            uint32_t rounding_bias = 0x00007FFF + ((x >> 16) & 1);
            data[i] = (x + rounding_bias) & 0xFFFF0000;
        }
    } else if (precision == "fp16") {
        snap_logits_to_fp16_inplace(logits, n_vocab);
    } else if (precision == "int8") {
        for (int i = 0; i < n_vocab; ++i) {
            int8_t i8 = (int8_t)std::round(std::clamp(logits[i] * 127.0f, -128.0f, 127.0f));
            logits[i] = i8 / 127.0f;
        }
    }
    // fp32: no-op
}

inline uint32_t get_compact(const std::vector<uint8_t>& target, bool negative = false) {
    if (target.size() != 32) {
        return 0;
    }
    
    // Convert bytes to integer (big endian) - find first non-zero byte
    int size = 0;
    uint64_t mantissa = 0;
    
    // Find the byte size (skip leading zeros)
    for (int i = 0; i < 32; i++) {
        if (target[i] != 0) {
            size = 32 - i;
            break;
        }
    }
    
    if (size == 0) {
        return 0;
    }
    
    // Extract mantissa based on size
    if (size <= 3) {
        // Shift target to fit into 3-byte mantissa
        int start_idx = 32 - size;
        for (int i = 0; i < size; i++) {
            mantissa = (mantissa << 8) | target[start_idx + i];
        }
        mantissa <<= (8 * (3 - size));
    } else {
        // Take the first 3 bytes of the significant part
        int start_idx = 32 - size;
        mantissa = (static_cast<uint64_t>(target[start_idx]) << 16) |
                   (static_cast<uint64_t>(target[start_idx + 1]) << 8) |
                   static_cast<uint64_t>(target[start_idx + 2]);
    }
    
    // If the sign bit (0x00800000) is set, shift mantissa down and bump exponent
    if (mantissa & 0x00800000) {
        mantissa >>= 8;
        size += 1;
    }
    
    // Compose compact: 1-byte exponent, 3-byte mantissa
    uint32_t compact = (size << 24) | (mantissa & 0x007fffff);
    if (negative && mantissa != 0) {
        compact |= 0x00800000;
    }
    
    return compact;
}

// Logger implementation
Logger::Logger(const std::string& log_dir) {
    std::string dir = log_dir.empty() ? get_env_var("MINER_LOG_DIR", "/data/miner_logs") : log_dir;
    fs::create_directories(dir);
    log_file_path = (fs::path(dir) / "pow_sampler.log").string();
}

void Logger::log(const std::string& message, const std::string& level) {
    try {
        std::ofstream file(log_file_path, std::ios::app);
        if (file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            file << "[" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") 
                 << "] [" << level << "] " << message << std::endl;
        }
    } catch (...) {
        // Silent failure
    }
}

// RowManager implementation
RowManager::RowManager(int max_rows) : max_rows(max_rows) {
    for (int i = 0; i < max_rows; ++i) {
        free_rows.push_back(i);
    }
}

std::optional<int> RowManager::get_row(int seq_id) {
    auto it = seqid_to_row.find(seq_id);
    if (it != seqid_to_row.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<int> RowManager::allocate_row(int seq_id) {
    if (seqid_to_row.find(seq_id) != seqid_to_row.end()) {
        return seqid_to_row[seq_id];
    }
    
    if (free_rows.empty()) {
        return std::nullopt;
    }
    
    int row = free_rows.front();
    free_rows.pop_front();
    seqid_to_row[seq_id] = row;
    allocation_order[seq_id] = next_allocation_id++;
    return row;
}

std::optional<int> RowManager::free_row(int seq_id) {
    auto it = seqid_to_row.find(seq_id);
    if (it != seqid_to_row.end()) {
        int row = it->second;
        seqid_to_row.erase(it);
        free_rows.push_back(row);
        allocation_order.erase(seq_id);
        return row;
    }
    return std::nullopt;
}

std::pair<std::optional<int>, std::optional<int>> RowManager::get_oldest_sequence(const std::vector<int32_t>& steps) {
    if (seqid_to_row.empty()) {
        return {std::nullopt, std::nullopt};
    }
    
    int oldest_seq = -1;
    int oldest_row = -1;
    int max_steps = -1;
    int oldest_alloc = INT_MAX;
    
    for (const auto& [seq_id, row] : seqid_to_row) {
        int step_count = steps[row];
        int alloc_order = allocation_order[seq_id];
        
        if (step_count > max_steps || (step_count == max_steps && alloc_order < oldest_alloc)) {
            oldest_seq = seq_id;
            oldest_row = row;
            max_steps = step_count;
            oldest_alloc = alloc_order;
        }
    }
    
    return {oldest_seq, oldest_row};
}

// RingBuffers implementation
RingBuffers::RingBuffers(int window_size, int max_rows) 
    : window_size(window_size), max_rows(max_rows) {
    
    // Initialize 3D arrays
    topk_logits.resize(window_size, std::vector<std::vector<float>>(max_rows, std::vector<float>(70, 0.0f)));
    topk_indices.resize(window_size, std::vector<std::vector<int32_t>>(max_rows, std::vector<int32_t>(70, 0)));
    logsumexp_stats.resize(window_size, std::vector<std::vector<float>>(max_rows, std::vector<float>(6, 0.0f)));
    
    // Initialize 2D arrays
    chosen_probs.resize(window_size, std::vector<float>(max_rows, 0.0f));
    chosen_tokens.resize(window_size, std::vector<int64_t>(max_rows, 0));
    attention_mask.resize(window_size, std::vector<bool>(max_rows, false));
    sampling_u.resize(window_size, std::vector<float>(max_rows, 0.0f));
    softmax_normalizers.resize(window_size, std::vector<float>(max_rows, 0.0f));
    
    // Initialize 1D array
    steps.resize(max_rows, 0);

    digest_buffer.assign(max_rows, std::vector<std::vector<uint8_t>>(window_size, std::vector<uint8_t>(32)));

    // v3 admission state (TIP-0003)
    admission_nonce.assign(max_rows, std::array<uint8_t, 32>{});
    admission_valid.assign(max_rows, 0);
    // v4 recorded nonce R (proof v4) — decoupled from the appended E4 above.
    v4_recorded_nonce.assign(max_rows, std::array<uint8_t, 32>{});
    v4_recorded_valid.assign(max_rows, 0);
}

void RingBuffers::write_admission_nonce(int row, const uint8_t* nonce32_or_null) {
    if (row < 0 || row >= max_rows) return;
    if (nonce32_or_null == nullptr) {
        admission_nonce[row].fill(0);
        admission_valid[row] = 0;
    } else {
        std::copy(nonce32_or_null, nonce32_or_null + 32, admission_nonce[row].begin());
        admission_valid[row] = 1;
    }
}

std::vector<int> RingBuffers::get_positions(const std::vector<int>& rows) {
    std::vector<int> positions;
    for (int row : rows) {
        positions.push_back(steps[row] % window_size);
    }
    return positions;
}

std::vector<std::vector<uint8_t>> RingBuffers::get_window_digests(int row) {
    std::vector<std::vector<uint8_t>> out;
    out.reserve(window_size);
    // steps[row] is how many tokens have been written so far
    for (int i = 0; i < window_size; ++i) {
        int idx = (steps[row] - window_size + i) % window_size;
        if (idx < 0) idx += window_size;
        out.push_back(digest_buffer[row][idx]);
    }
    return out;
}

void RingBuffers::clear_row(int row) {
    if (row < 0 || row >= max_rows) return;
    
    for (int i = 0; i < window_size; ++i) {
        std::fill(topk_logits[i][row].begin(), topk_logits[i][row].end(), 0.0f);
        std::fill(topk_indices[i][row].begin(), topk_indices[i][row].end(), 0);
        std::fill(logsumexp_stats[i][row].begin(), logsumexp_stats[i][row].end(), 0.0f);
        chosen_probs[i][row] = 0.0f;
        chosen_tokens[i][row] = 0;
        attention_mask[i][row] = false;
        sampling_u[i][row] = 0.0f;
        softmax_normalizers[i][row] = 0.0f;
    }
    steps[row] = 0;
    admission_nonce[row].fill(0);
    admission_valid[row] = 0;
    v4_recorded_nonce[row].fill(0);
    v4_recorded_valid[row] = 0;
}

void RingBuffers::clear_rows(const std::vector<int>& rows) {
    for (int row : rows) {
        clear_row(row);
    }
}

void RingBuffers::write_batch(const std::vector<int>& positions, 
                             const std::vector<int>& rows,
                             const std::unordered_map<std::string, std::any>& values_dict) {
    if (positions.empty() || rows.empty()) {
        return;
    }
    
    // Suppress unused parameter warning
    (void)values_dict;
    
    // For now, this is a placeholder implementation
    // In production, you would properly extract and write the values
    // based on the buffer type specified in the keys of values_dict
}

void RingBuffers::increment_steps(const std::vector<int>& rows) {
    for (int row : rows) {
        if (row >= 0 && row < max_rows) {
            steps[row]++;
        }
    }
}

std::unordered_map<std::string, std::any> RingBuffers::get_window(int row) {
    std::unordered_map<std::string, std::any> result;
    
    if (row < 0 || row >= max_rows) {
        return result;
    }
    
    int pos = steps[row] % window_size;
    
    // Get circular indices
    std::vector<int> indices;
    for (int i = 0; i < window_size; ++i) {
        indices.push_back((i + pos) % window_size);
    }
    
    // 1D SCALAR FIELDS (256 values)
    std::vector<int32_t> tokens_1d;
    std::vector<float> probs_1d;
    std::vector<float> u_values_1d;
    std::vector<float> normalizers_1d;
    std::vector<bool> mask_1d;
    
    // 2D VECTOR FIELDS (256 × vector_size)
    std::vector<std::vector<float>> topk_logits_2d;
    std::vector<std::vector<int32_t>> topk_indices_2d;
    std::vector<std::vector<float>> logsumexp_2d;
    
    for (int idx : indices) {
        // Scalars - just push the value
        tokens_1d.push_back(chosen_tokens[idx][row]);
        probs_1d.push_back(chosen_probs[idx][row]);
        u_values_1d.push_back(sampling_u[idx][row]);
        normalizers_1d.push_back(softmax_normalizers[idx][row]);
        mask_1d.push_back(attention_mask[idx][row]);
        
        // Vectors - push the entire vector
        topk_logits_2d.push_back(topk_logits[idx][row]);
        topk_indices_2d.push_back(topk_indices[idx][row]);
        logsumexp_2d.push_back(logsumexp_stats[idx][row]);
    }
    
    // Store with proper types
    result["chosen_tokens"] = tokens_1d;           // std::vector<int32_t>
    result["chosen_probs"] = probs_1d;             // std::vector<float>
    result["sampling_u"] = u_values_1d;            // std::vector<float>
    result["softmax_normalizers"] = normalizers_1d; // std::vector<float>
    result["attention_mask"] = mask_1d;            // std::vector<bool>
    result["topk_logits"] = topk_logits_2d;        // std::vector<std::vector<float>>
    result["topk_indices"] = topk_indices_2d;      // std::vector<std::vector<int32_t>>
    result["logsumexp_stats"] = logsumexp_2d;      // std::vector<std::vector<float>>
    
    return result;
}

// PowHasher implementation
PowHasher::PowHasher() : tick(0), request_id(0), difficulty(0.0f) {
    h_b.resize(32, 0);
    v.resize(32, 0);
    target.resize(32, 0);
    target[31] = 0xFF; // Default easy target
    temperature = 1.0f;
    top_p = 1.0f;
    top_k = 50;
    repetition_penalty = 1.0f;
    model_config_diff = "{}";        
}

void PowHasher::update_from_payload(const std::unordered_map<std::string, std::string>& payload) {
    if (payload.find("block_hash") != payload.end()) {
        h_b = hex_to_bytes(payload.at("block_hash"));
    }
    if (payload.find("vdf") != payload.end()) {
        v = hex_to_bytes(payload.at("vdf"));
    }
    if (payload.find("tick") != payload.end()) {
        tick = std::stoi(payload.at("tick"));
    }
    if (payload.find("request_id") != payload.end()) {
        request_id = std::stoi(payload.at("request_id"));
    }
    if (payload.find("proof_version") != payload.end()) {
        try {
            v4_mode_ = std::stoi(payload.at("proof_version")) >= pow_v4::V4_PROOF_VERSION;
        } catch (const std::exception&) { /* leave unchanged */ }
    }
    if (payload.find("target") != payload.end()) {
        std::string target_hex = payload.at("target");
        // Pad to 64 chars if needed
        if (target_hex.length() < 64) {
            target_hex = std::string(64 - target_hex.length(), '0') + target_hex;
        } else if (target_hex.length() > 64) {
            target_hex = target_hex.substr(target_hex.length() - 64);
        }
        target = hex_to_bytes(target_hex);
    }
    // Slice 11.4 — share target (optional). Empty / unset means "no
    // share mode for this lease"; check_share_solution will then
    // return all-false and emission is block-target-only.
    if (payload.find("share_target") != payload.end()) {
        std::string share_hex = payload.at("share_target");
        if (share_hex.empty()) {
            share_target.clear();
        } else {
            if (share_hex.length() < 64) {
                share_hex = std::string(64 - share_hex.length(), '0') + share_hex;
            } else if (share_hex.length() > 64) {
                share_hex = share_hex.substr(share_hex.length() - 64);
            }
            share_target = hex_to_bytes(share_hex);
        }
    }
    if (payload.find("difficulty") != payload.end()) {
        difficulty = std::stof(payload.at("difficulty"));
        // Exact integer for the v3 admission target derivation (§6). The
        // payload value is an integer chain param serialized as a string;
        // fall back through the float only if it ever isn't.
        try {
            difficulty_i64 = std::stoll(payload.at("difficulty"));
        } catch (const std::exception&) {
            difficulty_i64 = static_cast<int64_t>(difficulty);
        }
    }
    if (payload.find("header_prefix") != payload.end()) {
        header_prefix = hex_to_bytes(payload.at("header_prefix"));
    }
    if (payload.find("ipfs_cid") != payload.end()) {
        ipfs_cid = payload.at("ipfs_cid");
    }
    if (payload.find("temperature") != payload.end()) {
        temperature = std::stof(payload.at("temperature"));
    }
    if (payload.find("top_p") != payload.end()) {
        top_p = std::stof(payload.at("top_p"));
    }
    if (payload.find("top_k") != payload.end()) {
        top_k = std::stoi(payload.at("top_k"));
    }
    if (payload.find("repetition_penalty") != payload.end()) {
        repetition_penalty = std::stof(payload.at("repetition_penalty"));
    }
    if (payload.find("model_config_diff") != payload.end()) {
        model_config_diff = payload.at("model_config_diff");
    }
    if (payload.find("compute_precision") != payload.end()) {
        compute_precision = payload.at("compute_precision");
    }
    if (payload.find("model_identifier") != payload.end()) {
        model_identifier = payload.at("model_identifier");
    }    
}

// Utility function implementations
std::vector<uint8_t> hex_to_bytes(const std::string& hex_str) {
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < hex_str.length(); i += 2) {
        std::string byte_string = hex_str.substr(i, 2);
        uint8_t byte = static_cast<uint8_t>(std::stoul(byte_string, nullptr, 16));
        bytes.push_back(byte);
    }
    return bytes;
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    for (uint8_t b : bytes) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
    }
    return ss.str();
}

std::vector<uint8_t> tok_le_bytes(const std::vector<int64_t>& tokens) {
    std::vector<uint8_t> result;
    result.reserve(tokens.size() * 8);
    
    for (int64_t token : tokens) {
        // Little-endian conversion
        for (int i = 0; i < 8; ++i) {
            result.push_back((token >> (i * 8)) & 0xFF);
        }
    }
    return result;
}

std::vector<uint8_t> u32le(uint32_t value) {
    std::vector<uint8_t> result(4);
    result[0] = value & 0xFF;
    result[1] = (value >> 8) & 0xFF;
    result[2] = (value >> 16) & 0xFF;
    result[3] = (value >> 24) & 0xFF;
    return result;
}

std::vector<uint8_t> str_bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

std::vector<uint8_t> build_msg(const std::vector<uint8_t>& header_prefix,
                               const std::vector<uint8_t>& v,
                               const std::vector<uint8_t>& T8,
                               const std::vector<uint8_t>& j4,
                               const std::vector<uint8_t>& ctx_bytes,
                               const std::vector<uint8_t>& precision) {
    std::vector<uint8_t> msg;
    msg.insert(msg.end(), header_prefix.begin(), header_prefix.end());
    msg.insert(msg.end(), v.begin(), v.end());
    msg.insert(msg.end(), T8.begin(), T8.end());
    msg.insert(msg.end(), j4.begin(), j4.end());
    msg.insert(msg.end(), ctx_bytes.begin(), ctx_bytes.end());
    msg.insert(msg.end(), precision.begin(), precision.end());
    return msg;
}

float digest_to_u(const std::vector<uint8_t>& digest) {
    if (digest.size() < 4) return 0.0f;
    
    // Match Python's calculation method exactly
    float b0 = static_cast<float>(digest[0]);
    float b1 = static_cast<float>(digest[1]);
    float b2 = static_cast<float>(digest[2]);
    float b3 = static_cast<float>(digest[3]);
    
    float result = (b0 + b1 * 256.0f + b2 * 65536.0f + b3 * 16777216.0f) / 4294967296.0f;
    return result;
}

// Replace the old sha256_many function:
std::vector<std::vector<uint8_t>> sha256_many(const std::vector<std::vector<uint8_t>>& messages) {
    std::vector<std::vector<uint8_t>> results;
    
    for (const auto& msg : messages) {
        std::vector<uint8_t> result(32); // SHA256 produces 32 bytes
        unsigned int len = 32;
        
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        if (ctx) {
            if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                EVP_DigestUpdate(ctx, msg.data(), msg.size()) == 1 &&
                EVP_DigestFinal_ex(ctx, result.data(), &len) == 1) {
                results.push_back(result);
            }
            EVP_MD_CTX_free(ctx);
        }
    }
    
    return results;
}

// Helper function for single SHA256 hash:
std::vector<uint8_t> sha256_single(const std::vector<uint8_t>& msg) {
    std::vector<uint8_t> result(32);
    unsigned int len = 32;
    
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx) {
        if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
            EVP_DigestUpdate(ctx, msg.data(), msg.size()) == 1 &&
            EVP_DigestFinal_ex(ctx, result.data(), &len) == 1) {
            // Success
        }
        EVP_MD_CTX_free(ctx);
    }
    
    return result;
}

// ProofWriter implementation
ProofWriter::ProofWriter(const std::string& output_dir)
    : output_dir(output_dir.empty() ? get_env_var("PROOF_SAVE_DIR", "/tmp/llama_pow_proofs") : output_dir) {
    fs::create_directories(this->output_dir);
}

void ProofWriter::set_callback(std::function<void(const std::string&)> callback) {
    submit_callback = callback;
}

void ProofWriter::set_model_identifier(const std::string& identifier) {
    model_identifier = identifier;
}

void ProofWriter::set_ipfs_cid(const std::string& cid) {
    ipfs_cid = cid;
}

void ProofWriter::set_model_config_diff(const std::string& diff) {
    model_config_diff = diff;
}

void ProofWriter::set_sampling_params_diff(const std::string& diff) {
    sampling_params_diff = diff;
}

void ProofWriter::set_compute_precision(const std::string& precision) {
    compute_precision = precision;
}

void ProofWriter::set_proof_version(int version) {
    proof_version = version;
}

std::pair<std::vector<uint8_t>, std::unordered_map<std::string, std::any>>
ProofWriter::write_proof(
    int seq_id,
    int step_num,
    const std::unordered_map<std::string, std::any>& window_data,
    const std::vector<uint8_t>& digest,
    bool is_solution,
    const std::unordered_map<std::string, std::string>& pow_params,
    const std::unordered_map<std::string, std::any>& seq_info
) {
    // Proof-version agreement (TIP-0003), mirroring the Python
    // writer: the miner-api proxy stamps its own POW_PROOF_VERSION into the
    // pow payload (it forces the v3 fixed sampler profile keyed off THAT
    // env), while this writer's proof_version comes from THIS process's env.
    // Drift between the two means every emitted proof is verifier-rejected
    // (wrong profile or wrong carrier) — fail loudly on the first proof
    // instead. Absent stamp (old proxy image / direct callers) skips the
    // check.
    {
        auto stamped_it = pow_params.find("proof_version");
        if (stamped_it != pow_params.end() &&
            std::stoi(stamped_it->second) != proof_version) {
            throw std::invalid_argument(
                "POW_PROOF_VERSION disagreement: miner-api ingress stamped "
                "proof_version=" + stamped_it->second +
                " but this sampler process is configured for " +
                std::to_string(proof_version) +
                "; align the env on both processes (the proxy forces the v3 "
                "fixed profile only when ITS env is >= 3)");
        }
    }

    // 1) Build the proof dictionary exactly as before:
    std::unordered_map<std::string, std::any> proof;
    proof["sequence_id"] = seq_id;
    proof["steps"]       = step_num;
    proof["is_solution"] = is_solution;
    // Wire the proof version into the returned dict, not just the FlatBuffer
    // built below. The ZMQ egress path (pow_zmq_writer serialize_response) reads
    // proof_dict["version"] and DEFAULTS a missing key to 2, so a v3-configured
    // llama miner whose dict lacked this key emitted version=2 on the wire
    // regardless of POW_PROOF_VERSION -- accepted as Full_Green by the validator
    // then rejected by AcceptBlock (bad-proof-version-v3). Mirrors the Python
    // ProofProcessor path, which sets proof["version"] = uint32_t(proof_version_).
    proof["version"]     = static_cast<uint32_t>(proof_version);
    auto ts = std::to_string(
         std::chrono::system_clock::now().time_since_epoch().count()
    );
    proof["timestamp"] = ts;

    // copy all pow_params (hex strings)
    for (auto &kv : pow_params) {
        proof[kv.first] = kv.second;
    }
    // copy all window_data (float matrices)
    for (auto &kv : window_data) {
        proof[kv.first] = kv.second;
    }
    // store digest hex for the dict (we’ll convert back below)
    proof["hash"] = bytes_to_hex(digest);

    // v3 (TIP-0003): absent sampler params are a hard failure,
    // not a default — the recomputed B_cred prices the real params, so a
    // silently-defaulted value would desync miner and verifier. v2 keeps the
    // legacy fallbacks byte-for-byte.
    for (const char* key : {"temperature", "top_p", "top_k",
                            "repetition_penalty"}) {
        if (proof_version >= 3 && pow_params.find(key) == pow_params.end()) {
            throw std::invalid_argument(
                std::string("v3 proof missing required sampler param: ") + key);
        }
    }

    if (pow_params.find("temperature") != pow_params.end()) {
        proof["temperature"] = std::string(pow_params.at("temperature"));
    } else {
        proof["temperature"] = std::string("1.0");
    }

    if (pow_params.find("top_p") != pow_params.end()) {
        proof["top_p"] = std::string(pow_params.at("top_p"));
    } else {
        proof["top_p"] = std::string("1.0");
    }

    if (pow_params.find("top_k") != pow_params.end()) {
        proof["top_k"] = std::string(pow_params.at("top_k"));
    } else {
        proof["top_k"] = std::string("30");
    }

    if (pow_params.find("repetition_penalty") != pow_params.end()) {
        proof["repetition_penalty"] = std::string(pow_params.at("repetition_penalty"));
    } else {
        proof["repetition_penalty"] = std::string("1.0");
    }

    // 2. Add model config diff (if not already supplied via pow_params)
    if (proof.find("model_config_diff") == proof.end()) {
        proof["model_config_diff"] = std::string("{}");
    }

    // 4. Ensure model fields are present (even if empty)
    proof["model_identifier"] = model_identifier.empty() ? "unknown" : model_identifier;
    proof["compute_precision"] = compute_precision.empty() ? "fp16" : compute_precision;

    if (!ipfs_cid.empty())
        proof["ipfs_cid"] = ipfs_cid;

    // sequence‐specific info
    for (auto &kv : seq_info) {
        proof[kv.first] = kv.second;
    }
    
    if (pow_params.find("request_id") != pow_params.end()) {
        proof["request_id"] = pow_params.at("request_id");
    }
    if (pow_params.find("difficulty") != pow_params.end()) {
        proof["difficulty"] = pow_params.at("difficulty");
    }

    // v3 (TIP-0003, §9): the sampler-selected admission nonce
    // arrives as 64 lowercase hex in pow_params["admission_nonce"]; merge it
    // into model_config_diff (written verbatim into Proof.extra_flags below)
    // through the shared helper so existing keys survive. Placed after the
    // pow_params/window_data/seq_info copies and just before FlatBuffer
    // serialization — nothing can overwrite the merged member afterwards.
    // This path never re-grinds and never parses JSON beyond the one merge.
    {
        auto nonce_it = pow_params.find("admission_nonce");
        if (proof_version >= 3 && nonce_it != pow_params.end()) {
            std::string existing;
            try {
                existing = std::any_cast<std::string>(
                    proof.at("model_config_diff"));
            } catch (...) { existing = ""; }
            // v4 carries R under the v4 carrier (v4:1 + v3.stepbind:1 +
            // v3.admission_nonce); v3 carries the raw appended nonce.
            proof["model_config_diff"] =
                proof_version >= pow_v4::V4_PROOF_VERSION
                    ? pow_v4::merge_extra_flags_v4(existing, nonce_it->second)
                    : pow_v3::merge_extra_flags_v3(existing, nonce_it->second);
        }
    }

    // 2) Now *really* serialize into a FlatBuffer:
    flatbuffers::FlatBufferBuilder builder(4096);

    // --- convert your hex fields back to bytes ---
    auto target_bytes      = hex_to_bytes(std::any_cast<std::string>(proof.at("target")));
    auto vdf_bytes         = hex_to_bytes(std::any_cast<std::string>(proof.at("vdf")));
    auto block_hash_bytes  = hex_to_bytes(std::any_cast<std::string>(proof.at("block_hash")));
    auto header_pref_bytes = hex_to_bytes(std::any_cast<std::string>(proof.at("header_prefix")));
    // digest is already raw:
    auto hash_bytes        = digest;

    // --- FlatBuffer vectors for those byte‐arrays ---
    auto target_vec      = builder.CreateVector(target_bytes);
    auto vdf_vec         = builder.CreateVector(vdf_bytes);
    auto hash_vec        = builder.CreateVector(hash_bytes);
    auto block_hash_vec  = builder.CreateVector(block_hash_bytes);
    auto header_pref_vec = builder.CreateVector(header_pref_bytes);

    // --- pull out your 1D arrays from `proof` ---
    auto &chosen_tokens = std::any_cast<const std::vector<int32_t>&>(proof.at("chosen_tokens"));
    auto &chosen_probs  = std::any_cast<const std::vector<float>&>(    proof.at("chosen_probs"));
    auto cp_vec = builder.CreateVector(chosen_probs); 
    auto &sampling_u    = std::any_cast<const std::vector<float>&>(    proof.at("sampling_u"));
    auto &softmax_norm  = std::any_cast<const std::vector<float>&>(    proof.at("softmax_normalizers"));
    auto &prompt_tokens = std::any_cast<const std::vector<int32_t>&>(proof.at("prompt_tokens"));
    if (proof.find("pad_mask") == proof.end()) {
        proof["pad_mask"] = std::vector<bool>(prompt_tokens.size(), false);
    }
    auto &pad_mask      = std::any_cast<const std::vector<bool>&>(    proof.at("pad_mask"));

    std::vector<uint32_t> ct_u32(chosen_tokens.begin(), chosen_tokens.end());
    std::vector<uint32_t> pt_u32(prompt_tokens.begin(), prompt_tokens.end());
    auto ct_vec = builder.CreateVector(ct_u32);
    auto pt_vec = builder.CreateVector(pt_u32);
    auto su_vec = builder.CreateVector(sampling_u);
    auto sn_vec = builder.CreateVector(softmax_norm);
    // bool → uint8_t vector:
    std::vector<uint8_t> mask_bytes(pad_mask.begin(), pad_mask.end());
    auto pm_vec = builder.CreateVector(mask_bytes);

    // --- helper to build 2D arrays of floats / uint32 ---
    auto makeFloatArr = [&](auto &M){
      std::vector<flatbuffers::Offset<proof::FloatArray>> rows;
      rows.reserve(M.size());
      for (auto &r : M) {
        rows.push_back(
          proof::CreateFloatArray(builder, builder.CreateVector(r))
        );
      }
      return builder.CreateVector(rows);
    };
    auto makeUIntArr = [&](auto &M){
      std::vector<flatbuffers::Offset<proof::UIntArray>> rows;
      rows.reserve(M.size());
      for (auto &r : M) {
        std::vector<uint32_t> tmp(r.begin(), r.end());
        rows.push_back(
          proof::CreateUIntArray(builder, builder.CreateVector(tmp))
        );
      }
      return builder.CreateVector(rows);
    };

    auto &topk_logits = std::any_cast<const std::vector<std::vector<float>>&>( proof.at("topk_logits") );
    auto &topk_idx    = std::any_cast<const std::vector<std::vector<int32_t>>&>( proof.at("topk_indices") );
    auto &lse_stats   = std::any_cast<const std::vector<std::vector<float>>&>( proof.at("logsumexp_stats") );

    auto tkl_vec = makeFloatArr(topk_logits);
    auto tki_vec = makeUIntArr(topk_idx);
    auto lse_vec = makeFloatArr(lse_stats);

    // FlatBuffers requires strings to be created before entering the table.
    auto model_id_off = builder.CreateString(std::any_cast<std::string>(proof.at("model_identifier")));
    auto compute_precision_off = builder.CreateString(std::any_cast<std::string>(proof.at("compute_precision")));
    auto ipfs_cid_off = builder.CreateString(std::any_cast<std::string>(proof.at("ipfs_cid")));
    auto extra_flags_off = builder.CreateString(std::any_cast<std::string>(proof.at("model_config_diff")));

    // --- Build the Proof table ---
    proof::ProofBuilder pb(builder);
    // Wire version from the writer's proof_version member (TIP-0003
    // §3: version=3 turns on the v3 carrier rules) — never hardcoded.
    pb.add_version(static_cast<uint32_t>(proof_version));
    pb.add_tick(std::stoll(std::any_cast<std::string>(proof.at("tick"))));
    pb.add_timestamp(std::stoll(std::any_cast<std::string>(proof.at("timestamp"))));
    pb.add_is_solution(is_solution);
    pb.add_model_identifier(model_id_off);
    pb.add_compute_precision(compute_precision_off);
    pb.add_ipfs_cid(ipfs_cid_off);
    pb.add_extra_flags(extra_flags_off);
    pb.add_temperature(std::stof(std::any_cast<std::string>(proof.at("temperature"))));
    pb.add_top_p(std::stof(std::any_cast<std::string>(proof.at("top_p"))));
    pb.add_top_k(std::stoul(std::any_cast<std::string>(proof.at("top_k"))));
    pb.add_repetition_penalty(std::stof(std::any_cast<std::string>(proof.at("repetition_penalty"))));
    pb.add_target(target_vec);
    pb.add_vdf(vdf_vec);
    pb.add_hash(hash_vec);
    pb.add_block_hash(block_hash_vec);
    pb.add_header_prefix(header_pref_vec);
    pb.add_chosen_tokens(ct_vec);
    pb.add_chosen_probs(cp_vec);
    pb.add_sampling_u(su_vec);
    pb.add_softmax_normalizers(sn_vec);
    pb.add_prompt_tokens(pt_vec);
    pb.add_pad_mask(pm_vec);
    pb.add_topk_logits(tkl_vec);
    pb.add_topk_indices(tki_vec);
    pb.add_logsumexp_stats(lse_vec);
    auto proof_off = pb.Finish();

    // --- extract the real nonce from the digest (little-endian) ---
    uint32_t nonce =  
        static_cast<uint32_t>(digest[0])        |
        (static_cast<uint32_t>(digest[1]) << 8)  |
        (static_cast<uint32_t>(digest[2]) << 16) |
        (static_cast<uint32_t>(digest[3]) << 24);

    // --- Build the outer MiningResponse ---
    auto pow_blob_hash_vec = builder.CreateVector(digest);
    proof::MiningResponseBuilder rb(builder);
    rb.add_req_id(std::stoll(pow_params.at("request_id")));
    rb.add_nonce( nonce );  // or your actual nonce
    rb.add_adjusted_bits(0);
    rb.add_pow_blob_hash(pow_blob_hash_vec);
    rb.add_difficulty(std::stoul(pow_params.at("difficulty")));
    rb.add_pow_blob(proof_off);
    auto resp_off = rb.Finish();

    // builder.Finish(resp_off, /*file_id=*/"PROF");
    builder.Finish(resp_off);

    // 3) extract the raw bytes:
    auto ptr = builder.GetBufferPointer();
    auto sz  = builder.GetSize();
    std::vector<uint8_t> serialized(ptr, ptr + sz);

    // 4) Save proof files to output_dir
    try {
        std::filesystem::create_directories(output_dir);
        // Generate filename with seq_id, step_num, and timestamp
        std::string base_filename = "proof_" + std::to_string(seq_id) + "_" + 
                                std::to_string(step_num) + "_" + 
                                std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch()).count());
        // Save binary file
        std::string bin_path = output_dir + "/" + base_filename + ".bin";
        std::ofstream bin_file(bin_path, std::ios::binary);
        if (bin_file.is_open()) {
            bin_file.write(reinterpret_cast<const char*>(serialized.data()), serialized.size());
            bin_file.close();
        }
        // Save JSON file (simplified - convert std::any values to strings)
        std::string json_path = output_dir + "/" + base_filename + ".json";
        std::ofstream json_file(json_path);
        if (json_file.is_open()) {
            for (auto it = proof.begin(); it != proof.end(); ++it) {
                const auto &key   = it->first;
                const auto &value = it->second;
                json_file << "  \"" << key << "\": ";

                if (value.type() == typeid(std::string)) {
                    json_file << "\"" << std::any_cast<std::string>(value) << "\"";
                }
                else if (value.type() == typeid(int)) {
                    json_file << std::any_cast<int>(value);
                }
                else if (value.type() == typeid(bool)) {
                    json_file << (std::any_cast<bool>(value) ? "true" : "false");
                }
                else if (value.type() == typeid(std::vector<int32_t>)) {
                    auto &v = std::any_cast<const std::vector<int32_t>&>(value);
                    json_file << "[";
                    for (size_t i = 0; i < v.size(); ++i) {
                        if (i) json_file << ",";
                        json_file << v[i];
                    }
                    json_file << "]";
                }
                else if (value.type() == typeid(std::vector<float>)) {
                    auto &v = std::any_cast<const std::vector<float>&>(value);
                    json_file << "[";
                    for (size_t i = 0; i < v.size(); ++i) {
                        if (i) json_file << ",";
                        json_file << v[i];
                    }
                    json_file << "]";
                }
                else if (value.type() == typeid(std::vector<bool>)) {
                    auto &v = std::any_cast<const std::vector<bool>&>(value);
                    json_file << "[";
                    for (size_t i = 0; i < v.size(); ++i) {
                        if (i) json_file << ",";
                        json_file << (v[i] ? "true" : "false");
                    }
                    json_file << "]";
                }
                else if (value.type() == typeid(std::vector<std::vector<float>>)) {
                    auto &M = std::any_cast<const std::vector<std::vector<float>>&>(value);
                    json_file << "[";
                    for (size_t i = 0; i < M.size(); ++i) {
                        if (i) json_file << ",";
                        json_file << "[";
                        for (size_t j = 0; j < M[i].size(); ++j) {
                            if (j) json_file << ",";
                            json_file << M[i][j];
                        }
                        json_file << "]";
                    }
                    json_file << "]";
                }
                else if (value.type() == typeid(std::vector<std::vector<int32_t>>)) {
                    auto &M = std::any_cast<const std::vector<std::vector<int32_t>>&>(value);
                    json_file << "[";
                    for (size_t i = 0; i < M.size(); ++i) {
                        if (i) json_file << ",";
                        json_file << "[";
                        for (size_t j = 0; j < M[i].size(); ++j) {
                            if (j) json_file << ",";
                            json_file << M[i][j];
                        }
                        json_file << "]";
                    }
                    json_file << "]";
                }
                else {
                    // fallback: print the demangled type name
                    json_file << "\"<unsupported_type>\"";
                }

                // comma except on last item
                if (std::next(it) != proof.end()) json_file << ",";
                json_file << "\n";
            }
            json_file.close();
        }
        
    } catch (const std::exception& e) {
        // Silent failure for file operations
        std::cerr << "Failed to save proof files: " << e.what() << std::endl;
    }

    return { std::move(serialized), std::move(proof) };
}

std::vector<bool> check_hash_against_target(const std::vector<std::vector<uint8_t>>& digests,
                                           const std::vector<uint8_t>& target) {
    std::vector<bool> results;
    
    // Flip target for little-endian comparison
    std::vector<uint8_t> t_le(target.rbegin(), target.rend());
    
    for (const auto& digest : digests) {
        bool decided = false;
        bool result = false;
        
        // Compare from most significant byte
        for (int i = 31; i >= 0; --i) {
            if (!decided) {
                if (digest[i] < t_le[i]) {
                    result = true;
                    decided = true;
                } else if (digest[i] > t_le[i]) {
                    result = false;
                    decided = true;
                }
            }
        }
        
        if (!decided) {
            result = true; // Equal means valid
        }
        
        results.push_back(result);
    }
    
    return results;
}

std::vector<uint8_t> nbits_to_target(int nbits) {
    int exponent = nbits >> 24;
    int mantissa = nbits & 0x00ffffff;
    
    uint64_t target_int;
    if (exponent <= 3) {
        target_int = mantissa >> (8 * (3 - exponent));
    } else {
        target_int = static_cast<uint64_t>(mantissa) << (8 * (exponent - 3));
    }
    
    // Convert to 32-byte big-endian
    std::vector<uint8_t> target_bytes(32, 0);
    for (int i = 0; i < 8; ++i) {
        if (target_int > 0) {
            target_bytes[31 - i] = (target_int >> (i * 8)) & 0xFF;
        }
    }
    
    return target_bytes;
}

// Update batch_sample_tokens method:
std::tuple<std::vector<int64_t>, std::vector<float>, std::vector<std::vector<uint8_t>>>
PowHasher::batch_sample_tokens(const std::vector<std::vector<int64_t>>& contexts,
                              const std::vector<int32_t>& steps,
                              const std::vector<std::vector<float>>& cdfs,
                              const std::string& compute_precision,
                              const std::vector<const uint8_t*>& admission_nonces) {

    int batch_size = contexts.size();
    if (!admission_nonces.empty() &&
        admission_nonces.size() != contexts.size()) {
        throw std::invalid_argument(
            "batch_sample_tokens: admission_nonces size must match contexts");
    }
    std::vector<int64_t> token_ids;
    std::vector<float> us;
    std::vector<std::vector<uint8_t>> digests;

    for (int b = 0; b < batch_size; ++b) {
        // Convert context to bytes
        auto ctx_bytes = tok_le_bytes(contexts[b]);

        // Convert step to bytes
        auto j4 = u32le(steps[b]);

        // Convert tick to bytes (8 bytes for T8)
        auto T8 = u32le(tick);

        // Convert precision to bytes
        auto precision_bytes = str_bytes(compute_precision);

        // Use header_prefix if available, otherwise h_b
        const auto& header_data = header_prefix.empty() ? h_b : header_prefix;

        // Build message
        auto msg = build_msg(header_data, v, T8, j4, ctx_bytes, precision_bytes);

        // v3 (TIP-0003): append this row's admission nonce.
        if (!admission_nonces.empty() && admission_nonces[b] != nullptr) {
            msg.insert(msg.end(), admission_nonces[b], admission_nonces[b] + 32);
        }
        if (v4_mode_) {
            // v4: the Gumbel mode suffix follows E4 on every step preimage.
            msg.insert(msg.end(), pow_v4::SAMPLER_MODE_SUFFIX.begin(),
                       pow_v4::SAMPLER_MODE_SUFFIX.end());
        }

        // Compute hash using modern OpenSSL
        auto digest = sha256_single(msg);
        digests.push_back(digest);
        
        // Convert to uniform value
        float u = digest_to_u(digest);
        us.push_back(u);
        
        // Sample token using binary search on CDF
        auto it = std::lower_bound(cdfs[b].begin(), cdfs[b].end(), u);
        int64_t token_id = std::distance(cdfs[b].begin(), it);
        token_ids.push_back(token_id);
    }
    
    return {token_ids, us, digests};
}

// Update sample_token method:
std::tuple<int64_t, float, std::vector<uint8_t>>
PowHasher::sample_token(const std::vector<int64_t>& context,
                       int32_t step,
                       const std::vector<float>& cdf,
                       const uint8_t* admission_nonce32) {
    // Convert context to bytes
    auto ctx_bytes = tok_le_bytes(context);

    // Convert step to bytes
    auto j4 = u32le(step);

    // Convert tick to bytes (8 bytes total)
    auto T8 = u32le(tick);

    // Convert precision to bytes
    auto precision_bytes = str_bytes(compute_precision);

    // Use header_prefix if available, otherwise h_b
    const auto& header_data = header_prefix.empty() ? h_b : header_prefix;

    // Build message
    auto msg = build_msg(header_data, v, T8, j4, ctx_bytes, precision_bytes);

    // v3 (TIP-0003): the 32 admission-nonce bytes are appended
    // to the legacy preimage — the only change to the message shape.
    if (admission_nonce32 != nullptr) {
        msg.insert(msg.end(), admission_nonce32, admission_nonce32 + 32);
    }
    if (v4_mode_) {
        // v4: the Gumbel mode suffix follows E4 on the final/solution hash too.
        msg.insert(msg.end(), pow_v4::SAMPLER_MODE_SUFFIX.begin(),
                   pow_v4::SAMPLER_MODE_SUFFIX.end());
    }

    // Compute hash using modern OpenSSL
    auto digest = sha256_single(msg);

    // Convert to uniform value
    float u = digest_to_u(digest);
    
    // Sample token
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    int64_t token_id = std::distance(cdf.begin(), it);
    
    return {token_id, u, digest};
}

bool PowHasher::grind_admission(const std::vector<int64_t>& context,
                                int32_t step,
                                const std::vector<int64_t>& prefix_tokens,
                                const std::vector<uint8_t>& prefix_pad_mask,
                                uint64_t max_tries_factor,
                                std::array<uint8_t, 32>& out_nonce) {
    if (difficulty_i64 <= 0) {
        return false;
    }
    // Byte-identical to pow_v3.build_admission_preimage (§6/§9): msg_w is
    // this window's first-step preimage with NO nonce (header falls back to
    // block_hash exactly like the sampling path); the commitment binds the
    // full pre-window prefix; model_identifier must be the exact string the
    // proof will carry ("unknown" fallback matches write_proof).
    const auto& header_data = header_prefix.empty() ? h_b : header_prefix;
    auto msg_w = pow_v3::build_step_message(
        header_data, v,
        static_cast<uint32_t>(tick), static_cast<uint32_t>(step),
        context, compute_precision);
    auto commitment = pow_v3::prompt_commitment(prefix_tokens, prefix_pad_mask);
    auto target_le = pow_v3::admission_target_le(difficulty_i64);
    if (max_tries_factor < 1) {
        max_tries_factor = 1;
    }
    const uint64_t max_tries =
        pow_v3::admission_expected_tries(difficulty_i64) * max_tries_factor;
    auto nonce = pow_v3::admission_grind(
        msg_w,
        model_identifier.empty() ? "unknown" : model_identifier,
        target_le, max_tries, commitment);
    if (!nonce.has_value()) {
        return false;
    }
    out_nonce = *nonce;
    return true;
}

bool PowHasher::compute_v4_window_root(const std::vector<int64_t>& context,
                                       const std::vector<int64_t>& prefix_tokens,
                                       const std::vector<uint8_t>& prefix_pad_mask,
                                       std::array<uint8_t, 32>& out_e4,
                                       std::array<uint8_t, 32>& out_r) {
    // Same preimage bytes as grind_admission / the sampling path: msg_w0 is the
    // window's first-step message with NO trailing nonce; C binds the full
    // pre-window prefix. E4 is then COMPUTED (not grinded) from a fresh random
    // R — free tier admits any A, and R is recorded in the proof so the verifier
    // recomputes A and E4 (pow_v4::v4_root_from_fields).
    const auto& header_data = header_prefix.empty() ? h_b : header_prefix;
    auto msg_w0 = pow_v3::build_step_message(
        header_data, v,
        static_cast<uint32_t>(tick), static_cast<uint32_t>(0),
        context, compute_precision);
    auto commitment = pow_v3::prompt_commitment(prefix_tokens, prefix_pad_mask);

    std::array<uint8_t, 32> r{};
    if (RAND_bytes(r.data(), 32) != 1) {
        return false;  // CSPRNG failure — never mine a window we can't record
    }
    const std::string& mid = model_identifier.empty() ? std::string("unknown")
                                                       : model_identifier;
    pow_v4::V4Root root = pow_v4::derive_v4_root(msg_w0, commitment, mid, r);
    out_e4 = root.E4;
    out_r = r;
    return true;
}

std::vector<uint8_t> PowHasher::step_digest_for(const std::vector<int64_t>& context,
                                                int32_t step,
                                                const uint8_t* nonce32) {
    // Byte-identical to the preimage sample_token() hashes, but returns the raw
    // 32-byte digest D only — the v4 Gumbel race consumes D directly as its
    // seed (no CDF search). nonce32 is E4 for v4.
    auto ctx_bytes = tok_le_bytes(context);
    auto j4 = u32le(step);
    auto T8 = u32le(tick);
    auto precision_bytes = str_bytes(compute_precision);
    const auto& header_data = header_prefix.empty() ? h_b : header_prefix;
    auto msg = build_msg(header_data, v, T8, j4, ctx_bytes, precision_bytes);
    if (nonce32 != nullptr) {
        msg.insert(msg.end(), nonce32, nonce32 + 32);
    }
    if (v4_mode_) {
        // v4: the Gumbel mode suffix follows E4 on every step preimage.
        msg.insert(msg.end(), pow_v4::SAMPLER_MODE_SUFFIX.begin(),
                   pow_v4::SAMPLER_MODE_SUFFIX.end());
    }
    return sha256_single(msg);
}

// Update check_solution method:
std::vector<bool> PowHasher::check_solution(const std::vector<std::vector<uint8_t>>& digests) {
    std::vector<bool> results;

    for (const auto& digest : digests) {
        // Extract nonce (first 4 bytes)
        std::vector<uint8_t> nonce(digest.begin(), digest.begin() + 4);

        // Build complete 80-byte header
        std::vector<uint8_t> header;
        header.insert(header.end(), header_prefix.begin(), header_prefix.end());
        header.insert(header.end(), nonce.begin(), nonce.end());

        // First SHA-256
        auto hash1 = sha256_single(header);

        // Second SHA-256
        auto hash2 = sha256_single(hash1);

        // Check against target
        auto check_results = check_hash_against_target({hash2}, target);
        results.push_back(check_results[0]);
    }

    return results;
}

// Slice 11.4 — share-target companion to check_solution. Identical
// canonical hash computation, gated on the easier share_target.
// Returns all-false when share_target is empty (rollout-safe: no
// caller sees spurious shares before the broker enables share-mode).
std::vector<bool> PowHasher::check_share_solution(const std::vector<std::vector<uint8_t>>& digests) {
    if (share_target.empty()) {
        return std::vector<bool>(digests.size(), false);
    }
    std::vector<bool> results;
    results.reserve(digests.size());
    for (const auto& digest : digests) {
        std::vector<uint8_t> nonce(digest.begin(), digest.begin() + 4);
        std::vector<uint8_t> header;
        header.insert(header.end(), header_prefix.begin(), header_prefix.end());
        header.insert(header.end(), nonce.begin(), nonce.end());
        auto hash1 = sha256_single(header);
        auto hash2 = sha256_single(hash1);
        auto check_results = check_hash_against_target({hash2}, share_target);
        results.push_back(check_results[0]);
    }
    return results;
}

void PowSamplingCoordinator::ensure_capacity_(int n_vocab, int top_k) {
    // Grow primary buffers as needed; zero/clear only up to n_vocab to avoid O(capacity)
    if ((int)logits_.size() < n_vocab) {
        logits_.assign(n_vocab, 0.0f);
        probs_.assign(n_vocab, 0.0f);
        cdf_.assign(n_vocab, 0.0f);
        pretemp_desc_.reserve(n_vocab);
        pretemp_desc_.clear();
        keep_.assign(n_vocab, 0u);
    } else {
        std::fill(logits_.begin(), logits_.begin() + n_vocab, 0.0f);
        std::fill(probs_.begin(),  probs_.begin()  + n_vocab, 0.0f);
        std::fill(cdf_.begin(),    cdf_.begin()    + n_vocab, 0.0f);
        pretemp_desc_.clear();
        keep_.assign(n_vocab, 0u);
    }

    const int m_cap = (top_k > 0 && top_k < n_vocab) ? top_k : n_vocab;
    if ((int)cand_.capacity() < m_cap) cand_.reserve(m_cap);
    if ((int)cand_exp_.capacity() < m_cap) cand_exp_.reserve(m_cap);
}

PowSamplingCoordinator::SamplingResult
PowSamplingCoordinator::sample_token_complete(
    int seq_id,
    const float* raw_logits,
    int n_vocab,
    float temperature,
    int top_k,
    float top_p,
    const std::vector<int64_t>& context,
    const std::string& compute_precision = "fp32") 
{
    SamplingResult result;

    if (n_vocab <= 0) {
        throw std::runtime_error("sample_token_complete: n_vocab <= 0");
    }

    if (!activate_sequence_params(seq_id)) {
        throw std::runtime_error("No PoW params registered for sequence " + std::to_string(seq_id));
    }

    auto row_opt = row_manager->get_row(seq_id);
    if (!row_opt.has_value()) {
        throw std::runtime_error("No row allocated for sequence " + std::to_string(seq_id));
    }
    const int row = row_opt.value();
    int32_t step = ring_buffers->steps[row] % window_size;

    // v3 (TIP-0003): a window's admission nonce must exist
    // BEFORE its first sampled token — it enters every u of the window.
    const uint8_t* admission_nonce_ptr = nullptr;
    if (proof_version_ >= pow_v3::V3_PROOF_VERSION) {
        if (step == 0) {
            prepare_window_admission_(seq_id, row, context);
        }
        if (ring_buffers->admission_valid[row]) {
            admission_nonce_ptr = ring_buffers->admission_nonce[row].data();
        }
    }

    ensure_capacity_(n_vocab, top_k);
    const float ninf = -std::numeric_limits<float>::infinity();

    std::vector<float> snapped_logits;
    const float* working_logits = raw_logits;
    
    if (compute_precision != "fp32" && !compute_precision.empty()) {
        snapped_logits.assign(raw_logits, raw_logits + n_vocab);
        snap_logits_to_precision_inplace(snapped_logits.data(), n_vocab, compute_precision);
        working_logits = snapped_logits.data();
    }    

    // ---------- 1) Pre-temp sort once in the effective precision domain ----------
    pretemp_desc_.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) pretemp_desc_.emplace_back(working_logits[i], i);
    std::sort(pretemp_desc_.begin(), pretemp_desc_.end(),
              [](const auto& a, const auto& b){
                  if (a.first != b.first) return a.first > b.first; // desc by value
                  return a.second < b.second;                        // tie by id
              });

    // ---------- 2) Temperature scaling ----------
    if (temperature != 1.0f) {
        const float invT = 1.0f / temperature;
        for (int i = 0; i < n_vocab; ++i) logits_[i] = working_logits[i] * invT;
    } else {
        std::copy(working_logits, working_logits + n_vocab, logits_.begin());
    }

    // ---------- 3) logsumexp_full on temp before truncation ----------
    float max_log = logits_[0];
    for (int i = 1; i < n_vocab; ++i) max_log = std::max(max_log, logits_[i]);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp(double(logits_[i] - max_log));
    result.logsumexp_full = max_log + std::log(float(sum_exp));

    // ---------- 4) Stats (index 0 from temp; others pre-temp means) ----------
    result.logsumexp_stats.assign(6, 0.0f);
    result.logsumexp_stats[0] = result.logsumexp_full;
    if (n_vocab >= 50) {
        double s = 0.0; for (int i = 0; i < 50; ++i) s += pretemp_desc_[i].first;
        result.logsumexp_stats[1] = float(s / 50.0);
    }
    if (n_vocab >= 500) {
        const int hi = std::min(500, n_vocab);
        double s = 0.0; for (int i = 50; i < hi; ++i) s += pretemp_desc_[i].first;
        const int denom = std::max(1, hi - 50);
        result.logsumexp_stats[2] = float(s / double(denom));
    }
    if (n_vocab >= 2000) {
        const int hi = std::min(2000, n_vocab);
        double s = 0.0; for (int i = 500; i < hi; ++i) s += pretemp_desc_[i].first;
        const int denom = std::max(1, hi - 500);
        result.logsumexp_stats[3] = float(s / double(denom));
    }
    if (n_vocab > 2000) {
        double s = 0.0; for (int i = 2000; i < n_vocab; ++i) s += pretemp_desc_[i].first;
        const int denom = std::max(1, n_vocab - 2000);
        result.logsumexp_stats[4] = float(s / double(denom));
    }
    {
        double s = 0.0; for (const auto& p : pretemp_desc_) s += p.first;
        result.logsumexp_stats[5] = float(s / double(n_vocab));
    }

    // ---------- 5) TOP-K (strict exclusive <=) ----------
    const bool do_top_k = (top_k > 0 && top_k < n_vocab);
    int survivors = n_vocab;

    if (do_top_k) {
        const int kpos    = std::min(top_k, n_vocab) - 1;
        const float kth_pre = pretemp_desc_[kpos].first; // effective pre-temp threshold

        survivors = 0;
        for (int i = 0; i < n_vocab; ++i) {
            if (working_logits[i] > kth_pre) {        // strict exclusiveness (intentional)
                keep_[i] = 1u;
                ++survivors;
            } else {
                logits_[i] = ninf;
            }
        }

        // --- No-survivor fallback (due to ties at the threshold) ---
        if (survivors == 0) {
            // pick the single global argmax in the effective precision domain
            int argmax = 0;
            float best = working_logits[0];
            for (int i = 1; i < n_vocab; ++i) {
                if (working_logits[i] > best || (working_logits[i] == best && i < argmax)) {
                    best = working_logits[i]; argmax = i;
                }
            }
            // allow exactly one survivor
            std::fill(keep_.begin(), keep_.begin() + n_vocab, 0u);
            for (int i = 0; i < n_vocab; ++i) logits_[i] = ninf;
            keep_[argmax] = 1u;
            logits_[argmax] = (temperature != 1.0f) ? (working_logits[argmax] / temperature)
                                                    : working_logits[argmax];
            survivors = 1;
        }
    } else {
        std::fill(keep_.begin(), keep_.begin() + n_vocab, 1u);
        survivors = n_vocab;
    }

    // ---------- 6) TOP-P in survivor space only ----------
    const bool do_top_p = (top_p > 0.0f && top_p < 1.0f - 1e-12f);

    double keptZ = 0.0;
    if (do_top_p && survivors > 0) {
        cand_.clear(); cand_exp_.clear();
        cand_.reserve(survivors);
        cand_exp_.reserve(survivors);

        for (int i = 0; i < n_vocab; ++i) if (keep_[i]) cand_.emplace_back(logits_[i], i);

        // Desc sort by logit (tie: lower id first)
        std::sort(cand_.begin(), cand_.end(),
                  [](const auto& a, const auto& b){
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });

        // Compute unnormalized scores once
        const float max_survivor = cand_.front().first;
        double Z = 0.0;
        cand_exp_.resize(cand_.size());
        for (size_t r = 0; r < cand_.size(); ++r) {
            float e = std::exp(cand_[r].first - max_survivor);
            cand_exp_[r] = e;
            Z += e;
        }

        // Accumulate until reaching p (guarantees at least one kept)
        double cum = 0.0;
        size_t cut = 0;
        for (; cut < cand_.size(); ++cut) {
            cum += cand_exp_[cut] / Z;
            if (cum >= top_p) break;
        }

        // Mask tail after 'cut'
        for (size_t r = cut + 1; r < cand_.size(); ++r) {
            logits_[cand_[r].second] = ninf;
            keep_[cand_[r].second] = 0u;
        }

        keptZ = 0.0;
        for (size_t r = 0; r <= cut; ++r) keptZ += cand_exp_[r];
    }

    // ---------- 7) Post-truncation log_Z and final probabilities ----------
    result.softmax_log_z = -std::numeric_limits<float>::infinity();
    float max_kept = ninf;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits_[i] != ninf) {
            max_kept = std::max(max_kept, logits_[i]);
        }
    }
    if (max_kept != ninf) {
        double kept_sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] != ninf) {
                kept_sum += std::exp(double(logits_[i] - max_kept));
            }
        }
        result.softmax_log_z = max_kept + static_cast<float>(std::log(kept_sum));
    }

    if (do_top_p && survivors > 0) {
        if (keptZ > 0.0) {
            // Initialize to zero
            std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
            // Place normalized survivor probs directly
            for (size_t r = 0; r < cand_.size(); ++r) {
                const int id = cand_[r].second;
                if (keep_[id]) probs_[id] = float(cand_exp_[r] / keptZ);
            }
        } else {
            // Extremely degenerate numeric case: uniform over kept
            int kept = 0; for (int i = 0; i < n_vocab; ++i) if (keep_[i]) ++kept;
            const float u = kept > 0 ? 1.0f / kept : 0.0f;
            for (int i = 0; i < n_vocab; ++i) probs_[i] = keep_[i] ? u : 0.0f;
        }
    } else {
        // Standard masked softmax once
        float max_masked = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_vocab; ++i) if (logits_[i] != ninf) max_masked = std::max(max_masked, logits_[i]);

        double Z = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] == ninf) { probs_[i] = 0.0f; continue; }
            float e = std::exp(logits_[i] - max_masked);
            probs_[i] = e;
            Z += e;
        }
        if (Z > 0.0) {
            const float invZ = float(1.0 / Z);
            for (int i = 0; i < n_vocab; ++i) probs_[i] *= invZ;
        } else {
            std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
        }
    }

    // ---------- 8) CDF in vocab (token-id) order ----------
    float cumulative = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        cumulative += probs_[i];
        cdf_[i] = cumulative;
    }

    // ---------- 9) PoW sampling ----------
    int64_t token_id;
    float u_val;
    std::vector<uint8_t> digest;
    if (proof_version_ >= pow_v4::V4_PROOF_VERSION) {
        // Proof v4: the draw is the chain-bound Gumbel race over the survivor
        // support, NOT the v3 CDF inverse-transform. The per-step seed is the
        // SAME digest D the v3 path hashes, with E4 (admission_nonce_ptr)
        // appended. Support = top-k/top-p survivors (logits_[i] != ninf) in
        // ASCENDING id order with their temp-scaled logits; the race winner is
        // argmax(logit_i + gumbel_i), ties -> smallest id. (An additive
        // constant on all logits does not move the winner, so the un-normalised
        // survivor logits are the correct weights — as in credit_v4::ComputeStep.)
        digest = hasher->step_digest_for(context, step, admission_nonce_ptr);
        std::vector<int64_t> sup_ids;
        std::vector<float> sup_logits;
        sup_ids.reserve(n_vocab);
        sup_logits.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] != ninf) {
                sup_ids.push_back(i);
                sup_logits.push_back(logits_[i]);
            }
        }
        std::array<uint8_t, 32> seed{};
        std::copy(digest.begin(), digest.begin() + 32, seed.begin());
        gumbel::dlog_selftest();  // refuses a non-canonical FP env (FTZ/DAZ)
        gumbel::RaceResult w = gumbel::race(sup_ids, sup_logits, seed);
        token_id = w.winner_id;
        u_val = digest_to_u(digest);  // telemetry only; the verifier races, not u
    } else {
        std::tie(token_id, u_val, digest) =
            hasher->sample_token(context, step, cdf_, admission_nonce_ptr);
    }
    result.token_id   = token_id;
    result.u_value    = u_val;
    result.digest     = digest;
    result.token_prob = (token_id >= 0 && token_id < n_vocab)
        ? (proof_version_ >= pow_v4::V4_PROOF_VERSION ? probs_[token_id] : cdf_[token_id])
        : 0.0f;

    // ---------- 10) Telemetry: pre-temp top-50 + 20 probes ----------
    result.topk_logits.assign(70, 0.0f);
    result.topk_indices.assign(70, 0);

    const int topN = std::min(50, n_vocab);
    for (int i = 0; i < topN; ++i) {
        result.topk_logits[i]  = pretemp_desc_[i].first;   // effective pre-temp
        result.topk_indices[i] = pretemp_desc_[i].second;
    }
    const int probeN = 20;
    int probe_step = std::max(1, n_vocab / probeN);
    for (int i = 0; i < probeN; ++i) {
        int probe_idx = i * probe_step;
        if (probe_idx >= n_vocab) break;
        result.topk_logits[50 + i]  = working_logits[probe_idx]; // effective pre-temp
        result.topk_indices[50 + i] = probe_idx;
    }

    return result;
}


PowSamplingCoordinator::PowSamplingCoordinator(int ws, int mc) 
    : window_size(ws), max_concurrency(mc) {}

void PowSamplingCoordinator::apply_pow_params(
    const std::unordered_map<std::string, std::string>& pow_params) {
    current_pow_params = pow_params;
    hasher->update_from_payload(pow_params);
    if (pow_params.find("model_identifier") != pow_params.end()) {
        proof_writer->set_model_identifier(pow_params.at("model_identifier"));
    }
    if (pow_params.find("compute_precision") != pow_params.end()) {
        proof_writer->set_compute_precision(pow_params.at("compute_precision"));
    }
    if (pow_params.find("ipfs_cid") != pow_params.end()) {
        proof_writer->set_ipfs_cid(pow_params.at("ipfs_cid"));
    }
}

void PowSamplingCoordinator::prepare_window_admission_(
    int seq_id, int row, const std::vector<int64_t>& context) {
    // A nonce admits exactly one window (§6) — msg_w changes at every
    // boundary, so always clear before any grind decision. A stale nonce
    // left set would corrupt every u of the new window.
    ring_buffers->write_admission_nonce(row, nullptr);

    // The v3 proof's prompt_tokens must be the pre-window prefix the
    // admission commitment binds (prompt + all previously generated
    // tokens), not just the original prompt — the verifier rebuilds both
    // msg_w and the commitment from that field. v2 keeps the legacy
    // set-once prompt_tokens (this helper only runs for version >= 3).
    static const std::vector<int64_t> k_empty_prefix;
    auto arch_it = seq_archive_.find(seq_id);
    const std::vector<int64_t>& prefix =
        arch_it != seq_archive_.end() ? arch_it->second : k_empty_prefix;
    sequence_info_map[seq_id]["prompt_tokens"] =
        std::vector<int32_t>(prefix.begin(), prefix.end());

    if (proof_version_ >= pow_v4::V4_PROOF_VERSION) {
        // Proof v4: compute the window-constant root. E4 (the effective step
        // nonce) is APPENDED to every step of this window; R is RECORDED in the
        // proof so the verifier recomputes A/E4. Free tier STILL computes A — it
        // waives only the admission-target comparison, not the root (skipping it
        // yields a different E4 that passes vectors but fails live replay). v4
        // never runs the v3 admission grind (R is random, not grinded).
        const std::vector<uint8_t> pad_mask(prefix.size(), 0);
        std::array<uint8_t, 32> e4{}, r{};
        if (hasher->compute_v4_window_root(context, prefix, pad_mask, e4, r)) {
            ring_buffers->write_admission_nonce(row, e4.data());  // append E4
            ring_buffers->v4_recorded_nonce[row] = r;             // record R
            ring_buffers->v4_recorded_valid[row] = 1;
        } else {
            logger->log("v4 window-root derivation failed for seq " +
                        std::to_string(seq_id) + " — window skipped", "ERROR");
        }
        return;
    }

    if (admission_mode_ != "always") {
        return;  // v3-inert: mine nonce-less; [B_FLOOR, B_FREE) forfeited
    }
    if (!pow_v3::argon2_compiled()) {
        // Defensive: assert_v3_ready_() already failed startup for this.
        if (!admission_warned_no_grinder_) {
            admission_warned_no_grinder_ = true;
            logger->log(
                "v3 admission requested but this build lacks libargon2 — "
                "mining nonce-less; [B_FLOOR, B_FREE) windows will be "
                "forfeited", "ERROR");
        }
        return;
    }
    // Canonical v3 pad mask: all-false, one entry per prefix token (the
    // verifier derives the same shape when the proof omits it).
    const std::vector<uint8_t> pad_mask(prefix.size(), 0);
    std::array<uint8_t, 32> nonce;
    if (hasher->grind_admission(context, 0, prefix, pad_mask,
                                admission_max_tries_factor_, nonce)) {
        ring_buffers->write_admission_nonce(row, nonce.data());
    } else {
        logger->log(
            "v3 admission grind exhausted/skipped for seq " +
            std::to_string(seq_id) + " — window mined nonce-less", "WARN");
    }
}

bool PowSamplingCoordinator::activate_sequence_params(int seq_id) {
    if (active_pow_seq_id.has_value() && active_pow_seq_id.value() == seq_id) {
        return true;
    }
    auto it = seq_pow_params_map.find(seq_id);
    if (it == seq_pow_params_map.end()) {
        return false;
    }
    apply_pow_params(it->second);
    active_pow_seq_id = seq_id;
    return true;
}

int64_t PowSamplingCoordinator::now_ms_() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

void PowSamplingCoordinator::initialize(const std::string& log_dir, const std::string& proof_dir) {
    logger = std::make_unique<Logger>(log_dir);
    row_manager = std::make_unique<RowManager>(max_concurrency);
    ring_buffers = std::make_unique<RingBuffers>(window_size, max_concurrency);
    hasher = std::make_unique<PowHasher>();
    const std::string resolved_proof_dir = proof_dir.empty()
        ? get_env_var("PROOF_SAVE_DIR", "/tmp/llama_pow_proofs")
        : proof_dir;
    proof_writer = std::make_unique<ProofWriter>(resolved_proof_dir);
    submitter_ = std::make_unique<MiningResponseSubmitter>();
    seq_pow_params_map.clear();
    active_pow_seq_id.reset();
    seq_archive_.clear();
    cooldown_duration_ms_ = 1000 * std::max(
        0,
        std::stoi(get_env_var("MINING_SOLUTION_COOLDOWN_SEC", "0"))
    );
    cooldown_until_ms_.store(0);

    // v3 opt-in (TIP-0003), mirroring the vLLM sampler env
    // contract: POW_PROOF_VERSION=3 switches the emitted proof schema to the
    // v3 carrier; POW_V3_ADMISSION_MODE=always grinds an admission nonce at
    // every window boundary ('off' mines nonce-less and forfeits
    // [B_FLOOR, B_FREE) windows). Defaults stay 2/'off' until activation.
    proof_version_ = std::stoi(get_env_var("POW_PROOF_VERSION", "2"));
    admission_mode_ = get_env_var("POW_V3_ADMISSION_MODE", "off");
    // .strip().lower() like the Python side
    while (!admission_mode_.empty() && std::isspace((unsigned char)admission_mode_.front())) admission_mode_.erase(admission_mode_.begin());
    while (!admission_mode_.empty() && std::isspace((unsigned char)admission_mode_.back())) admission_mode_.pop_back();
    std::transform(admission_mode_.begin(), admission_mode_.end(), admission_mode_.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    try {
        admission_max_tries_factor_ = static_cast<uint64_t>(std::max(
            1, std::stoi(get_env_var("POW_V3_GRIND_MAX_TRIES_FACTOR", "16"))));
    } catch (const std::exception&) {
        admission_max_tries_factor_ = 16;
    }
    proof_writer->set_proof_version(proof_version_);
    if (proof_version_ >= pow_v3::V3_PROOF_VERSION) {
        assert_v3_ready_();
    }

    logger->log("PoW Sampling Coordinator initialized", "INFO");
}

void PowSamplingCoordinator::assert_v3_ready_() {
    if (!pow_v3::argon2_compiled()) {
        throw std::runtime_error(
            "POW_PROOF_VERSION >= 3 but this llama-server build lacks "
            "libargon2 (compiled without POW_V3_HAVE_ARGON2) — a v3 miner "
            "without Argon2 would silently forfeit every admission-band "
            "window. Rebuild with libargon2-dev or mine v2.");
    }
    // All-0xff little-endian target = max uint256: any digest except
    // all-0xff passes the strict <, so the first try wins.
    auto msg_w = pow_v3::build_step_message(
        std::vector<uint8_t>(76, 0), std::vector<uint8_t>(32, 0), 0, 0,
        std::vector<int64_t>{}, "selftest");
    auto commitment = pow_v3::prompt_commitment(
        std::vector<int64_t>{}, std::vector<uint8_t>{});
    std::array<uint8_t, 32> target_le;
    target_le.fill(0xff);
    auto nonce = pow_v3::admission_grind(msg_w, "v3-selftest", target_le, 4,
                                         commitment);
    if (!nonce.has_value()) {
        throw std::runtime_error(
            "v3 startup self-test: admission_grind found no nonce against "
            "the trivial target — Argon2 backend is broken.");
    }
    logger->log("v3 startup self-test passed: native Argon2 admission "
                "grinder operational", "INFO");
}

void PowSamplingCoordinator::update_pow_params(
    const std::unordered_map<std::string, std::string>& pow_params) {
    apply_pow_params(pow_params);
    active_pow_seq_id.reset();
}

void PowSamplingCoordinator::update_pow_params_for_sequence(
    int seq_id,
    const std::unordered_map<std::string, std::string>& pow_params) {
    auto it = seq_pow_params_map.find(seq_id);
    if (it != seq_pow_params_map.end() && it->second == pow_params) {
        return;
    }
    seq_pow_params_map[seq_id] = pow_params;
    if (active_pow_seq_id.has_value() && active_pow_seq_id.value() == seq_id) {
        apply_pow_params(pow_params);
    }
}

void PowSamplingCoordinator::ensure_sequences(
    const std::vector<int>& seq_ids,
    const std::unordered_map<int, std::vector<int64_t>>& prompt_mapping) 
{
    for (int seq_id : seq_ids) {
        // 1) Skip if already allocated
        if (row_manager->get_row(seq_id).has_value()) {
            continue;
        }

        // 3) Try to allocate a row
        auto new_row = row_manager->allocate_row(seq_id);
        if (!new_row.has_value()) {
            // a) no free row → evict the oldest and retry
            auto [oldest_sid, oldest_row] = 
                row_manager->get_oldest_sequence(ring_buffers->steps);
            if (oldest_sid.has_value()) {
                cleanup_sequence(oldest_sid.value());
                new_row = row_manager->allocate_row(seq_id);
            }
        }

        // 4) If allocation succeeded, clear that slot
        if (new_row.has_value()) {
            int row = *new_row;
            ring_buffers->clear_row(row);
            logger->log(
                "Allocated row " + std::to_string(row) + 
                " for sequence " + std::to_string(seq_id),
                "DEBUG"
            );
        }
    }
}

bool PowSamplingCoordinator::is_pow_sequence(int seq_id) const {
    return row_manager->get_row(seq_id).has_value();
}

void PowSamplingCoordinator::cleanup_sequence(int seq_id) {
    seq_pow_params_map.erase(seq_id);
    if (active_pow_seq_id.has_value() && active_pow_seq_id.value() == seq_id) {
        active_pow_seq_id.reset();
    }

    auto row_opt = row_manager->free_row(seq_id);
    if (row_opt.has_value()) {
        ring_buffers->clear_row(row_opt.value());
        logger->log("Cleaned up sequence " + std::to_string(seq_id), "DEBUG");
        sequence_info_map.erase(seq_id);
    }
    seq_archive_.erase(seq_id);
}

void PowSamplingCoordinator::check_solutions(const std::vector<int>& seq_ids) {
    if (cooldown_duration_ms_ > 0 && is_cooldown_active()) {
        logger->log(
            "Skipping solution checks during cooldown (" +
            std::to_string(cooldown_remaining_seconds()) + "s remaining)",
            "DEBUG"
        );
        return;
    }

    for (int seq_id : seq_ids) {
        if (!activate_sequence_params(seq_id)) {
            logger->log("Skipping solution check: no PoW params for sequence " + std::to_string(seq_id), "WARNING");
            continue;
        }
        auto seq_pow_it = seq_pow_params_map.find(seq_id);
        if (seq_pow_it == seq_pow_params_map.end()) {
            continue;
        }
        const auto & seq_pow_params = seq_pow_it->second;

        auto row_opt = row_manager->get_row(seq_id);
        if (!row_opt.has_value()) continue;
        int row = row_opt.value();
        int nsteps = ring_buffers->steps[row];

        // only check at window boundaries
        if (nsteps <= 0 || nsteps % window_size != 0) {
            continue;
        }

        logger->log(
            "Checking seq_id=" + std::to_string(seq_id) +
            " row=" + std::to_string(row) +
            " nsteps=" + std::to_string(nsteps),
            "DEBUG"
        );

        // Get full window data including the last token generated
        auto window_data = ring_buffers->get_window(row);
        auto& chosen_tokens = std::any_cast<const std::vector<int32_t>&>(window_data.at("chosen_tokens"));
        
        // Convert tokens to context for hash computation
        std::vector<int64_t> context(chosen_tokens.begin(), chosen_tokens.end());
        
        // Use step 0 (start of new cycle) consistent with Python
        int32_t step_offset = 0;

        // v3 (TIP-0003): the final target-critical digest folds
        // the SAME nonce every step of this window hashed — the derived
        // header nonce must match verifier replay. The next window's grind
        // has not run yet (it happens at that window's first sampled token),
        // so the ring buffer still holds this window's nonce.
        const uint8_t* admission_nonce_ptr = nullptr;
        if (proof_version_ >= pow_v3::V3_PROOF_VERSION &&
            ring_buffers->admission_valid[row]) {
            admission_nonce_ptr = ring_buffers->admission_nonce[row].data();
        }

        // Recompute digest with step 0 and full context including last token
        auto [token_id, u_val, digest] = hasher->sample_token(context, step_offset, {1.0f}, admission_nonce_ptr);

        // Slice 11.4 — dual-threshold check. Block-tier gates on
        // ``target``; share-tier on ``share_target`` (broker-derived,
        // empty when share mode is off). A digest meeting share but
        // not block is a sub-block share emission; the cpp branch
        // in proof_processor routes it to MiningResponseSubmitter::
        // submit_share when in broker mode.
        bool is_solution = hasher->check_solution({ digest })[0];
        bool is_share = hasher->check_share_solution({ digest })[0];

        bool proxy_audit_enabled = false;
        const char* proxy_env = std::getenv("POW_PROXY_ENABLE");
        if (proxy_env != nullptr) {
            std::string proxy_str(proxy_env);
            proxy_audit_enabled = (proxy_str == "1" || proxy_str == "true" || proxy_str == "True");
        }

        // Emit on EITHER threshold met (or audit). Without share-
        // target, is_share is always false → behaviour matches the
        // pre-slice-11 block-only path.
        if (!is_solution && !is_share && !proxy_audit_enabled) {
            continue;
        }

        // Extract nonce and proceed with processing
        uint32_t nonce =
            static_cast<uint32_t>(digest[0])        |
            (static_cast<uint32_t>(digest[1]) << 8) |
            (static_cast<uint32_t>(digest[2]) << 16)|
            (static_cast<uint32_t>(digest[3]) << 24);

        auto seq_info = sequence_info_map.count(seq_id)
            ? sequence_info_map[seq_id]
            : std::unordered_map<std::string, std::any>{};

        // v3 (§3/§9): hand the window's selected nonce to the writer as 64
        // lowercase hex — write_proof merges it into extra_flags through the
        // shared helper. The writer never re-grinds.
        auto proof_pow_params = seq_pow_params;
        if (proof_version_ >= pow_v4::V4_PROOF_VERSION) {
            // v4: RECORD R (not the appended E4). The verifier recomputes A and
            // E4 from R; write_proof emits the v4 carrier (v4:1 + v3.stepbind:1).
            if (ring_buffers->v4_recorded_valid[row]) {
                proof_pow_params["admission_nonce"] = bytes_to_hex(
                    std::vector<uint8_t>(ring_buffers->v4_recorded_nonce[row].begin(),
                                         ring_buffers->v4_recorded_nonce[row].end()));
            }
        } else if (admission_nonce_ptr != nullptr) {
            proof_pow_params["admission_nonce"] = bytes_to_hex(
                std::vector<uint8_t>(admission_nonce_ptr, admission_nonce_ptr + 32));
        }

        auto [fb_bytes, proof_dict] = proof_writer->write_proof(
            seq_id,
            step_offset,  // Use step 0
            window_data,
            digest,       // Use recomputed digest
            is_solution,  // Pass actual solution status
            proof_pow_params,
            seq_info
        );

        if (submitter_) {
            int64_t req_id = std::stoll(std::any_cast<std::string>(proof_dict.at("request_id")));
            
            // Submit proof for audit if proxy is enabled (for ALL proofs)
            if (proxy_audit_enabled) {
                submitter_->submit_proof_for_audit(req_id, proof_dict);
            }
            
            // Submit block hits as solutions; submit share-only hits only
            // in broker mode. local_miner's primary egress is Core Node and
            // must never receive sub-block proofs.
            if (is_solution || (is_share && submitter_->is_broker_mode())) {
                uint32_t diff = std::stoul(std::any_cast<std::string>(proof_dict.at("difficulty")));
                auto proof_hash = sha256_single(fb_bytes);
                auto target_it = seq_pow_params.find("target");
                if (target_it == seq_pow_params.end()) {
                    logger->log("Cannot submit proof: missing target for sequence " + std::to_string(seq_id), "ERROR");
                    continue;
                }
                auto target_bytes = hex_to_bytes(target_it->second);
                uint32_t adjusted_bits = get_compact(target_bytes);
                if (is_solution) {
                    submitter_->submit_solution(req_id, nonce, adjusted_bits, proof_hash, diff, proof_dict);
                } else {
                    submitter_->submit_share(req_id, nonce, adjusted_bits, proof_hash, diff, proof_dict);
                }
            }
            if (is_solution) {
                activate_solution_cooldown();
                
                logger->log(
                    "PoW solution found for seq " + std::to_string(seq_id) +
                    " at step 0 (window boundary)",
                    "INFO"
                );
                return;
            }
        }
    }
}

void PowSamplingCoordinator::activate_solution_cooldown() {
    if (cooldown_duration_ms_ <= 0) {
        return;
    }
    const int64_t now = now_ms_();
    const int64_t until = now + cooldown_duration_ms_;
    cooldown_until_ms_.store(until);
    logger->log(
        "Activated solution cooldown for " +
        std::to_string(cooldown_duration_ms_ / 1000) + "s",
        "INFO"
    );
}

bool PowSamplingCoordinator::is_cooldown_active() const {
    return cooldown_until_ms_.load() > now_ms_();
}

double PowSamplingCoordinator::cooldown_remaining_seconds() const {
    const int64_t remaining_ms = cooldown_until_ms_.load() - now_ms_();
    return remaining_ms > 0 ? static_cast<double>(remaining_ms) / 1000.0 : 0.0;
}

void PowSamplingCoordinator::set_prompt_tokens(int seq_id,
                                               const std::vector<int32_t>& tokens) {
    sequence_info_map[seq_id]["prompt_tokens"] = tokens;
    // Seed the prompt+generated archive (v3 pre-window prefix, §6).
    seq_archive_[seq_id].assign(tokens.begin(), tokens.end());
}

void PowSamplingCoordinator::set_completion_id(int seq_id,
                                               const std::string& completion_id) {
    sequence_info_map[seq_id]["completion_id"] = completion_id;
}

void PowSamplingCoordinator::record_complete_step(int seq_id, const SamplingResult& result, bool is_valid) {
    auto row_opt = row_manager->get_row(seq_id);
    if (!row_opt.has_value()) return;
    
    int row = row_opt.value();
    int pos = ring_buffers->steps[row] % window_size;
    
    // Store basic sampling results
    ring_buffers->chosen_tokens[pos][row] = result.token_id;
    ring_buffers->chosen_probs[pos][row] = result.token_prob;
    ring_buffers->sampling_u[pos][row] = result.u_value;
    ring_buffers->softmax_normalizers[pos][row] = result.softmax_log_z;
    ring_buffers->attention_mask[pos][row] = is_valid;
    ring_buffers->digest_buffer[row][pos] = result.digest;
    
    // Fix the comparison warnings
    for (size_t i = 0; i < 71 && i < result.topk_logits.size(); i++) {
        ring_buffers->topk_logits[pos][row][i] = result.topk_logits[i];
        ring_buffers->topk_indices[pos][row][i] = result.topk_indices[i];
    }
    
    for (size_t i = 0; i < 6 && i < result.logsumexp_stats.size(); i++) {
        ring_buffers->logsumexp_stats[pos][row][i] = result.logsumexp_stats[i];
    }

    // Extend the prompt+generated archive (v3 pre-window prefix, §6).
    auto arch_it = seq_archive_.find(seq_id);
    if (arch_it != seq_archive_.end()) {
        arch_it->second.push_back(result.token_id);
    }

    ring_buffers->increment_steps({row});
    
    logger->log("Recorded step for pos="+ std::to_string(pos) +" seq_id="+ std::to_string(seq_id), "DEBUG");
}
