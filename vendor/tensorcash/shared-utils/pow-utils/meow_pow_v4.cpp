// Minimal proof-v4 window-root derivation — byte-exact subset of bcore
// verification/pow_v4.cpp. See meow_pow_v4.h. The function bodies below are
// copied verbatim from pow_v4.cpp (sampler_domain_id / admit_message_v4 /
// admit_root_v4 / step_root_v4 / derive_v4_root); do not "clean up" — parity
// with the golden vectors is the only correctness criterion.
#include "meow_pow_v4.h"

#include <cstring>
#include <stdexcept>

namespace pow_v4 {

const std::array<uint8_t, ROOT_BYTES>& sampler_domain_id() {
    static const std::array<uint8_t, ROOT_BYTES> id = [] {
        std::vector<uint8_t> msg;
        msg.insert(msg.end(), SAMPLER_DOMAIN_PREFIX,
                   SAMPLER_DOMAIN_PREFIX + std::strlen(SAMPLER_DOMAIN_PREFIX));
        msg.insert(msg.end(), SAMPLER_MODE_SUFFIX.begin(), SAMPLER_MODE_SUFFIX.end());
        return pow_v3::step_digest(msg);  // single SHA-256
    }();
    return id;
}

std::vector<uint8_t> admit_message_v4(
    const std::vector<uint8_t>& msg_w0,
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::string& model_identifier,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce) {
    if (model_identifier.size() > 0xFFFF) {
        throw std::invalid_argument("model_identifier too long for u16le length prefix");
    }
    const std::array<uint8_t, ROOT_BYTES>& s4 = sampler_domain_id();
    std::vector<uint8_t> msg;
    msg.reserve(TAG_ADMIT_V4_LEN + msg_w0.size() + ROOT_BYTES + 2 +
                model_identifier.size() + ROOT_BYTES + ROOT_BYTES);
    msg.insert(msg.end(), TAG_ADMIT_V4, TAG_ADMIT_V4 + TAG_ADMIT_V4_LEN);
    msg.insert(msg.end(), msg_w0.begin(), msg_w0.end());
    msg.insert(msg.end(), prompt_commitment_digest.begin(), prompt_commitment_digest.end());
    const uint16_t mid_len = static_cast<uint16_t>(model_identifier.size());
    msg.push_back(static_cast<uint8_t>(mid_len & 0xFF));
    msg.push_back(static_cast<uint8_t>((mid_len >> 8) & 0xFF));
    msg.insert(msg.end(), model_identifier.begin(), model_identifier.end());
    msg.insert(msg.end(), admission_nonce.begin(), admission_nonce.end());
    msg.insert(msg.end(), s4.begin(), s4.end());
    return msg;
}

std::array<uint8_t, ROOT_BYTES> admit_root_v4(const std::vector<uint8_t>& admit_message) {
    // Same Argon2id profile and salt as the v3 admission puzzle; the domain
    // separation is TAG_ADMIT_V4 inside the message. Throws without libargon2.
    return pow_v3::argon2id_digest(admit_message);
}

std::array<uint8_t, ROOT_BYTES> step_root_v4(
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce,
    const std::array<uint8_t, ROOT_BYTES>& admit_root) {
    const std::array<uint8_t, ROOT_BYTES>& s4 = sampler_domain_id();
    std::vector<uint8_t> msg;
    msg.reserve(TAG_STEP_V4_LEN + 4 * ROOT_BYTES);
    msg.insert(msg.end(), TAG_STEP_V4, TAG_STEP_V4 + TAG_STEP_V4_LEN);
    msg.insert(msg.end(), prompt_commitment_digest.begin(), prompt_commitment_digest.end());
    msg.insert(msg.end(), admission_nonce.begin(), admission_nonce.end());
    msg.insert(msg.end(), admit_root.begin(), admit_root.end());
    msg.insert(msg.end(), s4.begin(), s4.end());
    return pow_v3::step_digest(msg);  // single SHA-256
}

V4Root derive_v4_root(
    const std::vector<uint8_t>& msg_w0,
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::string& model_identifier,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce) {
    V4Root root;
    root.A = admit_root_v4(admit_message_v4(msg_w0, prompt_commitment_digest,
                                            model_identifier, admission_nonce));
    root.E4 = step_root_v4(prompt_commitment_digest, admission_nonce, root.A);
    return root;
}

std::string merge_extra_flags_v4(const std::string& existing,
                                 const std::string& admission_nonce_hex) {
    std::string carrier =
        "{\"v4\":1,\"v3\":{\"stepbind\":1,\"admission_nonce\":\"" +
        admission_nonce_hex + "\"}";
    const size_t a = existing.find_first_not_of(" \t\r\n");
    if (a != std::string::npos) {
        const size_t b = existing.find_last_not_of(" \t\r\n");
        const std::string t = existing.substr(a, b - a + 1);
        if (t != "{}") {
            carrier += ",\"_diff\":" + t;  // preserve caller's object verbatim
        }
    }
    carrier += "}";
    return carrier;
}

}  // namespace pow_v4
