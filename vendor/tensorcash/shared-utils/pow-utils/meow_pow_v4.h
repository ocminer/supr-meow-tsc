#ifndef MEOW_POW_V4_H
#define MEOW_POW_V4_H
// Minimal proof-v4 window-root derivation for the supr-meow-tsc miner.
//
// A byte-exact subset of bcore verification/pow_v4.{h,cpp} — ONLY the root
// functions the PRODUCER needs (sampler_domain_id / admit_message_v4 /
// admit_root_v4 / step_root_v4 / derive_v4_root). The full pow_v4 (tier rules,
// priced target, decide_proof_mode) is verifier-side and pulls
// arith_uint256 + consensus/params deps this miner does not carry, so it is
// deliberately NOT vendored here. Parity with the golden contract is pinned by
// tests/v4_root_parity.cpp against tests/vectors/v4_vectors.json (the SAME
// vectors the live verifiers run), so this subset can never drift from the
// authoritative pow_v4.cpp without the parity test failing.
//
// Depends only on the miner's flat pow_v3.h (step_digest = single SHA-256,
// argon2id_digest = the v3 Argon2id profile) + std.
#include "pow_v3.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pow_v4 {

constexpr int V4_PROOF_VERSION = 4;
constexpr std::size_t ROOT_BYTES = 32;

// TAG_ADMIT_V4 prefixes the Argon2id admission-root message, TAG_STEP_V4 the
// SHA-256 step-root message (verification/pow_v4.h §domain separation).
constexpr char TAG_ADMIT_V4[] = "TC_V4_ADMIT_ROOT";
constexpr std::size_t TAG_ADMIT_V4_LEN = 16;
static_assert(sizeof(TAG_ADMIT_V4) == TAG_ADMIT_V4_LEN + 1, "TAG_ADMIT_V4 must be 16 bytes");
constexpr char TAG_STEP_V4[] = "TC_V4_STEP_ROOT!";
constexpr std::size_t TAG_STEP_V4_LEN = 16;
static_assert(sizeof(TAG_STEP_V4) == TAG_STEP_V4_LEN + 1, "TAG_STEP_V4 must be 16 bytes");

// S4 = SHA256(SAMPLER_DOMAIN_PREFIX || SAMPLER_MODE_SUFFIX). The v4 sampler is
// FROZEN as the chain-bound Gumbel race (gumbel_sampler.h), contract v2, head
// h0, arm ONE (draw 0 only, no switch); SAMPLER_MODE_SUFFIX is its 20-byte
// mode suffix (gumbel::mode_suffix_v2 for that frozen mode).
constexpr char SAMPLER_DOMAIN_PREFIX[] = "tensorcash/v4/sampler-domain/";
constexpr std::size_t SAMPLER_MODE_SUFFIX_BYTES = 20;
constexpr std::array<uint8_t, SAMPLER_MODE_SUFFIX_BYTES> SAMPLER_MODE_SUFFIX{{
    'T', 'C', 'G', 'M', 'O', 'D', 'E', '2',  // gumbel::MODE_TAG_V2
    2,                                        // gumbel::CONTRACT_VERSION_V2
    'h', '0',                                 // head h0
    1,                                        // arm ONE
    0, 0, 0, 0, 0, 0, 0, 0}};                 // policy digest slot: zero

struct V4Root {
    std::array<uint8_t, ROOT_BYTES> A{};   // admit root (Argon2id)
    std::array<uint8_t, ROOT_BYTES> E4{};  // effective step nonce (SHA-256)
};

// S4, 32 bytes, computed once (thread-safe).
const std::array<uint8_t, ROOT_BYTES>& sampler_domain_id();

// TAG_ADMIT_V4 || msg_w0 || C || u16le(len(model_id)) || model_id || R || S4.
std::vector<uint8_t> admit_message_v4(
    const std::vector<uint8_t>& msg_w0,
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::string& model_identifier,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce);

// A = Argon2id(admit_message), the unchanged v3 profile/salt; domain
// separation is TAG_ADMIT_V4 inside the message. Throws without libargon2.
std::array<uint8_t, ROOT_BYTES> admit_root_v4(const std::vector<uint8_t>& admit_message);

// E4 = SHA256(TAG_STEP_V4 || C || R || A || S4).
std::array<uint8_t, ROOT_BYTES> step_root_v4(
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce,
    const std::array<uint8_t, ROOT_BYTES>& admit_root);

// (A, E4) from the canonical inputs.
V4Root derive_v4_root(
    const std::vector<uint8_t>& msg_w0,
    const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
    const std::string& model_identifier,
    const std::array<uint8_t, ROOT_BYTES>& admission_nonce);

// The v4 extra_flags carrier the verifier reads (v4_dispatch: top-level
// "v4":1; v3.stepbind:1; v3.admission_nonce=R). admission_nonce_hex must be 64
// lowercase hex (bytes_to_hex guarantees it). A non-trivial existing
// extra_flags object is preserved verbatim under top-level "_diff" (the same
// escape hatch pow_v3::merge_extra_flags_v3 uses; the miner's diff is "{}").
std::string merge_extra_flags_v4(const std::string& existing,
                                 const std::string& admission_nonce_hex);

}  // namespace pow_v4
#endif  // MEOW_POW_V4_H
