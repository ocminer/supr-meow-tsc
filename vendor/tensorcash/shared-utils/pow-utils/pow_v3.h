#pragma once

// V3 prompt-binding / admission helpers (TIP-0003).
//
// C++ mirror of pow_v3.py — the two implementations must stay semantically
// IDENTICAL (the golden vectors in tests/vectors/v3_vectors.json are the
// contract). Self-contained on purpose: only std headers here, OpenSSL SHA-256
// and (optionally) libargon2 in the .cpp, so the file compiles standalone in
// the proof-processor module, in bcore, and in the standalone vector test.
//
// Scope (TIP-0003):
//   §3  extra_flags carrier   — merge_extra_flags_v3 / extract_admission_nonce_hex
//   §4  conservative B_cred   — mass_q63_for_step / credit_units_for_step /
//                               b_cred_units_from_bounds  (R=1024 table)
//   §5  tier rule             — tier_for_b_cred_units
//   §6  Argon2id admission    — admission_message / argon2id_digest /
//                               admission_expected_tries / admission_target_le /
//                               admission_valid / admission_grind
//   §7  v3 step hashing       — build_step_message / step_digest / step_u
//
// Consensus-determinism notes (same agreed deviations as pow_v3.py):
//   * B_cred accumulates in integer credit units via a checked-in R=1024 Q63
//     threshold table. Runtime tiering uses no log2/libm path; endpoint and
//     threshold rounding are conservative.
//   * expected_tries is integer-only from integer chain constants;
//     admission_target = (2^256 - 1) / expected_tries.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "bcred_table_r1024.h"  // R=1024 B_cred credit-threshold table (§4)

namespace pow_v3 {

// ------------------------------------------------------------------------- //
// Constants (values marked CALIBRATION are placeholders pending the TIP-0003
// calibration pass; they must come from consensus chain params at activation).
// Mirror pow_v3.py names/values exactly.
// ------------------------------------------------------------------------- //

constexpr int V3_PROOF_VERSION = 3;

constexpr std::size_t POW_WINDOW_SIZE = 256;      // mirrors pow_utils

constexpr std::size_t ADMISSION_NONCE_BYTES = 32;
constexpr std::size_t ADMISSION_NONCE_HEX_LEN = 64;  // exactly 64 lowercase hex

// Full-prefix commitment bound into admission (§6) — mirrors
// pow_v3.PROMPT_CTX_TAG / PROMPT_COMMITMENT_BYTES.
constexpr char PROMPT_CTX_TAG[] = "TC_V3_PROMPT_CTX";  // 16 bytes, no NUL
constexpr std::size_t PROMPT_CTX_TAG_LEN = 16;
constexpr std::size_t PROMPT_COMMITMENT_BYTES = 32;

// B_cred credit units (§4): R units == 1 bit. BCRED_R / BCRED_N_MAX /
// BCRED_THRESHOLD_Q63 come from bcred_table_r1024.h. Chain params carry the
// tiers as BITS (V3BFloorBits / V3BFreeBits); the tier comparison * R.
constexpr uint64_t BCRED_Q_ONE = 1ULL << 63;       // Q63 unit == mass 1.0
constexpr uint64_t B_FLOOR_BITS = 45;              // CALIBRATION (initial floor)
constexpr uint64_t B_FREE_BITS = 70;               // CALIBRATION (initial high tier)
constexpr uint64_t B_FLOOR_UNITS = B_FLOOR_BITS * BCRED_R;
constexpr uint64_t B_FREE_UNITS = B_FREE_BITS * BCRED_R;

// Per-step credit cap == the table's max index (32 bits worth of units). The
// 2*ATOL widening floors mass_q well above this (~12 bits), so the cap is
// purely defensive.
constexpr uint64_t B_STEP_MAX_UNITS = BCRED_N_MAX;

// Interval-mass tolerance — must equal the verifier's ATOL
// (services/verification-api/src/config/constants.py). ATOL_Q63_CEIL is the
// EXACT integer ceil(ATOL * 2^63); the mass widening adds 2*ATOL_Q63_CEIL. It
// is a checked-in constant (identical in pow_v3.py) so no float feeds the Q63
// arithmetic: ceil(0.0001_f64 * 2^63) == 922337203685478.
constexpr double ATOL = 0.0001;
constexpr uint64_t ATOL_Q63_CEIL = 922337203685478ULL;

// Consensus parser bounds for the v3 extra_flags carrier (§3) — identical in
// Python and C++; violations mean "no nonce claimed", never a parse crash.
constexpr std::size_t EXTRA_FLAGS_MAX_BYTES = 4096;  // CALIBRATION
constexpr int EXTRA_FLAGS_MAX_DEPTH = 8;             // CALIBRATION

// v3.0 sampler profile (§1, §2) — CONSENSUS-FIXED, not miner- or model-
// chosen. Enforcement (verifier side) is exact equality against the proof's
// sampler fields; mirrors pow_v3.SAMPLER_PROFILE_V3.
constexpr float SAMPLER_V3_TEMPERATURE = 1.0f;
constexpr float SAMPLER_V3_TOP_P = 1.0f;
constexpr uint32_t SAMPLER_V3_TOP_K = 50;
constexpr float SAMPLER_V3_REPETITION_PENALTY = 1.0f;

// ELIG_ALPHA = 0.04 as an exact rational (§1).
constexpr uint64_t ELIG_ALPHA_NUM = 4;
constexpr uint64_t ELIG_ALPHA_DEN = 100;

// ARGON_PROFILE (§1): Argon2id variant, memory, iterations, lanes, output len.
constexpr uint32_t ARGON2_TIME_COST = 1;          // CALIBRATION
constexpr uint32_t ARGON2_MEMORY_KIB = 8192;      // CALIBRATION (8 MiB)
constexpr uint32_t ARGON2_LANES = 1;              // CALIBRATION
constexpr std::size_t ARGON2_HASH_LEN = 32;
// Fixed public salt: pure domain-separation constant (the puzzle's entropy is
// in the message; Argon2 requires a salt >= 8 bytes). 16 bytes, never changes.
constexpr char ARGON2_SALT[] = "TC_V3_ADMISSION!";
constexpr std::size_t ARGON2_SALT_LEN = 16;
static_assert(sizeof(ARGON2_SALT) == ARGON2_SALT_LEN + 1,
              "ARGON2_SALT must be exactly 16 bytes");

// Reference timings in integer microseconds (§1) — integer so the target
// derivation is exact in every language.
constexpr uint64_t ARGON_REF_US = 4000;               // CALIBRATION: 4010us EPYC 7R13
constexpr uint64_t DECODE_US_AT_NORMALIZER = 6000000; // CALIBRATION: B=1 5.97s L40S -> tries=60
constexpr uint64_t MODEL_DIFFICULTY_NORMALIZER = 1000000;  // consensus/params.h

// ------------------------------------------------------------------------- //
// §3 — extra_flags carrier
// ------------------------------------------------------------------------- //

// Exactly 64 lowercase hex chars (consensus shape rule).
bool is_valid_admission_nonce_hex(const std::string& value);

// Producer-side merge of {"v3":{"admission_nonce":"<hex>"}} into an existing
// extra_flags / model_config_diff string WITHOUT a JSON library, mirroring the
// hand-splice idiom of the audit marker in proof_processor.cpp. Existing
// top-level members are preserved verbatim; any existing top-level "v3"
// member is removed via a string-escape-aware balanced scan and replaced, so
// the operation is idempotent. Inputs that are not a {...} JSON object are
// preserved under "_diff" (matching pow_v3.merge_extra_flags_v3); an empty /
// whitespace-only input yields the bare v3 object. Consensus does NOT require
// canonical JSON (§3) — this output is parseable JSON, not byte-canonical.
// Throws std::invalid_argument when nonce_hex fails the shape rule.
std::string merge_extra_flags_v3(const std::string& extra_flags,
                                 const std::string& nonce_hex);

// Consensus extraction + shape rule (§3 parser bounds), mirror of
// pow_v3.extract_admission_nonce. Returns the 64 lowercase hex chars of the
// nonce, or std::nullopt when no admission is claimed. NEVER throws.
//
// A nonce is claimed only by the exact shape
// {"v3":{"admission_nonce":"<64 lowercase hex>"}, ...}. ANY violation —
// empty/oversized input (> EXTRA_FLAGS_MAX_BYTES), invalid UTF-8,
// unparseable JSON, DUPLICATE object keys (any level), nesting deeper than
// EXTRA_FLAGS_MAX_DEPTH, non-object top level, "v3" not an object, key
// absent, or a nonce value that is not a string of exactly 64 lowercase hex
// chars — means NO nonce claimed. Implemented as a small bounded
// recursive-descent JSON validator (no JSON library) matching Python
// json.loads acceptance, including NaN/Infinity literals and \uXXXX escapes.
std::optional<std::string> extract_admission_nonce_hex(
    const std::string& extra_flags);

// ------------------------------------------------------------------------- //
// §7 — v3 step message (byte-exact mirror of pow_v3.build_step_message)
// ------------------------------------------------------------------------- //

// header_prefix | vdf | u32le(tick) | u32le(step) | ctx_window | precision
// [| nonce32]. ctx_window: the LAST window_size tokens of context_tokens,
// left-padded with zeros to exactly window_size entries, 8 bytes
// little-endian per token — identical to pow_utils tok_le_bytes on the
// rolling window and to QuickVerifier::ComputeUValue. nonce32_or_null: 32
// raw admission-nonce bytes appended when non-null (§7), null for the legacy
// v2 shape.
std::vector<uint8_t> build_step_message(
    const std::vector<uint8_t>& header_prefix,
    const std::vector<uint8_t>& vdf,
    uint32_t tick,
    uint32_t step,
    const std::vector<int64_t>& context_tokens,
    const std::string& precision,
    const uint8_t* nonce32_or_null = nullptr,
    std::size_t window_size = POW_WINDOW_SIZE);

// Single (not double) SHA-256 of the step message — mirrors
// pow_v3.step_u_from_message's digest.
std::array<uint8_t, 32> step_digest(const std::vector<uint8_t>& message);

// u = little-endian uint32 of the first 4 digest bytes / 2^32, as double —
// identical to pow_utils._digest_to_u and QuickVerifier::DigestToU.
double step_u_from_digest(const std::array<uint8_t, 32>& digest);

// ------------------------------------------------------------------------- //
// §6 — Argon2id admission puzzle
// ------------------------------------------------------------------------- //

// SHA256("TC_V3_PROMPT_CTX" | u32le(n_tokens) | prompt_tokens_i64le
// | u32le(n_mask) | pad_mask_u8) — commits to the FULL model-visible prefix
// of the window (§6). prompt_tokens is the proof's existing field (for later
// windows it already includes the previously generated tokens); token layout is
// the same 8-byte little-endian encoding the sampler preimage uses; pad_mask is
// one byte per bool in proof order and must have exactly one entry per prompt
// token. Mirrors pow_v3.prompt_commitment.
std::array<uint8_t, 32> prompt_commitment(
    const std::vector<int64_t>& prompt_tokens,
    const std::vector<uint8_t>& pad_mask);

// msg_w | prompt_commitment(32B) | u16le(len(model_identifier))
// | model_identifier | nonce (§6).
// msg_w is build_step_message(...) at the window's FIRST step WITHOUT the
// nonce appended (the nonce enters here explicitly). The prompt commitment
// binds the FULL model-visible prefix — msg_w's rolling window alone would
// let a miner vary out-of-window prefix tokens (which the model conditions
// on) and amortize one admission across decode paths. model_identifier is
// length-prefixed because it is the only variable-length field between two
// fixed-layout regions. Throws std::invalid_argument when model_identifier
// exceeds the u16le prefix.
std::vector<uint8_t> admission_message(const std::vector<uint8_t>& msg_w,
                                       const std::string& model_identifier,
                                       const uint8_t nonce[32],
                                       const std::array<uint8_t, 32>& prompt_commitment_digest);

// Raw Argon2id digest of the admission message (§6) with the ARGON_PROFILE
// constants above and the fixed 16-byte salt. Requires libargon2: built with
// -DPOW_V3_HAVE_ARGON2 this calls argon2id_hash_raw(); without it, it throws
// std::runtime_error (paths that never grind/verify admission stay buildable).
std::array<uint8_t, 32> argon2id_digest(const std::vector<uint8_t>& message);

// True iff argon2id_digest() above is functional in this binary (pow_v3.cpp
// compiled with POW_V3_HAVE_ARGON2). Startup capability guards use this to
// refuse a finite V3ActivationHeight on a binary that cannot verify
// admission — an argonless full node at a v3-active height would reject
// every consensus-valid admission-band block and fork off.
bool argon2_compiled() noexcept;

// Integer-exact §6 derivation (mirror of pow_v3.admission_expected_tries):
//     expected_tries = floor((alpha_num * decode_us_at_normalizer * normalizer)
//                            / (alpha_den * argon_ref_us * difficulty))
// clamped to >= 1. Registered `difficulty` is an INVERSE compute scalar: more
// FLOPs => LOWER difficulty => more tries. The numerator is computed in
// unsigned __int128 so the default chain constants can never overflow.
// Throws std::invalid_argument on difficulty <= 0 (and std::overflow_error
// in the never-expected case that the result exceeds uint64).
uint64_t admission_expected_tries(
    int64_t difficulty,
    uint64_t normalizer = MODEL_DIFFICULTY_NORMALIZER,
    uint64_t decode_us_at_normalizer = DECODE_US_AT_NORMALIZER,
    uint64_t elig_alpha_num = ELIG_ALPHA_NUM,
    uint64_t elig_alpha_den = ELIG_ALPHA_DEN,
    uint64_t argon_ref_us = ARGON_REF_US);

// admission_target = (2^256 - 1) / expected_tries, returned as the
// LITTLE-ENDIAN 32-byte value (the comparison domain of admission_valid).
// 256-bit-by-64-bit long division over four uint64 limbs — no bigint dep.
std::array<uint8_t, 32> admission_target_le(
    int64_t difficulty,
    uint64_t normalizer = MODEL_DIFFICULTY_NORMALIZER,
    uint64_t decode_us_at_normalizer = DECODE_US_AT_NORMALIZER,
    uint64_t elig_alpha_num = ELIG_ALPHA_NUM,
    uint64_t elig_alpha_den = ELIG_ALPHA_DEN,
    uint64_t argon_ref_us = ARGON_REF_US);

// uint256_le(digest) < uint256_le(target) — STRICT less-than, both read as
// little-endian uint256 (§6).
bool admission_valid(const std::array<uint8_t, 32>& digest,
                     const std::array<uint8_t, 32>& target_le);

// Native admission grind loop (§9): the vLLM sampler calls this (with the
// GIL released via the pybind11 wrapper) so no Python nonce loop exists.
// Starts from a std::random_device 32-byte nonce, increments it as a
// little-endian counter per try, and returns the first nonce whose Argon2id
// digest satisfies admission_valid, or std::nullopt after max_tries.
std::optional<std::array<uint8_t, 32>> admission_grind(
    const std::vector<uint8_t>& msg_w,
    const std::string& model_identifier,
    const std::array<uint8_t, 32>& target_le,
    uint64_t max_tries,
    const std::array<uint8_t, 32>& prompt_commitment_digest);

// ------------------------------------------------------------------------- //
// §4 — numerically conservative B_cred
// ------------------------------------------------------------------------- //

// EXACT floor/ceil of (x * 2^63) for a finite double x in [0, 1], via mantissa
// decomposition (frexp) and integer shifts — NOT (uint64)(x * 9.22e18), whose
// double multiply would round in the 52-bit mantissa and be FMA/platform
// dependent. Used to quantise interval endpoints for the conservative mass.
uint64_t f64_to_q63_floor(double x);
uint64_t f64_to_q63_ceil(double x);

// Conservative interval mass of one step in Q63 fixed point (§4). Quantises
// the ENDPOINTS (upper via ceil, lower via floor) and adds 2*atol_q63_ceil;
// every rounding direction OVER-estimates the mass so credit never over-counts.
// Throws std::invalid_argument on NaN/Inf/upper < lower. Clamped to [0, 2^63].
uint64_t mass_q63_for_step(double lower, double upper,
                           uint64_t atol_q63_ceil = ATOL_Q63_CEIL);

// Credit units for one step: the largest n in [0, N_MAX] with
// BCRED_THRESHOLD_Q63[n] >= mass_q63 (§4). n == 0 always qualifies (mass is
// clamped to <= 2^63 == threshold[0]), so 0 <= credit <= N_MAX (per-step cap).
// Throws std::invalid_argument on mass_q63 == 0 (invalid interval earns none).
uint64_t credit_units_for_step(uint64_t mass_q63);

// Sum of per-step credit units over the window (§4). A SEPARATE integer
// accumulator (<= 256 * N_MAX), exact and reduction-order independent. Both
// quick and full verification call this on their bounds. Throws
// std::invalid_argument on length mismatch or invalid bounds — proof invalid.
uint64_t b_cred_units_from_bounds(const std::vector<double>& lower_bounds,
                                  const std::vector<double>& upper_bounds,
                                  uint64_t atol_q63_ceil = ATOL_Q63_CEIL);

// ------------------------------------------------------------------------- //
// §5 — tier rule
// ------------------------------------------------------------------------- //

enum class Tier {
    Invalid,            // B_cred < B_FLOOR
    AdmissionRequired,  // B_FLOOR <= B_cred < B_FREE
    Free,               // B_cred >= B_FREE
};

// String names matching pow_v3.TIER_* ("invalid" | "admission_required" |
// "free") for logs and the cross-language vectors.
const char* tier_name(Tier tier);

// B_cred < B_FLOOR -> invalid; < B_FREE -> admission required; else free. All
// comparisons in integer credit units (R units == 1 bit). Callers must also
// enforce: a PRESENT admission nonce is verified regardless of tier (present
// => valid), and absent + admission_required => invalid (§5).
Tier tier_for_b_cred_units(uint64_t b_cred_units,
                           uint64_t b_floor_units = B_FLOOR_UNITS,
                           uint64_t b_free_units = B_FREE_UNITS);

}  // namespace pow_v3
