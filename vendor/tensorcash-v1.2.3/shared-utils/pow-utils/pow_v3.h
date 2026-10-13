#pragma once

// V3 prompt-binding / admission helpers (TIP-0003).
//
// C++ mirror of pow_v3.py — the two implementations must stay semantically
// IDENTICAL (the golden vectors in tests/vectors/v3_vectors.json are the
// contract). Self-contained on purpose: only std headers here, OpenSSL SHA-256
// and (optionally) libargon2 in the .cpp, so the file compiles standalone in
// the proof-processor module, in bcore, and in the standalone vector test.
//
// Scope (TIP-0003 sections):
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
#include "term_table_r1024.h"   // R=1024 anti-parrot TERM_Q32 decay table (§8)

namespace pow_v3 {

// ------------------------------------------------------------------------- //
// Constants (values marked CALIBRATION are placeholders pending TIP-0003 §12; they must come from consensus chain params at activation).
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

// Effective step-binding nonce (bcore consensus StepBindHeight). Domain tag for
// the derivation that folds the whole-prompt commitment into the value appended
// to EVERY step preimage, so any prompt/pad_mask mutation perturbs every u draw
// at once. Mirrors pow_v3.STEP_BIND_TAG (Python) and the bcore verifier.
constexpr char STEP_BIND_TAG[] = "TC_V3_STEP_BIND1";  // 16 bytes, no NUL
constexpr std::size_t STEP_BIND_TAG_LEN = 16;

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

// Producer-side StepBind-capable carrier merge. Rebuilds the proof-owned v3
// object with an optional admission nonce and optional exact stepbind=1 marker.
// Used by the native ProofProcessor, including nonce-less free-tier proofs.
std::string merge_extra_flags_v3_fields(
    const std::string& extra_flags,
    const std::optional<std::string>& nonce_hex,
    bool stepbind);

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

// Exact integer v3.stepbind=1 under the same bounded strict parser. Any other
// shape/value/parse failure is false. Mirrors Python extract_stepbind_flag.
bool extract_stepbind_flag(const std::string& extra_flags);

enum class StepBindFlagState : uint8_t {
    ABSENT = 0,
    CLAIMED = 1,
    MALFORMED = 2,
};
StepBindFlagState extract_stepbind_flag_state(const std::string& extra_flags);

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

// Effective step-binding nonce E (StepBindHeight):
//   E = SHA256(STEP_BIND_TAG(16) | prompt_commitment(32) | present(1) | R(32))
// R = the claimed 32-byte admission nonce when present, else 32 zero bytes.
// When the StepBind gate is active this E — passed to build_step_message as
// nonce32_or_null — is what every step preimage and the final hash append, so
// every step commits to the whole prompt+pad_mask (global coupling ⇒ no
// per-step separability). present distinguishes an absent nonce from a claimed
// all-zero nonce. Mirrors pow_v3::effective_step_nonce (bcore) and the Python
// reference. The admission message (admission_message) is UNCHANGED.
std::array<uint8_t, 32> effective_step_nonce(
    const std::array<uint8_t, 32>& prompt_commitment_digest,
    bool nonce_present,
    const uint8_t nonce[32]);

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

// CONSERVATIVE LOWER interval mass in Q63 (§8): OPPOSITE rounding of
// mass_q63_for_step — floor(upper), ceil(lower), MINUS 2*atol — so it
// UNDER-estimates the true mass. Clamped to [0, 2^63]. Used only for the
// hard-pin test. Throws std::invalid_argument on NaN/Inf/upper < lower.
// Vendored byte-identical from bcore src/verification/pow_v3.{h,cpp}; the
// canonical credit kernel (credit_v4.cpp step 9) calls it, so this copy must
// exist here too — check_vendored_consensus_sources.py pins the body.
uint64_t mass_lower_q63_for_step(double lower, double upper,
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

// ------------------------------------------------------------------------- //
// §8 — anti-parrot metrics. C++-ONLY for the initial consensus ship: the C++
// contextual gate (VerifySequenceLightVectorized, reached from ConnectBlock /
// ContextualCheckBlock) is the sole consensus authority, so there is NO
// pow_v3.py mirror and NO cross-language golden vector for anti-parrot yet
// (both DEFERRED; the separate B_CRED cross-language contract is untouched).
// Every value is integer (Q63/Q32); no float feeds a decision. Q32 quantities
// are held in uint64 because the Q32 unit (2^32) does not fit uint32.
//
// KNOWN LIMITATION (order-2): the collision statistics are all PAIRWISE (order
// 2) — sum over step pairs of a similarity S(s,t). They measure how few
// distinct whole-CDF states the path realizes, and catch constant / repeated /
// token-renamed streams. They do NOT, by construction, detect higher-order
// structure whose every 2-marginal looks diverse — e.g. an order-4 rotation
// through 4 mutually-distinct states in a fixed cycle has near-maximal pairwise
// diversity (each pair differs) yet is deterministic at lag 4. Catching that
// needs an order-k (k>2) statistic (block/lagged self-similarity), which is
// DEFERRED. The N_states_eff count below still lower-bounds the realized state
// set, but a 4-state rotation reads N_states_eff ~4 (not collapsed). Documented
// so the gate is not mistaken for a general repetition detector.
// ------------------------------------------------------------------------- //

// Per-step 1-bit credit cap R; cumulative reuse cap (== TERM index cap); Q32
// similarity unit; and the hard-pin threshold on the conservative LOWER mass
// (p >= 1 - 2^-12).
constexpr uint64_t AP_R_STEP_CAP = BCRED_R;
constexpr uint64_t AP_MAXRS_CUM_CAP = BCRED_N_MAX;
constexpr uint64_t AP_Q32_ONE = 1ULL << 32;
constexpr uint64_t AP_HARD_PIN_THRESHOLD_Q63 = BCRED_Q_ONE - (1ULL << 51);

// Gate thresholds (§8/§12). These are the CONSENSUS values: the gate is
// scheduled on mainnet at AntiParrotHeight = 24150 (kernel/chainparams.cpp)
// and evaluates exactly these constants from that height. Changing any of
// them is a consensus change and needs its own coordinated activation.
//
// GATE POLICY (FINAL): anti-parrot is a STRUCTURAL validity rule and returns
// INVALID ONLY — it never makes an admission decision (Braw tiering + the
// admission band are owned by VerifyV3TierAndAdmission from CHAIN PARAMS, not
// duplicated here). A proof is INVALID iff any of: hard_run > L_HARD,
// state_HARD (state collapse), Bcap < BCAP_ABS_FLOOR, or backload_HARD.
// Everything else is FREE from anti-parrot's standpoint. The GREY bands and
// state_deficit are emitted TELEMETRY only and do NOT gate. See
// anti_parrot_reject_reason.
constexpr uint64_t AP_L_HARD = 80;                            // CONSENSUS (active from AntiParrotHeight): hard-pinned run
constexpr uint64_t AP_BCAP_ABS_FLOOR_UNITS = 20 * BCRED_R;    // CONSENSUS (active from AntiParrotHeight): absolute Bcap floor (20 bits)
// State-collapse thresholds, calibrated against the CONSERVATIVE (post-ATOL,
// UPPER-bound) corpus distribution. state_HARD is an OR of TWO joint tests
// (see anti_parrot_reject_reason / pow_v3.cpp compute):
//   state_HARD == (kappa_hi_u < KU_HARD  AND kappa_hi_id < KID_HARD)   // density
//              OR (N_states_hi_u < NU_HARD AND N_states_hi_id < NID_HARD) // count
// kappa catches RECURRENCE-DENSITY collapse (long low-density cycles);
// N_states catches SHORT cycles a density threshold conflates by length
// (kappa_hi ~1/3 at 3 steps but ~0.05 at 70 steps). EITHER joint condition
// triggers. Dev-sample (328 blocks; 12 caught by the state gate) thresholds:
//   kappa_hi_u  -> 0.09  (honest >= 0.1152) ; kappa_hi_id -> 0.045 (honest >= 0.0640)
//   N_states_hi_u -> 12  (parrots <= 11, honest >= 16) ; N_states_hi_id -> 6 (parrots <= 4, honest >= 8)
// The two arms are NOT redundant: 10 of the 12 state-caught blocks separate on
// BOTH kappa AND N_states, but 2 SHORT-CYCLE blocks (h21215 kappa_u=0.314
// kappa_id=0.136; h23416 kappa_u=0.316 kappa_id=0.171) have kappa OVERLAPPING
// the honest range and are caught ONLY by the N_states arm. The OR catches all
// 12 with zero honest false positives in this sample. Activated on mainnet at
// AntiParrotHeight = 24150 with these values.
constexpr uint64_t AP_KU_HARD_NUM = 9, AP_KU_HARD_DEN = 100;    // CONSENSUS (active from AntiParrotHeight): kappa_hi_u < 0.09 (gap .056..115)
constexpr uint64_t AP_KID_HARD_NUM = 9, AP_KID_HARD_DEN = 200;  // CONSENSUS (active from AntiParrotHeight): kappa_hi_id < 0.045 (gap .029..064)
constexpr uint64_t AP_NSTATES_U_HARD = 12;  // CONSENSUS (active from AntiParrotHeight): N_states_hi_u < 12 (gap 9..16)
constexpr uint64_t AP_NSTATES_ID_HARD = 6;  // CONSENSUS (active from AntiParrotHeight): N_states_hi_id < 6 (gap 4..8)
// GREY = a wider SHADOW band; each needs its OWN empirical gap before it may be
// promoted to invalid (grey_enforced). Placeholders, telemetry only for now.
constexpr uint64_t AP_KU_GREY_NUM = 10, AP_KU_GREY_DEN = 100;   // CONSENSUS (active from AntiParrotHeight): kappa_hi_u < 0.10 (shadow)
constexpr uint64_t AP_KID_GREY_NUM = 55, AP_KID_GREY_DEN = 1000;// CONSENSUS (active from AntiParrotHeight): kappa_hi_id < 0.055 (shadow)
constexpr uint64_t AP_NSTATES_U_GREY = 14;  // CONSENSUS (active from AntiParrotHeight): grey/shadow band (u)
constexpr uint64_t AP_NSTATES_ID_GREY = 7;  // CONSENSUS (active from AntiParrotHeight): grey/shadow band (id)
constexpr uint64_t AP_BACKLOAD_HARD_T = 96;                     // CONSENSUS (active from AntiParrotHeight): (maxRs-Bcap) > 96 => HARD (invalid)
constexpr uint64_t AP_BACKLOAD_GREY_T = 80;                     // CONSENSUS (active from AntiParrotHeight): (maxRs-Bcap) > 80 => GREY (telemetry)

// Cache-budgeted reuse score (§8) — a ROOT-ANCHORED single forward pass from
// step 0, distinct from maxRs (which RESTARTS the reuse accumulation at every
// checkpoint). The difference is exactly a 10-bit cache allowance (CACHE_CREDIT:
// the first ~2^10 cached prefix states are free) plus a 2-bit per-step branching
// cap (STEP_CAP). It does NOT improve fractional-credit handling over maxRs —
// maxRs is ALREADY exact-fractional; whole-bit rounding lives only in the legacy
// reuse gate, which is untouched. reuse_cache_q32 accumulates the per-step
// TERM_Q32 decay of the cache-adjusted cumulative credit; <= 256*2^32 == 2^40.
//
// HONEST LIMIT: the <= 2.0x modeled-advantage bound below is CONDITIONAL on the
// ~2^10 cached-prefix-state model AND on the conservative credit rc being a sound
// coverage proxy; larger cache budgets need the rc_12_2 shadow (reuse_cache12).
// It is NOT proof of skipped work. VALIDATION: on the full chain rc catches block
// 23961 (rc=233) plus 20 qash soft-trunk campaign blocks (hard_run~35 < 80,
// maxRs 65-126 < 160, Bcap 33-71) that hardrun/backload/kappa/Bcap all miss;
// clean empty gap [122.55..130.19] around 128; overlaps only 3 of the 41 existing
// INVALID, so rc does NOT replace the rest of the suite.
constexpr uint64_t AP_REUSE_CACHE_BITS = 10;                    // CONSENSUS (active from AntiParrotHeight): 2^10 cached prefix states
constexpr uint64_t AP_REUSE_CACHE_CREDIT = AP_REUSE_CACHE_BITS * BCRED_R;   // free-cache credit allowance
constexpr uint64_t AP_REUSE_STEP_CAP = 2 * BCRED_R;            // CONSENSUS (active from AntiParrotHeight): STEP_CAP == 2 B_CRED bits/step
constexpr uint64_t AP_REUSE_TERM_C_MAX = 32 * BCRED_R;         // TERM table max index (== BCRED_N_MAX)
constexpr uint64_t AP_REUSE_CACHE_CAP_Q32 = 128ULL << 32;      // CONSENSUS (active from AntiParrotHeight): CAP == 128 cached-token equiv (advantage <= 2.0x)
// SHADOW telemetry only (NEVER gates): identical accumulation with a 12-bit
// cache allowance, for calibrating larger cache budgets (rc_12_2).
constexpr uint64_t AP_REUSE_CACHE12_BITS = 12;                 // CONSENSUS (active from AntiParrotHeight): shadow 2^12 cache budget
constexpr uint64_t AP_REUSE_CACHE12_CREDIT = AP_REUSE_CACHE12_BITS * BCRED_R;

// Small-output-alphabet detector (§8) — SHADOW telemetry thresholds. The
// primary discriminator is Keff_out (dust-robust); K15_16 is a secondary
// count tripwire. The rule is CONJUNCTIVE (small alphabet AND supplied/collapsed
// AND model-concentrated) — NEVER standalone (Keff<8 alone false-positives on
// ~7 honest blocks). Consensus value, active from AntiParrotHeight.
constexpr uint64_t AP_KEFF0 = 8;             // CONSENSUS (active from AntiParrotHeight): Keff_out < 8 (dust-robust)
constexpr uint64_t AP_K1516_0 = 8;           // CONSENSUS (active from AntiParrotHeight): K15_16_out <= 8 (count tripwire)
constexpr uint64_t AP_NSID_RELAXED = 6;      // CONSENSUS (active from AntiParrotHeight): N_states_hi_id < 6 (collapsed support)
constexpr uint64_t AP_VUNIQUE_MIN = 12;      // CONSENSUS (active from AntiParrotHeight): Vunique_out < 12 ~ empty (corpus min 15)

// state_deficit monitoring metric: kappa in integer PPM units; KU_REF/KID_REF
// == the grey thresholds in the same units. Emitted only; does NOT gate.
constexpr uint64_t AP_KAPPA_UNITS_SCALE = 1000000ULL;
constexpr uint64_t AP_KU_REF_UNITS = AP_KU_GREY_NUM * AP_KAPPA_UNITS_SCALE / AP_KU_GREY_DEN;    // 100000
constexpr uint64_t AP_KID_REF_UNITS = AP_KID_GREY_NUM * AP_KAPPA_UNITS_SCALE / AP_KID_GREY_DEN; // 80000

// backload_surface return codes.
enum class BackloadSurface { Free = 0, Grey = 1, Hard = 2 };

// floor(x * 2^32) for a finite double x in [0, 1], via EXACT integer
// arithmetic (mirror of f64_to_q63_floor at Q32). Returns a value in
// [0, 2^32] held in uint64. A generic Q32 conversion helper (the anti-parrot
// collisions themselves are Q63; this is retained for Q32 callers/tests).
uint64_t f64_to_q32_floor(double x);

// §8 metric bundle. The collision sums are CONSERVATIVE LOWER bounds in Q63
// (similarity unit 2^63); they can reach ~2^99, so they are unsigned __int128
// (self <= 2^91). self_collision is the exact forced-diagonal sum w_t^2 * 2^63.
// self / total_collision_v is an UPPER bound on kappa (recurrence density);
// bcap^2 * 2^63 / total_collision_v is an UPPER bound on the state count.
struct AntiParrotMetrics {
    uint64_t braw = 0;                 // sum of per-step credit units
    uint64_t bcap = 0;                 // sum of min(credit_units, R)
    uint64_t hard_run = 0;             // longest consecutive hard-pinned run
    uint64_t maxrs_q32 = 0;            // restarted conditional reuse score (Q32)
    // Cache-budgeted reuse score (§8): ROOT-ANCHORED single pass from step 0 with
    // a 10-bit free-cache allowance and a 2-bit per-step branching cap (see the
    // AP_REUSE_* constants). <= 256*2^32 == 2^40, fits u64.
    uint64_t reuse_cache_q32 = 0;      // gating reuse score (10-bit cache)
    uint64_t reuse_cache12_q32 = 0;    // SHADOW telemetry ONLY (12-bit cache) — NEVER gates
    unsigned __int128 self_collision = 0;     // sum w_t^2 * 2^63 (exact forced diagonal)
    unsigned __int128 total_collision_u = 0;  // sum w_s w_t S_u_lo(s,t)  — same-u interval LOWER bound
    unsigned __int128 total_collision_id = 0; // sum w_s w_t S_id_lo(s,t) — support LOWER bound
    // CONSERVATIVE UPPER bound on the effective whole-CDF state COUNT ==
    // bcap^2 * 2^63 / total_collision_v (>= the true count; in [1, K]). The
    // length-invariant discriminator (kappa == N_states/K_steps conflates the
    // count with the length). floor.
    uint64_t n_states_eff_u = 0;
    uint64_t n_states_eff_id = 0;
    // Derived flags consumed by the consensus INVALID decision (conservative:
    // reject only when even the UPPER bound crosses the threshold).
    bool state_hard = false;           // INVALID: N_states_hi_u < N0_U AND N_states_hi_id < N0_ID
    bool backload_hard = false;        // INVALID: (maxRs-Bcap) > BACKLOAD_HARD_T
    bool reuse_cache_hard = false;     // INVALID: reuse_cache_q32 > REUSE_CACHE_CAP_Q32 (structural; admission never cures it)
    // GREY structural state: SHADOW telemetry now; becomes INVALID (never
    // admission) once grey-enforcement is switched on (anti_parrot_reject_reason
    // grey_enforced). state_deficit is pure telemetry and never gates.
    bool state_grey = false;           // N_states_hi_u < N0_U_GREY AND N_states_hi_id < N0_ID_GREY
    bool backload_grey = false;        // GREY_T < (maxRs-Bcap) <= HARD_T
    uint64_t state_deficit = 0;        // conservative kappa_hi deficit vs grey ref (telemetry)

    // ---- Small-output-alphabet detector (§8) — SHADOW TELEMETRY ONLY. -------
    // A PROTOCOL prohibition on small-alphabet output generation (NOT a claim
    // that proof computation was skipped — a small alphabet can carry genuine
    // history-dependent inference). Output-only counts c(t) over the N chosen
    // tokens, plus ONE CDF-reuse term (MassA). Emitted for calibration; it does
    // NOT gate until a future sub-height flips the dormant grey guard, and only
    // after model-generated fixtures validate the thresholds. Defence-in-depth,
    // ORTHOGONAL to Lcopy (contiguous copy) — do not conflate.
    uint64_t vunique_out = 0;          // # distinct output token ids (extreme tripwire; <12 ~ empty)
    uint64_t k15_16_out = 0;           // smallest #ids covering >= ceil(15*N/16) output positions
    uint64_t sum_c2 = 0;               // sum of c(t)^2; Keff_out == N^2 / sum_c2 (dust-robust)
    uint64_t cprompt_ppm = 0;          // fraction of output tokens present in the prompt set (ppm)
    uint64_t massA_hi_steps = 0;       // # credited steps with conservative MassA(A) >= 7/8
    uint64_t credited_steps = 0;       // K (denominator for "most credited steps")
    bool small_output_alphabet = false;// Keff_out < KEFF0 OR K15_16_out <= K1516_0
    bool small_alphabet_hit = false;   // the CONJUNCTIVE rule (see anti_parrot_reject_reason)
};

// Anti-parrot aggregator (§8) — a CONSERVATIVE Q63 interval-bound certification,
// structurally analogous to B_cred (identical ATOL widening; no re-softmax, no
// fine quantisation of raw probabilities). Per-step inputs, from the SAME
// sampler-pass CDF B_cred uses:
//   credit_mass_q63 : conservative UPPER mass of the CHOSEN token (Q63; B_cred).
//   hard_mass_q63   : conservative LOWER mass of the CHOSEN token (Q63).
//   topk_ids        : the WHOLE per-step top-k token ids, id-sorted, active
//                     (prob > 0), <= top_k, deduped upstream.
//   topk_cdf_hi     : the aligned id-sorted CUMULATIVE CDF endpoint after each
//                     token (double in [0,1]); token j spans [prev_endpoint,
//                     this_endpoint].
// Each endpoint is ATOL-expanded and quantised to Q63 (l_max = ceil(min(1,
// l+ATOL)*2^63), h_min = floor(max(0, h-ATOL)*2^63)); the definitely-present
// token mass is m_lo = max(0, h_min - l_max). The collision LOWER bounds are:
//   S_u_lo  = sum over shared ids of max(0, min(h_min) - max(l_max)) (same-u).
//   S_id_lo = sum over shared ids of min(m_lo) (support).
// The diagonal (s == t) is forced to S == 1 (2^63). Throws std::invalid_argument
// on per-step length mismatch.
//
// chosen_ids / prompt_token_ids (optional; empty => the small-alphabet SHADOW
// telemetry is left zero) drive the output-vocabulary detector: chosen_ids is
// the per-step chosen token id (size N, the step count), prompt_token_ids is
// the UNPADDED prompt token id set (the caller drops pad_mask positions).
AntiParrotMetrics compute_anti_parrot_metrics(
    const std::vector<uint64_t>& credit_mass_q63,
    const std::vector<uint64_t>& hard_mass_q63,
    const std::vector<std::vector<uint32_t>>& topk_ids,
    const std::vector<std::vector<double>>& topk_cdf_hi,
    const std::vector<uint32_t>& chosen_ids = {},
    const std::vector<uint32_t>& prompt_token_ids = {});

// kappa < num/den, division-free: self*den < total*num. total == 0 (no
// credited overlap) => NOT below the threshold (kappa undefined).
bool anti_parrot_kappa_lt(unsigned __int128 self_collision,
                          unsigned __int128 total_collision,
                          uint64_t num, uint64_t den);

// N_states_eff < n0, division-free: bcap^2 * Q32 < n0 * total_collision.
// total == 0 (no credited overlap) => NOT below (undefined).
bool anti_parrot_nstates_lt(uint64_t bcap_units,
                            unsigned __int128 total_collision, uint64_t n0);

// Keff_out < k0, division-free: n^2 < k0 * sum_c2 (n == N steps, sum_c2 ==
// sum of chosen-count^2 <= N^2 <= 65536). sum_c2 == 0 (no counts) => not below.
bool anti_parrot_keff_lt(uint64_t n, uint64_t sum_c2, uint64_t k0);

// The joint maxRs-Bcap backloading surface (§8). Initial boundary is the
// integer line on (maxRs_real - Bcap_real) tokens; a frozen 2-D LUT can
// replace the body WITHOUT touching call sites. Signed integer delta
// (maxrs_q32 - (bcap_units << 22)) compared to (T << 32).
BackloadSurface backload_surface(uint64_t maxrs_q32, uint64_t bcap_units);

// Height-gated anti-parrot gate (§8). Returns a reject reason string, or
// nullptr to accept. Anti-parrot returns INVALID ONLY (never admission): a
// proof is rejected iff hard_run > L_HARD, state_HARD, Bcap < BCAP_ABS_FLOOR,
// or backload_HARD. Braw tiering / admission is NOT re-checked here (owned by
// VerifyV3TierAndAdmission). `grey_enforced` is a DORMANT HOOK: it defaults
// false and NOTHING passes true (no activation path is wired yet), so grey is
// always free today. If a future calibration promotes it, a GREY structural
// state (state_grey || backload_grey) becomes INVALID ("anti-parrot-grey") —
// never admission. The constants above are the consensus values active from
// AntiParrotHeight. QuickVerifier::VerifyAntiParrot calls this.
const char* anti_parrot_reject_reason(const AntiParrotMetrics& metrics,
                                      bool grey_enforced = false);

}  // namespace pow_v3
