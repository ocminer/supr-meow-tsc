#pragma once

// Proof v4 fork-contract primitives, sampler-independent part
// (the proof-v4 specification,
// sections 2 and 4; work package 1 plus the target/height parts of 3 and 4).
//
// C++ mirror of shared-utils/pow-utils/pow_v4.py. The two implementations
// must stay semantically IDENTICAL; the golden vectors in
// shared-utils/pow-utils/tests/vectors/v4_vectors.json are the contract
// (src/test/pow_v4_tests.cpp carries literal copies, as pow_v3_vectors_tests
// does for v3). Scope of this file:
//
//   * strict v4 carrier extraction: top-level extra_flags member "v4" that is
//     EXACTLY the JSON integer 1 (extract_v4_flag / extract_v4_flag_state);
//   * the proof MODE decision from (proof.version, v4 flag) with explicit
//     contradiction handling (decide_proof_mode);
//   * the height matrix (CheckProofModeAtHeight) over consensus/params.h
//     ProofV4Height X / V3PenaltyHeight Y / V3SunsetHeight Z;
//   * the price multiplier M = P/Q (ProofPriceMultiplier) and the exactly
//     computed priced target floor(T0 * Q / P) (PricedTarget), plus the
//     "hash meets target" comparison with its zero-target edge explicit;
//   * the v4 ROOT (plan section 3, consensus contract document
//     the proof-v4 specification): the Argon2id
//     admission root A over (msg_w0, C, model_id, R, S4) and the effective
//     step nonce E4 = SHA256(TAG_STEP_V4 || C || R || A || S4) that every
//     draw and the final hash append from ProofV4Height
//     (admit_message_v4 / admit_root_v4 / step_root_v4 / derive_v4_root);
//   * the v4 tier rule on A (V4TierAdmits): R is required in both tiers,
//     the free tier accepts every A, the admission tier requires A to meet
//     the existing model-priced admission target with the existing strict
//     comparison.
//
//   * the v4 PROFILE PRICE (V4ProfilePrice, plan section 5): exact integer
//     relative-deficit envelope plus absolute tail envelope over the
//     validated per-step credit vector, M = P/1000 in [1, 3]; the MECHANISM
//     is the rule, every coefficient is PROVISIONAL pending the
//     normal-corpus calibration (PRICE_* block).
//
// Deliberately NOT here: the CALIBRATED price coefficients (the PRICE_*
// constants are hypotheses). The sampler IS decided (SAMPLER_MODE_SUFFIX:
// the single Gumbel race); its replay lives in QuickVerifier with the
// vendored gumbel_sampler.h. The v3 primitives in pow_v3.{h,cpp} are vendored
// byte-identical from shared-utils and are NOT edited for v4; this file
// reuses their public constants, the same parser bounds, the same Argon2id
// profile and salt, and the same SHA-256 helper.

#include <arith_uint256.h>
#include <consensus/params.h>
#include <verification/pow_v3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace pow_v4 {

// Serialized proof version that carries the v4 claim (existing CProofBlob
// field; no schema change). The claim is valid only together with the strict
// carrier flag below.
constexpr uint8_t V4_PROOF_VERSION = 4;

// From V3PenaltyHeight (Y) a v3 proof is not surcharged by a flat factor: it is
// priced with the FULL v4 price (profile x tail x structure over the SAME
// validated evidence a v4 proof supplies), see ProofPriceMultiplier. There is
// no separate v3 penalty constant any more.

// ------------------------------------------------------------------------- //
// v4 root constants (plan section 3). Mirror pow_v4.py exactly.
// ------------------------------------------------------------------------- //

constexpr std::size_t ROOT_BYTES = 32;

// Fixed ASCII domain tags, 16 bytes each, no NUL (same shape as the v3
// STEP_BIND_TAG / PROMPT_CTX_TAG tags). TAG_ADMIT_V4 prefixes the Argon2id
// admission-root message, TAG_STEP_V4 prefixes the SHA-256 step-root message.
constexpr char TAG_ADMIT_V4[] = "TC_V4_ADMIT_ROOT";
constexpr std::size_t TAG_ADMIT_V4_LEN = 16;
static_assert(sizeof(TAG_ADMIT_V4) == TAG_ADMIT_V4_LEN + 1, "TAG_ADMIT_V4 must be 16 bytes");
constexpr char TAG_STEP_V4[] = "TC_V4_STEP_ROOT!";
constexpr std::size_t TAG_STEP_V4_LEN = 16;
static_assert(sizeof(TAG_STEP_V4) == TAG_STEP_V4_LEN + 1, "TAG_STEP_V4 must be 16 bytes");

// Sampler-domain identifier S4 = SHA256(SAMPLER_DOMAIN_PREFIX || SAMPLER_MODE_SUFFIX),
// 32 bytes, constant in both languages. The v4 sampler is FROZEN as the
// single chain-bound Gumbel race (verification/gumbel_sampler.h), contract
// v2, head h0, arm ONE (draw 0 only, no switch, RACE_ATOL 0), and
// SAMPLER_MODE_SUFFIX is that sampler's 20-byte mode suffix:
//
//   "TCGMODE2" | u8(2) | "h0" | u8(1 = ONE) | 8 x 0x00
//
// The last eight bytes are the v2 policy-digest slot, ZERO by definition:
// consensus has no switch policy (that slot is never
// SHA256("xhg.two.policy|" ...)[:8] of any research policy). The same 20
// bytes are appended LAST to every mode-V4 step preimage and to the final
// hash, after the effective nonce E4 (QuickVerifier::BuildStepMessage), so
// the sampler identity binds every draw twice: through S4 in the root and
// through the suffix in the digest itself. S4 hex:
//   ff256c6cd52e61675e7afb667fa923c261d1f3188b60ef6cf033536cf64b6ebe
constexpr char SAMPLER_DOMAIN_PREFIX[] = "tensorcash/v4/sampler-domain/";
constexpr std::size_t SAMPLER_MODE_SUFFIX_BYTES = 20;
constexpr std::array<uint8_t, SAMPLER_MODE_SUFFIX_BYTES> SAMPLER_MODE_SUFFIX{{
    'T', 'C', 'G', 'M', 'O', 'D', 'E', '2',  // gumbel::MODE_TAG_V2
    2,                                        // gumbel::CONTRACT_VERSION_V2
    'h', '0',                                 // head h0
    1,                                        // arm ONE
    0, 0, 0, 0, 0, 0, 0, 0}};                 // policy digest slot: zero
// Thread-safe, computed once.
const std::array<uint8_t, ROOT_BYTES>& sampler_domain_id();

// ------------------------------------------------------------------------- //
// Strict v4 carrier (plan section 2)
// ------------------------------------------------------------------------- //

// Consensus needs to distinguish a genuinely absent v4 declaration from a
// present member with the wrong value, exactly as pow_v3::StepBindFlagState
// does for stepbind:
//   ABSENT    - no well-formed top-level "v4" member (this includes every
//               parser-bound violation: empty/whitespace-only or oversized
//               input, invalid UTF-8, unparseable JSON, duplicate keys at any
//               level, nesting deeper than the bound, non-object top level);
//   CLAIMED   - top-level "v4" whose raw JSON token is exactly `1`;
//   MALFORMED - top-level "v4" present with any other value (true, "1", 1.0,
//               1e0, -1, 0, 2, null, an array, an object, NaN, ...).
// A "v4" member nested inside the "v3" object (or anywhere below top level)
// is not the carrier and leaves the state ABSENT. Never throws.
enum class V4FlagState : uint8_t {
    ABSENT = 0,
    CLAIMED = 1,
    MALFORMED = 2,
};
V4FlagState extract_v4_flag_state(const std::string& extra_flags);

// true iff extract_v4_flag_state(extra_flags) == CLAIMED. Uses the SAME
// bounded validating parser rules as pow_v3::extract_admission_nonce_hex
// (pow_v3::EXTRA_FLAGS_MAX_BYTES / EXTRA_FLAGS_MAX_DEPTH, duplicate-key
// rejection, json.loads acceptance grammar). Mirrors pow_v4.extract_v4_flag.
bool extract_v4_flag(const std::string& extra_flags);

// ------------------------------------------------------------------------- //
// Proof mode (plan section 2: "the carrier selects the hash/replay semantics;
// the height selects whether those semantics are legal")
// ------------------------------------------------------------------------- //

enum class ProofMode : uint8_t {
    Legacy = 0,   // proof.version < 3 (v1/v2 rules)
    V3 = 1,       // proof.version == 3 and no v4 declaration at all
    V4 = 2,       // proof.version == 4 and v4 flag CLAIMED
    Invalid = 3,  // every contradiction; see decide_proof_mode
};
const char* proof_mode_name(ProofMode mode);

// PRECISE DEFINITION (the plan leaves "version >= 3 with no v4 flag" open;
// this is the chosen semantic, listed for review):
//   version <  3 : Legacy when the flag is ABSENT, else Invalid (a v4
//                  declaration on a pre-v3 proof is a contradiction);
//   version == 3 : V3 when the flag is ABSENT, else Invalid (a present "v4"
//                  member, whether 1 or malformed, contradicts version 3 and
//                  must not silently fall back to v3);
//   version == 4 : V4 when the flag is CLAIMED, else Invalid (version 4
//                  without the exact flag, or with a malformed one, never
//                  falls back to v3);
//   version >= 5 : Invalid ("unsupported future claims in the new regime").
// The decision is a pure function; whether Legacy/V3/V4 are LEGAL at a height
// is CheckProofModeAtHeight. The verifier applies the strict decision only
// inside the v4 regime (IsProofV4Active or IsV3Penalized); below X the carrier is
// INERT and every version >= 3 is verified under the v3 rules
// (effective_proof_mode below), exactly as today.
ProofMode decide_proof_mode(uint8_t version, V4FlagState flag);
ProofMode decide_proof_mode(uint8_t version, const std::string& extra_flags);

// Height-aware mode: the mode the verifier actually replays with.
//   height <  X (or X unset): the v4 carrier is inert. version < 3 is
//                             Legacy, every version >= 3 is V3 regardless of
//                             any top-level "v4" member (claimed, malformed
//                             or absent): today's behaviour, unchanged.
//   height >= X             : decide_proof_mode (strict).
// A negative height is "unknown context" and is treated as below X here;
// the verifier separately refuses to replay a v4-carrier proof without
// context (QuickVerifier::PrepareV4 deferral), it never reaches this
// fallback for one.
ProofMode effective_proof_mode(uint8_t version, const std::string& extra_flags,
                               int height, const Consensus::Params& params);

// The plan's height matrix (consensus/params.h ProofV4Height block):
//   Legacy : legal below X, rejected at/after X;
//   V3     : legal below Z (priced at the full v4 price in [Y, Z) by
//            ProofPriceMultiplier), rejected at/after Z;
//   V4     : below X "validated under v3 rules; the v4 claim is inert"
//            (returns true; effective_proof_mode maps it to V3, so v4
//            replay semantics are never selected below X); legal at/after X;
//   Invalid: always rejected.
// `height` is the CANDIDATE block's own height. Returns false with `error`
// set on rejection. Pure; no chain state.
bool CheckProofModeAtHeight(ProofMode mode, int height,
                            const Consensus::Params& params, std::string& error);

// ------------------------------------------------------------------------- //
// v4 root (plan section 3). Byte layouts, all lengths fixed except model_id:
//
//   admit_message = TAG_ADMIT_V4(16) || msg_w0 || C(32)
//                   || u16le(len(model_id)) || model_id || R(32) || S4(32)
//   A             = Argon2id(admit_message)   (pow_v3::argon2id_digest: 8 MiB,
//                   t=1, one lane, 32 bytes, fixed salt "TC_V3_ADMISSION!")
//   E4            = SHA256(TAG_STEP_V4(16) || C(32) || R(32) || A(32) || S4(32))
//
// msg_w0 is the EXISTING first-step sampler message WITHOUT any trailing
// nonce (QuickVerifier::BuildStepMessage(prompt_tokens, step 0,
// include_nonce=false) == pow_v3::build_step_message(..., nullptr)); it
// already carries the header prefix, VDF, tick, the rolling first-step
// context window and the precision string. C is the existing
// pow_v3::prompt_commitment over the FULL pre-window prefix and pad_mask. R
// is the mandatory 32-byte admission nonce from the existing v3 carrier
// (extra_flags.v3.admission_nonce). Nothing here is serialized: the verifier
// recomputes A and E4 from proof fields and never trusts a claimed A.
// ------------------------------------------------------------------------- //

// Throws std::invalid_argument when model_identifier exceeds the u16le prefix.
std::vector<uint8_t> admit_message_v4(const std::vector<uint8_t>& msg_w0,
                                      const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                                      const std::string& model_identifier,
                                      const std::array<uint8_t, ROOT_BYTES>& admission_nonce);

// A = Argon2id(admit_message). Throws std::runtime_error when libargon2 is
// unavailable in this binary (POW_V3_HAVE_ARGON2 not compiled) or fails; the
// verifier maps that to a local infrastructure failure, never to acceptance.
std::array<uint8_t, ROOT_BYTES> admit_root_v4(const std::vector<uint8_t>& admit_message);

// E4 = SHA256(TAG_STEP_V4 || C || R || A || S4). Pure SHA-256, never throws.
std::array<uint8_t, ROOT_BYTES> step_root_v4(const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                                             const std::array<uint8_t, ROOT_BYTES>& admission_nonce,
                                             const std::array<uint8_t, ROOT_BYTES>& admit_root);

struct V4Root {
    std::array<uint8_t, ROOT_BYTES> A{};
    std::array<uint8_t, ROOT_BYTES> E4{};
};

// A and E4 together (admit_message_v4 -> admit_root_v4 -> step_root_v4).
// Same exceptions as the parts.
V4Root derive_v4_root(const std::vector<uint8_t>& msg_w0,
                      const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                      const std::string& model_identifier,
                      const std::array<uint8_t, ROOT_BYTES>& admission_nonce);

// v4 tier rule on the recomputed root (plan section 3 table). The tier comes
// from the SAME conservative B_cred pass v3 uses (pow_v3::tier_for_b_cred_units
// over the chain-param thresholds); R presence is enforced separately and
// before this (R is required in every tier).
//   Invalid           -> false (B_cred < B_FLOOR, as today)
//   AdmissionRequired -> pow_v3::admission_valid(A, target_le): A read as a
//                        little-endian uint256 must be STRICTLY below the
//                        existing model-priced admission target
//   Free              -> true (one evaluation of A, every A eligible)
bool V4TierAdmits(pow_v3::Tier tier,
                  const std::array<uint8_t, ROOT_BYTES>& admit_root,
                  const std::array<uint8_t, ROOT_BYTES>& admission_target_le);

// ------------------------------------------------------------------------- //
// Price (plan section 4)
// ------------------------------------------------------------------------- //

// Rational price multiplier M = P / Q applied to the short-header target:
// T_effective = floor(T0 * Q / P). {1,1} means "existing target".
// {0,0} is the fail-closed sentinel ProofPriceMultiplier returns for
// ProofMode::Invalid; PricedTarget rejects P == 0, so such a proof can never
// meet any target.
struct ProofPrice {
    uint64_t P{1};
    uint64_t Q{1};
    bool operator==(const ProofPrice& o) const { return P == o.P && Q == o.Q; }
};

// Validated per-step credit metrics from the existing sequence/CDF pass
// (pow_v3 B_cred units, R units == 1 bit): the SAME conservative credit
// vector tiering uses, never chosen_probs or a claimed entropy figure. The
// verifier fills this after VerifyV3TierAndAdmission / VerifyAntiParrot so the
// v4 price is derived from validated evidence only. `valid` is false when no
// pass ran (for example a Legacy proof), in which case V4 pricing has no
// evidence to price.
struct CreditMetrics {
    std::vector<uint64_t> step_units;  // per-step credit units, proof order
    uint64_t total_units{0};           // exact integer sum of step_units
    bool valid{false};
};

// ------------------------------------------------------------------------- //
// v4 profile price (plan section 5). PROVISIONAL COEFFICIENTS.
//
// Every number in this block is a HYPOTHESIS pending the normal-corpus
// calibration (plan section 5: "The 0.05 allowance, interpolation and
// 20/32-bit thresholds are hypotheses, not measured economic break-even
// values"). The MECHANISM (relative prefix deficit, piecewise-linear price,
// absolute tail envelope, max, clip, exact integer arithmetic with floor
// rounding) is the consensus rule; changing any constant below is a
// coordinated consensus change that regenerates the profile_price vectors.
// Mirror pow_v4.py (PRICE_* constants) exactly.
//
// Representation. M = P / PRICE_Q with the FIXED denominator PRICE_Q = 1000.
// The anchors 1.00, 1.35 and 3.00 are exact in thousandths (1000, 1350,
// 3000); a power-of-two denominator cannot represent 1.35 = 27/20 exactly
// (20 carries the factor 5), so a fixed decimal denominator is the smallest
// exact family; 1000 rather than the minimal 20 gives the interpolated
// segments 0.001 resolution. PricedTarget treats P == Q (1000/1000) as the
// base target; the pair is never normalised.
//
// Integer formulation (W = 256 steps, c_t credit units of step t = 1..W,
// prefix sums C_t, total B = C_W):
//   d_t = t * B - W * C_t           (t/W - C_t/B scaled by S = W * B, exact)
//   D   = max(0, max_t d_t)          (d_W = 0 always, so the range 1..W and
//                                    the convention C_0 = 0 / d_0 = 0 agree)
//   d   = D / S                      (never materialised; every comparison
//                                    is the cross-multiplied form)
//   P_rel(D, S): anchors (d, P) = (1/20, 1000), (1/4, 1350), (3/4, 3000);
//       d <= 1/20       : 1000
//       1/20 < d <= 1/4 : 1000 + floor(350 * (20 D - S) / (4 S))
//       1/4  < d <= 3/4 : 1350 + floor(1650 * (4 D - S) / (2 S))
//       3/4  < d        : 3000
//     (each segment is the exact rational interpolation
//      P_lo + (P_hi - P_lo) * (d - d_lo) / (d_hi - d_lo) rounded DOWN to the
//      thousandth; InterpolatePriceFloor derives both from the anchor table)
//   P_64 = 1000 + floor(2000 * max(0, 20480 - C_192) / 20480)
//   P_32 = 1000 + floor(2000 * max(0, 32768 - C_224) / 32768)
//   P    = min(3000, max(1000, P_rel, P_64, P_32))
//
// Rounding proof. Every division is a floor on a non-negative numerator, so
// each component P is the unique integer with
// P / 1000 <= M_exact < (P + 1) / 1000: the integer price never exceeds the
// exact rational price and undershoots it by less than one thousandth (it
// rounds in the miner's favour, never against). At every anchor the exact
// price is a whole number of thousandths (the numerator is an exact
// multiple of the denominator), so the floor is exact there and the curve
// is continuous at the anchors; inside a segment the numerator grows with D
// while the denominator is fixed, so P_rel is monotone non-decreasing in D.
// The same holds for P_64 / P_32 in the prefix shortfall.
//
// Edge cases (all tested): B == 0 returns the cap (3000/1000) before any
// division (B = 0 is impossible past the tier floor; the tier gate rejects,
// the price merely must not divide by zero and must not be more lenient
// than the cap); C_t == B early makes every later d_t <= 0 (no deficit);
// all credit in one step k gives d = (k - 1) / W; a vector whose length is
// not W, any entry above the per-step cap pow_v3::B_STEP_MAX_UNITS, a
// total_units that does not equal the sum, or metrics with valid == false
// are REJECTED with the fail-closed sentinel {0,0} (PricedTarget rejects
// P == 0, so no hash can meet it).
//
// Overflow bound: c_t <= 32768 => B <= 2^23, S = W * B <= 2^31, D <= 2^31;
// the largest intermediate is 1650 * (20 D) * 4 < 2^49, inside uint64_t.
// ------------------------------------------------------------------------- //

constexpr std::size_t PRICE_W = pow_v3::POW_WINDOW_SIZE;  // 256; other lengths reject
constexpr uint64_t PRICE_Q = 1000;        // fixed denominator: M = P / 1000
constexpr uint64_t PRICE_MIN_P = 1000;    // 1.00x (clip floor)
constexpr uint64_t PRICE_MAX_P = 3000;    // 3.00x (clip cap)

// Relative-deficit anchor (d = d_num / d_den, price P thousandths).
struct PriceAnchor {
    uint64_t d_num;
    uint64_t d_den;
    uint64_t P;
};
// PROVISIONAL: (0.05, 1.00), (0.25, 1.35), (0.75, 3.00); strictly increasing.
constexpr std::array<PriceAnchor, 3> PRICE_ANCHORS{{{1, 20, 1000}, {1, 4, 1350}, {3, 4, 3000}}};

// Absolute tail envelope: credit before the last 64 steps (prefix 192) must
// reach 20 bits, before the last 32 steps (prefix 224) 32 bits, else the
// price rises linearly to the cap (P = 1000 + 2000 * shortfall / threshold).
// PROVISIONAL thresholds.
struct PriceAbsEnvelope {
    std::size_t prefix_steps;
    uint64_t threshold_units;
};
constexpr std::array<PriceAbsEnvelope, 2> PRICE_ABS_ENVELOPES{{{192, 20 * pow_v3::BCRED_R},
                                                               {224, 32 * pow_v3::BCRED_R}}};
constexpr uint64_t PRICE_ABS_SLOPE_P = 2000;  // thousandths per unit shortfall
constexpr ProofPrice PRICE_FAIL_CLOSED{0, 0};  // no admissible target

// Integer floor interpolation between two anchors for d = D / S (both
// arguments exact integers), exposed for the tests:
//   P_lo + floor((P_hi - P_lo) * (D * k_lo - n_lo * S) * k_hi
//                / (S * (n_hi * k_lo - n_lo * k_hi)))
// Precondition: d_lo <= d <= d_hi (numerator and denominator non-negative).
uint64_t InterpolatePriceFloor(uint64_t D, uint64_t S, const PriceAnchor& lo, const PriceAnchor& hi);

// v4 profile price over the validated per-step credit vector (the SAME
// conservative pow_v3 credit units tiering uses, length PRICE_W). Returns
// {P, PRICE_Q} with PRICE_MIN_P <= P <= PRICE_MAX_P per the integer rule
// above, or PRICE_FAIL_CLOSED when the metrics are invalid (see the edge
// cases). Never throws. Mirrors pow_v4.v4_profile_price. COEFFICIENTS
// PROVISIONAL (PRICE_ANCHORS / PRICE_ABS_ENVELOPES).
ProofPrice V4ProfilePrice(const CreditMetrics& credits);

// ------------------------------------------------------------------------- //
// v4 tail factor and total price. PROVISIONAL COEFFICIENTS.
//
// The profile price charges credit that arrives late RELATIVE to the
// window's own total and credit missing before the tail. It does not charge
// a tail carrying a large ABSOLUTE amount of credit. The tail factor does:
//
//   M_tail  = min(2, max(1, B_64 / theta_64, B_32 / theta_32))
//   M_total = M_profile * M_tail
//
// B_L is the credit in the last L steps, in pow_v3 credit units. theta_64 =
// 96 bits and theta_32 = 48 bits are hypotheses from the 2026-09-11 corpus
// calibration; the MECHANISM is the rule. Mirror pow_v4.py exactly.
//
// Integer formulation, per tail (L, T):
//   B_L    = sum of the last L credit units
//   P_L    = 1000 + floor(1000 * max(0, B_L - T) / T)
//   P_tail = min(2000, max(1000, P_64, P_32))
//   P_base = floor(P_profile * P_tail / 1000),  {P, PRICE_Q},
//            1000 <= P_base <= PRICE_BASE_MAX_P (6000)
//
// P_base is NOT the whole v4 price any more: the STRUCTURE factor below
// multiplies it, and the v4 maximum is PRICE_TOTAL_MAX_P == 12000.
//
// Rounding: each factor and the product are floored to the thousandth, so
// P / 1000 never exceeds the exact product and falls short of it by less
// than 0.0061, always in the miner's favour. P_L is exactly 1000 at B_L = T
// and exactly 2000 at B_L = 2T.
//
// Overflow bound: c_t <= 2^15 and L <= 64 give B_L <= 2^21, so
// 1000 * B_L < 2^31; P_profile * P_tail <= 6,000,000 < 2^23. PricedTarget
// multiplies only T0 mod P by Q, so its split is unchanged.
//
// The factor reads the validated credit vector only; admission credit, the
// tier thresholds and the tier decision are untouched.
// ------------------------------------------------------------------------- //

// Tail envelope: credit in the last `tail_steps` steps above
// `threshold_units` raises the factor linearly, reaching the cap at twice
// the threshold. PROVISIONAL thresholds.
struct PriceTailEnvelope {
    std::size_t tail_steps;
    uint64_t threshold_units;
};
constexpr std::array<PriceTailEnvelope, 2> PRICE_TAIL_ENVELOPES{{{64, 96 * pow_v3::BCRED_R},
                                                                 {32, 48 * pow_v3::BCRED_R}}};
constexpr uint64_t PRICE_TAIL_SLOPE_P = 1000;  // thousandths per unit of relative excess
constexpr uint64_t PRICE_TAIL_MAX_P = 2000;    // 2.00x, per-factor cap
// The profile x tail product on its own: the v4 price BEFORE the structure
// factor below. This used to be the whole v4 maximum (PRICE_TOTAL_MAX_P);
// it is not any more, and a range check that still reads 6000 is a bug.
constexpr uint64_t PRICE_BASE_MAX_P = PRICE_MAX_P * PRICE_TAIL_MAX_P / PRICE_Q;  // 6.00x

// Tail factor P_tail in thousandths, PRICE_MIN_P <= P_tail <= PRICE_TAIL_MAX_P,
// or 0 when the metrics are invalid (the same validation as V4ProfilePrice).
// Mirrors pow_v4.v4_tail_factor.
uint64_t V4TailFactor(const CreditMetrics& credits);

// The BASE v4 price (profile x tail, without the structure factor):
// {floor(P_profile * P_tail / PRICE_Q), PRICE_Q}, PRICE_MIN_P <= P <=
// PRICE_BASE_MAX_P, or PRICE_FAIL_CLOSED when the metrics are invalid. Never
// throws. Mirrors pow_v4.v4_base_price(credits) (== v4_total_price with
// structure omitted). The consensus price a v4 proof owes is the two-argument
// overload below; ProofPriceMultiplier refuses mode V4 without the structure
// bundle in BOTH languages, so no caller reaches this one by omission.
ProofPrice V4TotalPrice(const CreditMetrics& credits);

// ------------------------------------------------------------------------- //
// v4 STRUCTURE price: low-state and near-pinned windows. PROVISIONAL
// COEFFICIENTS.
//
// Source: the proof-v4 reference tests, the structure-price specification
// and its reference module structure_price.py.
// Every coefficient below (PIN_NEAR_*, PIN_FREE_COUNT, PIN_RAMP_DEN,
// STATE_U_ONSET, STATE_ID_ONSET, PRICE_STRUCTURE_MAX_P) is a HYPOTHESIS
// screened on a 995-window development corpus, exactly as the PRICE_*
// block above is; the MECHANISM is the consensus rule. Only the requested
// hard boundary "reject at a near-pin count of 175" is an input
// requirement rather than a curve choice. Changing any constant is a
// coordinated consensus change that regenerates the structure_price vectors.
// The 995-window screen is NOT an attack-cost calibration and the 2x ceiling
// is a pricing choice, NOT a bound on possible forking advantage.
//
// (a) NEAR-PIN COUNT. Over EXACTLY PRICE_W = 256 generated positions, N
// counts those whose MOST PROBABLE token's conservative CDF lower mass
// (credit_v4::StepResult::p_max_lower_q63, the canonical CDF's own value —
// never the chosen token's wire scalar and never an inverse-CDF endpoint)
// satisfies
//
//     PIN_NEAR_NUM * p_max_lower_q63 > PIN_NEAR_DEN * 2^63
//
// i.e. strictly above 497/500 = 0.994. With the existing 0.0002 lower-bound
// allowance that is a nominal mass above roughly 0.9942.
//
//     N >= PIN_REJECT_COUNT (175)  ->  the proof is INVALID.
//
// That is a validity gate, not a finite high price: V4StructureRejects is
// checked on every consensus entry point independently of hash luck, and
// V4PinPrice returns the 0 sentinel for such a count so no price path can
// silently pay for it instead. 174 is the last permitted count. The existing
// anti-parrot hard rules and the tier/admission rules are unchanged and keep
// rejecting on their own.
//
// For N <= 174 the pin factor is a quadratic ramp with a free region:
//
//     e      = max(0, N - PIN_FREE_COUNT)
//     P_pin  = PRICE_Q + floor(PRICE_Q * e^2 / PIN_RAMP_DEN^2)   [<= 2000]
//
// PIN_RAMP_DEN == (PIN_REJECT_COUNT - 1) - PIN_FREE_COUNT == 46, so N = 174
// gives exactly 2000 (2.000x) and N <= 128 exactly 1000. The quadratic is an
// explicit design choice — small charges on entry, larger ones near the hard
// boundary — and does not follow any established GPU-cost or accepted-fork
// equation. Overflow: e <= 46, so PRICE_Q * e^2 <= 2,116,000 < 2^22.
//
// (b) LOW-STATE. The exact conservative collision totals from the SAME
// anti-parrot aggregation the hard state gate reads
// (pow_v3::AntiParrotMetrics::bcap / total_collision_u / total_collision_id)
// give the effective state counts as N_u = D / collision_u and
// N_id = D / collision_id with D = bcap^2 * 2^63. The factor is
//
//     M_state = clamp(min(STATE_U_ONSET / N_u, STATE_ID_ONSET / N_id), 1, 2)
//
// evaluated DIVISION-FREE from the totals, never from the displayed floored
// counts n_states_eff_*:
//
//     D        = bcap_units^2 * 2^63
//     P_state  = clamp(floor(PRICE_Q * min(STATE_U_ONSET  * collision_u,
//                                          STATE_ID_ONSET * collision_id) / D),
//                      PRICE_MIN_P, PRICE_STRUCTURE_MAX_P)
//
// The min makes this a JOINT signal: both statistics must be low relative to
// their onsets before anything is charged. These counts measure overlap of
// credit-weighted observed CDFs; they are NOT counts of reusable KV objects
// or of accepted suffixes, and this factor is a structural deterrent, not a
// measurement of cache profitability or of skipped inference.
//
// Overflow: bcap_units <= STATE_BCAP_MAX_UNITS = 2^18 for a valid window, so
// D <= 2^99 (the proposal's bound) and the largest intermediate is
// PRICE_Q * STATE_U_ONSET * collision_u <= 1000 * 40 * 2^99 < 2^115, inside
// unsigned __int128. Zero or out-of-range inputs are NOT given a free price:
// V4StatePrice returns the 0 sentinel and V4TotalPrice fails closed, so a
// defective or missing aggregation can never cheapen a proof.
//
// (c) COMPOSITION.
//
//     P_structure = max(P_pin, P_state)                    [1000 .. 2000]
//     P_total     = floor(P_base * P_structure / PRICE_Q)  [1000 .. 12000]
//
// The two indicators may describe the same low-diversity structure, so the
// MAXIMUM (not the product) bounds the new factor at 2x. The existing
// trunk/tail price stays multiplicative with it; some overlap with the trunk
// factor remains possible and is an explicit design choice, not a claim of
// independent attack gains.
//
// The v4 maximum therefore rises from 6x to 12x. PRICE_TOTAL_MAX_P below is
// 12000 and every range assumption must be updated with it; silently keeping
// the old 6x cap would erase part of the charge. Rounding is unchanged: one
// floor to the thousandth, then the single division PricedTarget already
// does on the decoded target. V3 uses this same full price from Y; no
// surcharge enters header bits or proof serialization, and a true v4 proof
// must satisfy this body-derived boundary even when the ordinary header
// target passes. Overflow: 6000 * 2000 = 12,000,000 < 2^24.
// ------------------------------------------------------------------------- //

// Near-pin predicate: PIN_NEAR_NUM * p_max_lower_q63 > PIN_NEAR_DEN * 2^63
// (strictly above 497/500 == 0.994). PROVISIONAL.
constexpr uint64_t PIN_NEAR_NUM = 500;
constexpr uint64_t PIN_NEAR_DEN = 497;
// Hard boundary: this count and above is INVALID (requested input, not a
// curve choice). The last permitted count is PIN_REJECT_COUNT - 1 == 174.
constexpr uint64_t PIN_REJECT_COUNT = 175;
// Free region: counts up to and including this pay nothing. PROVISIONAL.
constexpr uint64_t PIN_FREE_COUNT = 128;
// Quadratic ramp denominator; fixed by the two above so the last permitted
// count is exactly the cap.
constexpr uint64_t PIN_RAMP_DEN = 46;
static_assert(PIN_RAMP_DEN == PIN_REJECT_COUNT - 1 - PIN_FREE_COUNT,
              "the pin ramp must reach the cap exactly at the last legal count");
static_assert(PIN_FREE_COUNT < PIN_REJECT_COUNT - 1, "pin ramp must have width");
static_assert(PIN_REJECT_COUNT <= PRICE_W, "the pin count is over the 256-step window");

// State onsets, in effective-state counts. PROVISIONAL.
constexpr uint64_t STATE_U_ONSET = 40;
constexpr uint64_t STATE_ID_ONSET = 24;
// Largest capped credit a valid window can carry: 256 steps x R units
// (pow_v3::AP_R_STEP_CAP == pow_v3::BCRED_R). Bounds D <= 2^99.
constexpr uint64_t STATE_BCAP_MAX_UNITS =
    static_cast<uint64_t>(PRICE_W) * pow_v3::BCRED_R;

// Cap of the structure factor, and the raised v4 maximum.
constexpr uint64_t PRICE_STRUCTURE_MAX_P = 2000;    // 2.00x
constexpr uint64_t PRICE_TOTAL_MAX_P =
    PRICE_BASE_MAX_P * PRICE_STRUCTURE_MAX_P / PRICE_Q;  // 12.00x

// The validated structure evidence for ONE v4-priced proof, filled by the
// verifier from the canonical Quick pass: `near_pin_count` from the per-step
// credit_v4 p_max_lower_q63 over exactly PRICE_W positions, and
// `bcap_units` / `collision_u` / `collision_id` from the SAME
// pow_v3::compute_anti_parrot_metrics aggregation that already runs on the
// same CDF (no second CDF pass, no second softmax). For a mode-V4 proof
// `collision_u` is collision_race (see below); for a mode-V3 proof in
// [Y, Z) it is the u-interval collision. `valid` is false when no such pass
// ran or its shape was wrong; an invalid bundle is never priced free — it
// fails closed.
struct StructureMetrics {
    uint64_t near_pin_count{0};
    uint64_t bcap_units{0};
    unsigned __int128 collision_u{0};
    unsigned __int128 collision_id{0};
    bool valid{false};
};

// The near-pin predicate on ONE position's canonical maximum-token
// conservative lower mass: PIN_NEAR_NUM * p > PIN_NEAR_DEN * 2^63, evaluated
// in unsigned __int128 (PIN_NEAR_NUM * 2^63 overflows uint64_t). A mass
// above 2^63 is not a valid Q63 mass and is never counted. Mirrors the
// predicate inside pow_v4.v4_near_pin_count.
bool V4IsNearPinned(uint64_t p_max_lower_q63);

// N over EXACTLY PRICE_W canonical maximum-token lower masses
// (credit_v4::StepResult::p_max_lower_q63, one per replayed position).
// Returns false and leaves `count` untouched when the evidence is not
// exactly PRICE_W values each within [0, 2^63]: a short, long or malformed
// vector is not a complete count and must never be treated as one.
// Mirrors pow_v4.v4_near_pin_count.
bool V4NearPinCount(const std::vector<uint64_t>& p_max_lower_q63, uint64_t& count);

// P_pin in thousandths for a near-pin count, PRICE_MIN_P <= P <=
// PRICE_STRUCTURE_MAX_P. Returns 0 (the "no price" sentinel) for a count at
// or above PIN_REJECT_COUNT — such a proof is REJECTED by
// V4StructureRejects, never priced — and for a count above PRICE_W.
// Mirrors pow_v4.v4_pin_price.
uint64_t V4PinPrice(uint64_t near_pin_count);

// P_state in thousandths, PRICE_MIN_P <= P <= PRICE_STRUCTURE_MAX_P, or 0
// when the metrics are invalid: !valid, bcap outside (0,
// STATE_BCAP_MAX_UNITS], or either collision total zero or above
// D = bcap^2 * 2^63. Division-free from the exact totals. Never throws.
// Mirrors pow_v4.v4_state_price.
uint64_t V4StatePrice(const StructureMetrics& structure);

// P_structure = max(P_pin, P_state) in thousandths, or 0 when the metrics
// are invalid or either component has no price (a rejecting count included).
// Mirrors pow_v4.v4_structure_price.
uint64_t V4StructurePrice(const StructureMetrics& structure);

// The hard boundary, independent of price and of hash luck: true iff
// structure.near_pin_count >= PIN_REJECT_COUNT. `reason` is set to the
// consensus error string when it returns true. Mirrors
// pow_v4.v4_structure_rejects.
bool V4StructureRejects(const StructureMetrics& structure, std::string& reason);

// The price a v4 proof owes: {floor(P_base * P_structure / PRICE_Q),
// PRICE_Q} with PRICE_MIN_P <= P <= PRICE_TOTAL_MAX_P, or PRICE_FAIL_CLOSED
// when EITHER the credit metrics or the structure metrics are invalid (fail
// closed both ways). Never throws. Mirrors
// pow_v4.v4_total_price(credits, structure).
ProofPrice V4TotalPrice(const CreditMetrics& credits, const StructureMetrics& structure);

// ------------------------------------------------------------------------- //
// collision_race: the mode-V4 replacement of the u-interval collision.
//
// The anti-parrot aggregation (pow_v3::compute_anti_parrot_metrics) measures
// how few distinct whole-CDF states a window realises with two pairwise
// similarities: S_u_lo, the same-u interval overlap of two rows (the
// probability that the same u draws the same token from both inverse CDFs),
// and S_id_lo, the support overlap. Under the Gumbel race sampler
// (SAMPLER_MODE_SUFFIX) a position is not decided by a u interval but by the
// argmax of log-weights plus per-token Gumbel noise, so the u construction
// has no meaning for a mode-V4 proof. Its analogue is the SAME-NOISE
// AGREEMENT probability: with the same Gumbel vector g applied to rows A and
// B, token i wins both iff i = argmax_j(log b_j + g_j) with
// b_j = max(p_A(j)/p_A(i), p_B(j)/p_B(i)) and b_i = 1, hence
//
//     P(i wins both) = 1 / sum_j max(p_A(j)/p_A(i), p_B(j)/p_B(i))
//     S_race(A, B)   = sum over i with p_A(i) > 0 and p_B(i) > 0
//
// (a token absent from a row has p = 0 there and can win neither race in
// that row; the sum over i is at most 1). RaceAgreementQ63 evaluates it in
// binary64 from the kernel's binary32 probabilities (credit_v4
// StepResult::cdf_p), in a FIXED order — union ids ascending for i and for
// j, ratios as true divisions, max, sequential addition — and rounds it
// CONSERVATIVELY: the denominator is multiplied by (1 + 2^-40), which
// dominates the accumulated round-to-nearest error of the at most ~300
// operations of a 100-id union, and 1 / den_up is floored to Q63
// (pow_v3::f64_to_q63_floor). The result is therefore a LOWER bound on the
// exact agreement probability, in the SAME direction as S_u_lo / S_id_lo:
// the state gate rejects only when even the conservative bound says
// collapse, and the state price factor never exceeds the exact one (it
// rounds in the miner's favour, never against). Every operation is an IEEE
// basic op, so the value is bit-identical on every platform that satisfies
// credit_v4::FloatEnvironmentOk (the source is compiled with contraction off).
//
// RaceCollisionTotal aggregates it exactly as compute_anti_parrot_metrics
// aggregates S_u_lo: weights w = min(credit_units, AP_R_STEP_CAP) from the
// per-step credit mass (only credited steps take part), the diagonal forced
// to 2^63, the sum over ORDERED credited pairs of w_x * w_y * S_race in
// unsigned 128-bit. ApplyRaceCollision installs it as total_collision_u and
// recomputes the derived u-arm fields (n_states_eff_u, state_hard,
// state_grey, state_deficit) with the same formulas. collision_id is
// untouched (the support overlap is not a u construction); mode V3 keeps
// collision_u at every height. StructureMetrics::collision_u of a mode-V4
// proof therefore carries collision_race, which is what V4StatePrice reads.
// ------------------------------------------------------------------------- //

// (1 + 2^-40) as a binary64 constant: the conservative denominator factor.
constexpr double RACE_COLLISION_DEN_UP = 1.0 + 9.094947017729282e-13;

// S_race(A, B) in Q63 for two rows given as ascending unique ids with their
// binary32 probabilities (every p > 0, exactly the kernel's cdf_ids /
// cdf_p). Symmetric. Throws std::invalid_argument on malformed rows (length
// mismatch, unsorted or duplicate ids, a non-positive or non-finite p).
uint64_t RaceAgreementQ63(const std::vector<uint32_t>& ids_a, const std::vector<float>& p_a,
                          const std::vector<uint32_t>& ids_b, const std::vector<float>& p_b);

// The window total over the credited steps (see above). Inputs per step:
// the credit mass (Q63) the weights derive from, the kernel's cdf_ids and
// cdf_p. Throws std::invalid_argument on a length mismatch.
unsigned __int128 RaceCollisionTotal(const std::vector<uint64_t>& credit_mass_q63,
                                     const std::vector<std::vector<uint32_t>>& cdf_ids,
                                     const std::vector<std::vector<float>>& cdf_p);

// Install collision_race as the metrics' u-arm total and recompute the
// derived fields that read it (mirrors the tail of compute_anti_parrot_metrics).
void ApplyRaceCollision(pow_v3::AntiParrotMetrics& metrics, unsigned __int128 total_collision_race);

// M(h, proof):
//   Legacy       -> {1,1}
//   V3           -> V4TotalPrice(credits, structure) when params.IsV3Penalized
//                   (height), i.e. Y <= h < Z: the FULL v4 price with the same
//                   evidence rules (exactly PRICE_W steps, a valid structure
//                   bundle, fail closed {0,0} otherwise) and NO floor or flat
//                   surcharge; {1,1} outside the band
//   V4           -> V4TotalPrice(credits, structure)
//   Invalid      -> {0,0}                              (fail closed)
// The verifier fills `credits` and `structure` for a v3 proof inside [Y, Z)
// from the canonical kernel replay exactly as it does for a v4 proof
// (QuickVerifier::VerifySequenceLightVectorized), so a v3 miner from Y owes
// the price a v4 miner owes for the same window shape.
ProofPrice ProofPriceMultiplier(ProofMode mode, int height,
                                const Consensus::Params& params,
                                const CreditMetrics& credits,
                                const StructureMetrics& structure);

// floor(T0 * Q / P) computed EXACTLY without 256-bit overflow. Returns
// std::nullopt when P == 0 (undefined) or when the result would not fit in
// 256 bits (Q > P can raise the target above 2^256 - 1); callers treat
// nullopt as "no admissible target" (reject). Q == 0 yields the zero target,
// which under HashMeetsTarget admits only the all-zero hash.
//
// Exactness proof (see the .cpp for the implementation):
//   Write T0 = q * P + r with q = floor(T0 / P) and 0 <= r < P (256-by-64-bit
//   long division in arith_uint256; r fits in 64 bits because r < P < 2^64).
//   Then T0 * Q / P = q * Q + (r * Q) / P, and since q * Q is an integer,
//   floor(T0 * Q / P) = q * Q + floor((r * Q) / P).
//   r * Q < 2^64 * 2^64 = 2^128 is computed exactly in unsigned __int128, and
//   floor((r * Q) / P) < Q < 2^64 fits uint64_t. q * Q is formed in
//   arith_uint256 only after checking q <= (2^256 - 1) / Q, and the final
//   addition is checked for wrap-around, so every intermediate is exact and
//   an overflow is reported rather than truncated.
std::optional<arith_uint256> PricedTarget(const arith_uint256& T0, uint64_t P, uint64_t Q);

// The consensus comparison for the priced boundary: hash <= target (the same
// direction VerifyHeaderPoW uses for the base target). Equality passes; a
// zero target admits only the all-zero hash.
bool HashMeetsTarget(const arith_uint256& hash, const arith_uint256& target);

}  // namespace pow_v4
