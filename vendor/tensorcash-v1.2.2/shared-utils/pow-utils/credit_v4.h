// Canonical proof v4 credit kernel: declared top-k row -> chosen-token CDF
// interval -> conservative credit units, with a fully specified scalar
// evaluation sequence.
//
// Why a separate kernel. The v3 path (QuickVerifier::VerifySampleStep) builds
// the same quantities with std::exp / std::log from the platform libm, which
// IEEE 754 does not require to be correctly rounded, so two conforming
// platforms may disagree in the last bit. Every other operation on the path
// is a basic IEEE operation or an integer. This kernel replaces the two libm
// calls with CORE-MATH cr_exp / cr_log (correctly rounded binary64, vendored
// under verification/coremath with its provenance) and pins the rest of the
// evaluation contract:
//
//   * binary32 logits in, binary64 accumulation, the conversion points below
//     are the only binary64 -> binary32 roundings;
//   * round-to-nearest-ties-to-even, gradual underflow (no flush-to-zero or
//     denormals-are-zero: x86 MXCSR FTZ/DAZ, AArch64 FPCR FZ, FIZ and AH),
//     FLT_EVAL_METHOD == 0; FloatEnvironmentOk checks these at run time and
//     the kernel refuses to run otherwise;
//   * no floating-point contraction in this file or the vendored sources
//     (-ffp-contract=off per source in CMake); CORE-MATH's explicit
//     __builtin_fma calls are part of its algorithm and are kept;
//   * every reduction is sequential in the stated order.
//
// Evaluation sequence for one step (v3/v4 pinned sampler profile:
// temperature 1, top_p 1, top_k 50, repetition penalty 1; temperature 1 and
// repetition penalty 1 are exact identities and top_p 1 disables the top-p
// branch, so none of them contributes an operation):
//
//   1. dedupe: for each token id keep the maximum logit (first occurrence on
//      equality); order the survivors by id ascending.
//   2. top-k: if k < n, t = the k-th largest logit (descending sort); every
//      logit <= t becomes -inf (ties at t are masked too). If that masks
//      every entry, restore the single entry with the largest pre-mask logit
//      (smallest id on ties).
//   3. m = the first maximum over the row; lse = -inf if m == -inf, else
//      lse = double(m) + cr_log(S) with S = sum over finite entries, in id
//      order, of cr_exp(double(x) - double(m)).
//   4. p_i = float(cr_exp(double(x_i) - lse)) for finite x_i, else 0.
//   5. CDF over entries with p_i > 0 in id order: c = c + double(p_i) in
//      binary64 from c = 0, cdf_i = float(c).
//   6. chosen token at position j: lower = j > 0 ? cdf_{j-1} : 0, upper =
//      cdf_j; absent: position -1, lower = upper = 0.
//   7. in bounds: u > lower - ATOL and u <= upper + ATOL, binary32.
//   8. mass = pow_v3::mass_q63_for_step(lower, upper) (exact integer
//      quantisation plus 2 * ATOL_Q63_CEIL), credit =
//      pow_v3::credit_units_for_step(mass) (integer table).
//   9. p_max_lower_q63 = the CONSERVATIVE LOWER interval mass, in the SAME
//      Q63 units and with the SAME ATOL handling
//      (pow_v3::mass_lower_q63_for_step: floor(upper), ceil(lower), minus
//      2 * ATOL_Q63_CEIL), of the MOST PROBABLE token of this row — not of
//      the chosen token, and never of a wire-supplied scalar. The most
//      probable token is the first maximum of the masked row in id order
//      (the same `m` step 3 takes): softmax is strictly monotone, so the
//      largest logit is the largest probability, and the binary32 rounding
//      of step 4 can create ties but can never reorder. Its CDF interval is
//      read from the same cdf_hi array step 5 built, so this is a property
//      of the canonical CDF only. Its probability is 1/S >= 1/TOP_K > 0, so
//      it is always present in the CDF.
//
// This is bit-for-bit the v3 path's sequence with the two libm calls
// replaced, so on inputs where the platform libm happens to round correctly
// the outputs coincide. It is used for v4-mode proofs only; v3 and legacy
// proofs keep the v3 path unchanged (see QuickVerifier).
//
// The same source is vendored byte-identical into
// shared-utils/pow-utils/credit_v4.{h,cpp}; build_credit_v4_lib.sh there
// builds the C ABI below as a shared library for the Python verifier, so
// every consumer of v4 credit runs this code rather than a re-implementation.

#ifndef TENSORCASH_VERIFICATION_CREDIT_V4_H
#define TENSORCASH_VERIFICATION_CREDIT_V4_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace credit_v4 {

// Bumped on any change to the evaluation sequence; the C ABI reports it so a
// consumer can refuse a mismatched library.
//   1 -> 2: step 9 (p_max_lower_q63) added to StepResult and to the batch
//           C ABI. Steps 1-8 are byte-for-byte unchanged; the bump exists so
//           a consumer that needs the new output refuses an old library
//           instead of reading an absent field.
//   2 -> 3: ComputeWindowStructure / tc_credit_v4_structure_batch added --
//           the ONE call that returns everything the v4 structure rule needs
//           for a window (per-step credit and p_max_lower_q63, the near-pin
//           count N, bcap_units and both 128-bit collision totals) from a
//           single pass over the raw declared rows. Steps 1-9 and
//           tc_credit_v4_batch are byte-for-byte unchanged; the bump exists
//           so a consumer that needs the window call refuses an old library
//           instead of falling back to rebuilding the CDFs itself.
//   3 -> 4: the RACE sampler outputs: StepResult gains p_chosen (step 4's
//           binary32 probability of the chosen token), the step-2 masked row
//           as support_ids / support_logits (the Gumbel race support) and
//           cdf_p (step 4's probabilities aligned with cdf_ids, the
//           collision_race input); ComputeWindowStructure / a new
//           tc_credit_v4_race_structure_batch evaluate the mode-V4 evidence
//           interval (0, p_chosen) and the same-noise collision_race
//           (pow_v4.h) in place of the u-interval collision. Steps 1-9 and
//           the existing C ABI are byte-for-byte unchanged.
//   4 -> 5: race verification and structure in one call with trusted step digests.
//   5 -> 6: the candidate-band V90 aggregate (verification/candidate_band.h):
//           ComputeWindowStructure accumulates it from the SAME id-ordered
//           CDF endpoints every step already produced (StepResult::cdf_ids /
//           cdf_hi, identical under both sampler modes) and reports N, T and
//           L31 in WindowStructure; tc_credit_v4_candidate_band_batch exposes
//           the same aggregate to non-C++ consumers. Steps 1-9 and every
//           existing C ABI entry are byte-for-byte unchanged; the bump exists
//           so a consumer that enforces the condition refuses an older
//           library instead of silently omitting the check.
constexpr int32_t KERNEL_VERSION = 6;

// The pinned sampler top-k (pow_v3::SAMPLER_V3_TOP_K) and the verifier ATOL
// (quick_verifier.h ATOL, services/verification-api constants.py).
constexpr uint32_t TOP_K = 50;
constexpr float ATOL = 0.0001f;

enum class Status : int32_t {
    Ok = 0,
    BadFloatEnvironment = 1,  // rounding mode, flush-to-zero or DAZ not canonical
    BadInput = 2,             // empty row, non-finite logit, invalid interval
};

struct StepResult {
    float lower{0.0f};
    float upper{0.0f};
    int32_t position{-1};      // index of the chosen id in the CDF, -1 if absent
    bool in_bounds{false};     // step 7, meaningful only when u was supplied
    uint64_t mass_q63{0};
    uint64_t credit_units{0};
    // Step 9: conservative LOWER mass (Q63) of the MOST PROBABLE token of
    // this row, from the canonical CDF. Consumed by the v4 near-pin
    // structure rule (pow_v4::StructureMetrics::near_pin_count); it is not a
    // credit quantity and never enters mass_q63 / credit_units.
    uint64_t p_max_lower_q63{0};
    std::vector<uint32_t> cdf_ids;  // active ids, ascending
    std::vector<float> cdf_hi;      // their float32 CDF upper endpoints
    // Step 4's binary32 probabilities aligned with cdf_ids (every entry > 0):
    // the per-token input of the mode-V4 collision_race (pow_v4.h).
    std::vector<float> cdf_p;
    // Step 4's binary32 probability of the chosen token, 0 when absent. Under
    // the mode-V4 race sampler the evidence interval is (0, p_chosen): its
    // mass, credit and hard mass are pow_v3::mass_q63_for_step(0, p_chosen)
    // and friends (see the race outputs of ComputeWindowStructure).
    float p_chosen{0.0f};
    // The step-2 masked row, i.e. the Gumbel race SUPPORT: ids ascending
    // (dedupe order) with their binary32 post-transform log-weights, finite
    // entries only (a masked entry is simply not listed). Under the pinned
    // profile no arithmetic transform applies, so these are the declared
    // logits after dedupe / top-k / the all-masked fallback.
    std::vector<uint32_t> support_ids;
    std::vector<float> support_logits;
};

// Everything the proof v4 STRUCTURE rule needs for ONE window, produced by a
// SINGLE pass over the window's raw declared rows. Every field is the value
// the node already computes on its own Quick pass; nothing here is a second
// derivation:
//
//   * the per-step arrays are exactly what ComputeStep reports step by step;
//   * `bcap_units`, `collision_u` and `collision_id` come from ONE
//     pow_v3::compute_anti_parrot_metrics over the canonical id-ordered CDF
//     this pass already built (no second softmax, no second CDF);
//   * `near_pin_count` is pow_v4::V4NearPinCount over the per-step
//     p_max_lower_q63, i.e. the consensus predicate, not a copy of it.
//
// `valid` is true only when the window is a COMPLETE pow_v4::PRICE_W-position
// window, so that `near_pin_count` is a complete count. On a short or long
// window the per-step arrays and the aggregation are still filled (a caller
// may want them for telemetry) but `near_pin_count` stays 0 and `valid` stays
// false -- and 0 would read as "no near-pinned position", which is the
// CHEAPEST possible answer, so a caller must fail closed on !valid rather than
// price it. pow_v4::StructureMetrics::valid takes this flag.
struct WindowStructure {
    // Per step, in step order; n_steps entries each. Under race_sampler
    // (mode V4) lower is 0 and upper is p_chosen for every step, and
    // collision_u carries collision_race (pow_v4::RaceCollisionTotal).
    std::vector<float> lower;
    std::vector<float> upper;
    std::vector<float> p_chosen;
    std::vector<int32_t> position;
    std::vector<uint8_t> in_bounds;   // 0/1; all 0 when no u was supplied
    std::vector<uint64_t> mass_q63;
    std::vector<uint64_t> credit_units;
    std::vector<uint64_t> p_max_lower_q63;
    // Window aggregates.
    uint64_t near_pin_count{0};
    uint64_t bcap_units{0};
    unsigned __int128 collision_u{0};
    unsigned __int128 collision_id{0};
    bool valid{false};
    // Candidate-band V90 condition (verification/candidate_band.h; kernel
    // v6), accumulated from the SAME StepResult::cdf_ids / cdf_hi every step
    // above produced -- the CDF interval widths, never cdf_p, never the
    // ATOL-widened masses, identical under both sampler modes. N (active
    // rows), T and L31 are exact integers; `candidate_complete` is true only
    // when the window is exactly candidate_band::WINDOW_ROWS rows and every
    // row was accepted by the accumulator. On an incomplete window the three
    // aggregates read 0, which is the CHEAPEST possible answer (no rejection),
    // so a consumer that enforces the condition must fail closed on
    // !candidate_complete rather than read them. `candidate_reject` is
    // candidate_band::Rejects over the three, false when incomplete.
    uint64_t candidate_active_rows{0};
    unsigned __int128 candidate_total{0};
    unsigned __int128 candidate_largest_terms{0};
    bool candidate_complete{false};
    bool candidate_reject{false};
};

// True when the calling thread's floating-point environment is the one the
// contract assumes. Checked by ComputeStep on every call (cheap) so a caller
// that changed the environment cannot get a silently different answer.
bool FloatEnvironmentOk(std::string* why = nullptr);

// One step over a raw declared row (duplicates allowed). u may be null, in
// which case in_bounds is left false. Never throws.
Status ComputeStep(const float* logits, const uint32_t* ids, std::size_t n,
                   uint32_t chosen, const float* u, StepResult& out,
                   std::string* err = nullptr);

// One pass over a whole window's raw declared rows (concatenated exactly as
// tc_credit_v4_batch takes them: row s occupies logits/ids[offset_s,
// offset_s + row_len[s]) with offset_s the prefix sum of row_len). u may be
// null. Returns the first failing step's Status; on failure `out.valid` is
// false and `out.lower.size()` is the index of the failing step (the only
// field a caller may read then). Never throws. This is the ONLY call a consumer of the structure rule needs:
// it exists so the near-pin count, bcap and the collision totals have exactly
// one implementation, reached the same way by the node, the miner and the
// Python verifier.
//
// race_sampler selects the mode-V4 evaluation (the Gumbel race sampler,
// pow_v4::SAMPLER_MODE_SUFFIX): the per-step evidence interval is
// (0, p_chosen) instead of the chosen token's CDF interval (credit, mass,
// hard mass and the anti-parrot weights all read it), in_bounds is left 0
// (the race, not a u band, decides acceptance) and the window's
// collision_u is the same-noise collision_race over the kernel's binary32
// probabilities (pow_v4::RaceCollisionTotal) instead of the u-interval
// overlap; collision_id, bcap and the near-pin count are unchanged. The race
// verdict itself (winner == chosen) is NOT evaluated here: it needs the
// position seed D, which only the verifier holds.
Status ComputeWindowStructure(const uint32_t* row_len, const float* logits,
                              const uint32_t* ids, const uint32_t* chosen,
                              const float* u, std::size_t n_steps,
                              WindowStructure& out, std::string* err = nullptr,
                              bool race_sampler = false,
                              const uint8_t* digests = nullptr);

// The two correctly rounded primitives, exposed for conformance vectors.
double CanonicalExp(double x);
double CanonicalLog(double x);

}  // namespace credit_v4

extern "C" {

// KERNEL_VERSION.
int32_t tc_credit_v4_version(void);

// Batch entry for non-C++ consumers. Rows are concatenated: row s occupies
// logits/ids[offset_s, offset_s + row_len[s]) with offset_s the prefix sum of
// row_len. u may be null (in_bounds is then written as 0). in_bounds may be
// null, and so may p_max_lower_q63 (step 9; a consumer that does not need the
// near-pin evidence passes null). Every output array has n_steps entries.
// Returns the Status value; on failure *failed_step (if non-null) holds the
// offending step and the outputs from that step on are unspecified.
int32_t tc_credit_v4_batch(uint64_t n_steps, const uint32_t* row_len,
                           const float* logits, const uint32_t* ids,
                           const uint32_t* chosen, const float* u,
                           float* lower, float* upper, int32_t* position,
                           uint8_t* in_bounds, uint64_t* mass_q63,
                           uint64_t* credit_units, uint64_t* p_max_lower_q63,
                           uint64_t* failed_step);

// Sibling of tc_credit_v4_batch that also returns the window structure
// aggregate (kernel v3): ONE pass, one anti-parrot aggregation, everything the
// v4 structure rule needs. The per-step arguments and their semantics are
// tc_credit_v4_batch's; the additions are the window scalars.
//
// The two collision totals are up to 99 bits and cannot cross the C ABI as one
// integer, so each is returned as a HIGH / LOW pair of uint64
// (total == (hi << 64) | lo) which the binding recomposes. `structure_valid`
// is WindowStructure::valid: 0 means near_pin_count is NOT a complete count
// and the caller must fail closed, never price it as 0.
//
// Every pointer except `u`, `in_bounds` and `failed_step` is required.
int32_t tc_credit_v4_structure_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids,
    const uint32_t* chosen, const float* u,
    float* lower, float* upper, int32_t* position,
    uint8_t* in_bounds, uint64_t* mass_q63, uint64_t* credit_units,
    uint64_t* p_max_lower_q63,
    uint64_t* near_pin_count, uint64_t* bcap_units,
    uint64_t* collision_u_hi, uint64_t* collision_u_lo,
    uint64_t* collision_id_hi, uint64_t* collision_id_lo,
    uint8_t* structure_valid, uint64_t* failed_step);

// Kernel v5: the same outputs for the single race. digests is n_steps * 32
// bytes (draw 0), or null for pricing only. in_bounds reports exact winner
// equality; a null digest input always produces false, never a verdict.
int32_t tc_credit_v4_race_verify_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids,
    const uint32_t* chosen, const uint8_t* digests,
    float* lower, float* upper, int32_t* position,
    uint8_t* in_bounds, uint64_t* mass_q63, uint64_t* credit_units,
    uint64_t* p_max_lower_q63,
    uint64_t* near_pin_count, uint64_t* bcap_units,
    uint64_t* collision_u_hi, uint64_t* collision_u_lo,
    uint64_t* collision_id_hi, uint64_t* collision_id_lo,
    uint8_t* structure_valid, uint64_t* failed_step);

// The mode-V4 (race sampler) sibling of tc_credit_v4_structure_batch:
// ComputeWindowStructure with race_sampler = true. Same arguments and
// semantics, plus `p_chosen` (required; n_steps binary32 entries: step 4's
// probability of the chosen token, the evidence interval's upper end). lower
// / upper are written as (0, p_chosen); mass_q63 / credit_units are of that
// interval; collision_u_hi/lo carry collision_race; in_bounds is always 0
// (u is ignored and may be null). kernel v4.
int32_t tc_credit_v4_race_structure_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids,
    const uint32_t* chosen,
    float* lower, float* upper, float* p_chosen, int32_t* position,
    uint64_t* mass_q63, uint64_t* credit_units,
    uint64_t* p_max_lower_q63,
    uint64_t* near_pin_count, uint64_t* bcap_units,
    uint64_t* collision_u_hi, uint64_t* collision_u_lo,
    uint64_t* collision_id_hi, uint64_t* collision_id_lo,
    uint8_t* structure_valid, uint64_t* failed_step);

// Kernel v6: return credit, structure and candidate-band metrics from ONE
// canonical window pass. The original structure/race entry points retain
// their signatures. race_sampler is 0 (CDF) or 1 (single Gumbel race);
// digests may be null for pricing, but then in_bounds never establishes a
// race verdict. Candidate totals use exact high/low uint64 limbs. Every
// candidate output pointer is required; candidate_complete must be checked.
int32_t tc_credit_v4_window_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids,
    const uint32_t* chosen, const float* u,
    float* lower, float* upper, int32_t* position,
    uint8_t* in_bounds, uint64_t* mass_q63, uint64_t* credit_units,
    uint64_t* p_max_lower_q63,
    uint64_t* near_pin_count, uint64_t* bcap_units,
    uint64_t* collision_u_hi, uint64_t* collision_u_lo,
    uint64_t* collision_id_hi, uint64_t* collision_id_lo,
    uint8_t* structure_valid, uint64_t* failed_step,
    uint8_t race_sampler, const uint8_t* digests,
    uint64_t* candidate_active_rows,
    uint64_t* candidate_total_hi, uint64_t* candidate_total_lo,
    uint64_t* candidate_largest_hi, uint64_t* candidate_largest_lo,
    uint8_t* candidate_complete, uint8_t* candidate_reject);

// Kernel v6: the candidate-band V90 aggregate of a window
// (verification/candidate_band.h) from ONE canonical pass over the raw
// declared rows (ComputeWindowStructure; the aggregate does not depend on
// the sampler mode, u or the chosen tokens beyond the rows' own CDFs, so
// this entry takes neither u nor digests). The per-step arguments are
// tc_credit_v4_batch's. Outputs: N (active rows), T and L31 as HIGH / LOW
// uint64 pairs (each is up to 92 bits), `complete` (WindowStructure::
// candidate_complete: 0 means the window is not exactly 256 accepted rows
// and the aggregates must NOT be read -- fail closed, never "no rejection")
// and `reject` (candidate_band::Rejects). Every pointer except `failed_step`
// is required. Returns the Status value; on failure *failed_step (if
// non-null) is the failing step.
int32_t tc_credit_v4_candidate_band_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids, const uint32_t* chosen,
    uint64_t* active_rows,
    uint64_t* total_hi, uint64_t* total_lo,
    uint64_t* largest_terms_hi, uint64_t* largest_terms_lo,
    uint8_t* complete, uint8_t* reject, uint64_t* failed_step);

double tc_credit_v4_exp(double x);
double tc_credit_v4_log(double x);

}  // extern "C"

#endif  // TENSORCASH_VERIFICATION_CREDIT_V4_H
