// Canonical proof v4 credit kernel; the evaluation contract is in credit_v4.h.
// Compiled with -ffp-contract=off (CMake per-source property). Vendored
// byte-identical into shared-utils/pow-utils/credit_v4.cpp.

#include <verification/credit_v4.h>

#include <verification/candidate_band.h>
#include <verification/pow_v3.h>
#include <verification/pow_v4.h>
#include <verification/gumbel_sampler.h>
#include <array>
#include <cstring>

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__)
#error "credit_v4: the float environment check supports x86-64 and AArch64 only"
#endif

#if FLT_EVAL_METHOD != 0
#error "credit_v4 requires FLT_EVAL_METHOD == 0 (no excess precision)"
#endif

extern "C" double cr_exp(double);
extern "C" double cr_log(double);

namespace credit_v4 {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "binary32 required");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559, "binary64 required");
static_assert(TOP_K == pow_v3::SAMPLER_V3_TOP_K, "pinned top-k drifted from pow_v3");

double CanonicalExp(double x) { return cr_exp(x); }
double CanonicalLog(double x) { return cr_log(x); }

bool FloatEnvironmentOk(std::string* why) {
    if (std::fegetround() != FE_TONEAREST) {
        if (why) *why = "rounding mode is not round-to-nearest";
        return false;
    }
#if defined(__x86_64__) || defined(_M_X64)
    const unsigned int csr = _mm_getcsr();
    if (csr & (1u << 15)) {
        if (why) *why = "MXCSR flush-to-zero is set";
        return false;
    }
    if (csr & (1u << 6)) {
        if (why) *why = "MXCSR denormals-are-zero is set";
        return false;
    }
    if (csr & (3u << 13)) {
        if (why) *why = "MXCSR rounding control is not round-to-nearest";
        return false;
    }
#elif defined(__aarch64__)
    uint64_t fpcr = 0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (fpcr & (1ull << 24)) {
        if (why) *why = "FPCR flush-to-zero is set";
        return false;
    }
    // FEAT_AFP (Armv8.7): FIZ flushes denormal inputs to zero and AH selects
    // the alternate floating-point behaviour (including denormal handling).
    // Both change results; on cores without FEAT_AFP the bits read as zero.
    if (fpcr & 1ull) {
        if (why) *why = "FPCR flush-inputs-to-zero (FIZ) is set";
        return false;
    }
    if (fpcr & (1ull << 1)) {
        if (why) *why = "FPCR alternate handling (AH) is set";
        return false;
    }
    if (fpcr & (3ull << 22)) {
        if (why) *why = "FPCR rounding mode is not round-to-nearest";
        return false;
    }
#endif
    return true;
}

namespace {

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

void Fail(std::string* err, const char* what) {
    if (err) *err = what;
}

}  // namespace

Status ComputeStep(const float* logits, const uint32_t* ids, std::size_t n,
                   uint32_t chosen, const float* u, StepResult& out,
                   std::string* err) {
    out = StepResult{};
    if (!FloatEnvironmentOk(err)) return Status::BadFloatEnvironment;
    if (n == 0 || logits == nullptr || ids == nullptr) {
        Fail(err, "empty row");
        return Status::BadInput;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            Fail(err, "non-finite logit");
            return Status::BadInput;
        }
    }

    // 1. dedupe, keep the maximum per id; std::map yields ids ascending.
    std::map<uint32_t, float> tok_to_max;
    for (std::size_t i = 0; i < n; ++i) {
        auto it = tok_to_max.find(ids[i]);
        if (it == tok_to_max.end() || logits[i] > it->second) {
            tok_to_max[ids[i]] = logits[i];
        }
    }
    std::vector<uint32_t> row_ids;
    std::vector<float> row;
    row_ids.reserve(tok_to_max.size());
    row.reserve(tok_to_max.size());
    for (const auto& [tok, val] : tok_to_max) {
        row_ids.push_back(tok);
        row.push_back(val);
    }

    // 2. top-k, ties at the k-th value masked too, then the all-masked
    // fallback to the single best pre-mask entry (smallest id on ties).
    const std::vector<float> pre_mask = row;
    if (TOP_K > 0 && TOP_K < row.size()) {
        std::vector<float> sorted = row;
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        const float threshold = sorted[TOP_K - 1];
        for (float& x : row) {
            if (x <= threshold) x = NEG_INF;
        }
    }
    if (std::all_of(row.begin(), row.end(), [](float x) { return !std::isfinite(x); })) {
        std::size_t best = 0;
        for (std::size_t i = 1; i < pre_mask.size(); ++i) {
            if (pre_mask[i] > pre_mask[best] ||
                (pre_mask[i] == pre_mask[best] && row_ids[i] < row_ids[best])) {
                best = i;
            }
        }
        std::fill(row.begin(), row.end(), NEG_INF);
        row[best] = pre_mask[best];
    }
    // The race support: the masked row's finite entries, in id order.
    out.support_ids.reserve(row.size());
    out.support_logits.reserve(row.size());
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (row[i] > NEG_INF) {
            out.support_ids.push_back(row_ids[i]);
            out.support_logits.push_back(row[i]);
        }
    }

    // 3. log-sum-exp, sequential over the row in id order. max_element
    // returns the FIRST maximum, i.e. the smallest id among the maxima; that
    // index is also the most probable token step 9 reads (softmax is
    // strictly monotone in the logit).
    const std::vector<float>::const_iterator max_it =
        std::max_element(row.begin(), row.end());
    const std::size_t max_index = static_cast<std::size_t>(max_it - row.begin());
    const float m = *max_it;
    double lse;
    if (m == NEG_INF) {
        lse = -std::numeric_limits<double>::infinity();
    } else {
        double sum = 0.0;
        for (float x : row) {
            if (x > NEG_INF) sum += cr_exp(static_cast<double>(x) - static_cast<double>(m));
        }
        lse = static_cast<double>(m) + cr_log(sum);
    }

    // 4. probabilities, one binary64 -> binary32 rounding each.
    std::vector<float> probs(row.size(), 0.0f);
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (row[i] > NEG_INF) {
            probs[i] = static_cast<float>(cr_exp(static_cast<double>(row[i]) - lse));
        }
    }

    // 5. CDF over positive probabilities in id order (row_ids is ascending).
    // The most probable token's CDF slot is recorded on the way (step 9); it
    // is always taken because its probability is 1/S >= 1/TOP_K > 0.
    double cumulative = 0.0;
    std::size_t max_cdf_position = out.cdf_ids.max_size();  // "not present"
    out.cdf_ids.reserve(row.size());
    out.cdf_hi.reserve(row.size());
    out.cdf_p.reserve(row.size());
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (probs[i] > 0.0f) {
            if (i == max_index) max_cdf_position = out.cdf_ids.size();
            cumulative += static_cast<double>(probs[i]);
            out.cdf_ids.push_back(row_ids[i]);
            out.cdf_hi.push_back(static_cast<float>(cumulative));
            out.cdf_p.push_back(probs[i]);
        }
    }

    // 6. the chosen token's interval (and, for the race sampler, its
    // probability).
    for (std::size_t i = 0; i < out.cdf_ids.size(); ++i) {
        if (out.cdf_ids[i] == chosen) {
            out.position = static_cast<int32_t>(i);
            out.lower = i > 0 ? out.cdf_hi[i - 1] : 0.0f;
            out.upper = out.cdf_hi[i];
            out.p_chosen = out.cdf_p[i];
            break;
        }
    }

    // 7. membership, binary32 as in the v3 path.
    if (u != nullptr) {
        out.in_bounds = (*u > out.lower - ATOL) && (*u <= out.upper + ATOL);
    }

    // 8. conservative credit, integer from here on.
    // 9. the most probable token's conservative LOWER mass, same ATOL
    // handling (pow_v3::mass_lower_q63_for_step). Not a credit quantity; the
    // near-pin structure rule is the only consumer. If the slot were absent
    // (mathematically impossible: p_max >= 1/TOP_K) the value stays 0, which
    // the near-pin predicate never counts.
    try {
        out.mass_q63 = pow_v3::mass_q63_for_step(static_cast<double>(out.lower),
                                                 static_cast<double>(out.upper));
        out.credit_units = pow_v3::credit_units_for_step(out.mass_q63);
        if (max_cdf_position < out.cdf_hi.size()) {
            const float max_lower =
                max_cdf_position > 0 ? out.cdf_hi[max_cdf_position - 1] : 0.0f;
            const float max_upper = out.cdf_hi[max_cdf_position];
            out.p_max_lower_q63 = pow_v3::mass_lower_q63_for_step(
                static_cast<double>(max_lower), static_cast<double>(max_upper));
        }
    } catch (const std::invalid_argument& e) {
        if (err) *err = e.what();
        return Status::BadInput;
    }
    return Status::Ok;
}

Status ComputeWindowStructure(const uint32_t* row_len, const float* logits,
                              const uint32_t* ids, const uint32_t* chosen,
                              const float* u, std::size_t n_steps,
                              WindowStructure& out, std::string* err,
                              bool race_sampler, const uint8_t* digests) {
    out = WindowStructure{};
    if (n_steps == 0 || row_len == nullptr || logits == nullptr ||
        ids == nullptr || chosen == nullptr) {
        Fail(err, "empty window");
        return Status::BadInput;
    }

    out.lower.reserve(n_steps);
    out.upper.reserve(n_steps);
    out.p_chosen.reserve(n_steps);
    out.position.reserve(n_steps);
    out.in_bounds.reserve(n_steps);
    out.mass_q63.reserve(n_steps);
    out.credit_units.reserve(n_steps);
    out.p_max_lower_q63.reserve(n_steps);

    // The anti-parrot aggregation's per-step inputs, filled from THIS pass:
    // the chosen token's conservative UPPER mass (the one B_cred is summed
    // from, so the collision weights are the kernel's own credit units), its
    // conservative LOWER mass, and the canonical id-ordered CDF the same
    // ComputeStep call built. No second softmax and no second CDF anywhere.
    std::vector<uint64_t> credit_mass_q63(n_steps, 0);
    std::vector<uint64_t> hard_mass_q63(n_steps, 0);
    std::vector<std::vector<uint32_t>> topk_ids(n_steps);
    std::vector<std::vector<double>> topk_cdf_hi(n_steps);
    std::vector<std::vector<float>> topk_p(n_steps);
    // The candidate-band V90 accumulator (kernel v6) reads the SAME
    // id-ordered binary32 CDF endpoints, row by row, before anything else
    // consumes them. A row it refuses is malformed evidence and fails the
    // window like any other bad row; a window that is not exactly
    // candidate_band::WINDOW_ROWS rows leaves candidate_complete false.
    candidate_band::Accumulator band;
    bool band_rows_ok = true;

    std::size_t offset = 0;
    StepResult r;
    for (std::size_t s = 0; s < n_steps; ++s) {
        const Status st = ComputeStep(logits + offset, ids + offset, row_len[s],
                                      chosen[s], (u != nullptr && !race_sampler) ? u + s : nullptr,
                                      r, err);
        if (st != Status::Ok) return st;   // out.lower.size() == s: the failing step
        // The evidence interval: the chosen token's CDF interval, or under the
        // race sampler (0, p_chosen) — its mass and credit are recomputed from
        // that interval with the same integer rules (steps 8-9 of the
        // contract, on the other endpoints).
        float lower = r.lower;
        float upper = r.upper;
        uint64_t mass = r.mass_q63;
        uint64_t credit = r.credit_units;
        uint64_t hard = 0;
        try {
            if (race_sampler) {
                lower = 0.0f;
                upper = r.p_chosen;
                mass = pow_v3::mass_q63_for_step(0.0, static_cast<double>(upper));
                credit = pow_v3::credit_units_for_step(mass);
            }
            hard = pow_v3::mass_lower_q63_for_step(static_cast<double>(lower),
                                                   static_cast<double>(upper));
        } catch (const std::invalid_argument& e) {
            if (err) *err = e.what();
            return Status::BadInput;       // out.lower.size() == s: the failing step
        }
        if (race_sampler && digests != nullptr) {
            try {
                gumbel::dlog_selftest();
                std::array<uint8_t, 32> seed{};
                std::memcpy(seed.data(), digests + 32 * s, 32);
                const std::vector<int64_t> support(r.support_ids.begin(), r.support_ids.end());
                const auto winner = gumbel::race(support, r.support_logits, seed);
                r.in_bounds = winner.winner_pos >= 0 &&
                    winner.winner_id == static_cast<int64_t>(chosen[s]);
            } catch (const std::exception& e) {
                if (err) *err = e.what();
                return Status::BadFloatEnvironment;
            }
        }
        out.lower.push_back(lower);
        out.upper.push_back(upper);
        out.p_chosen.push_back(r.p_chosen);
        out.position.push_back(r.position);
        out.in_bounds.push_back(r.in_bounds ? 1u : 0u);
        out.mass_q63.push_back(mass);
        out.credit_units.push_back(credit);
        out.p_max_lower_q63.push_back(r.p_max_lower_q63);

        credit_mass_q63[s] = mass;
        hard_mass_q63[s] = hard;
        if (band_rows_ok) {
            band_rows_ok = band.AddRow(r.cdf_ids.data(), r.cdf_hi.data(), r.cdf_ids.size(), nullptr);
        }
        topk_ids[s] = r.cdf_ids;
        topk_cdf_hi[s].reserve(r.cdf_hi.size());
        for (float hi : r.cdf_hi) topk_cdf_hi[s].push_back(static_cast<double>(hi));
        topk_p[s] = r.cdf_p;
        offset += row_len[s];
    }

    // ONE anti-parrot aggregation, the same pow_v3 code the node runs on the
    // same canonical CDF: bcap_units and both exact 128-bit collision totals.
    // Under the race sampler the u-interval collision is replaced by the
    // same-noise collision_race over the same rows (pow_v4.h).
    try {
        const pow_v3::AntiParrotMetrics ap = pow_v3::compute_anti_parrot_metrics(
            credit_mass_q63, hard_mass_q63, topk_ids, topk_cdf_hi);
        out.bcap_units = ap.bcap;
        out.collision_u = race_sampler
                              ? pow_v4::RaceCollisionTotal(credit_mass_q63, topk_ids, topk_p)
                              : ap.total_collision_u;
        out.collision_id = ap.total_collision_id;
    } catch (const std::invalid_argument& e) {
        if (err) *err = e.what();
        out.valid = false;
        return Status::BadInput;
    }

    // N, through the consensus predicate itself. V4NearPinCount refuses
    // anything that is not exactly PRICE_W masses in [0, 2^63], so a short or
    // long window leaves the count 0 AND valid false; 0 is the cheapest
    // possible count, so the flag is what a caller must read.
    uint64_t count = 0;
    if (pow_v4::V4NearPinCount(out.p_max_lower_q63, count)) {
        out.near_pin_count = count;
        out.valid = true;
    } else if (err) {
        *err = "the near-pin count needs the whole window of canonical "
               "maximum-token lower masses";
    }

    // The candidate-band aggregate over exactly WINDOW_ROWS accepted rows
    // (kernel v6). Finish refuses any other row count, which leaves the
    // three aggregates 0 and candidate_complete false: a consumer must fail
    // closed on the flag, never read 0 as "no rejection".
    candidate_band::Metrics band_metrics;
    if (band_rows_ok && band.Finish(band_metrics, nullptr)) {
        out.candidate_active_rows = band_metrics.active_rows;
        out.candidate_total = band_metrics.total;
        out.candidate_largest_terms = band_metrics.largest_terms;
        out.candidate_complete = true;
        out.candidate_reject = band_metrics.reject;
    }
    return Status::Ok;
}

}  // namespace credit_v4

extern "C" {

int32_t tc_credit_v4_version(void) { return credit_v4::KERNEL_VERSION; }

int32_t tc_credit_v4_batch(uint64_t n_steps, const uint32_t* row_len,
                           const float* logits, const uint32_t* ids,
                           const uint32_t* chosen, const float* u,
                           float* lower, float* upper, int32_t* position,
                           uint8_t* in_bounds, uint64_t* mass_q63,
                           uint64_t* credit_units, uint64_t* p_max_lower_q63,
                           uint64_t* failed_step) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        lower == nullptr || upper == nullptr || position == nullptr ||
        mass_q63 == nullptr || credit_units == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    uint64_t offset = 0;
    credit_v4::StepResult r;
    for (uint64_t s = 0; s < n_steps; ++s) {
        const credit_v4::Status st = credit_v4::ComputeStep(
            logits + offset, ids + offset, row_len[s], chosen[s],
            u != nullptr ? u + s : nullptr, r, nullptr);
        if (st != credit_v4::Status::Ok) {
            if (failed_step) *failed_step = s;
            return static_cast<int32_t>(st);
        }
        lower[s] = r.lower;
        upper[s] = r.upper;
        position[s] = r.position;
        if (in_bounds) in_bounds[s] = (u != nullptr && r.in_bounds) ? 1 : 0;
        mass_q63[s] = r.mass_q63;
        credit_units[s] = r.credit_units;
        if (p_max_lower_q63) p_max_lower_q63[s] = r.p_max_lower_q63;
        offset += row_len[s];
    }
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

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
    uint8_t* candidate_complete, uint8_t* candidate_reject) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        lower == nullptr || upper == nullptr || position == nullptr ||
        mass_q63 == nullptr || credit_units == nullptr || p_max_lower_q63 == nullptr ||
        near_pin_count == nullptr || bcap_units == nullptr ||
        collision_u_hi == nullptr || collision_u_lo == nullptr ||
        collision_id_hi == nullptr || collision_id_lo == nullptr ||
        structure_valid == nullptr || race_sampler > 1 ||
        (race_sampler == 0 && digests != nullptr) ||
        candidate_active_rows == nullptr || candidate_total_hi == nullptr ||
        candidate_total_lo == nullptr || candidate_largest_hi == nullptr ||
        candidate_largest_lo == nullptr || candidate_complete == nullptr ||
        candidate_reject == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    credit_v4::WindowStructure ws;
    const credit_v4::Status st = credit_v4::ComputeWindowStructure(
        row_len, logits, ids, chosen, u, static_cast<std::size_t>(n_steps), ws, nullptr, race_sampler != 0, digests);
    if (st != credit_v4::Status::Ok) {
        // ComputeWindowStructure stops at the first failing step and clears
        // its output, so the step index is the number of steps it completed.
        if (failed_step) *failed_step = static_cast<uint64_t>(ws.lower.size());
        return static_cast<int32_t>(st);
    }
    for (uint64_t s = 0; s < n_steps; ++s) {
        lower[s] = ws.lower[s];
        upper[s] = ws.upper[s];
        position[s] = ws.position[s];
        if (in_bounds) in_bounds[s] = ((race_sampler != 0) ? (digests != nullptr) : (u != nullptr)) ? ws.in_bounds[s] : 0;
        mass_q63[s] = ws.mass_q63[s];
        credit_units[s] = ws.credit_units[s];
        p_max_lower_q63[s] = ws.p_max_lower_q63[s];
    }
    *near_pin_count = ws.near_pin_count;
    *bcap_units = ws.bcap_units;
    // 128 bits do not cross the ABI as one integer: high / low uint64 pair.
    *collision_u_hi = static_cast<uint64_t>(ws.collision_u >> 64);
    *collision_u_lo = static_cast<uint64_t>(ws.collision_u);
    *collision_id_hi = static_cast<uint64_t>(ws.collision_id >> 64);
    *collision_id_lo = static_cast<uint64_t>(ws.collision_id);
    *structure_valid = ws.valid ? 1 : 0;
    *candidate_active_rows = ws.candidate_active_rows;
    *candidate_total_hi = static_cast<uint64_t>(ws.candidate_total >> 64);
    *candidate_total_lo = static_cast<uint64_t>(ws.candidate_total);
    *candidate_largest_hi = static_cast<uint64_t>(ws.candidate_largest_terms >> 64);
    *candidate_largest_lo = static_cast<uint64_t>(ws.candidate_largest_terms);
    *candidate_complete = ws.candidate_complete ? 1 : 0;
    *candidate_reject = ws.candidate_reject ? 1 : 0;
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

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
    uint8_t* structure_valid, uint64_t* failed_step) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        lower == nullptr || upper == nullptr || position == nullptr ||
        mass_q63 == nullptr || credit_units == nullptr || p_max_lower_q63 == nullptr ||
        near_pin_count == nullptr || bcap_units == nullptr ||
        collision_u_hi == nullptr || collision_u_lo == nullptr ||
        collision_id_hi == nullptr || collision_id_lo == nullptr ||
        structure_valid == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    credit_v4::WindowStructure ws;
    const credit_v4::Status st = credit_v4::ComputeWindowStructure(
        row_len, logits, ids, chosen, u, static_cast<std::size_t>(n_steps), ws);
    if (st != credit_v4::Status::Ok) {
        // ComputeWindowStructure stops at the first failing step and clears
        // its output, so the step index is the number of steps it completed.
        if (failed_step) *failed_step = static_cast<uint64_t>(ws.lower.size());
        return static_cast<int32_t>(st);
    }
    for (uint64_t s = 0; s < n_steps; ++s) {
        lower[s] = ws.lower[s];
        upper[s] = ws.upper[s];
        position[s] = ws.position[s];
        if (in_bounds) in_bounds[s] = (u != nullptr) ? ws.in_bounds[s] : 0;
        mass_q63[s] = ws.mass_q63[s];
        credit_units[s] = ws.credit_units[s];
        p_max_lower_q63[s] = ws.p_max_lower_q63[s];
    }
    *near_pin_count = ws.near_pin_count;
    *bcap_units = ws.bcap_units;
    // 128 bits do not cross the ABI as one integer: high / low uint64 pair.
    *collision_u_hi = static_cast<uint64_t>(ws.collision_u >> 64);
    *collision_u_lo = static_cast<uint64_t>(ws.collision_u);
    *collision_id_hi = static_cast<uint64_t>(ws.collision_id >> 64);
    *collision_id_lo = static_cast<uint64_t>(ws.collision_id);
    *structure_valid = ws.valid ? 1 : 0;
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

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
    uint8_t* structure_valid, uint64_t* failed_step) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        lower == nullptr || upper == nullptr || position == nullptr ||
        mass_q63 == nullptr || credit_units == nullptr || p_max_lower_q63 == nullptr ||
        near_pin_count == nullptr || bcap_units == nullptr ||
        collision_u_hi == nullptr || collision_u_lo == nullptr ||
        collision_id_hi == nullptr || collision_id_lo == nullptr ||
        structure_valid == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    credit_v4::WindowStructure ws;
    const credit_v4::Status st = credit_v4::ComputeWindowStructure(
        row_len, logits, ids, chosen, nullptr, static_cast<std::size_t>(n_steps), ws,
        nullptr, true, digests);
    if (st != credit_v4::Status::Ok) {
        // ComputeWindowStructure stops at the first failing step and clears
        // its output, so the step index is the number of steps it completed.
        if (failed_step) *failed_step = static_cast<uint64_t>(ws.lower.size());
        return static_cast<int32_t>(st);
    }
    for (uint64_t s = 0; s < n_steps; ++s) {
        lower[s] = ws.lower[s];
        upper[s] = ws.upper[s];
        position[s] = ws.position[s];
        if (in_bounds) in_bounds[s] = (digests != nullptr) ? ws.in_bounds[s] : 0;
        mass_q63[s] = ws.mass_q63[s];
        credit_units[s] = ws.credit_units[s];
        p_max_lower_q63[s] = ws.p_max_lower_q63[s];
    }
    *near_pin_count = ws.near_pin_count;
    *bcap_units = ws.bcap_units;
    // 128 bits do not cross the ABI as one integer: high / low uint64 pair.
    *collision_u_hi = static_cast<uint64_t>(ws.collision_u >> 64);
    *collision_u_lo = static_cast<uint64_t>(ws.collision_u);
    *collision_id_hi = static_cast<uint64_t>(ws.collision_id >> 64);
    *collision_id_lo = static_cast<uint64_t>(ws.collision_id);
    *structure_valid = ws.valid ? 1 : 0;
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

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
    uint8_t* structure_valid, uint64_t* failed_step) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        lower == nullptr || upper == nullptr || p_chosen == nullptr || position == nullptr ||
        mass_q63 == nullptr || credit_units == nullptr || p_max_lower_q63 == nullptr ||
        near_pin_count == nullptr || bcap_units == nullptr ||
        collision_u_hi == nullptr || collision_u_lo == nullptr ||
        collision_id_hi == nullptr || collision_id_lo == nullptr ||
        structure_valid == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    credit_v4::WindowStructure ws;
    const credit_v4::Status st = credit_v4::ComputeWindowStructure(
        row_len, logits, ids, chosen, nullptr, static_cast<std::size_t>(n_steps), ws,
        nullptr, /*race_sampler=*/true);
    if (st != credit_v4::Status::Ok) {
        if (failed_step) *failed_step = static_cast<uint64_t>(ws.lower.size());
        return static_cast<int32_t>(st);
    }
    for (uint64_t s = 0; s < n_steps; ++s) {
        lower[s] = ws.lower[s];
        upper[s] = ws.upper[s];
        p_chosen[s] = ws.p_chosen[s];
        position[s] = ws.position[s];
        mass_q63[s] = ws.mass_q63[s];
        credit_units[s] = ws.credit_units[s];
        p_max_lower_q63[s] = ws.p_max_lower_q63[s];
    }
    *near_pin_count = ws.near_pin_count;
    *bcap_units = ws.bcap_units;
    *collision_u_hi = static_cast<uint64_t>(ws.collision_u >> 64);
    *collision_u_lo = static_cast<uint64_t>(ws.collision_u);
    *collision_id_hi = static_cast<uint64_t>(ws.collision_id >> 64);
    *collision_id_lo = static_cast<uint64_t>(ws.collision_id);
    *structure_valid = ws.valid ? 1 : 0;
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

int32_t tc_credit_v4_candidate_band_batch(
    uint64_t n_steps, const uint32_t* row_len,
    const float* logits, const uint32_t* ids, const uint32_t* chosen,
    uint64_t* active_rows,
    uint64_t* total_hi, uint64_t* total_lo,
    uint64_t* largest_terms_hi, uint64_t* largest_terms_lo,
    uint8_t* complete, uint8_t* reject, uint64_t* failed_step) {
    if (row_len == nullptr || logits == nullptr || ids == nullptr || chosen == nullptr ||
        active_rows == nullptr || total_hi == nullptr || total_lo == nullptr ||
        largest_terms_hi == nullptr || largest_terms_lo == nullptr ||
        complete == nullptr || reject == nullptr) {
        if (failed_step) *failed_step = 0;
        return static_cast<int32_t>(credit_v4::Status::BadInput);
    }
    // The aggregate reads the rows' own CDFs only: no u, no digests, and the
    // sampler mode changes nothing about them (StepResult::cdf_hi is the
    // same under both), so the plain evaluation serves every consumer.
    credit_v4::WindowStructure ws;
    const credit_v4::Status st = credit_v4::ComputeWindowStructure(
        row_len, logits, ids, chosen, nullptr, static_cast<std::size_t>(n_steps), ws);
    if (st != credit_v4::Status::Ok) {
        if (failed_step) *failed_step = static_cast<uint64_t>(ws.lower.size());
        return static_cast<int32_t>(st);
    }
    *active_rows = ws.candidate_active_rows;
    // 128 bits do not cross the ABI as one integer: high / low uint64 pair.
    *total_hi = static_cast<uint64_t>(ws.candidate_total >> 64);
    *total_lo = static_cast<uint64_t>(ws.candidate_total);
    *largest_terms_hi = static_cast<uint64_t>(ws.candidate_largest_terms >> 64);
    *largest_terms_lo = static_cast<uint64_t>(ws.candidate_largest_terms);
    *complete = ws.candidate_complete ? 1 : 0;
    *reject = ws.candidate_reject ? 1 : 0;
    return static_cast<int32_t>(credit_v4::Status::Ok);
}

// Conformance-only helpers: they expose the primitives for vector checks and
// are NOT a credit entry point. They carry the same environment guard as
// ComputeStep, reported as a quiet NaN, so a caller in a non-canonical
// environment cannot read a platform-dependent value out of them.
double tc_credit_v4_exp(double x) {
    if (!credit_v4::FloatEnvironmentOk()) return std::numeric_limits<double>::quiet_NaN();
    return credit_v4::CanonicalExp(x);
}
double tc_credit_v4_log(double x) {
    if (!credit_v4::FloatEnvironmentOk()) return std::numeric_limits<double>::quiet_NaN();
    return credit_v4::CanonicalLog(x);
}

}  // extern "C"
