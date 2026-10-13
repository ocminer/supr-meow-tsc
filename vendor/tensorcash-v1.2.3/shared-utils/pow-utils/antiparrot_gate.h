// The §8 anti-parrot HARD gate over a window's RAW declared rows, as a
// header-only core the NATIVE miner path can call.
//
// WHAT THIS IS. Exactly the evaluation `tc_antiparrot_window_batch`
// (antiparrot_abi.cpp) performs for the stateless verifier and the pool verify
// pack, and therefore exactly what the node's canonical-kernel capture does
// (QuickVerifier::VerifySequenceLightVectorized, canonical_credit branch, then
// ComputeAntiParrotMetrics / VerifyAntiParrot):
//
//   credit_v4::ComputeStep per declared row with NO u  ->  the evidence
//   interval (lower, upper), or (0, p_chosen) under the mode-V4 race sampler
//   ->  pow_v3::mass_q63_for_step / mass_lower_q63_for_step  ->  ONE
//   pow_v3::compute_anti_parrot_metrics over the canonical id-ordered CDF
//   (plus pow_v4::ApplyRaceCollision under the race sampler)  ->
//   pow_v3::anti_parrot_reject_reason.
//
// Every consensus quantity above is an out-of-line call into the vendored,
// flag-pinned translation units (credit_v4.cpp, pow_v3.cpp, pow_v4.cpp): this
// header performs NO floating-point arithmetic of its own, only exact
// float -> double widenings and vector moves, so a caller compiled with other
// options cannot get a different answer.
//
// WHY A HEADER. antiparrot_abi.cpp is a translation unit of libtc_credit_v4
// only; the miner links pow_utils.cpp instead (proof_processor.so and
// llama.cpp's server-context). A header needs no build-system entry anywhere,
// so the native gate reaches the miner without adding a source to the nine
// image recipes that spell libtc_credit_v4's sources out by name.
//
// KEEP IN STEP with antiparrot_abi.cpp: both are pinned to the SAME committed
// bcore expectations (tests/vectors/antiparrot_mainnet/expected_bcore.json) --
// tests/test_antiparrot_mainnet_vectors.py through the C ABI, and
// tests/test_v4_anti_parrot_gate_cpp.cpp through this header -- so a drift in
// either shows up as a failing vector rather than as a silent disagreement.

#ifndef TENSORCASH_POW_UTILS_ANTIPARROT_GATE_H
#define TENSORCASH_POW_UTILS_ANTIPARROT_GATE_H

#include "credit_v4.h"
#include "pow_v3.h"
#include "pow_v4.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace antiparrot_gate {

// Everything the node's gate reads, so a caller can log what the node logs.
struct Verdict {
    // credit_v4::Status of the kernel pass (Ok / BadInput /
    // BadFloatEnvironment). `ok` is false for anything but Ok: the evidence
    // could not be gated here at all.
    int32_t status{static_cast<int32_t>(credit_v4::Status::Ok)};
    bool ok{true};
    // The node's own reject reason string, "" to accept. Only meaningful
    // when `ok`.
    std::string reason;
    // pow_v3::AntiParrotMetrics, in the order antiparrot_abi.cpp reports them.
    uint64_t braw{0};
    uint64_t bcap{0};
    uint64_t hard_run{0};
    uint64_t maxrs_q32{0};
    uint64_t reuse_cache_q32{0};
    uint64_t n_states_eff_u{0};
    uint64_t n_states_eff_id{0};
    // The step the kernel refused, when !ok and the refusal came from a row.
    std::size_t failed_step{0};

    bool rejected() const { return ok && !reason.empty(); }
};

// The gate over a window's RAW declared rows (the very rows the proof writer
// is about to serialize).
//
//   logits / ids   one row per step, equal length within a step, duplicates
//                  allowed (the kernel dedupes keeping the maximum logit).
//   chosen         the chosen token id per step.
//   prompt_unpadded  the UNPADDED prompt token ids (pad-masked slots already
//                  dropped by the caller, as the verifier's
//                  _verify_anti_parrot does); may be empty, which leaves the
//                  small-alphabet shadow telemetry at zero exactly as the C
//                  ABI's n_prompt == 0 does.
//   race_sampler   the proof's MODE, never a default: true for a mode-V4
//                  proof (evidence interval (0, p_chosen) and the same-noise
//                  collision_race), false for the CDF bounds a mode-V3 proof
//                  is gated over. The divergence is not cosmetic - on the
//                  mainnet h24150 window it is n_states_eff_u 44 against 41.
inline Verdict evaluate_window(const std::vector<std::vector<float>>& logits,
                               const std::vector<std::vector<uint32_t>>& ids,
                               const std::vector<uint32_t>& chosen,
                               const std::vector<uint32_t>& prompt_unpadded,
                               bool race_sampler) {
    Verdict out;
    const std::size_t n = chosen.size();
    if (n == 0 || logits.size() != n || ids.size() != n) {
        out.status = static_cast<int32_t>(credit_v4::Status::BadInput);
        out.ok = false;
        return out;
    }

    std::vector<uint64_t> credit_mass(n), hard_mass(n);
    std::vector<std::vector<uint32_t>> cdf_ids(n);
    std::vector<std::vector<double>> topk_cdf_hi(n);
    std::vector<std::vector<float>> cdf_p;
    if (race_sampler) cdf_p.resize(n);

    credit_v4::StepResult r;
    for (std::size_t s = 0; s < n; ++s) {
        if (logits[s].size() != ids[s].size() || logits[s].empty()) {
            out.status = static_cast<int32_t>(credit_v4::Status::BadInput);
            out.ok = false;
            out.failed_step = s;
            return out;
        }
        const credit_v4::Status st = credit_v4::ComputeStep(
            logits[s].data(), ids[s].data(), logits[s].size(), chosen[s],
            /*u=*/nullptr, r);
        if (st != credit_v4::Status::Ok) {
            out.status = static_cast<int32_t>(st);
            out.ok = false;
            out.failed_step = s;
            return out;
        }
        // The mode-V4 race evidence interval is (0, p_chosen); the CDF sampler
        // keeps the chosen token's own bounds. Same selection as
        // tc_antiparrot_window_batch and credit_v4::ComputeWindowStructure.
        const float lower = race_sampler ? 0.0f : r.lower;
        const float upper = race_sampler ? r.p_chosen : r.upper;
        credit_mass[s] = pow_v3::mass_q63_for_step(lower, upper);
        hard_mass[s] = pow_v3::mass_lower_q63_for_step(lower, upper);
        topk_cdf_hi[s].reserve(r.cdf_hi.size());
        for (float endpoint : r.cdf_hi) {
            topk_cdf_hi[s].push_back(static_cast<double>(endpoint));
        }
        cdf_ids[s] = std::move(r.cdf_ids);
        if (race_sampler) cdf_p[s] = std::move(r.cdf_p);
    }

    pow_v3::AntiParrotMetrics m;
    try {
        m = pow_v3::compute_anti_parrot_metrics(credit_mass, hard_mass, cdf_ids,
                                                topk_cdf_hi, chosen,
                                                prompt_unpadded);
        if (race_sampler) {
            pow_v4::ApplyRaceCollision(
                m, pow_v4::RaceCollisionTotal(credit_mass, cdf_ids, cdf_p));
        }
    } catch (const std::invalid_argument&) {
        // A garbage interval never earns credit: the proof is invalid (§8),
        // exactly as QuickVerifier::ComputeAntiParrotMetrics treats it (and as
        // antiparrot_abi.cpp's Evaluate returns BadInput).
        out.status = static_cast<int32_t>(credit_v4::Status::BadInput);
        out.ok = false;
        return out;
    }

    out.braw = m.braw;
    out.bcap = m.bcap;
    out.hard_run = m.hard_run;
    out.maxrs_q32 = m.maxrs_q32;
    out.reuse_cache_q32 = m.reuse_cache_q32;
    out.n_states_eff_u = m.n_states_eff_u;
    out.n_states_eff_id = m.n_states_eff_id;
    const char* reason = pow_v3::anti_parrot_reject_reason(m);
    out.reason = (reason == nullptr) ? std::string() : std::string(reason);
    return out;
}

}  // namespace antiparrot_gate

#endif  // TENSORCASH_POW_UTILS_ANTIPARROT_GATE_H
