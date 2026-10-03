// Copyright (c) 2026 TensorCash
// C ABI for the node's unchanged, integer-only post-sunset admission rules.
#include "flat_region.h"
#include "prompt_scaffold.h"
#include <verification/credit_v4.h>
#include <verification/pow_v3.h>
#include <algorithm>
#include <exception>
#include <vector>

namespace {
constexpr uint64_t W = 256;
constexpr size_t OUTPUTS = 11;
// Status 3 means canonical credit disagrees with the earlier replay; 4 means
// an internal failure. Both are infrastructure abstentions, never acceptance.
int Evaluate(const std::vector<std::vector<float>>& rows, bool cdf,
             const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
             uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out)
{
    std::vector<uint32_t> p;
    std::vector<uint8_t> mask;
    if (np) p.assign(prompt, prompt + np);
    if (nm) mask.assign(pad, pad + nm);
    const std::vector<uint32_t> tokens(chosen, chosen + W);
    const std::vector<uint64_t> c(credits, credits + W);
    prompt_scaffold::Metrics scaffold;
    if (!prompt_scaffold::ComputeWindow(p, mask, tokens, c, scaffold)) return 2;
    for (size_t i = 0; i < 4; ++i) out[i] = scaffold.covered[i];
    out[4] = scaffold.reject;
    // Preserve QuickVerifier's order and avoid unnecessary flatness work.
    if (scaffold.reject) return 0;
    flat_region::Metrics flat;
    if (!flat_region::ComputeWindow(rows, c, cdf, flat)) return 2;
    out[5] = flat.capped_units;
    out[6] = flat.flat_capped_units;
    out[7] = flat.flat_credited_rows;
    out[8] = flat.eligible;
    out[9] = flat.reject;
    out[10] = 1; // flat metric computed (eligible=false means search skipped)
    return 0;
}

bool Valid(uint64_t n, const uint32_t* lengths, const float* values,
           const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
           uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out)
{
    if (out) std::fill_n(out, OUTPUTS, 0);
    if (n != W || !lengths || !values || !chosen || !credits || !out ||
        (np && !prompt) || (nm && (!pad || nm != np))) return false;
    for (size_t i = 0; i < W; ++i) if (credits[i] > flat_region::MAX_STEP_CREDIT) return false;
    for (size_t i = 0; i < nm; ++i) if (pad[i] > 1) return false;
    return true;
}
} // namespace

extern "C" int32_t tc_sunset_version() { return 1; }

// Evidence adapter for conformance and diagnostics. Live verification uses
// the raw-row entry below, so caller-supplied probabilities cannot be trusted.
extern "C" int32_t tc_sunset_evidence_batch(
    uint64_t n, const uint32_t* lengths, const float* values, uint8_t cdf,
    const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
    uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out)
{
    if (!Valid(n, lengths, values, chosen, np, prompt, nm, pad, credits, out) || cdf > 1) return 2;
    try {
        std::vector<std::vector<float>> rows(W);
        size_t offset = 0;
        for (size_t i = 0; i < W; ++i) {
            if (!lengths[i] || lengths[i] > flat_region::MAX_ROW) return 2;
            rows[i].assign(values + offset, values + offset + lengths[i]);
            offset += lengths[i];
        }
        return Evaluate(rows, cdf != 0, chosen, np, prompt, nm, pad, credits, out);
    } catch (const std::exception&) { return 4; }
}

// RAW declared rows -> canonical probabilities -> node predicates. Credits
// must equal the previous race replay. Sampling membership remains the job
// of the preceding window kernel; this entry never substitutes for it.
extern "C" int32_t tc_sunset_window_batch(
    uint64_t n, const uint32_t* lengths, const float* logits, const uint32_t* ids,
    const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
    uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out,
    uint64_t* failed_step)
{
    if (!Valid(n, lengths, logits, chosen, np, prompt, nm, pad, credits, out) || !ids) return 2;
    try {
        std::vector<std::vector<float>> rows(W);
        size_t offset = 0;
        credit_v4::StepResult step;
        for (size_t i = 0; i < W; ++i) {
            if (failed_step) *failed_step = i;
            if (!lengths[i] || lengths[i] > 70) return 2;
            const auto status = credit_v4::ComputeStep(logits + offset, ids + offset,
                lengths[i], chosen[i], nullptr, step);
            if (status != credit_v4::Status::Ok) return static_cast<int32_t>(status);
            if (step.position < 0 || !(step.p_chosen > 0)) return 2;
            const auto units = pow_v3::credit_units_for_step(pow_v3::mass_q63_for_step(0, step.p_chosen));
            if (credits[i] != units) return 3;
            rows[i] = std::move(step.cdf_p);
            offset += lengths[i];
        }
        return Evaluate(rows, false, chosen, np, prompt, nm, pad, credits, out);
    } catch (const std::exception&) { return 4; }
}
