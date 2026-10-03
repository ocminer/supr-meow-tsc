// =============================================================================
// precheck.cpp — see precheck.h.
//
// Built in two flavours:
//   * MEOW_PRECHECK_ENABLED: links the unmodified TensorCash v1.2.2 rule
//     sources (target meow_tc122) and compiles this file with the same pinned
//     floating-point flags and the same namespace renames (pow_v3, pow_v4,
//     credit_v4, gumbel -> tcpc_*), so the names below resolve to the
//     upstream copies, never to the miner's own producer sources.
//   * otherwise (MSVC: no unsigned __int128): a stub that reports itself
//     unavailable; the miner then submits every proof as v0.7.0 did.
//
// Every consensus quantity is computed by an upstream function. This file
// only marshals the FlatBuffer into vectors, calls the rules in the
// verifier's order (services/verification-api proof_verifier.py Quick path:
// kernel strict check, structure, candidate band, sunset, tier, anti-parrot)
// and maps the results to a verdict.
// =============================================================================
#include "precheck.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>

namespace meow::precheck {

namespace {

constexpr const char* kGateNames[kGateCount] = {
    "strict", "near-pin", "candidate-band", "prompt-scaffold",
    "flat-region", "credit-tier", "anti-parrot",
};

}  // namespace

const char* gate_name(Gate g) {
    const int i = static_cast<int>(g);
    return (i >= 0 && i < kGateCount) ? kGateNames[i] : "-";
}

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Off:    return "off";
        case Mode::Shadow: return "shadow";
        case Mode::On:     return "on";
    }
    return "?";
}

bool parse_mode(const std::string& s, Mode& out) {
    if (s == "off" || s == "0" || s == "false")      { out = Mode::Off;    return true; }
    if (s == "shadow")                                { out = Mode::Shadow; return true; }
    if (s == "on" || s == "1" || s == "true")        { out = Mode::On;     return true; }
    return false;
}

bool parse_gates(const std::string& spec, uint32_t& mask, std::string& error) {
    uint32_t m = 0;
    size_t pos = 0;
    bool any = false;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string tok = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = (comma == std::string::npos) ? spec.size() + 1 : comma + 1;
        while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
        while (!tok.empty() && tok.back() == ' ') tok.pop_back();
        if (tok.empty()) continue;
        any = true;
        bool remove = false;
        if (tok[0] == '-') { remove = true; tok.erase(tok.begin()); }
        uint32_t bits = 0;
        if (tok == "all") bits = implemented_gates();
        else if (tok == "none") bits = 0;
        else {
            for (int i = 0; i < kGateCount; ++i)
                if (tok == kGateNames[i]) bits = 1u << i;
            if (!bits) { error = "unknown gate '" + tok + "'"; return false; }
        }
        if (tok == "none" && !remove) m = 0;
        else if (remove) m &= ~bits;
        else m |= bits;
    }
    if (!any) { error = "empty gate list"; return false; }
    mask = m & implemented_gates();
    return true;
}

std::string gates_string(uint32_t mask) {
    if (mask == implemented_gates() && mask) return "all";
    std::string s;
    for (int i = 0; i < kGateCount; ++i)
        if (mask & (1u << i)) { if (!s.empty()) s += ","; s += kGateNames[i]; }
    return s.empty() ? "none" : s;
}

}  // namespace meow::precheck

// =============================================================================
#if defined(MEOW_PRECHECK_ENABLED)
// =============================================================================

#include "blockheader_generated.h"   // proof::MiningResponse (fbgen)
#include "../vendor/nlohmann/json.hpp"

// Upstream TensorCash v1.2.2 (renamed namespaces via compile definitions).
#include "credit_v4.h"
#include "candidate_band.h"
#include "pow_v3.h"
#include "pow_v4.h"
#include "antiparrot_gate.h"
#include "prompt_scaffold.h"
#include "flat_region.h"

#include <cfenv>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

// The canonical-row sunset entry point (sunset_abi.cpp): the very call the
// verifier's v4_sunset makes. Declared here because upstream ships no header.
extern "C" int32_t tc_sunset_window_batch(
    uint64_t n, const uint32_t* lengths, const float* logits, const uint32_t* ids,
    const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
    uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out,
    uint64_t* failed_step);

namespace meow::precheck {

namespace {

constexpr size_t kWindow = 256;     // pow_v4::PRICE_W / POW_WINDOW_SIZE
constexpr size_t kMaxRow = 70;      // 50 top-k + 20 probes, as sunset_abi.cpp bounds it
// v4_credit.V4_CHOSEN_PROB_ATOL: the declared chosen_probs must equal the
// canonical p_chosen within this float32 tolerance.
constexpr float kChosenProbAtol = 1e-5f;

static_assert(kWindow == static_cast<size_t>(pow_v4::PRICE_W), "window drifted from pow_v4");

std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 0xF]); }
    return s;
}

// Shared state of one evaluation: the window, the ONE canonical kernel pass
// every gate reads, and the sunset outputs (scaffold and flat share a call).
struct Ctx {
    const Window& w;
    credit_v4::WindowStructure ws;
    bool kernel_ok = false;
    bool sunset_done = false;
    int32_t sunset_status = -1;
    uint64_t sunset_out[11] = {};
    uint64_t sunset_failed = 0;
    std::vector<uint8_t> pad_norm;   // pow_v3.normalize_pad_mask
    explicit Ctx(const Window& win) : w(win) {}
};

// Result of one gate: pass, reject (with the verifier's reason), or abstain.
struct GateResult {
    Outcome outcome = Outcome::Pass;
    std::string reason;
    static GateResult pass() { return {}; }
    static GateResult reject(std::string r) { return {Outcome::Reject, std::move(r)}; }
    static GateResult unavailable(std::string r) { return {Outcome::Unavailable, std::move(r)}; }
};

// ---- strict: v4_credit.first_strict_violation over the kernel pass --------
// The kernel itself refusing a row is a proof failure at the verifier
// ("canonical credit kernel rejected step"), so it is reported here too.
GateResult gate_strict(Ctx& c) {
    const auto& ws = c.ws;
    for (size_t s = 0; s < kWindow; ++s) {
        // Race mode: lower = 0, upper = p_chosen.
        if (ws.position[s] < 0 || !(ws.upper[s] > ws.lower[s]))
            return GateResult::reject("v4: chosen token absent from the active row at step " + std::to_string(s));
        if (s >= c.w.chosen_probs.size())
            return GateResult::reject("v4: chosen_probs does not match the row's upper endpoint at step " + std::to_string(s));
        const float diff = std::fabs(c.w.chosen_probs[s] - ws.upper[s]);
        if (!(diff <= kChosenProbAtol))
            return GateResult::reject("v4: chosen_probs does not match the row's upper endpoint at step " + std::to_string(s));
    }
    return GateResult::pass();
}

// ---- near-pin: proof v4 structure rule (pow_v4::V4StructureRejects) -------
GateResult gate_near_pin(Ctx& c) {
    const auto& ws = c.ws;
    if (!ws.valid) return GateResult::unavailable("structure aggregate incomplete");
    pow_v4::StructureMetrics m;
    m.near_pin_count = ws.near_pin_count;
    m.bcap_units = ws.bcap_units;
    m.collision_u = ws.collision_u;
    m.collision_id = ws.collision_id;
    m.valid = ws.valid;
    std::string why;
    if (pow_v4::V4StructureRejects(m, why) || ws.near_pin_count >= pow_v4::PIN_REJECT_COUNT) {
        // The text the verifier logs (v4_structure.enforce).
        const uint64_t at = pow_v4::PIN_REJECT_COUNT;
        return GateResult::reject("proof v4: near-pinned window rejected: N = " +
            std::to_string(ws.near_pin_count) + " >= " + std::to_string(at) +
            " (the last legal near-pin count is " + std::to_string(at - 1) + ")");
    }
    return GateResult::pass();
}

// ---- candidate band: kernel v6 aggregate (candidate_band::Rejects) -------
GateResult gate_candidate_band(Ctx& c) {
    const auto& ws = c.ws;
    if (!ws.candidate_complete) return GateResult::unavailable("candidate-band aggregate incomplete");
    if (ws.candidate_reject) {
        candidate_band::Metrics m;
        m.active_rows = ws.candidate_active_rows;
        m.total = ws.candidate_total;
        m.largest_terms = ws.candidate_largest_terms;
        return GateResult::reject(candidate_band::RejectReason(m));
    }
    return GateResult::pass();
}

// ---- sunset: tc_sunset_window_batch, shared by scaffold and flat ---------
void run_sunset(Ctx& c) {
    if (c.sunset_done) return;
    c.sunset_done = true;
    const Window& w = c.w;
    // pow_v3.normalize_pad_mask: an empty mask is all-visible.
    if (w.pad.empty()) c.pad_norm.assign(w.prompt.size(), 0);
    else c.pad_norm = w.pad;
    if (c.pad_norm.size() != w.prompt.size()) { c.sunset_status = -2; return; }
    c.sunset_status = tc_sunset_window_batch(
        kWindow, w.row_len.data(), w.logits.data(), w.ids.data(), w.chosen.data(),
        w.prompt.size(), w.prompt.empty() ? nullptr : w.prompt.data(),
        c.pad_norm.size(), c.pad_norm.empty() ? nullptr : c.pad_norm.data(),
        c.ws.credit_units.data(), c.sunset_out, &c.sunset_failed);
}

// Status mapping of sunset_native._result / v4_sunset.metrics_from_rows:
// 2 (CreditV4InputError) is a proof failure, 1/3/4 abstain.
GateResult sunset_status_result(const Ctx& c) {
    switch (c.sunset_status) {
        case 0:  return GateResult::pass();
        case 2:  return GateResult::reject("post-sunset admission: invalid evidence at step " +
                                           std::to_string(c.sunset_failed));
        case -2: return GateResult::unavailable("pad_mask length differs from prompt length");
        default: return GateResult::unavailable("sunset kernel status " + std::to_string(c.sunset_status));
    }
}

GateResult gate_prompt_scaffold(Ctx& c) {
    run_sunset(c);
    GateResult st = sunset_status_result(c);
    if (st.outcome != Outcome::Pass) return st;
    if (c.sunset_out[4]) {
        prompt_scaffold::Metrics m;
        for (int i = 0; i < 4; ++i) m.covered[i] = static_cast<unsigned>(c.sunset_out[i]);
        m.reject = true;
        return GateResult::reject(prompt_scaffold::RejectReason(m));
    }
    return GateResult::pass();
}

GateResult gate_flat_region(Ctx& c) {
    run_sunset(c);
    GateResult st = sunset_status_result(c);
    if (st.outcome != Outcome::Pass) return st;
    // A rejecting scaffold skips the flat computation (sunset_abi.cpp); that
    // only happens here when the scaffold gate is masked off.
    if (c.sunset_out[4]) return GateResult::unavailable("flat region not computed (scaffold rejected)");
    if (c.sunset_out[10] != 1) return GateResult::unavailable("flat-region metrics incomplete");
    if (c.sunset_out[9]) {
        flat_region::Metrics m;
        m.capped_units = c.sunset_out[5];
        m.flat_capped_units = c.sunset_out[6];
        m.flat_credited_rows = static_cast<unsigned>(c.sunset_out[7]);
        m.eligible = c.sunset_out[8] != 0;
        m.reject = true;
        return GateResult::reject(flat_region::RejectReason(m));
    }
    return GateResult::pass();
}

// ---- credit tier: B_cred = sum of the kernel's race credit ----------------
// Below B_FLOOR (45 bits) the proof is invalid outright. Between B_FLOOR and
// B_FREE (70 bits) the v4 tier rule needs A below the admission target, which
// an ungrinded R practically never meets; withholding that band is the
// miner's existing policy (the float estimate in poi.cpp drops the same band
// whenever its upper-bound estimate is already below 70 bits).
GateResult gate_credit_tier(Ctx& c) {
    uint64_t units = 0;
    for (uint64_t u : c.ws.credit_units) units += u;
    const auto tier = pow_v3::tier_for_b_cred_units(units);
    char bits[32];
    std::snprintf(bits, sizeof(bits), "%.2f", static_cast<double>(units) / static_cast<double>(pow_v3::BCRED_R));
    if (tier == pow_v3::Tier::Invalid)
        return GateResult::reject(std::string("v3 proof below B_FLOOR (B_cred ") + bits + " bits)");
    if (tier == pow_v3::Tier::AdmissionRequired)
        return GateResult::reject(std::string("admission tier: B_cred ") + bits +
                                  " bits below B_FREE (70); withheld by miner policy");
    return GateResult::pass();
}

// ---- anti-parrot: antiparrot_gate::evaluate_window, race mode -------------
GateResult gate_anti_parrot(Ctx& c) {
    const Window& w = c.w;
    std::vector<std::vector<float>> rl(kWindow);
    std::vector<std::vector<uint32_t>> ri(kWindow);
    size_t off = 0;
    for (size_t s = 0; s < kWindow; ++s) {
        rl[s].assign(w.logits.begin() + off, w.logits.begin() + off + w.row_len[s]);
        ri[s].assign(w.ids.begin() + off, w.ids.begin() + off + w.row_len[s]);
        off += w.row_len[s];
    }
    // _verify_anti_parrot: pad-masked slots dropped when the mask matches.
    std::vector<uint32_t> unpadded;
    if (w.pad.size() == w.prompt.size()) {
        for (size_t i = 0; i < w.prompt.size(); ++i) if (!w.pad[i]) unpadded.push_back(w.prompt[i]);
    } else {
        unpadded = w.prompt;
    }
    const auto v = antiparrot_gate::evaluate_window(rl, ri, w.chosen, unpadded, /*race_sampler=*/true);
    if (!v.ok) {
        if (v.status == static_cast<int32_t>(credit_v4::Status::BadInput))
            return GateResult::reject("v3 anti-parrot: invalid step evidence at step " + std::to_string(v.failed_step));
        return GateResult::unavailable("anti-parrot kernel status " + std::to_string(v.status));
    }
    if (!v.reason.empty()) return GateResult::reject("v3 anti-parrot rejected (" + v.reason + ")");
    return GateResult::pass();
}

using GateFn = GateResult (*)(Ctx&);
struct Step { Gate gate; GateFn fn; };

// Verifier order (proof_verifier.py Quick kernel path).
constexpr Step kPipeline[] = {
    {Gate::Strict,         gate_strict},
    {Gate::NearPin,        gate_near_pin},
    {Gate::CandidateBand,  gate_candidate_band},
    {Gate::PromptScaffold, gate_prompt_scaffold},
    {Gate::FlatRegion,     gate_flat_region},
    {Gate::CreditTier,     gate_credit_tier},
    {Gate::AntiParrot,     gate_anti_parrot},
};

}  // namespace

bool available() { return true; }
const char* rules_version() { return "tensorcash v1.2.2 (48d1344)"; }

uint32_t implemented_gates() {
    uint32_t m = 0;
    for (const auto& s : kPipeline) m |= gate_bit(s.gate);
    return m;
}

bool ensure_float_environment(std::string* why) {
    if (credit_v4::FloatEnvironmentOk(why)) return true;
    std::fesetround(FE_TONEAREST);
#if defined(__x86_64__) || defined(_M_X64)
    // Clear FTZ (15), DAZ (6) and the rounding-control bits (13-14).
    _mm_setcsr(_mm_getcsr() & ~((1u << 15) | (1u << 6) | (3u << 13)));
#elif defined(__aarch64__)
    uint64_t fpcr = 0;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr &= ~((1ull << 24) | 1ull | (1ull << 1) | (3ull << 22));
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
    return credit_v4::FloatEnvironmentOk(why);
}

bool decode(const uint8_t* data, size_t n, Window& out, std::string& why) noexcept {
    try {
        out = Window{};
        if (!data || n == 0) { why = "empty proof"; return false; }
        flatbuffers::Verifier::Options opts;
        opts.max_tables = 4u << 20;
        flatbuffers::Verifier ver(data, n, opts);
        if (!ver.VerifyBuffer<proof::MiningResponse>(nullptr)) { why = "MiningResponse does not verify"; return false; }
        const auto* mr = flatbuffers::GetRoot<proof::MiningResponse>(data);
        const auto* pf = mr ? mr->pow_blob() : nullptr;
        if (!pf) { why = "no pow_blob"; return false; }
        if (pf->hash()) out.hash_hex = hex(pf->hash()->data(), pf->hash()->size());
        out.is_solution = pf->is_solution();
        if (pf->version() != 4) { why = "proof version " + std::to_string(pf->version()) + " is not 4"; return false; }
        // The v4 carrier: {"v4":1,...} in extra_flags.
        bool v4 = false;
        if (pf->extra_flags()) {
            const auto j = nlohmann::json::parse(pf->extra_flags()->str(), nullptr, false);
            if (j.is_object() && j.contains("v4") && j["v4"].is_number_integer() && j["v4"].get<int64_t>() == 1) v4 = true;
        }
        if (!v4) { why = "no proof v4 carrier in extra_flags"; return false; }
        const auto* ct = pf->chosen_tokens();
        const auto* cp = pf->chosen_probs();
        const auto* tl = pf->topk_logits();
        const auto* ti = pf->topk_indices();
        if (!ct || ct->size() != kWindow) { why = "chosen_tokens is not 256 entries"; return false; }
        if (!tl || !ti || tl->size() != kWindow || ti->size() != kWindow) { why = "top-k rows are not 256"; return false; }
        out.chosen.assign(ct->begin(), ct->end());
        if (cp) out.chosen_probs.assign(cp->begin(), cp->end());
        out.row_len.resize(kWindow);
        out.logits.reserve(kWindow * kMaxRow);
        out.ids.reserve(kWindow * kMaxRow);
        for (size_t s = 0; s < kWindow; ++s) {
            const auto* lr = tl->Get(static_cast<flatbuffers::uoffset_t>(s));
            const auto* ir = ti->Get(static_cast<flatbuffers::uoffset_t>(s));
            const auto* lv = lr ? lr->values() : nullptr;
            const auto* iv = ir ? ir->values() : nullptr;
            if (!lv || !iv || lv->size() != iv->size() || lv->size() == 0 || lv->size() > kMaxRow) {
                why = "row " + std::to_string(s) + " has an unexpected shape";
                return false;
            }
            out.row_len[s] = lv->size();
            out.logits.insert(out.logits.end(), lv->begin(), lv->end());
            out.ids.insert(out.ids.end(), iv->begin(), iv->end());
        }
        if (const auto* pt = pf->prompt_tokens()) out.prompt.assign(pt->begin(), pt->end());
        if (const auto* pm = pf->pad_mask())
            for (flatbuffers::uoffset_t i = 0; i < pm->size(); ++i) out.pad.push_back(pm->Get(i) ? 1 : 0);
        return true;
    } catch (const std::exception& e) {
        why = std::string("decode error: ") + e.what();
        return false;
    } catch (...) {
        why = "decode error";
        return false;
    }
}

Verdict evaluate_window(const Window& w, uint32_t gate_mask) noexcept {
    Verdict v;
    v.hash_hex = w.hash_hex;
    v.is_solution = w.is_solution;
    try {
        std::string why;
        if (!ensure_float_environment(&why)) { v.reason = "float environment: " + why; return v; }
        if (w.row_len.size() != kWindow || w.chosen.size() != kWindow) { v.reason = "not a 256-step window"; return v; }
        Ctx c(w);
        std::string err;
        const auto st = credit_v4::ComputeWindowStructure(
            w.row_len.data(), w.logits.data(), w.ids.data(), w.chosen.data(),
            /*u=*/nullptr, kWindow, c.ws, &err, /*race_sampler=*/true, /*digests=*/nullptr);
        if (st == credit_v4::Status::BadFloatEnvironment) { v.reason = "float environment refused by kernel"; return v; }
        if (st != credit_v4::Status::Ok) {
            // A row the kernel refuses is a proof failure at the verifier.
            if (gate_mask & gate_bit(Gate::Strict)) {
                v.outcome = Outcome::Reject;
                v.gate = Gate::Strict;
                v.reason = "proof v4: canonical credit kernel rejected step " +
                           std::to_string(c.ws.lower.size()) + ": " + err;
            } else {
                v.reason = "canonical kernel refused the window: " + err;
            }
            return v;
        }
        if (c.ws.credit_units.size() != kWindow || c.ws.position.size() != kWindow) {
            v.reason = "kernel returned an incomplete window";
            return v;
        }
        for (const auto& s : kPipeline) {
            if (!(gate_mask & gate_bit(s.gate))) continue;
            GateResult r = s.fn(c);
            if (r.outcome == Outcome::Reject) {
                v.outcome = Outcome::Reject;
                v.gate = s.gate;
                v.reason = std::move(r.reason);
                return v;
            }
            if (r.outcome == Outcome::Unavailable) {
                // Fail open: a gate that cannot decide never blocks a proof,
                // and the later gates still run.
                if (v.reason.empty()) v.reason = std::string(gate_name(s.gate)) + ": " + r.reason;
            }
        }
        v.outcome = v.reason.empty() ? Outcome::Pass : Outcome::Unavailable;
        return v;
    } catch (const std::exception& e) {
        v.outcome = Outcome::Unavailable;
        v.reason = std::string("internal error: ") + e.what();
        return v;
    } catch (...) {
        v.outcome = Outcome::Unavailable;
        v.reason = "internal error";
        return v;
    }
}

Verdict evaluate(const uint8_t* data, size_t n, uint32_t gate_mask) noexcept {
    const auto t0 = std::chrono::steady_clock::now();
    Window w;
    std::string why;
    Verdict v;
    if (!decode(data, n, w, why)) {
        v.outcome = Outcome::Unavailable;
        v.reason = why;
        v.hash_hex = w.hash_hex;
        v.is_solution = w.is_solution;
    } else {
        v = evaluate_window(w, gate_mask);
    }
    v.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return v;
}

}  // namespace meow::precheck

// =============================================================================
#else  // !MEOW_PRECHECK_ENABLED: compiled out (MSVC has no unsigned __int128)
// =============================================================================

namespace meow::precheck {

bool available() { return false; }
const char* rules_version() { return "unavailable in this build"; }
uint32_t implemented_gates() { return 0; }
bool ensure_float_environment(std::string*) { return true; }

bool decode(const uint8_t*, size_t, Window& out, std::string& why) noexcept {
    out = Window{};
    why = "precheck unavailable in this build";
    return false;
}

Verdict evaluate_window(const Window& w, uint32_t) noexcept {
    Verdict v;
    v.hash_hex = w.hash_hex;
    v.reason = "precheck unavailable in this build";
    return v;
}

Verdict evaluate(const uint8_t*, size_t, uint32_t) noexcept {
    Verdict v;
    v.reason = "precheck unavailable in this build";
    return v;
}

}  // namespace meow::precheck

#endif
