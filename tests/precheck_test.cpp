// CPU test for the pre-submit check (src/precheck.cpp), run through ctest
// (-DMEOW_BUILD_TESTS=ON). No GPU, no model, no network.
//
//   1. upstream golden vectors (vendor/tensorcash-v1.2.2/.../tests/vectors):
//      candidate-band mainnet windows, prompt-scaffold and flat-region cases,
//      with the verdicts upstream's own tests assert;
//   2. the five anti-parrot mainnet windows against expected_bcore.json:
//      every metric in CDF mode (as upstream's C++ gate test), and the
//      verdict through the pre-check (race mode, as live v4 proofs);
//   3. synthetic windows for near-pin, strict, credit tier, scaffold and
//      flat region, including the verifier's gate order;
//   4. the FlatBuffer path: a serialized MiningResponse is decoded and
//      checked, and malformed bytes fail open (Unavailable).
//
// usage: meow-precheck-test <vectors dir> <rendered anti-parrot .bin>

#include "precheck.h"

#include "blockheader_generated.h"
#include "nlohmann/json.hpp"

// Upstream (renamed namespaces via compile definitions, as in meow_precheck).
#include "antiparrot_gate.h"
#include "credit_v4.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

extern "C" int32_t tc_sunset_evidence_batch(
    uint64_t n, const uint32_t* lengths, const float* values, uint8_t cdf,
    const uint32_t* chosen, uint64_t np, const uint32_t* prompt,
    uint64_t nm, const uint8_t* pad, const uint64_t* credits, uint64_t* out);

namespace pc = meow::precheck;
using nlohmann::json;

static int g_fail = 0, g_ok = 0;
#define CHECK(cond, what)                                                       \
    do {                                                                        \
        if (cond) { ++g_ok; }                                                   \
        else { ++g_fail; std::cout << "  FAIL " << what << "  (" << __FILE__   \
                                   << ":" << __LINE__ << ")\n"; }               \
    } while (0)

static float f_bits(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint32_t as_u32(const json& v) {
    if (v.is_string()) return static_cast<uint32_t>(std::stoul(v.get<std::string>(), nullptr, 0));
    return v.get<uint32_t>();
}
static json load_json(const std::string& p) {
    std::ifstream in(p);
    if (!in) { std::cout << "cannot open " << p << "\n"; std::exit(2); }
    return json::parse(in);
}

// Fill chosen_probs with the canonical p_chosen, as an honest miner writes it.
static void fill_chosen_probs(pc::Window& w) {
    credit_v4::WindowStructure ws;
    std::string err;
    const auto st = credit_v4::ComputeWindowStructure(w.row_len.data(), w.logits.data(), w.ids.data(),
        w.chosen.data(), nullptr, w.chosen.size(), ws, &err, true, nullptr);
    w.chosen_probs = (st == credit_v4::Status::Ok) ? ws.p_chosen : std::vector<float>(w.chosen.size(), 0.f);
}

static pc::Window window_from_rows(const std::vector<std::vector<float>>& lg,
                                   const std::vector<std::vector<uint32_t>>& id,
                                   const std::vector<uint32_t>& chosen) {
    pc::Window w;
    for (size_t s = 0; s < lg.size(); ++s) {
        w.row_len.push_back(static_cast<uint32_t>(lg[s].size()));
        w.logits.insert(w.logits.end(), lg[s].begin(), lg[s].end());
        w.ids.insert(w.ids.end(), id[s].begin(), id[s].end());
    }
    w.chosen = chosen;
    w.hash_hex = std::string(64, 'a');
    fill_chosen_probs(w);
    return w;
}

static pc::Window band_window(const json& c) {
    std::vector<std::vector<float>> lg;
    std::vector<std::vector<uint32_t>> id;
    for (const auto& r : c["raw_rows"]) {
        std::vector<float> l; std::vector<uint32_t> i;
        for (const auto& b : r["logit_bits"]) l.push_back(f_bits(as_u32(b)));
        for (const auto& x : r["ids"]) i.push_back(x.get<uint32_t>());
        lg.push_back(l); id.push_back(i);
    }
    std::vector<uint32_t> ch;
    for (const auto& x : c["chosen"]) ch.push_back(x.get<uint32_t>());
    return window_from_rows(lg, id, ch);
}

// tests/test_sunset_native.py raw_window(kind).
static pc::Window sunset_window(const std::string& kind) {
    std::vector<std::vector<float>> lg;
    std::vector<std::vector<uint32_t>> id;
    std::vector<uint32_t> ch;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t first;
        if (kind == "scaffold") {
            const bool slot = i % 5 == 4;
            first = slot ? 20000 + i * 4 : 100 + i % 5;
            id.push_back({first, 50000 + i});
            lg.push_back(slot ? std::vector<float>{0.f, 0.f}
                              : std::vector<float>{(float)std::log(.95), (float)std::log(.05)});
        } else if (kind == "flat") {
            const bool slot = i % 4 == 3;
            first = 20000 + i * 4;
            std::vector<uint32_t> ids;
            for (uint32_t j = 0; j < (slot ? 4u : 2u); ++j) ids.push_back(first + j);
            id.push_back(ids);
            lg.push_back(slot ? std::vector<float>(4, 0.f)
                              : std::vector<float>{(float)std::log(.99), (float)std::log(.01)});
        } else {
            first = 20000 + i * 4;
            id.push_back({first, first + 1});
            lg.push_back({0.f, 0.f});
        }
        ch.push_back(first);
    }
    pc::Window w = window_from_rows(lg, id, ch);
    w.prompt = {100, 101, 102, 103};
    return w;
}

// test_v4_anti_parrot_gate_cpp.cpp pinned_window: `pinned` near-pinned rows.
static pc::Window pinned_window(uint32_t pinned) {
    std::vector<std::vector<float>> lg;
    std::vector<std::vector<uint32_t>> id;
    std::vector<uint32_t> ch;
    for (uint32_t s = 0; s < 256; ++s) {
        lg.push_back(s < pinned ? std::vector<float>{10.f, 0.f} : std::vector<float>{0.f, 0.f});
        id.push_back({0, 1});
        ch.push_back(0);
    }
    return window_from_rows(lg, id, ch);
}

static std::vector<uint8_t> serialize(const pc::Window& w, uint8_t version, const std::string& flags,
                                      bool is_solution) {
    flatbuffers::FlatBufferBuilder fbb(1 << 20);
    std::vector<flatbuffers::Offset<proof::FloatArray>> lrows;
    std::vector<flatbuffers::Offset<proof::UIntArray>> irows;
    size_t off = 0;
    for (uint32_t len : w.row_len) {
        std::vector<float> l(w.logits.begin() + off, w.logits.begin() + off + len);
        std::vector<uint32_t> i(w.ids.begin() + off, w.ids.begin() + off + len);
        lrows.push_back(proof::CreateFloatArray(fbb, fbb.CreateVector(l)));
        irows.push_back(proof::CreateUIntArray(fbb, fbb.CreateVector(i)));
        off += len;
    }
    std::vector<uint8_t> hash(32);
    for (int i = 0; i < 32; ++i) hash[i] = static_cast<uint8_t>(i * 7 + 1);
    auto v_hash = fbb.CreateVector(hash);
    auto v_flags = fbb.CreateString(flags);
    auto v_ct = fbb.CreateVector(w.chosen);
    auto v_cp = fbb.CreateVector(w.chosen_probs);
    auto v_pt = fbb.CreateVector(w.prompt);
    auto v_pm = fbb.CreateVector(w.pad);
    auto v_tl = fbb.CreateVector(lrows);
    auto v_ti = fbb.CreateVector(irows);
    proof::ProofBuilder pb(fbb);
    pb.add_version(version);
    pb.add_hash(v_hash);
    pb.add_is_solution(is_solution);
    pb.add_extra_flags(v_flags);
    pb.add_temperature(1.0f);
    pb.add_top_p(1.0f);
    pb.add_top_k(50);
    pb.add_repetition_penalty(1.0f);
    pb.add_chosen_tokens(v_ct);
    pb.add_chosen_probs(v_cp);
    pb.add_prompt_tokens(v_pt);
    pb.add_pad_mask(v_pm);
    pb.add_topk_logits(v_tl);
    pb.add_topk_indices(v_ti);
    auto p = pb.Finish();
    proof::MiningResponseBuilder mb(fbb);
    mb.add_req_id(7);
    mb.add_pow_blob(p);
    fbb.Finish(mb.Finish());
    return std::vector<uint8_t>(fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize());
}

static uint32_t mask(std::initializer_list<pc::Gate> gs) {
    uint32_t m = 0;
    for (auto g : gs) m |= pc::gate_bit(g);
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "usage: meow-precheck-test <vectors dir> [antiparrot.bin]\n"; return 2; }
    const std::string vec = argv[1];
    std::cout << "precheck rules: " << pc::rules_version() << "\n";

    // ---- environment and option parsing ---------------------------------
    std::string why;
    CHECK(pc::available(), "pre-check compiled in");
    CHECK(pc::ensure_float_environment(&why), "canonical float environment: " + why);
    CHECK(pc::implemented_gates() == (1u << pc::kGateCount) - 1, "all seven gates implemented");
    {
        uint32_t m = 0; std::string e;
        CHECK(pc::parse_gates("all,-credit-tier", m, e) && m == (pc::implemented_gates() & ~pc::gate_bit(pc::Gate::CreditTier)),
              "parse_gates all,-credit-tier");
        CHECK(pc::parse_gates("near-pin,anti-parrot", m, e) && m == mask({pc::Gate::NearPin, pc::Gate::AntiParrot}),
              "parse_gates list");
        CHECK(!pc::parse_gates("bogus", m, e), "parse_gates rejects unknown names");
        pc::Mode md;
        CHECK(pc::parse_mode("shadow", md) && md == pc::Mode::Shadow && !pc::parse_mode("x", md), "parse_mode");
    }
    std::cout << "ok: environment and options\n";

    // ---- candidate band: upstream mainnet windows -----------------------
    {
        const json doc = load_json(vec + "/candidate_band_vectors.json");
        int n = 0;
        for (const auto& c : doc["windows"]) {
            pc::Window w = band_window(c);
            const auto& e = c["expected"];
            const auto v = pc::evaluate_window(w, mask({pc::Gate::CandidateBand}));
            const bool reject = e["reject"].get<bool>();
            CHECK((v.outcome == pc::Outcome::Reject) == reject, "band verdict " + c["name"].get<std::string>());
            if (reject) {
                const std::string want = "N=" + std::to_string(e["active_rows"].get<int>()) +
                    " T=" + e["total"].get<std::string>() + " L31=" + e["largest_terms"].get<std::string>();
                CHECK(v.gate == pc::Gate::CandidateBand && v.reason.find(want) != std::string::npos &&
                      v.reason.find("bad-candidate-band") != std::string::npos,
                      "band reason " + v.reason);
            }
            // Through the FlatBuffer decoder too.
            const auto bytes = serialize(w, 4, R"({"v4":1,"v3":{"stepbind":1}})", false);
            const auto v2 = pc::evaluate(bytes.data(), bytes.size(), mask({pc::Gate::CandidateBand}));
            CHECK(v2.outcome == v.outcome && v2.reason == v.reason, "band verdict via FlatBuffer");
            ++n;
        }
        CHECK(n == 2, "two candidate-band mainnet windows");
        std::cout << "ok: candidate-band mainnet windows (" << n << ")\n";
    }

    // ---- prompt scaffold: upstream vectors through tc_sunset ------------
    {
        const json doc = load_json(vec + "/prompt_scaffold_vectors.json");
        const unsigned min_cov[4] = {77, 48, 52, 48};
        std::vector<uint32_t> lens(256, 1);
        std::vector<float> vals(256, 1.0f);
        int n = 0, rejects = 0;
        for (const auto& c : doc["cases"]) {
            std::vector<uint32_t> ch = c["chosen"].get<std::vector<uint32_t>>();
            std::vector<uint32_t> pr = c["prompt"].get<std::vector<uint32_t>>();
            std::vector<uint8_t> pad;
            for (const auto& b : c["pad"]) pad.push_back(b.is_boolean() ? b.get<bool>() : b.get<int>());
            std::vector<uint64_t> cr = c["credits"].get<std::vector<uint64_t>>();
            uint64_t out[11] = {};
            const int st = tc_sunset_evidence_batch(256, lens.data(), vals.data(), 0, ch.data(),
                pr.size(), pr.data(), pad.size(), pad.empty() ? nullptr : pad.data(), cr.data(), out);
            bool want_reject = false, cov_ok = st == 0;
            for (int i = 0; i < 4; ++i) {
                const unsigned want = c["covered"][i].get<unsigned>();
                cov_ok = cov_ok && out[i] == want;
                want_reject = want_reject || want >= min_cov[i];
            }
            CHECK(cov_ok && (out[4] != 0) == want_reject, "scaffold case " + c["id"].get<std::string>());
            rejects += want_reject;
            ++n;
        }
        CHECK(n == 49, "49 prompt-scaffold cases");
        std::cout << "ok: prompt-scaffold vectors (" << n << ", " << rejects << " rejecting)\n";
    }

    // ---- flat region: upstream vectors through tc_sunset ----------------
    {
        const json doc = load_json(vec + "/flat_region_vectors.json");
        int n = 0;
        for (const auto& c : doc["cases"]) {
            std::vector<uint32_t> lens(256, 1);
            std::vector<float> vals;
            for (const auto& b : c["bits"]) vals.push_back(f_bits(as_u32(b)));
            lens[0] = static_cast<uint32_t>(vals.size());
            vals.insert(vals.end(), 255, 1.0f);
            std::vector<uint32_t> ch(256, 1);
            std::vector<uint64_t> cr(256, 0);
            cr[0] = 1024;
            uint64_t out[11] = {};
            const int st = tc_sunset_evidence_batch(256, lens.data(), vals.data(), c["cdf"].get<bool>() ? 1 : 0,
                ch.data(), 0, nullptr, 0, nullptr, cr.data(), out);
            if (!c["valid"].get<bool>()) CHECK(st == 2, "flat invalid case " + c["id"].get<std::string>());
            else CHECK(st == 0 && (out[9] != 0) == c["qualifies"].get<bool>(), "flat case " + c["id"].get<std::string>());
            ++n;
        }
        CHECK(n == 125, "125 flat-region cases");
        std::cout << "ok: flat-region vectors (" << n << ")\n";
    }

    // ---- sunset gates on canonical raw windows (test_sunset_native) -----
    {
        const uint32_t m = mask({pc::Gate::PromptScaffold, pc::Gate::FlatRegion});
        auto v = pc::evaluate_window(sunset_window("scaffold"), m);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::PromptScaffold &&
              v.reason.rfind("bad-prompt-scaffold:", 0) == 0, "raw scaffold window: " + v.reason);
        v = pc::evaluate_window(sunset_window("flat"), m);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::FlatRegion &&
              v.reason.rfind("bad-flat-region:", 0) == 0, "raw flat window: " + v.reason);
        v = pc::evaluate_window(sunset_window("pass"), m);
        CHECK(v.outcome == pc::Outcome::Pass, "raw passing window: " + v.reason);
        std::cout << "ok: sunset raw windows\n";
    }

    // ---- anti-parrot: mainnet windows vs expected_bcore.json ------------
    if (argc >= 3) {
        std::ifstream in(argv[2], std::ios::binary);
        CHECK(static_cast<bool>(in), std::string("rendered anti-parrot vectors ") + argv[2]);
        char magic[4] = {};
        in.read(magic, 4);
        CHECK(std::memcmp(magic, "MAP1", 4) == 0, "anti-parrot vector magic");
        uint32_t count = 0;
        in.read(reinterpret_cast<char*>(&count), 4);
        int n = 0;
        for (uint32_t k = 0; k < count && in; ++k) {
            uint32_t hdr[6];
            in.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
            const uint32_t height = hdr[0], steps = hdr[1], row = hdr[2], np = hdr[3], npad = hdr[4], rl = hdr[5];
            std::string reason(rl, '\0');
            in.read(reason.data(), rl);
            uint64_t m[7];
            in.read(reinterpret_cast<char*>(m), sizeof(m));
            std::vector<float> lg(size_t(steps) * row);
            std::vector<uint32_t> id(size_t(steps) * row), ch(steps), pr(np);
            std::vector<uint8_t> pad(npad);
            in.read(reinterpret_cast<char*>(lg.data()), lg.size() * 4);
            in.read(reinterpret_cast<char*>(id.data()), id.size() * 4);
            in.read(reinterpret_cast<char*>(ch.data()), ch.size() * 4);
            in.read(reinterpret_cast<char*>(pr.data()), pr.size() * 4);
            in.read(reinterpret_cast<char*>(pad.data()), pad.size());
            std::vector<std::vector<float>> rlg(steps);
            std::vector<std::vector<uint32_t>> rid(steps);
            for (uint32_t s = 0; s < steps; ++s) {
                rlg[s].assign(lg.begin() + size_t(s) * row, lg.begin() + size_t(s + 1) * row);
                rid[s].assign(id.begin() + size_t(s) * row, id.begin() + size_t(s + 1) * row);
            }
            std::vector<uint32_t> unpadded;
            if (pad.size() == pr.size()) { for (size_t i = 0; i < pr.size(); ++i) if (!pad[i]) unpadded.push_back(pr[i]); }
            else unpadded = pr;
            const std::string h = "h" + std::to_string(height);
            // Bcore equality, CDF mode (what expected_bcore.json captured).
            const auto g = antiparrot_gate::evaluate_window(rlg, rid, ch, unpadded, /*race*/false);
            const uint64_t got[7] = {g.braw, g.bcap, g.hard_run, g.maxrs_q32, g.reuse_cache_q32,
                                     g.n_states_eff_u, g.n_states_eff_id};
            CHECK(g.ok && g.reason == reason && std::memcmp(got, m, sizeof(m)) == 0,
                  h + " anti-parrot metrics equal bcore");
            // Through the pre-check (race mode, as a live v4 proof).
            pc::Window w = window_from_rows(rlg, rid, ch);
            w.prompt = pr; w.pad = pad;
            const auto v = pc::evaluate_window(w, mask({pc::Gate::AntiParrot}));
            if (reason.empty()) CHECK(v.outcome == pc::Outcome::Pass, h + " pre-check accepts: " + v.reason);
            else CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::AntiParrot &&
                       v.reason == "v3 anti-parrot rejected (" + reason + ")", h + " pre-check: " + v.reason);
            std::cout << "  " << h << ": bcore '" << (reason.empty() ? "accept" : reason)
                      << "', pre-check " << (v.outcome == pc::Outcome::Pass ? "pass" : v.reason) << "\n";
            ++n;
        }
        CHECK(n == 5, "five anti-parrot mainnet windows");
        std::cout << "ok: anti-parrot mainnet windows (" << n << ")\n";
    } else {
        std::cout << "SKIP: anti-parrot mainnet windows (no rendered .bin given)\n";
        ++g_fail;
    }

    // ---- near-pin, gate order, strict, credit tier ----------------------
    {
        const uint32_t all = pc::implemented_gates();
        auto v = pc::evaluate_window(pinned_window(175), mask({pc::Gate::NearPin}));
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::NearPin &&
              v.reason.find("N = 175 >= 175") != std::string::npos, "175 near-pinned rows reject: " + v.reason);
        v = pc::evaluate_window(pinned_window(174), mask({pc::Gate::NearPin}));
        CHECK(v.outcome == pc::Outcome::Pass, "174 near-pinned rows pass the near-pin gate");
        // All gates: strict passes (honest chosen_probs), near-pin is first.
        v = pc::evaluate_window(pinned_window(175), all);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::NearPin, "verifier order: near-pin first");
        // 174 pinned rows: near-pin passes, the next failing gate reports.
        v = pc::evaluate_window(pinned_window(174), all);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate != pc::Gate::NearPin && v.gate != pc::Gate::Strict,
              std::string("174 rows fail a later gate: ") + pc::gate_name(v.gate));
        // Strict: a declared chosen_probs off by more than 1e-5.
        pc::Window w = pinned_window(10);
        w.chosen_probs[37] += 1e-3f;
        v = pc::evaluate_window(w, all);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::Strict &&
              v.reason.find("at step 37") != std::string::npos, "strict chosen_probs mismatch: " + v.reason);
        // Strict: chosen token absent from its row.
        w = pinned_window(10);
        w.chosen[5] = 99999;
        fill_chosen_probs(w);
        v = pc::evaluate_window(w, all);
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::Strict &&
              v.reason.find("absent") != std::string::npos, "strict absent token: " + v.reason);
        // Credit tier: 256 near-certain rows earn almost no credit.
        v = pc::evaluate_window(pinned_window(256), mask({pc::Gate::CreditTier}));
        CHECK(v.outcome == pc::Outcome::Reject && v.gate == pc::Gate::CreditTier &&
              v.reason.find("B_FLOOR") != std::string::npos, "credit tier below floor: " + v.reason);
        v = pc::evaluate_window(pinned_window(0), mask({pc::Gate::CreditTier}));
        CHECK(v.outcome == pc::Outcome::Pass, "256 coin-flip rows (256 bits) clear the credit tier");
        std::cout << "ok: near-pin, order, strict, credit tier\n";
    }

    // ---- FlatBuffer path, fail-open -------------------------------------
    {
        const json doc = load_json(vec + "/candidate_band_vectors.json");
        pc::Window w = band_window(doc["windows"][1]);   // the accepted mainnet window
        w.prompt = {1, 2, 3};
        w.pad = {0, 0, 0};
        auto bytes = serialize(w, 4, R"({"v4":1,"v3":{"stepbind":1,"admission_nonce":")" +
                               std::string(64, 'a') + R"("}})", true);
        auto v = pc::evaluate(bytes.data(), bytes.size(), pc::implemented_gates());
        CHECK(v.outcome != pc::Outcome::Unavailable, "decoded proof is evaluated: " + v.reason);
        CHECK(v.hash_hex.size() == 64 && v.hash_hex.rfind("0108", 0) == 0, "hash_hex is Proof.hash in wire order");
        CHECK(v.is_solution, "is_solution carried");
        std::cout << "  accepted mainnet window, all gates: "
                  << (v.outcome == pc::Outcome::Pass ? "pass" : std::string(pc::gate_name(v.gate)) + ": " + v.reason)
                  << " (" << v.ms << " ms)\n";
        const auto v3 = serialize(w, 3, R"({"v4":1})", false);
        v = pc::evaluate(v3.data(), v3.size(), pc::implemented_gates());
        CHECK(v.outcome == pc::Outcome::Unavailable, "proof version 3 fails open");
        const auto nov4 = serialize(w, 4, R"({"v3":{"stepbind":1}})", false);
        v = pc::evaluate(nov4.data(), nov4.size(), pc::implemented_gates());
        CHECK(v.outcome == pc::Outcome::Unavailable, "missing v4 carrier fails open");
        bytes.resize(bytes.size() / 2);
        v = pc::evaluate(bytes.data(), bytes.size(), pc::implemented_gates());
        CHECK(v.outcome == pc::Outcome::Unavailable, "truncated bytes fail open");
        v = pc::evaluate(nullptr, 0, pc::implemented_gates());
        CHECK(v.outcome == pc::Outcome::Unavailable, "empty input fails open");
        std::cout << "ok: FlatBuffer path and fail-open\n";
    }

    std::cout << (g_fail ? "FAILED" : "PASSED") << ": " << g_ok << " checks ok, " << g_fail << " failed\n";
    return g_fail ? 1 : 0;
}
