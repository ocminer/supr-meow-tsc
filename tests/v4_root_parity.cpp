// Parity test: the miner's meow_pow_v4 (+ the miner's OWN pow_v3.cpp/argon2)
// must reproduce the golden v4_vectors.json "root" group byte-for-byte. This
// pins the vendored root-derivation subset to the authoritative contract IN
// THE MINER'S BUILD CONTEXT — a different pow_v3::step_digest or argon2 profile
// would fail here, before it could reach a live block.
#include "meow_pow_v4.h"
#include <openssl/sha.h>
#include "json.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using nlohmann::json;
static int g_pass = 0, g_fail = 0;
static void ok(bool c, const std::string& g, const std::string& n, const std::string& d = "") {
    if (c) ++g_pass; else { ++g_fail; std::cerr << "  FAIL [" << g << "] " << n << (d.empty() ? "" : "  " + d) << "\n"; }
}
static std::vector<uint8_t> from_hex(const std::string& h) {
    std::vector<uint8_t> o; o.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) o.push_back((uint8_t)std::stoi(h.substr(i, 2), nullptr, 16));
    return o;
}
static std::array<uint8_t, 32> to32(const std::vector<uint8_t>& v) {
    std::array<uint8_t, 32> a{}; for (size_t i = 0; i < 32 && i < v.size(); ++i) a[i] = v[i]; return a;
}
static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef"; std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 0xf]); } return s;
}
template <size_t N> static std::string hex(const std::array<uint8_t, N>& a) { return hex(a.data(), N); }
static std::string sha_hex(const std::vector<uint8_t>& v) {
    std::array<uint8_t, 32> d{}; SHA256(v.data(), v.size(), d.data()); return hex(d);
}

int main(int argc, char** argv) {
    std::ifstream f(argc > 1 ? argv[1] : "v4_vectors.json");
    json d; f >> d;
    if (!d.contains("root") || d["root"].empty()) return 2;
    for (const auto& v : d["root"]) {
        auto msg_w0 = from_hex(v["msg_w0_hex"]);
        auto C = to32(from_hex(v["prompt_commitment_hex"]));
        auto R = to32(from_hex(v["admission_nonce_hex"]));
        std::string model = v["model_identifier"];
        auto am = pow_v4::admit_message_v4(msg_w0, C, model, R);
        ok(am.size() == v["admit_message_len"].get<size_t>(), "admit_msg_len", v["name"]);
        ok(sha_hex(am) == v["admit_message_sha256"].get<std::string>(), "admit_msg_sha", v["name"]);
        auto root = pow_v4::derive_v4_root(msg_w0, C, model, R);
        ok(hex(root.A) == v["admit_root_hex"].get<std::string>(), "admit_root_A", v["name"],
           "got=" + hex(root.A));
        ok(hex(root.E4) == v["step_root_hex"].get<std::string>(), "step_root_E4", v["name"],
           "got=" + hex(root.E4));
    }
    std::cerr << "\n==== meow_pow_v4 root parity: " << g_pass << " passed, " << g_fail << " failed ====\n";
    return g_fail == 0 && g_pass == 24 ? 0 : 1;
}
