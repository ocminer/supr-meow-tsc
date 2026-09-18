#include "gumbel_sampler.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
using nlohmann::json;
std::array<uint8_t, 32> bytes(const std::string& text) {
    if (text.size() != 64) throw std::runtime_error("bad vector length");
    std::array<uint8_t,32> a{};
    for (int i=0;i<32;++i) a[i] = uint8_t(std::stoi(text.substr(i*2,2), nullptr,16));
    return a;
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream file(argv[1]); json d; file >> d;
    gumbel::dlog_selftest();
    int checked=0;
    for (const auto& v : d.at("seed")) {
        const auto digest = bytes(v["digest"]);
        const auto got = gumbel::seed_from_digest(digest.data(), v["draw"].get<uint64_t>());
        if (got != bytes(v["seed"])) return 1;
        ++checked;
    }
    for (const auto& v : d.at("uniform")) {
        const auto seed = bytes(v["seed"]);
        const auto message = gumbel::expansion_message(seed, v["token_id"].get<uint32_t>());
        if (gumbel::bytes_hex(message.data(), message.size()) != v["message"].get<std::string>()) return 1;
        if (gumbel::token_uniform(seed,v["token_id"].get<uint32_t>()) != v["u"].get<double>()) return 1;
        ++checked;
    }
    for (const auto& v : d.at("dlog")) {
        const auto input_bits = v["x_bits"].get<int64_t>();
        double x;
        std::memcpy(&x, &input_bits, sizeof(x));
        const double y = gumbel::dlog(x);
        int64_t bits;
        std::memcpy(&bits, &y, sizeof(bits));
        if (bits != v["dlog_bits"].get<int64_t>()) {
            std::cerr << "Deterministic log differs from reference bits\n";
            return 1;
        }
        ++checked;
    }
    if (checked != 479) return 2;
    std::cout << "Gumbel deterministic log and " << checked << " seed/uniform/log vectors passed\n";
}
