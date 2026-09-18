#include "vdf.h"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    std::string error;
    if (!meow::Vdf::self_test(error, 1000)) { std::cerr << error << '\n'; return 1; }
    std::vector<uint8_t> parent(32);
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = uint8_t(i * 7 + 1);
    const auto proof = meow::Vdf::prove(parent, 1000);
    const auto hex = meow::Vdf::to_hex(proof);
    if (argc == 1) { std::cout << hex << '\n'; return 0; }
    std::ifstream file(argv[1]);
    std::string expected;
    file >> expected;
    if (expected.empty() || hex != expected) { std::cerr << "VDF differs from Linux reference\n"; return 1; }
    auto corrupt = proof;
    corrupt[corrupt.size()/2] ^= 1;
    if (meow::Vdf::verify(parent, corrupt, 1000)) { std::cerr << "Corrupt VDF accepted\n"; return 1; }
    std::cout << "VDF reference, round trip, wrong challenge and corrupt proof checks passed\n";
}
