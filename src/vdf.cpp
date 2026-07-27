#include "vdf.h"

#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <mutex>

// chiavdf. ORDER MATTERS and is not cosmetic: verifier.h pulls in the prelude
// that defines `integer` and `form`. Putting create_discriminant.h first fails
// with a wall of "'form' does not name a type". Same order bcore uses.
#include "verifier.h"
#include "prover_slow.h"
#include "create_discriminant.h"

namespace meow {
namespace {

// chiavdf routes GMP allocation through these. Installing the plain libc
// functions keeps it away from any custom allocator the rest of the binary
// might use — llama.cpp and CUDA both allocate heavily on other threads.
void* MpAlloc(size_t n)                          { return std::malloc(n); }
void* MpRealloc(void* p, size_t, size_t n)       { return std::realloc(p, n); }
void  MpFree(void* p, size_t)                    { std::free(p); }

void ensure_runtime() {
    static std::once_flag once;
    std::call_once(once, [] {
        mp_set_memory_functions(MpAlloc, MpRealloc, MpFree);
        allow_integer_constructor = true;
    });
}

// chiavdf's fixed-point paths require truncation rounding. The FPU mode is
// per-thread and llama.cpp/CUDA do not set it, so it is set around each call
// and restored — leaving it changed globally would perturb inference.
struct RoundingGuard {
    int prev;
    RoundingGuard()  : prev(std::fegetround()) { std::fesetround(FE_TOWARDZERO); }
    ~RoundingGuard() { std::fesetround(prev); }
    RoundingGuard(const RoundingGuard&) = delete;
    RoundingGuard& operator=(const RoundingGuard&) = delete;
};

}  // namespace

std::vector<uint8_t> Vdf::prove(const std::vector<uint8_t>& prev_block_hash,
                                uint64_t iterations, uint32_t discriminant_bits) {
    if (prev_block_hash.size() != 32 || iterations == 0) return {};
    try {
        ensure_runtime();
        RoundingGuard rounding;

        std::vector<uint8_t> seed(prev_block_hash.begin(), prev_block_hash.end());
        integer D = CreateDiscriminant(seed, discriminant_bits);
        form x = form::from_abd(integer(2), integer(1), D);
        x.reduce();
        // Returns y||proof in the layout the verifier expects.
        return ProveSlow(D, x, iterations, "");
    } catch (...) {
        return {};
    }
}

bool Vdf::verify(const std::vector<uint8_t>& prev_block_hash,
                 const std::vector<uint8_t>& proof, uint64_t iterations,
                 uint32_t discriminant_bits, uint32_t recursion) {
    if (prev_block_hash.size() != 32 || proof.empty() || iterations == 0) return false;
    try {
        ensure_runtime();
        RoundingGuard rounding;

        std::vector<uint8_t> seed(prev_block_hash.begin(), prev_block_hash.end());
        integer D = CreateDiscriminant(seed, discriminant_bits);
        form x = form::from_abd(integer(2), integer(1), D);
        x.reduce();
        const int real_bits = D.num_bits();
        std::vector<uint8_t> x_bytes = SerializeForm(x, real_bits);

        return CheckProofOfTimeNWesolowski(
            D, x_bytes.data(), proof.data(), static_cast<int32_t>(proof.size()),
            iterations, real_bits, static_cast<int32_t>(recursion));
    } catch (...) {
        return false;
    }
}

bool Vdf::self_test(std::string& error, uint64_t iterations) {
    std::vector<uint8_t> prev(32);
    for (size_t i = 0; i < prev.size(); ++i) prev[i] = static_cast<uint8_t>(i * 7 + 1);

    const auto proof = prove(prev, iterations);
    if (proof.empty()) { error = "VDF prove() produced nothing"; return false; }
    if (!verify(prev, proof, iterations)) {
        error = "VDF proof failed its own verification — the chiavdf build is wrong "
                "(check GMP, and that FPU rounding is truncation)";
        return false;
    }
    // A proof must not verify against a different challenge; if it does, the
    // challenge is not actually binding and every proof we emit is garbage.
    std::vector<uint8_t> other = prev;
    other[0] ^= 0xFF;
    if (verify(other, proof, iterations)) {
        error = "VDF proof verified against the WRONG challenge — binding is broken";
        return false;
    }
    return true;
}

std::string Vdf::to_hex(const std::vector<uint8_t>& v) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(v.size() * 2);
    for (uint8_t b : v) { s.push_back(d[b >> 4]); s.push_back(d[b & 0xF]); }
    return s;
}

}  // namespace meow
