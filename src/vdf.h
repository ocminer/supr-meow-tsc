// =============================================================================
// vdf.h — Wesolowski verifiable delay function over the parent block hash.
//
// Every TSC proof carries one. It is sequential BY DESIGN — that is the whole
// point of a VDF — so it belongs on the CPU and cannot be parallelised or moved
// to the GPU. It runs alongside inference rather than competing with it.
//
// It also cannot be stubbed: a proof without a valid VDF is rejected by the
// network, so this is a hard dependency of the mining path.
//
// The class-group arithmetic underneath is chiavdf's; this wrapper exists to
// keep its global state (GMP allocators, FPU rounding mode) contained and to
// give the miner a small, obvious interface.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace meow {

class Vdf {
public:
    // Discriminant size in bits. The chain uses 1024; changing it produces
    // proofs the network will not accept.
    static constexpr uint32_t kDiscriminantBits = 1024;

    // Prove `iterations` sequential squarings from the challenge derived from
    // `prev_block_hash` (32 bytes). Returns y||proof as the verifier expects,
    // or empty on failure. Blocking and slow — that is the design.
    static std::vector<uint8_t> prove(const std::vector<uint8_t>& prev_block_hash,
                                      uint64_t iterations,
                                      uint32_t discriminant_bits = kDiscriminantBits);

    // Check a proof. Used to self-test the integration, and worth calling once
    // at startup: a miner that emits VDFs the network rejects should find out
    // immediately, not after hours of wasted inference.
    static bool verify(const std::vector<uint8_t>& prev_block_hash,
                       const std::vector<uint8_t>& proof,
                       uint64_t iterations,
                       uint32_t discriminant_bits = kDiscriminantBits,
                       uint32_t recursion = 0);

    // prove() then verify(); returns false with `error` set if they disagree.
    static bool self_test(std::string& error, uint64_t iterations = 1000);

    static std::string to_hex(const std::vector<uint8_t>& v);
};

}  // namespace meow
