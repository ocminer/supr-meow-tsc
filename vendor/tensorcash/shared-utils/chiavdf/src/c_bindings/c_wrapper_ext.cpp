// Minimal C API wrapper exposing verify_from_hash-style entrypoint for VDF verification.
// NOTE: This implementation depends on chiavdf's C++ verifier and GMP. It is a
// stepping stone toward a future portable (non-GMP/ASM) implementation.

#include "c_wrapper.h"

#include <vector>
#include <cstdint>
#include <cstring>

#include "../verifier.h"
#include "../proof_common.h"
#include "../create_discriminant.h"

extern "C" {

// Verify from challenge hash (32 bytes) with discriminant size in bits and iteration count.
// Returns true if the proof verifies, false otherwise.
// This mirrors the Python binding fastvdf.verify_from_hash signature.
bool verify_from_hash_wrapper(
    const uint8_t* challenge_hash,
    size_t challenge_size,
    const uint8_t* proof_blob,
    size_t proof_blob_size,
    uint32_t discriminant_size_bits,
    uint64_t num_iterations,
    uint64_t recursion)
{
    try {
        if (challenge_hash == nullptr || challenge_size != 32) return false;
        if (proof_blob == nullptr || proof_blob_size == 0) return false;
        if (discriminant_size_bits == 0) return false;

        // Build discriminant from 32-byte challenge
        std::vector<uint8_t> seed(challenge_hash, challenge_hash + challenge_size);
        integer D = CreateDiscriminant(seed, discriminant_size_bits);

        // x = form(2,1,D)
        form x = form::from_abd(integer(2), integer(1), D);
        x.reduce();

        // Verify
        return CheckProofOfTimeNWesolowski(
            D,
            SerializeForm(x, D.num_bits()).data(),
            proof_blob,
            static_cast<int32_t>(proof_blob_size),
            num_iterations,
            D.num_bits(),
            static_cast<int32_t>(recursion)
        );
    } catch (...) {
        return false;
    }
}

} // extern "C"

