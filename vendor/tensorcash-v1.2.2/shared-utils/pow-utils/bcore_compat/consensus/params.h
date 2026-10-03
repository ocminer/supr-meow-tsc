#pragma once
// bcore include-path shim for the vendored pow_v4.{h,cpp}.
//
// bcore's Consensus::Params is a node-side struct; the miner build only needs
// the three proof v4 heights and the predicates pow_v4.cpp reads
// (effective_proof_mode, CheckProofModeAtHeight, ProofPriceMultiplier). The
// miner never calls those height-aware functions: jobs select the mode
// through the "v4": 1 work-unit flag, never a height guess. The defaults are
// "never", exactly like bcore.
#include <limits>

namespace Consensus {

struct Params {
    int ProofV4Height{std::numeric_limits<int>::max()};     // X
    int V3PenaltyHeight{std::numeric_limits<int>::max()};   // Y
    int V3SunsetHeight{std::numeric_limits<int>::max()};    // Z

    bool IsProofV4Active(int height) const
    {
        return height >= 0 && height >= ProofV4Height;
    }
    bool IsV3Penalized(int height) const
    {
        return height >= 0 && height >= V3PenaltyHeight && height < V3SunsetHeight;
    }
    bool IsV3Sunset(int height) const
    {
        return height >= 0 && height >= V3SunsetHeight;
    }
};

}  // namespace Consensus
