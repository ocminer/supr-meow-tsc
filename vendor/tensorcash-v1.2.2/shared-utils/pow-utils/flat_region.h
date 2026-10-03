// Copyright (c) 2026 TensorCash
// Integer flat-region admission; byte-identical shared-utils mirror.
#ifndef TENSORCASH_VERIFICATION_FLAT_REGION_H
#define TENSORCASH_VERIFICATION_FLAT_REGION_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace flat_region {
constexpr size_t WINDOW = 256;
constexpr size_t MAX_ROW = 50;
constexpr size_t MIN_REGION = 4;
constexpr uint64_t CREDIT_R = 1024;
constexpr uint64_t MAX_STEP_CREDIT = 32768;
constexpr uint64_t MAX_CAPPED_UNITS = 80 * CREDIT_R;
constexpr char REJECT_TAG[] = "bad-flat-region";

struct Metrics {
    uint64_t capped_units{0};
    uint64_t flat_capped_units{0};
    unsigned flat_credited_rows{0};
    // Above the ceiling (or at zero credit), validate inputs but skip the
    // region search. Flat counters are unavailable, not measured zeroes.
    bool eligible{false};
    bool reject{false};
};

// Consume canonical binary32 probabilities, NOT declared chosen_probs.
// The cdf adapter exists for historical research/conformance: exact interval
// differences, not floating subtraction. Live v4 always uses probabilities.
// Zero masses do not count towards support. Float values are read as bits;
// every sum, square and comparison thereafter is exact integer arithmetic.
bool ComputeRow(const float* values, size_t count, bool cdf,
                bool& qualifies, std::string* error = nullptr);
bool ComputeWindow(const std::vector<std::vector<float>>& rows,
                   const std::vector<uint64_t>& credits, bool cdf,
                   Metrics& out, std::string* error = nullptr);
// Safe for arbitrary arguments; only canonical bounded aggregates can reject.
bool Rejects(uint64_t capped_units, uint64_t flat_capped_units);
std::string RejectReason(const Metrics& metrics);
} // namespace flat_region
#endif
