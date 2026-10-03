// Copyright (c) 2026 TensorCash
// Prompt-supplied scaffolds with credit-bearing holes, on a 256-token window.
// All arithmetic is integer; consume canonical credit units, NEVER chosen_probs.
// QuickVerifier enforces this only for v4 at/after the configured v3 sunset.
// Keep byte-identical to shared-utils/pow-utils/prompt_scaffold.h.
#ifndef TENSORCASH_VERIFICATION_PROMPT_SCAFFOLD_H
#define TENSORCASH_VERIFICATION_PROMPT_SCAFFOLD_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace prompt_scaffold {

constexpr size_t WINDOW = 256;
constexpr uint64_t MAX_STEP_CREDIT = 32768;
constexpr uint64_t LOW_CREDIT = 256;  // <= 0.25 bit, R = 1024
constexpr uint64_t MIN_HOLE_CREDIT = 512; // >= 0.5 bit
constexpr size_t MAX_HOLE_TOKENS = 8;
constexpr int MIN_HOLES = 3;
constexpr size_t MIN_REPETITIONS = 3;
constexpr char REJECT_TAG[] = "bad-prompt-scaffold";
// Branch order: low-credit R4x3/128, text C12/64, low-credit R3x3/64,
// text R3x3/64. Exact equivalents of >=60%, >=75%, >=80%, >=75%.
constexpr std::array<unsigned, 4> MIN_COVERED{{77, 48, 52, 48}};

using Mask = std::array<bool, WINDOW>;
struct Metrics {
    std::array<unsigned, 4> covered{}; // maximum coverage of an eligible local window
    bool reject{false};
};

namespace detail {

// Index ONLY continuation fragments: at most 256-K+1 keys. Ordered maps give
// exact matching without hash collisions or prompt-sized allocations. Each
// matched key is erased once its output occurrences have all been marked.
// Masked prompt tokens break a span; they are neither content nor removed in
// a way that could join two formerly nonadjacent source spans.
template <size_t K>
Mask Copied(const std::vector<uint32_t>& prompt, const std::vector<uint8_t>& pad,
            const std::vector<uint32_t>& chosen, bool require_repeats)
{
    using Key = std::array<uint32_t, K>;
    std::map<Key, std::vector<size_t>> occurrences;
    for (size_t start = 0; start + K <= WINDOW; ++start) {
        Key key;
        std::copy_n(chosen.begin() + start, K, key.begin());
        occurrences[key].push_back(start);
    }
    if (require_repeats) {
        for (auto it = occurrences.begin(); it != occurrences.end();) {
            size_t count = 0, next = 0;
            for (size_t start : it->second) {
                if (start >= next) {
                    ++count;
                    next = start + K;
                }
            }
            if (count < MIN_REPETITIONS) it = occurrences.erase(it);
            else ++it;
        }
    }
    Mask mask{};
    size_t consecutive = 0;
    for (size_t end = 0; end < prompt.size() && !occurrences.empty(); ++end) {
        if (!pad.empty() && pad[end]) {
            consecutive = 0;
            continue;
        }
        if (consecutive < K) ++consecutive;
        if (consecutive < K) continue;
        Key key;
        std::copy_n(prompt.begin() + (end + 1 - K), K, key.begin());
        const auto it = occurrences.find(key);
        if (it == occurrences.end()) continue;
        for (size_t start : it->second) {
            std::fill_n(mask.begin() + start, K, true);
        }
        occurrences.erase(it);
    }
    return mask;
}

inline Mask LowCredit(Mask mask, const std::vector<uint64_t>& credits)
{
    for (size_t i = 0; i < WINDOW; ++i) mask[i] = mask[i] && credits[i] <= LOW_CREDIT;
    return mask;
}

// A hole is a maximal uncovered run INTERNAL TO THE CONTINUATION. A local
// window qualifies when it contains at least three complete holes of <=8
// tokens carrying >=512 units each. The flanking covered tokens may lie just
// outside that local window. Prefix/suffix runs of the continuation never
// count. This boundary convention matches the frozen research reference.
inline unsigned LocalCoverage(const Mask& mask, const std::vector<uint64_t>& credits,
                              size_t width)
{
    std::array<unsigned, WINDOW + 1> prefix{};
    std::array<int, WINDOW + 2> hole_delta{};
    for (size_t i = 0; i < WINDOW; ++i) prefix[i + 1] = prefix[i] + unsigned(mask[i]);
    size_t i = 0;
    while (i < WINDOW) {
        if (mask[i]) { ++i; continue; }
        const size_t start = i;
        uint64_t sum = 0;
        while (i < WINDOW && !mask[i]) sum += credits[i++];
        if (start == 0 || i == WINDOW || i - start > MAX_HOLE_TOKENS || sum < MIN_HOLE_CREDIT) continue;
        const size_t low = i > width ? i - width : 0;
        const size_t high = std::min(start, WINDOW - width);
        if (low <= high) {
            ++hole_delta[low];
            --hole_delta[high + 1];
        }
    }
    unsigned best = 0;
    int holes = 0;
    for (size_t start = 0; start + width <= WINDOW; ++start) {
        holes += hole_delta[start];
        if (holes >= MIN_HOLES) best = std::max(best, prefix[start + width] - prefix[start]);
    }
    return best;
}

} // namespace detail

// false means malformed evidence, never a clean/non-flagged window. Counters
// are <=256 and total credit <=256*32768=2^23; sums fit even uint32. Widened
// uint64 here also makes the bound explicit. No floating point or hashing.
inline bool ComputeWindow(const std::vector<uint32_t>& prompt,
                          const std::vector<uint8_t>& pad,
                          const std::vector<uint32_t>& chosen,
                          const std::vector<uint64_t>& credits,
                          Metrics& out, std::string* error = nullptr)
{
    out = Metrics{};
    if (error) error->clear();
    const auto fail = [&](const char* why) {
        if (error) *error = why;
        return false;
    };
    if (chosen.size() != WINDOW || credits.size() != WINDOW) return fail("expected 256 tokens and credits");
    if (!pad.empty() && pad.size() != prompt.size()) return fail("pad mask length differs from prompt");
    for (uint8_t value : pad) if (value > 1) return fail("pad mask is not binary");
    for (uint64_t value : credits) if (value > MAX_STEP_CREDIT) return fail("credit exceeds canonical per-step cap");

    const Mask r3 = detail::Copied<3>(prompt, pad, chosen, true);
    const Mask r4 = detail::Copied<4>(prompt, pad, chosen, true);
    // Union of matching 12-grams equals union of ALL matching spans >=12.
    const Mask c12 = detail::Copied<12>(prompt, pad, chosen, false);
    out.covered = {{detail::LocalCoverage(detail::LowCredit(r4, credits), credits, 128),
                    detail::LocalCoverage(c12, credits, 64),
                    detail::LocalCoverage(detail::LowCredit(r3, credits), credits, 64),
                    detail::LocalCoverage(r3, credits, 64)}};
    for (size_t branch = 0; branch < MIN_COVERED.size(); ++branch) {
        out.reject = out.reject || out.covered[branch] >= MIN_COVERED[branch];
    }
    return true;
}

inline std::string RejectReason(const Metrics& metrics)
{
    return std::string(REJECT_TAG) + ": local copied-token coverage " +
        std::to_string(metrics.covered[0]) + "/128," +
        std::to_string(metrics.covered[1]) + "/64," +
        std::to_string(metrics.covered[2]) + "/64," +
        std::to_string(metrics.covered[3]) + "/64 with at least three credit-bearing holes";
}

} // namespace prompt_scaffold
#endif // TENSORCASH_VERIFICATION_PROMPT_SCAFFOLD_H
