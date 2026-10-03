// Candidate-band V90 condition for the Quick pass: consensus rule over the
// canonical id-ordered CDF of a COMPLETE 256-row continuation
// (docs/candidate-band.md; tested
// standalone reference tests/vectors/).
//
// The condition rejects a window when at least MIN_ACTIVE_ROWS (32) rows
// contain a candidate in the probability band (.01, .50) and at most
// TOP_TERMS (31) token identities account for at least RATIO_NUM/RATIO_DEN
// (9/10) of the variation in those band probabilities. Every qualifying
// candidate of every row counts, selected or not; the chosen token, the u
// draw, credit weights and the ATOL-widened masses play no part.
//
// Exact integer computation. For each row, with c_j the binary32 CDF
// endpoints in token-id order (credit_v4::StepResult::cdf_hi):
//
//   E_j = floor(2^32 * c_j)          exactly, from the binary32 encoding
//   q_j = E_j - E_{j-1}              (E_{-1} = 0), no renormalisation
//   included iff LOWEST_INCLUDED <= q_j <= HIGHEST_INCLUDED
//     (42,949,673 <= q <= 2,147,483,647, i.e. 100*q > 2^32 and 2*q < 2^32)
//
// A row with at least one included candidate is ACTIVE; N counts them. Per
// token id i, over its included occurrences: A_i = sum q, B_i = sum q^2
// (missing / excluded occurrences contribute 0 on every active row). Then
//
//   D_i = N * B_i - A_i^2           (>= 0 by Cauchy-Schwarz; exact)
//   T   = sum_i D_i
//   L31 = sum of the TOP_TERMS largest D_i
//   reject = (N >= MIN_ACTIVE_ROWS) && (RATIO_DEN * L31 >= RATIO_NUM * T)
//
// The comparison is inclusive at exactly 90%; T == 0 with N >= 32 rejects
// (the research V90 = 0 convention); N < 32 never rejects through this rule
// (insufficient coverage, reported as such). Malformed evidence (non-finite,
// negative or descending endpoints, unsorted or duplicate ids, an empty row,
// a row wider than the pinned top-k, a window that is not exactly
// WINDOW_ROWS rows) is an error, never an undercoverage result.
//
// Widths. q and A_i fit uint64 (A_i < 2^39); B_i, the products, D_i, T and
// L31 use unsigned __int128 (T, L31 < 2^92 with at most 12,800 token ids;
// 10 * L31 and 9 * T < 2^96). Widen BEFORE multiplying.
//
// Everything here is integer arithmetic on the binary32 ENCODINGS, so it is
// platform-independent by construction; the endpoints it consumes must come
// from the canonical credit kernel (credit_v4.h) for the result to be one.
// Header-only; vendored byte-identical into shared-utils/pow-utils/
// candidate_band.h and pinned by scripts/check_vendored_consensus_sources.py.

#ifndef TENSORCASH_VERIFICATION_CANDIDATE_BAND_H
#define TENSORCASH_VERIFICATION_CANDIDATE_BAND_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace candidate_band {

using u128 = unsigned __int128;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "candidate_band: binary32 required");

// Fixed-point scale Q = 2^32 and the inclusive integer band
// (100 * q > Q and 2 * q < Q).
constexpr uint64_t Q32 = uint64_t{1} << 32;
constexpr uint64_t LOWEST_INCLUDED = Q32 / 100 + 1;   // 42,949,673
constexpr uint64_t HIGHEST_INCLUDED = Q32 / 2 - 1;    // 2,147,483,647
static_assert(LOWEST_INCLUDED == 42949673ULL, "band lower edge");
static_assert(HIGHEST_INCLUDED == 2147483647ULL, "band upper edge");

// Window shape the condition is defined over.
constexpr uint64_t WINDOW_ROWS = 256;        // pow_v3::POW_WINDOW_SIZE
constexpr uint64_t MAX_ROW_CANDIDATES = 50;  // credit_v4::TOP_K
// Coverage and concentration parameters.
constexpr uint64_t MIN_ACTIVE_ROWS = 32;
constexpr uint64_t TOP_TERMS = 31;   // V90 < 32 <=> the 31 largest terms suffice
constexpr uint64_t RATIO_NUM = 9;    // ... for at least 9/10 of the total
constexpr uint64_t RATIO_DEN = 10;

// The consensus reason tag (connection-path reject reason and the verifier
// error prefix carry it verbatim).
constexpr char REJECT_TAG[] = "bad-candidate-band";

// Exact floor(x * 2^32) from a binary32 encoding. Either signed zero and
// every subnormal give 0. Returns false (out untouched) for a negative
// non-zero value, an infinity, a NaN or a value whose scaled floor would
// not fit uint64 (x >= 2^32). Never shifts a 64-bit integer by 64 or more.
inline bool EndpointQ32(uint32_t bits, uint64_t& out)
{
    const uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude == 0) {
        out = 0;
        return true;
    }
    if (bits >> 31) return false;  // negative, non-zero
    const uint32_t exponent = (bits >> 23) & 255U;
    if (exponent == 255) return false;  // inf / NaN
    if (exponent == 0) {
        out = 0;  // subnormal: below 2^-126 < 2^-32
        return true;
    }
    // x = M * 2^(e - 150) with M = 2^23 + fraction; floor(x * 2^32) =
    // floor(M * 2^(e - 118)).
    const uint64_t mantissa = (uint64_t{1} << 23) | (bits & 0x7fffffU);
    const int shift = static_cast<int>(exponent) - 118;
    if (shift >= 0) {
        if (shift > 40) return false;  // x >= 2^32: not a CDF endpoint
        out = mantissa << shift;
        return true;
    }
    out = (-shift >= 64) ? 0 : (mantissa >> (-shift));
    return true;
}

inline uint32_t Float32Bits(float x)
{
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

inline bool InBand(uint64_t q)
{
    return q >= LOWEST_INCLUDED && q <= HIGHEST_INCLUDED;
}

// The rejection predicate on the finished aggregates. Safe for aggregates
// produced by Accumulator (10 * L31 and 9 * T are below 2^96).
inline bool Rejects(uint64_t active_rows, u128 total, u128 largest_terms)
{
    return active_rows >= MIN_ACTIVE_ROWS &&
           u128{RATIO_DEN} * largest_terms >= u128{RATIO_NUM} * total;
}

struct Metrics {
    uint64_t active_rows{0};   // N
    u128 total{0};             // T
    u128 largest_terms{0};     // L31
    // Diagnostic only: the smallest k with RATIO_DEN * (sum of the k largest
    // D_i) >= RATIO_NUM * T, 0 when T == 0 (the research "V90").
    uint64_t v90{0};
    uint64_t token_ids{0};     // number of distinct token ids with a term
    bool eligible{false};      // active_rows >= MIN_ACTIVE_ROWS
    bool reject{false};        // Rejects(active_rows, total, largest_terms)
};

// Decimal rendering of a 128-bit total (JSON numbers cannot carry it; the
// verifier error string and every export carry the decimal text).
inline std::string U128ToString(u128 v)
{
    if (v == 0) return "0";
    std::string out;
    while (v != 0) {
        out.push_back(static_cast<char>('0' + static_cast<unsigned>(v % 10)));
        v /= 10;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

// One window, fed row by row. AddRow returns false with `err` set on
// malformed evidence; Finish returns false when the row count is not
// exactly WINDOW_ROWS. Neither throws.
class Accumulator {
    struct Moments {
        uint64_t sum{0};
        u128 squares{0};
    };
    std::map<uint32_t, Moments> m_moments;
    uint64_t m_rows{0};
    uint64_t m_active{0};

public:
    bool AddRow(const uint32_t* ids, const float* cdf_hi, std::size_t count, std::string* err)
    {
        if (m_rows >= WINDOW_ROWS) {
            if (err) *err = "more than " + std::to_string(WINDOW_ROWS) + " rows";
            return false;
        }
        if (count == 0 || ids == nullptr || cdf_hi == nullptr) {
            if (err) *err = "empty CDF row";
            return false;
        }
        if (count > MAX_ROW_CANDIDATES) {
            if (err) *err = "CDF row wider than the pinned top-k";
            return false;
        }
        ++m_rows;
        bool any = false;
        uint64_t previous_q = 0;
        uint32_t previous_bits = 0;
        for (std::size_t j = 0; j < count; ++j) {
            if (j > 0 && ids[j] <= ids[j - 1]) {
                if (err) *err = "CDF ids not strictly increasing";
                return false;
            }
            const uint32_t bits = Float32Bits(cdf_hi[j]);
            uint64_t endpoint_q = 0;
            if (!EndpointQ32(bits, endpoint_q)) {
                if (err) *err = "CDF endpoint is negative, non-finite or out of range";
                return false;
            }
            // Compare the original encodings too: quantisation must not hide
            // a descending pair that lands on the same integer.
            const uint32_t positive_bits = bits & 0x7fffffffU;
            if (positive_bits < previous_bits || endpoint_q < previous_q) {
                if (err) *err = "descending CDF";
                return false;
            }
            const uint64_t q = endpoint_q - previous_q;
            previous_q = endpoint_q;
            previous_bits = positive_bits;
            if (!InBand(q)) continue;
            any = true;
            Moments& m = m_moments[ids[j]];
            m.sum += q;
            m.squares += u128{q} * q;  // widen BEFORE multiplying
        }
        if (any) ++m_active;
        return true;
    }

    uint64_t Rows() const { return m_rows; }

    bool Finish(Metrics& out, std::string* err) const
    {
        out = Metrics{};
        if (m_rows != WINDOW_ROWS) {
            if (err) {
                *err = "expected " + std::to_string(WINDOW_ROWS) + " continuation rows, got " +
                       std::to_string(m_rows);
            }
            return false;
        }
        out.active_rows = m_active;
        std::vector<u128> terms;
        terms.reserve(m_moments.size());
        for (const auto& entry : m_moments) {
            const Moments& m = entry.second;
            const u128 first = u128{m_active} * m.squares;
            const u128 second = u128{m.sum} * m.sum;
            if (first < second) {
                // Impossible for unique ids per row (Cauchy-Schwarz); an
                // accumulator defect, reported rather than clamped.
                if (err) *err = "negative exact variance term";
                return false;
            }
            const u128 d = first - second;
            out.total += d;
            terms.push_back(d);
        }
        std::sort(terms.begin(), terms.end(), [](u128 a, u128 b) { return a > b; });
        u128 prefix = 0;
        for (std::size_t j = 0; j < terms.size(); ++j) {
            prefix += terms[j];
            if (j < TOP_TERMS) out.largest_terms += terms[j];
            if (out.total != 0 && out.v90 == 0 &&
                u128{RATIO_DEN} * prefix >= u128{RATIO_NUM} * out.total) {
                out.v90 = j + 1;
            }
        }
        out.token_ids = terms.size();
        out.eligible = m_active >= MIN_ACTIVE_ROWS;
        out.reject = Rejects(m_active, out.total, out.largest_terms);
        return true;
    }
};

// The whole window from the captured canonical CDF rows (one vector of ids
// and one of binary32 endpoints per row, as the verifier captures them).
inline bool ComputeWindow(const std::vector<std::vector<uint32_t>>& cdf_ids,
                          const std::vector<std::vector<float>>& cdf_hi,
                          Metrics& out, std::string* err)
{
    out = Metrics{};
    if (cdf_ids.size() != cdf_hi.size()) {
        if (err) *err = "CDF id / endpoint row counts differ";
        return false;
    }
    Accumulator acc;
    for (std::size_t s = 0; s < cdf_ids.size(); ++s) {
        if (cdf_ids[s].size() != cdf_hi[s].size()) {
            if (err) *err = "CDF id / endpoint lengths differ at row " + std::to_string(s);
            return false;
        }
        std::string row_err;
        if (!acc.AddRow(cdf_ids[s].data(), cdf_hi[s].data(), cdf_ids[s].size(), &row_err)) {
            if (err) *err = row_err + " at row " + std::to_string(s);
            return false;
        }
    }
    return acc.Finish(out, err);
}

// The consensus error string for a rejecting window (identical in the
// Python mirror pow_v4.candidate_band_rejects): the tag, then N, T and L31
// as exact decimals.
inline std::string RejectReason(const Metrics& m)
{
    return std::string("proof v4 candidate band rejected (") + REJECT_TAG + "): N=" +
           std::to_string(m.active_rows) + " T=" + U128ToString(m.total) + " L31=" +
           U128ToString(m.largest_terms) + " (at most " + std::to_string(TOP_TERMS) +
           " token identities account for at least " + std::to_string(RATIO_NUM) + "/" +
           std::to_string(RATIO_DEN) + " of the (.01, .50) candidate variance over " +
           std::to_string(m.active_rows) + " active rows)";
}

}  // namespace candidate_band

#endif  // TENSORCASH_VERIFICATION_CANDIDATE_BAND_H
