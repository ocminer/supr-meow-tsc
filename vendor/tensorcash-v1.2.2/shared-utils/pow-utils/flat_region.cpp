// Copyright (c) 2026 TensorCash
// Keep byte-identical to shared-utils/pow-utils/flat_region.cpp.
#include "flat_region.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <boost/multiprecision/cpp_int.hpp>

namespace flat_region {
namespace {
using Wide = boost::multiprecision::uint512_t;
using Fast = unsigned __int128;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

bool Fail(std::string* error, const char* message)
{
    if (error) *error = message;
    return false;
}

// Exact x * 2^149, including subnormals and signed zero. A binary32 in
// [0,1] needs <=150 bits. CDF endpoints may round just above one; accept
// [0,2) for that adapter only. No floating-point operation is performed.
bool Units(float value, bool cdf, Wide& out)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t magnitude = bits & 0x7fffffffU;
    if (magnitude == 0) { out = 0; return true; }
    if ((bits >> 31) || magnitude >= (cdf ? 0x40000000U : 0x3f800001U)) return false;
    const unsigned exponent = (bits >> 23) & 255U;
    const uint32_t fraction = bits & 0x7fffffU;
    if (exponent == 0) out = fraction;
    else out = Wide((1U << 23) | fraction) << (exponent - 1);
    return true;
}

bool Decode(const float* values, size_t count, bool cdf,
            std::array<Wide, MAX_ROW>& p, size_t& n, Wide& total, std::string* error)
{
    n = 0; total = 0;
    if (!values || count == 0 || count > MAX_ROW) return Fail(error, "invalid canonical row width");
    Wide previous = 0;
    for (size_t i = 0; i < count; ++i) {
        Wide value;
        if (!Units(values[i], cdf, value)) return Fail(error, "invalid canonical probability encoding");
        if (cdf) {
            if (value < previous) return Fail(error, "descending canonical CDF");
            const Wide endpoint = value;
            value -= previous;
            previous = endpoint;
        }
        if (value != 0) { p[n++] = value; total += value; }
    }
    if (total == 0) return Fail(error, "zero canonical row mass");
    return true;
}

template <typename Integer>
bool Search(std::array<Integer, MAX_ROW>& p, size_t n, const Integer& total)
{
    std::sort(p.begin(), p.begin() + n, std::greater<Integer>());
    std::array<Integer, MAX_ROW> squares;
    for (size_t i = 0; i < n; ++i) squares[i] = p[i] * p[i];
    for (size_t a = 0; a + MIN_REGION <= n; ++a) {
        Integer mass = 0, sum_squares = 0;
        for (size_t b = a; b < n; ++b) {
            mass += p[b]; sum_squares += squares[b];
            const size_t length = b - a + 1;
            // >=1/4 row mass and >=9/10 collision effective-support ratio.
            // Scale cancels: no normalization, logs, sqrt, or division.
            if (length >= MIN_REGION && 4 * mass >= total &&
                10 * mass * mass >= (9 * length) * sum_squares) return true;
        }
    }
    return false;
}

bool Qualifies(std::array<Wide, MAX_ROW>& p, size_t n, const Wide& total)
{
    if (n < MIN_REGION) return false;
    // Remove only COMMON powers of two, exactly. Most canonical rows then
    // fit the faster 128-bit path. No small probabilities are discarded.
    size_t shift = 512;
    // lsb() returns unsigned; name the type so the comparison is not a
    // deduction failure where it is narrower than size_t (a bit index
    // below 512 always fits).
    for (size_t i = 0; i < n; ++i)
        shift = std::min<size_t>(shift, static_cast<size_t>(boost::multiprecision::lsb(p[i])));
    const Wide reduced_total = total >> shift;
    if (boost::multiprecision::msb(reduced_total) < 59) {
        std::array<Fast, MAX_ROW> reduced{};
        for (size_t i = 0; i < n; ++i) reduced[i] = (p[i] >> shift).convert_to<uint64_t>();
        // total <2^59; 9*n*sum(p^2) <=450*total^2 <2^127.
        return Search(reduced, n, Fast(reduced_total.convert_to<uint64_t>()));
    }
    // Each unscaled mass <2^150, total <50*2^150 <2^156.
    // Both sides of every comparison are <2^321, safely inside 512 bits.
    return Search(p, n, total);
}
} // namespace

bool ComputeRow(const float* values, size_t count, bool cdf,
                bool& qualifies, std::string* error)
{
    qualifies = false;
    if (error) error->clear();
    std::array<Wide, MAX_ROW> p;
    size_t n; Wide total;
    if (!Decode(values, count, cdf, p, n, total, error)) return false;
    qualifies = Qualifies(p, n, total);
    return true;
}

bool Rejects(uint64_t capped_units, uint64_t flat_capped_units)
{
    return capped_units > 0 && capped_units <= MAX_CAPPED_UNITS &&
           flat_capped_units <= capped_units && 4 * flat_capped_units >= 3 * capped_units;
}

bool ComputeWindow(const std::vector<std::vector<float>>& rows,
                   const std::vector<uint64_t>& credits, bool cdf,
                   Metrics& out, std::string* error)
{
    out = Metrics{};
    if (error) error->clear();
    if (rows.size() != WINDOW || credits.size() != WINDOW) return Fail(error, "expected 256 rows and credits");
    Metrics metrics;
    for (uint64_t credit : credits) {
        if (credit > MAX_STEP_CREDIT) return Fail(error, "credit exceeds canonical per-step cap");
        metrics.capped_units += std::min(credit, CREDIT_R);
    }
    metrics.eligible = metrics.capped_units > 0 && metrics.capped_units <= MAX_CAPPED_UNITS;
    for (size_t i = 0; i < WINDOW; ++i) {
        std::array<Wide, MAX_ROW> p;
        size_t n; Wide total;
        if (!Decode(rows[i].data(), rows[i].size(), cdf, p, n, total, error)) return false;
        if (metrics.eligible && credits[i] != 0 && Qualifies(p, n, total)) {
            metrics.flat_capped_units += std::min(credits[i], CREDIT_R);
            ++metrics.flat_credited_rows;
        }
    }
    metrics.reject = Rejects(metrics.capped_units, metrics.flat_capped_units);
    out = metrics;
    return true;
}

std::string RejectReason(const Metrics& metrics)
{
    return std::string(REJECT_TAG) + ": flat-region capped credit " +
        std::to_string(metrics.flat_capped_units) + "/" + std::to_string(metrics.capped_units) +
        " units (>=75%, total <=80 bits; R=1024)";
}
} // namespace flat_region
