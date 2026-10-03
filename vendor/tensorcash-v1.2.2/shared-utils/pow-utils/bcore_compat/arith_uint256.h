// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.
//
// bcore include-path shim for the vendored pow_v4.{h,cpp}: a header-only,
// C++17-compatible subset of bcore src/arith_uint256.{h,cpp} (base_uint
// arithmetic, comparison, GetLow64, bits) sufficient for pow_v4::PricedTarget
// and pow_v4::HashMeetsTarget. Omitted on purpose: the uint256 conversions,
// the compact encoding, GetHex/ToString (they need the node's uint256 and
// strencodings). The miner never calls the price functions; this exists only
// so the vendored copy compiles byte-identical to bcore's.

#ifndef BITCOIN_ARITH_UINT256_H
#define BITCOIN_ARITH_UINT256_H

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

class uint_error : public std::runtime_error {
public:
    explicit uint_error(const std::string& str) : std::runtime_error(str) {}
};

/** Template base class for unsigned big integers. */
template<unsigned int BITS>
class base_uint
{
protected:
    static_assert(BITS / 32 > 0 && BITS % 32 == 0, "Template parameter BITS must be a positive multiple of 32.");
    static constexpr int WIDTH = BITS / 32;
    /** Big integer represented with 32-bit digits, least-significant first. */
    uint32_t pn[WIDTH];
public:

    base_uint()
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] = 0;
    }

    base_uint(const base_uint& b)
    {
        for (int i = 0; i < WIDTH; i++)
            pn[i] = b.pn[i];
    }

    base_uint& operator=(const base_uint& b)
    {
        if (this != &b) {
            for (int i = 0; i < WIDTH; i++)
                pn[i] = b.pn[i];
        }
        return *this;
    }

    base_uint(uint64_t b)
    {
        pn[0] = (unsigned int)b;
        pn[1] = (unsigned int)(b >> 32);
        for (int i = 2; i < WIDTH; i++)
            pn[i] = 0;
    }

    base_uint operator~() const
    {
        base_uint ret;
        for (int i = 0; i < WIDTH; i++)
            ret.pn[i] = ~pn[i];
        return ret;
    }

    base_uint operator-() const
    {
        base_uint ret;
        for (int i = 0; i < WIDTH; i++)
            ret.pn[i] = ~pn[i];
        ++ret;
        return ret;
    }

    base_uint& operator=(uint64_t b)
    {
        pn[0] = (unsigned int)b;
        pn[1] = (unsigned int)(b >> 32);
        for (int i = 2; i < WIDTH; i++)
            pn[i] = 0;
        return *this;
    }

    base_uint& operator<<=(unsigned int shift)
    {
        base_uint<BITS> a(*this);
        for (int i = 0; i < WIDTH; i++)
            pn[i] = 0;
        int k = shift / 32;
        shift = shift % 32;
        for (int i = 0; i < WIDTH; i++) {
            if (i + k + 1 < WIDTH && shift != 0)
                pn[i + k + 1] |= (a.pn[i] >> (32 - shift));
            if (i + k < WIDTH)
                pn[i + k] |= (a.pn[i] << shift);
        }
        return *this;
    }

    base_uint& operator>>=(unsigned int shift)
    {
        base_uint<BITS> a(*this);
        for (int i = 0; i < WIDTH; i++)
            pn[i] = 0;
        int k = shift / 32;
        shift = shift % 32;
        for (int i = 0; i < WIDTH; i++) {
            if (i - k - 1 >= 0 && shift != 0)
                pn[i - k - 1] |= (a.pn[i] << (32 - shift));
            if (i - k >= 0)
                pn[i - k] |= (a.pn[i] >> shift);
        }
        return *this;
    }

    base_uint& operator+=(const base_uint& b)
    {
        uint64_t carry = 0;
        for (int i = 0; i < WIDTH; i++)
        {
            uint64_t n = carry + pn[i] + b.pn[i];
            pn[i] = n & 0xffffffff;
            carry = n >> 32;
        }
        return *this;
    }

    base_uint& operator-=(const base_uint& b)
    {
        *this += -b;
        return *this;
    }

    base_uint& operator+=(uint64_t b64)
    {
        base_uint b;
        b = b64;
        *this += b;
        return *this;
    }

    base_uint& operator-=(uint64_t b64)
    {
        base_uint b;
        b = b64;
        *this += -b;
        return *this;
    }

    base_uint& operator*=(uint32_t b32)
    {
        uint64_t carry = 0;
        for (int i = 0; i < WIDTH; i++) {
            uint64_t n = carry + (uint64_t)b32 * pn[i];
            pn[i] = n & 0xffffffff;
            carry = n >> 32;
        }
        return *this;
    }

    base_uint& operator*=(const base_uint& b)
    {
        base_uint<BITS> a;
        for (int j = 0; j < WIDTH; j++) {
            uint64_t carry = 0;
            for (int i = 0; i + j < WIDTH; i++) {
                uint64_t n = carry + a.pn[i + j] + (uint64_t)pn[j] * b.pn[i];
                a.pn[i + j] = n & 0xffffffff;
                carry = n >> 32;
            }
        }
        *this = a;
        return *this;
    }

    base_uint& operator/=(const base_uint& b)
    {
        base_uint<BITS> div = b;     // make a copy, so we can shift.
        base_uint<BITS> num = *this; // make a copy, so we can subtract.
        *this = 0;                   // the quotient.
        int num_bits = num.bits();
        int div_bits = div.bits();
        if (div_bits == 0)
            throw uint_error("Division by zero");
        if (div_bits > num_bits) // the result is certainly 0.
            return *this;
        int shift = num_bits - div_bits;
        div <<= shift; // shift so that div and num align.
        while (shift >= 0) {
            if (num >= div) {
                num -= div;
                pn[shift / 32] |= (1U << (shift & 31)); // set a bit of the result.
            }
            div >>= 1; // shift back.
            shift--;
        }
        // num now contains the remainder of the division.
        return *this;
    }

    base_uint& operator++()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && ++pn[i] == 0)
            i++;
        return *this;
    }

    base_uint operator++(int)
    {
        // postfix operator
        const base_uint ret = *this;
        ++(*this);
        return ret;
    }

    base_uint& operator--()
    {
        // prefix operator
        int i = 0;
        while (i < WIDTH && --pn[i] == std::numeric_limits<uint32_t>::max())
            i++;
        return *this;
    }

    base_uint operator--(int)
    {
        // postfix operator
        const base_uint ret = *this;
        --(*this);
        return ret;
    }

    /** Numeric ordering (unlike base_blob::Compare) */
    int CompareTo(const base_uint& b) const
    {
        for (int i = WIDTH - 1; i >= 0; i--) {
            if (pn[i] < b.pn[i])
                return -1;
            if (pn[i] > b.pn[i])
                return 1;
        }
        return 0;
    }

    bool EqualTo(uint64_t b) const
    {
        for (int i = WIDTH - 1; i >= 2; i--) {
            if (pn[i])
                return false;
        }
        if (pn[1] != (b >> 32))
            return false;
        if (pn[0] != (b & 0xfffffffful))
            return false;
        return true;
    }

    friend inline base_uint operator+(const base_uint& a, const base_uint& b) { return base_uint(a) += b; }
    friend inline base_uint operator-(const base_uint& a, const base_uint& b) { return base_uint(a) -= b; }
    friend inline base_uint operator*(const base_uint& a, const base_uint& b) { return base_uint(a) *= b; }
    friend inline base_uint operator/(const base_uint& a, const base_uint& b) { return base_uint(a) /= b; }
    friend inline base_uint operator>>(const base_uint& a, int shift) { return base_uint(a) >>= shift; }
    friend inline base_uint operator<<(const base_uint& a, int shift) { return base_uint(a) <<= shift; }
    friend inline base_uint operator*(const base_uint& a, uint32_t b) { return base_uint(a) *= b; }
    friend inline bool operator==(const base_uint& a, const base_uint& b) { return memcmp(a.pn, b.pn, sizeof(a.pn)) == 0; }
    friend inline bool operator!=(const base_uint& a, const base_uint& b) { return !(a == b); }
    friend inline bool operator<(const base_uint& a, const base_uint& b) { return a.CompareTo(b) < 0; }
    friend inline bool operator<=(const base_uint& a, const base_uint& b) { return a.CompareTo(b) <= 0; }
    friend inline bool operator>(const base_uint& a, const base_uint& b) { return a.CompareTo(b) > 0; }
    friend inline bool operator>=(const base_uint& a, const base_uint& b) { return a.CompareTo(b) >= 0; }
    friend inline bool operator==(const base_uint& a, uint64_t b) { return a.EqualTo(b); }
    friend inline bool operator!=(const base_uint& a, uint64_t b) { return !a.EqualTo(b); }

    unsigned int size() const
    {
        return sizeof(pn);
    }

    /**
     * Returns the position of the highest bit set plus one, or zero if the
     * value is zero.
     */
    unsigned int bits() const
    {
        for (int pos = WIDTH - 1; pos >= 0; pos--) {
            if (pn[pos]) {
                for (int nbits = 31; nbits > 0; nbits--) {
                    if (pn[pos] & 1U << nbits)
                        return 32 * pos + nbits + 1;
                }
                return 32 * pos + 1;
            }
        }
        return 0;
    }

    uint64_t GetLow64() const
    {
        static_assert(WIDTH >= 2, "Assertion WIDTH >= 2 failed (WIDTH = BITS / 32). BITS is a template parameter.");
        return pn[0] | (uint64_t)pn[1] << 32;
    }
};

/** 256-bit unsigned big integer. */
class arith_uint256 : public base_uint<256> {
public:
    arith_uint256() = default;
    arith_uint256(const base_uint<256>& b) : base_uint<256>(b) {}
    arith_uint256(uint64_t b) : base_uint<256>(b) {}
};

#endif // BITCOIN_ARITH_UINT256_H
