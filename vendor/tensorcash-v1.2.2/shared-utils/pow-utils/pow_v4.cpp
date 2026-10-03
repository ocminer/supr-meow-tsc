// Proof v4 fork-contract primitives (sampler-independent part), see pow_v4.h.
//
// Must remain semantically identical to shared-utils/pow-utils/pow_v4.py; the
// golden vectors in shared-utils/pow-utils/tests/vectors/v4_vectors.json are
// the contract (src/test/pow_v4_tests.cpp carries literal copies).

#include <verification/pow_v4.h>

#include <verification/pow_v3.h>

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace pow_v4 {

namespace {

// --- bounded validating JSON parser for the v4 carrier ---------------------
// A faithful derivative of pow_v3.cpp's file-local CarrierParser (which is
// deliberately not exported: pow_v3.{h,cpp} are vendored byte-identical from
// shared-utils and are not edited for v4). Same acceptance as Python
// json.loads (strict UTF-8, \uXXXX escapes incl. surrogate pairs, strict
// number grammar, NaN/Infinity literals, no raw control chars in strings, no
// trailing garbage) PLUS the identical consensus bounds from pow_v3.h:
// duplicate object keys reject (any level), any value deeper than
// pow_v3::EXTRA_FLAGS_MAX_DEPTH rejects (top-level value = depth 1), input
// longer than pow_v3::EXTRA_FLAGS_MAX_BYTES rejects. The only difference is
// WHAT is captured: the raw token of the TOP-LEVEL "v4" member. Nothing else
// is materialised except object keys (for duplicate detection). Never throws;
// every violation is "no v4 declaration" (ABSENT).

bool is_json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

size_t skip_ws(const std::string& s, size_t i) {
    while (i < s.size() && is_json_ws(s[i])) ++i;
    return i;
}

bool is_blank(const std::string& s) {
    return s.find_first_not_of(" \t\r\n") == std::string::npos;
}

// Strict UTF-8 validation (rejects overlongs, surrogates, > U+10FFFF),
// equivalent to Python bytes.decode("utf-8", errors="strict").
bool is_valid_utf8(const std::string& s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) { ++i; continue; }
        int len;
        uint32_t cp;
        if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else return false;
        if (i + len > n) return false;
        for (int k = 1; k < len; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (len == 2 && cp < 0x80) return false;             // overlong
        if (len == 3 && cp < 0x800) return false;            // overlong
        if (len == 4 && cp < 0x10000) return false;          // overlong
        if (cp >= 0xD800 && cp <= 0xDFFF) return false;      // surrogate
        if (cp > 0x10FFFF) return false;                     // out of range
        i += len;
    }
    return true;
}

class V4CarrierParser {
public:
    explicit V4CarrierParser(const std::string& s) : s_(s) {}

    // Full whole-input parse; ABSENT on any parse failure or when no
    // top-level "v4" member exists, CLAIMED iff its raw token is exactly "1",
    // MALFORMED for any other present value.
    V4FlagState extract_state() {
        i_ = skip_ws(s_, 0);
        if (i_ >= s_.size() || s_[i_] != '{') return V4FlagState::ABSENT;  // non-object top
        if (!parse_object(/*depth=*/1, /*top=*/true)) return V4FlagState::ABSENT;
        i_ = skip_ws(s_, i_);
        if (i_ != s_.size()) return V4FlagState::ABSENT;  // trailing garbage
        if (!v4_seen_) return V4FlagState::ABSENT;
        return v4_one_ ? V4FlagState::CLAIMED : V4FlagState::MALFORMED;
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    bool v4_seen_ = false;
    bool v4_one_ = false;  // top-level v4 captured as the exact token "1"

    bool parse_value(int depth, bool top) {
        // depth semantics mirror pow_v3._depth_of: this VALUE sits at
        // `depth`; any value beyond the bound rejects (an empty container AT
        // the bound is still fine because no child value exists).
        if (depth > pow_v3::EXTRA_FLAGS_MAX_DEPTH) return false;
        i_ = skip_ws(s_, i_);
        if (i_ >= s_.size()) return false;
        switch (s_[i_]) {
            case '{': return parse_object(depth, top);
            case '[': return parse_array(depth);
            case '"': return parse_string(nullptr);
            case 't': return parse_literal("true");
            case 'f': return parse_literal("false");
            case 'n': return parse_literal("null");
            case 'N': return parse_literal("NaN");        // json.loads default
            case 'I': return parse_literal("Infinity");   // json.loads default
            default:  return parse_number();
        }
    }

    bool parse_object(int depth, bool top) {
        // s_[i_] == '{'
        ++i_;
        std::vector<std::string> keys;  // duplicate keys reject (any level)
        i_ = skip_ws(s_, i_);
        if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
        while (true) {
            i_ = skip_ws(s_, i_);
            std::string key;
            if (i_ >= s_.size() || s_[i_] != '"' || !parse_string(&key))
                return false;
            for (const auto& k : keys)
                if (k == key) return false;  // duplicate (unescaped compare)
            keys.push_back(key);
            i_ = skip_ws(s_, i_);
            if (i_ >= s_.size() || s_[i_] != ':') return false;
            ++i_;
            i_ = skip_ws(s_, i_);
            if (top && key == "v4") {
                // v4 flag: exact integer 1 only. Capture the raw value token
                // and compare to "1"; "1.0", "1e0", "true", "\"1\"", 0, 2,
                // -1, null, [1], {} all differ. Mirrors pow_v4.extract_v4_flag.
                const size_t vstart = i_;
                v4_seen_ = true;
                if (!parse_value(depth + 1, /*top=*/false)) return false;
                v4_one_ = (s_.substr(vstart, i_ - vstart) == "1");
            } else {
                if (!parse_value(depth + 1, /*top=*/false)) return false;
            }
            i_ = skip_ws(s_, i_);
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == '}') { ++i_; return true; }
            return false;
        }
    }

    bool parse_array(int depth) {
        // s_[i_] == '['
        ++i_;
        i_ = skip_ws(s_, i_);
        if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
        while (true) {
            if (!parse_value(depth + 1, /*top=*/false)) return false;
            i_ = skip_ws(s_, i_);
            if (i_ >= s_.size()) return false;
            if (s_[i_] == ',') { ++i_; i_ = skip_ws(s_, i_); continue; }
            if (s_[i_] == ']') { ++i_; return true; }
            return false;
        }
    }

    bool parse_literal(const char* lit) {
        size_t len = std::strlen(lit);
        if (s_.compare(i_, len, lit) != 0) return false;
        i_ += len;
        return true;
    }

    // JSON string with full unescaping into *out (when non-null): needed for
    // duplicate-key comparison ("a" duplicates "a", as in Python).
    bool parse_string(std::string* out) {
        // s_[i_] == '"'
        ++i_;
        while (i_ < s_.size()) {
            unsigned char c = static_cast<unsigned char>(s_[i_]);
            if (c == '"') { ++i_; return true; }
            if (c < 0x20) return false;  // raw control char (json strict)
            if (c != '\\') {
                if (out) out->push_back(static_cast<char>(c));
                ++i_;
                continue;
            }
            // escape sequence
            ++i_;
            if (i_ >= s_.size()) return false;
            char e = s_[i_];
            ++i_;
            switch (e) {
                case '"': case '\\': case '/':
                    if (out) out->push_back(e);
                    break;
                case 'b': if (out) out->push_back('\b'); break;
                case 'f': if (out) out->push_back('\f'); break;
                case 'n': if (out) out->push_back('\n'); break;
                case 'r': if (out) out->push_back('\r'); break;
                case 't': if (out) out->push_back('\t'); break;
                case 'u': {
                    uint32_t cp;
                    if (!parse_u16_escape(&cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() &&
                        s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        // try surrogate pair
                        size_t save = i_;
                        i_ += 2;
                        uint32_t lo;
                        if (parse_u16_escape(&lo) && lo >= 0xDC00 &&
                            lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            i_ = save;  // lone high surrogate, keep as-is
                        }
                    }
                    if (out) append_utf8(*out, cp);
                    break;
                }
                default:
                    return false;
            }
        }
        return false;  // unterminated
    }

    bool parse_u16_escape(uint32_t* cp) {
        // i_ points at the first of 4 hex digits (after "\u")
        if (i_ + 4 > s_.size()) return false;
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s_[i_ + k];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return false;
        }
        i_ += 4;
        *cp = v;
        return true;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    // Strict json.loads number grammar: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
    // plus the -Infinity literal (leading '-' path only reaches here).
    bool parse_number() {
        size_t start = i_;
        if (i_ < s_.size() && s_[i_] == '-') {
            ++i_;
            if (i_ < s_.size() && s_[i_] == 'I')
                return parse_literal("Infinity");
        }
        if (i_ >= s_.size()) return false;
        if (s_[i_] == '0') {
            ++i_;
        } else if (s_[i_] >= '1' && s_[i_] <= '9') {
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        } else {
            return false;
        }
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return false;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (i_ >= s_.size() || s_[i_] < '0' || s_[i_] > '9') return false;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
        }
        return i_ > start;
    }
};

}  // namespace

// ------------------------------------------------------------------------- //
// Strict v4 carrier
// ------------------------------------------------------------------------- //

V4FlagState extract_v4_flag_state(const std::string& extra_flags) {
    // Same parser bounds and pre-checks as pow_v3::extract_stepbind_flag_state;
    // every violation is ABSENT, never a throw.
    if (extra_flags.size() > pow_v3::EXTRA_FLAGS_MAX_BYTES) return V4FlagState::ABSENT;
    if (is_blank(extra_flags)) return V4FlagState::ABSENT;
    if (!is_valid_utf8(extra_flags)) return V4FlagState::ABSENT;
    return V4CarrierParser(extra_flags).extract_state();
}

bool extract_v4_flag(const std::string& extra_flags) {
    return extract_v4_flag_state(extra_flags) == V4FlagState::CLAIMED;
}

// ------------------------------------------------------------------------- //
// Proof mode
// ------------------------------------------------------------------------- //

const char* proof_mode_name(ProofMode mode) {
    switch (mode) {
        case ProofMode::Legacy: return "legacy";
        case ProofMode::V3: return "v3";
        case ProofMode::V4: return "v4";
        case ProofMode::Invalid: return "invalid";
    }
    return "invalid";
}

ProofMode decide_proof_mode(uint8_t version, V4FlagState flag) {
    if (version < pow_v3::V3_PROOF_VERSION) {
        return flag == V4FlagState::ABSENT ? ProofMode::Legacy : ProofMode::Invalid;
    }
    if (version == pow_v3::V3_PROOF_VERSION) {
        return flag == V4FlagState::ABSENT ? ProofMode::V3 : ProofMode::Invalid;
    }
    if (version == V4_PROOF_VERSION) {
        return flag == V4FlagState::CLAIMED ? ProofMode::V4 : ProofMode::Invalid;
    }
    return ProofMode::Invalid;  // version >= 5: unsupported future claim
}

ProofMode decide_proof_mode(uint8_t version, const std::string& extra_flags) {
    return decide_proof_mode(version, extract_v4_flag_state(extra_flags));
}

ProofMode effective_proof_mode(uint8_t version, const std::string& extra_flags,
                               int height, const Consensus::Params& params) {
    if (!params.IsProofV4Active(height)) {
        // Below X (and for an unknown height) the carrier is inert: today's
        // rule, "version >= 3 verifies under the v3 rules", unchanged.
        return version < pow_v3::V3_PROOF_VERSION ? ProofMode::Legacy : ProofMode::V3;
    }
    return decide_proof_mode(version, extra_flags);
}

bool CheckProofModeAtHeight(ProofMode mode, int height,
                            const Consensus::Params& params, std::string& error) {
    switch (mode) {
        case ProofMode::Invalid:
            error = "proof v4: invalid proof mode (proof.version and the strict "
                    "top-level \"v4\" carrier contradict each other, or the version "
                    "is unsupported)";
            return false;
        case ProofMode::Legacy:
            if (params.IsProofV4Active(height)) {
                error = "proof v4: legacy proof version (< 3) not allowed at height " +
                        std::to_string(height) + " (at/after ProofV4Height " +
                        std::to_string(params.ProofV4Height) + ")";
                return false;
            }
            return true;
        case ProofMode::V3:
            if (params.IsV3Sunset(height)) {
                error = "proof v4: v3 proof rejected at height " + std::to_string(height) +
                        " (at/after V3SunsetHeight " + std::to_string(params.V3SunsetHeight) + ")";
                return false;
            }
            return true;
        case ProofMode::V4:
            // Below X: "validated under v3 rules; the v4 claim is inert".
            // The claim is not a rejection reason; effective_proof_mode maps
            // it to V3 and the verifier never installs v4 root semantics
            // below X. At/after X the mode is legal.
            return true;
    }
    error = "proof v4: unknown proof mode";
    return false;
}

// ------------------------------------------------------------------------- //
// v4 root
// ------------------------------------------------------------------------- //

const std::array<uint8_t, ROOT_BYTES>& sampler_domain_id() {
    static const std::array<uint8_t, ROOT_BYTES> id = [] {
        std::vector<uint8_t> msg;
        msg.insert(msg.end(), SAMPLER_DOMAIN_PREFIX,
                   SAMPLER_DOMAIN_PREFIX + std::strlen(SAMPLER_DOMAIN_PREFIX));
        msg.insert(msg.end(), SAMPLER_MODE_SUFFIX.begin(), SAMPLER_MODE_SUFFIX.end());
        return pow_v3::step_digest(msg);  // single SHA-256
    }();
    return id;
}

std::vector<uint8_t> admit_message_v4(const std::vector<uint8_t>& msg_w0,
                                      const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                                      const std::string& model_identifier,
                                      const std::array<uint8_t, ROOT_BYTES>& admission_nonce) {
    if (model_identifier.size() > 0xFFFF) {
        throw std::invalid_argument("model_identifier too long for u16le length prefix");
    }
    const std::array<uint8_t, ROOT_BYTES>& s4 = sampler_domain_id();
    std::vector<uint8_t> msg;
    msg.reserve(TAG_ADMIT_V4_LEN + msg_w0.size() + ROOT_BYTES + 2 +
                model_identifier.size() + ROOT_BYTES + ROOT_BYTES);
    msg.insert(msg.end(), TAG_ADMIT_V4, TAG_ADMIT_V4 + TAG_ADMIT_V4_LEN);
    msg.insert(msg.end(), msg_w0.begin(), msg_w0.end());
    msg.insert(msg.end(), prompt_commitment_digest.begin(), prompt_commitment_digest.end());
    const uint16_t mid_len = static_cast<uint16_t>(model_identifier.size());
    msg.push_back(static_cast<uint8_t>(mid_len & 0xFF));
    msg.push_back(static_cast<uint8_t>((mid_len >> 8) & 0xFF));
    msg.insert(msg.end(), model_identifier.begin(), model_identifier.end());
    msg.insert(msg.end(), admission_nonce.begin(), admission_nonce.end());
    msg.insert(msg.end(), s4.begin(), s4.end());
    return msg;
}

std::array<uint8_t, ROOT_BYTES> admit_root_v4(const std::vector<uint8_t>& admit_message) {
    // Same Argon2id profile and salt as the v3 admission puzzle; the domain
    // separation is TAG_ADMIT_V4 inside the message. Throws std::runtime_error
    // without libargon2.
    return pow_v3::argon2id_digest(admit_message);
}

std::array<uint8_t, ROOT_BYTES> step_root_v4(const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                                             const std::array<uint8_t, ROOT_BYTES>& admission_nonce,
                                             const std::array<uint8_t, ROOT_BYTES>& admit_root) {
    const std::array<uint8_t, ROOT_BYTES>& s4 = sampler_domain_id();
    std::vector<uint8_t> msg;
    msg.reserve(TAG_STEP_V4_LEN + 4 * ROOT_BYTES);
    msg.insert(msg.end(), TAG_STEP_V4, TAG_STEP_V4 + TAG_STEP_V4_LEN);
    msg.insert(msg.end(), prompt_commitment_digest.begin(), prompt_commitment_digest.end());
    msg.insert(msg.end(), admission_nonce.begin(), admission_nonce.end());
    msg.insert(msg.end(), admit_root.begin(), admit_root.end());
    msg.insert(msg.end(), s4.begin(), s4.end());
    return pow_v3::step_digest(msg);  // single SHA-256
}

V4Root derive_v4_root(const std::vector<uint8_t>& msg_w0,
                      const std::array<uint8_t, ROOT_BYTES>& prompt_commitment_digest,
                      const std::string& model_identifier,
                      const std::array<uint8_t, ROOT_BYTES>& admission_nonce) {
    V4Root root;
    root.A = admit_root_v4(admit_message_v4(msg_w0, prompt_commitment_digest,
                                            model_identifier, admission_nonce));
    root.E4 = step_root_v4(prompt_commitment_digest, admission_nonce, root.A);
    return root;
}

bool V4TierAdmits(pow_v3::Tier tier,
                  const std::array<uint8_t, ROOT_BYTES>& admit_root,
                  const std::array<uint8_t, ROOT_BYTES>& admission_target_le) {
    switch (tier) {
        case pow_v3::Tier::Invalid:
            return false;
        case pow_v3::Tier::AdmissionRequired:
            // The existing strict little-endian comparison, unchanged.
            return pow_v3::admission_valid(admit_root, admission_target_le);
        case pow_v3::Tier::Free:
            return true;
    }
    return false;
}

// ------------------------------------------------------------------------- //
// Price
// ------------------------------------------------------------------------- //

uint64_t InterpolatePriceFloor(uint64_t D, uint64_t S, const PriceAnchor& lo, const PriceAnchor& hi) {
    // (d - d_lo) / (d_hi - d_lo)
    //   = ((D k_lo - n_lo S) / (S k_lo)) / ((n_hi k_lo - n_lo k_hi) / (k_lo k_hi))
    //   = (D k_lo - n_lo S) k_hi / (S (n_hi k_lo - n_lo k_hi))
    // Inside the segment D * k_lo >= n_lo * S and the anchors are strictly
    // increasing, so both sides are non-negative and floor is the plain
    // unsigned division. Bounds: see the overflow note in pow_v4.h.
    const uint64_t numerator = (hi.P - lo.P) * (D * lo.d_den - lo.d_num * S) * hi.d_den;
    const uint64_t denominator = S * (hi.d_num * lo.d_den - lo.d_num * hi.d_den);
    return lo.P + numerator / denominator;
}

ProofPrice V4ProfilePrice(const CreditMetrics& credits) {
    // Evidence shape: the validated vector of exactly PRICE_W per-step credit
    // units, each within the per-step cap, with a consistent total. Anything
    // else is not priceable and fails closed (plan section 5: prices come
    // from the validated credit vector only).
    if (!credits.valid) return PRICE_FAIL_CLOSED;
    if (credits.step_units.size() != PRICE_W) return PRICE_FAIL_CLOSED;
    uint64_t total = 0;
    for (const uint64_t units : credits.step_units) {
        if (units > pow_v3::B_STEP_MAX_UNITS) return PRICE_FAIL_CLOSED;
        total += units;  // <= 2^23, no wrap
    }
    if (total != credits.total_units) return PRICE_FAIL_CLOSED;
    if (total == 0) {
        // Cap before any division: the tier gate rejects B < B_FLOOR anyway.
        return ProofPrice{PRICE_MAX_P, PRICE_Q};
    }

    // D = max(0, max_t (t * B - W * C_t)), S = W * B, with the prefix sums
    // the absolute envelopes read captured on the way.
    const uint64_t S = static_cast<uint64_t>(PRICE_W) * total;
    uint64_t D = 0;
    uint64_t prefix = 0;
    std::array<uint64_t, PRICE_ABS_ENVELOPES.size()> prefix_at{};
    for (std::size_t i = 0; i < PRICE_W; ++i) {
        prefix += credits.step_units[i];
        const uint64_t t = static_cast<uint64_t>(i) + 1;
        for (std::size_t e = 0; e < PRICE_ABS_ENVELOPES.size(); ++e) {
            if (t == PRICE_ABS_ENVELOPES[e].prefix_steps) prefix_at[e] = prefix;
        }
        const uint64_t t_total = t * total;
        const uint64_t w_prefix = static_cast<uint64_t>(PRICE_W) * prefix;
        if (t_total > w_prefix && t_total - w_prefix > D) D = t_total - w_prefix;
    }

    // Relative envelope: cross-multiplied anchor comparisons (d <= n/k iff
    // D * k <= n * S), floor interpolation inside a segment, the last
    // anchor's price beyond it.
    uint64_t p_relative = PRICE_ANCHORS.back().P;
    if (D * PRICE_ANCHORS.front().d_den <= PRICE_ANCHORS.front().d_num * S) {
        p_relative = PRICE_ANCHORS.front().P;
    } else {
        for (std::size_t a = 0; a + 1 < PRICE_ANCHORS.size(); ++a) {
            const PriceAnchor& lo = PRICE_ANCHORS[a];
            const PriceAnchor& hi = PRICE_ANCHORS[a + 1];
            if (D * hi.d_den <= hi.d_num * S) {
                p_relative = InterpolatePriceFloor(D, S, lo, hi);
                break;
            }
        }
    }

    // Absolute envelopes on the prefix credit before the tail.
    uint64_t p = p_relative;
    for (std::size_t e = 0; e < PRICE_ABS_ENVELOPES.size(); ++e) {
        const uint64_t threshold = PRICE_ABS_ENVELOPES[e].threshold_units;
        const uint64_t shortfall = prefix_at[e] < threshold ? threshold - prefix_at[e] : 0;
        const uint64_t p_abs = PRICE_MIN_P + (PRICE_ABS_SLOPE_P * shortfall) / threshold;
        if (p_abs > p) p = p_abs;
    }
    if (p < PRICE_MIN_P) p = PRICE_MIN_P;
    if (p > PRICE_MAX_P) p = PRICE_MAX_P;
    return ProofPrice{p, PRICE_Q};
}

uint64_t V4TailFactor(const CreditMetrics& credits) {
    // Same evidence shape as V4ProfilePrice; anything else is 0 (no factor).
    if (!credits.valid) return 0;
    if (credits.step_units.size() != PRICE_W) return 0;
    uint64_t total = 0;
    for (const uint64_t units : credits.step_units) {
        if (units > pow_v3::B_STEP_MAX_UNITS) return 0;
        total += units;
    }
    if (total != credits.total_units) return 0;

    uint64_t p_late = PRICE_MIN_P;
    for (const PriceTailEnvelope& envelope : PRICE_TAIL_ENVELOPES) {
        uint64_t b_tail = 0;  // <= 64 * 2^15 = 2^21
        for (std::size_t i = PRICE_W - envelope.tail_steps; i < PRICE_W; ++i) {
            b_tail += credits.step_units[i];
        }
        const uint64_t threshold = envelope.threshold_units;
        const uint64_t excess = b_tail > threshold ? b_tail - threshold : 0;
        const uint64_t p_tail = PRICE_MIN_P + (PRICE_TAIL_SLOPE_P * excess) / threshold;
        if (p_tail > p_late) p_late = p_tail;
    }
    if (p_late > PRICE_TAIL_MAX_P) p_late = PRICE_TAIL_MAX_P;
    return p_late;
}

ProofPrice V4TotalPrice(const CreditMetrics& credits) {
    const ProofPrice profile = V4ProfilePrice(credits);
    if (profile == PRICE_FAIL_CLOSED) return PRICE_FAIL_CLOSED;
    const uint64_t p_late = V4TailFactor(credits);
    if (p_late == 0) return PRICE_FAIL_CLOSED;
    // <= 3000 * 2000 = 6,000,000, then floored to the thousandth.
    return ProofPrice{(profile.P * p_late) / PRICE_Q, PRICE_Q};
}

// ------------------------------------------------------------------------- //
// Structure price (near-pin count + low state). PROVISIONAL COEFFICIENTS;
// the rule is documented in pow_v4.h.
// ------------------------------------------------------------------------- //

bool V4IsNearPinned(uint64_t p_max_lower_q63) {
    // 500 * 2^63 does not fit in uint64_t, so both sides are formed in 128
    // bits. A mass above 2^63 is not a valid Q63 mass: never counted.
    if (p_max_lower_q63 > pow_v3::BCRED_Q_ONE) return false;
    return static_cast<unsigned __int128>(PIN_NEAR_NUM) * p_max_lower_q63 >
           static_cast<unsigned __int128>(PIN_NEAR_DEN) *
               static_cast<unsigned __int128>(pow_v3::BCRED_Q_ONE);
}

bool V4NearPinCount(const std::vector<uint64_t>& p_max_lower_q63, uint64_t& count) {
    if (p_max_lower_q63.size() != PRICE_W) return false;
    uint64_t n = 0;
    for (const uint64_t mass : p_max_lower_q63) {
        if (mass > pow_v3::BCRED_Q_ONE) return false;
        if (V4IsNearPinned(mass)) ++n;
    }
    count = n;
    return true;
}

uint64_t V4PinPrice(uint64_t near_pin_count) {
    // A count that is not a legal count over the 256-position window has no
    // price; a count at or above the hard boundary is REJECTED (see
    // V4StructureRejects), never priced.
    if (near_pin_count > static_cast<uint64_t>(PRICE_W)) return 0;
    if (near_pin_count >= PIN_REJECT_COUNT) return 0;
    const uint64_t excess =
        near_pin_count > PIN_FREE_COUNT ? near_pin_count - PIN_FREE_COUNT : 0;
    // excess <= PIN_RAMP_DEN == 46, so the numerator is below 2^22 and the
    // result is at most exactly PRICE_Q + PRICE_Q == PRICE_STRUCTURE_MAX_P.
    uint64_t p = PRICE_Q + (PRICE_Q * excess * excess) / (PIN_RAMP_DEN * PIN_RAMP_DEN);
    if (p < PRICE_MIN_P) p = PRICE_MIN_P;
    if (p > PRICE_STRUCTURE_MAX_P) p = PRICE_STRUCTURE_MAX_P;
    return p;
}

uint64_t V4StatePrice(const StructureMetrics& structure) {
    // Evidence shape: a completed anti-parrot aggregation over a valid
    // window. Zero or out-of-range totals are NOT a free price (they would
    // be the cheapest possible outcome); they are the 0 sentinel and the
    // caller fails closed.
    if (!structure.valid) return 0;
    if (structure.bcap_units == 0 || structure.bcap_units > STATE_BCAP_MAX_UNITS) return 0;
    // D = bcap^2 * 2^63 <= 2^99.
    const unsigned __int128 d =
        static_cast<unsigned __int128>(structure.bcap_units) * structure.bcap_units *
        (static_cast<unsigned __int128>(1) << 63);
    if (structure.collision_u == 0 || structure.collision_u > d) return 0;
    if (structure.collision_id == 0 || structure.collision_id > d) return 0;
    // N_u = D / collision_u and N_id = D / collision_id, so
    // onset / N = onset * collision / D. The min is the JOINT signal.
    // Largest intermediate: 1000 * 40 * 2^99 < 2^115.
    const unsigned __int128 by_u =
        static_cast<unsigned __int128>(STATE_U_ONSET) * structure.collision_u;
    const unsigned __int128 by_id =
        static_cast<unsigned __int128>(STATE_ID_ONSET) * structure.collision_id;
    const unsigned __int128 numerator = by_u < by_id ? by_u : by_id;
    const unsigned __int128 scaled =
        (static_cast<unsigned __int128>(PRICE_Q) * numerator) / d;
    if (scaled <= static_cast<unsigned __int128>(PRICE_MIN_P)) return PRICE_MIN_P;
    if (scaled >= static_cast<unsigned __int128>(PRICE_STRUCTURE_MAX_P)) {
        return PRICE_STRUCTURE_MAX_P;
    }
    return static_cast<uint64_t>(scaled);
}

uint64_t V4StructurePrice(const StructureMetrics& structure) {
    if (!structure.valid) return 0;
    const uint64_t p_pin = V4PinPrice(structure.near_pin_count);
    if (p_pin == 0) return 0;  // a rejecting or impossible count is never priced
    const uint64_t p_state = V4StatePrice(structure);
    if (p_state == 0) return 0;
    // max, NOT the product: the two indicators may describe the same
    // low-diversity structure, so the new factor is bounded by 2x.
    return p_pin > p_state ? p_pin : p_state;
}

bool V4StructureRejects(const StructureMetrics& structure, std::string& reason) {
    if (structure.near_pin_count >= PIN_REJECT_COUNT) {
        reason = "proof v4 structure rejected (near-pinned-count-at-least-" +
                 std::to_string(PIN_REJECT_COUNT) + "): " +
                 std::to_string(structure.near_pin_count) + " of " +
                 std::to_string(static_cast<uint64_t>(PRICE_W)) +
                 " positions have a most-probable-token conservative lower mass above " +
                 std::to_string(PIN_NEAR_DEN) + "/" + std::to_string(PIN_NEAR_NUM);
        return true;
    }
    return false;
}

ProofPrice V4TotalPrice(const CreditMetrics& credits, const StructureMetrics& structure) {
    const ProofPrice base = V4TotalPrice(credits);
    if (base == PRICE_FAIL_CLOSED) return PRICE_FAIL_CLOSED;
    const uint64_t p_structure = V4StructurePrice(structure);
    if (p_structure == 0) return PRICE_FAIL_CLOSED;
    // <= 6000 * 2000 = 12,000,000 < 2^24, then floored to the thousandth.
    return ProofPrice{(base.P * p_structure) / PRICE_Q, PRICE_Q};
}

// ------------------------------------------------------------------------- //
// collision_race (documented in pow_v4.h). Compiled with -ffp-contract=off.
// ------------------------------------------------------------------------- //

uint64_t RaceAgreementQ63(const std::vector<uint32_t>& ids_a, const std::vector<float>& p_a,
                          const std::vector<uint32_t>& ids_b, const std::vector<float>& p_b) {
    if (ids_a.size() != p_a.size() || ids_b.size() != p_b.size()) {
        throw std::invalid_argument("collision_race: ids/probabilities length mismatch");
    }
    const auto check_row = [](const std::vector<uint32_t>& ids, const std::vector<float>& p) {
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i > 0 && ids[i] <= ids[i - 1]) {
                throw std::invalid_argument("collision_race: ids must be ascending and unique");
            }
            if (!(p[i] > 0.0f) || !std::isfinite(p[i])) {
                throw std::invalid_argument("collision_race: probabilities must be finite and positive");
            }
        }
    };
    check_row(ids_a, p_a);
    check_row(ids_b, p_b);

    // The ascending union with each row's probability (0 = absent), built by
    // a two-pointer merge so the order is the id order.
    struct Entry { uint32_t id; double pa; double pb; };
    std::vector<Entry> u;
    u.reserve(ids_a.size() + ids_b.size());
    std::size_t ia = 0, ib = 0;
    while (ia < ids_a.size() || ib < ids_b.size()) {
        if (ib >= ids_b.size() || (ia < ids_a.size() && ids_a[ia] < ids_b[ib])) {
            u.push_back({ids_a[ia], static_cast<double>(p_a[ia]), 0.0});
            ++ia;
        } else if (ia >= ids_a.size() || ids_b[ib] < ids_a[ia]) {
            u.push_back({ids_b[ib], 0.0, static_cast<double>(p_b[ib])});
            ++ib;
        } else {
            u.push_back({ids_a[ia], static_cast<double>(p_a[ia]), static_cast<double>(p_b[ib])});
            ++ia;
            ++ib;
        }
    }

    unsigned __int128 total = 0;
    for (const Entry& i : u) {
        if (!(i.pa > 0.0 && i.pb > 0.0)) continue;
        double den = 0.0;
        for (const Entry& j : u) {
            const double ra = j.pa > 0.0 ? j.pa / i.pa : 0.0;
            const double rb = j.pb > 0.0 ? j.pb / i.pb : 0.0;
            den = den + (ra >= rb ? ra : rb);
        }
        const double den_up = den * RACE_COLLISION_DEN_UP;
        total += pow_v3::f64_to_q63_floor(1.0 / den_up);
    }
    const unsigned __int128 one = static_cast<unsigned __int128>(pow_v3::BCRED_Q_ONE);
    return static_cast<uint64_t>(total > one ? one : total);
}

unsigned __int128 RaceCollisionTotal(const std::vector<uint64_t>& credit_mass_q63,
                                     const std::vector<std::vector<uint32_t>>& cdf_ids,
                                     const std::vector<std::vector<float>>& cdf_p) {
    const std::size_t n = credit_mass_q63.size();
    if (cdf_ids.size() != n || cdf_p.size() != n) {
        throw std::invalid_argument("collision_race: per-step input length mismatch");
    }
    // The SAME weights compute_anti_parrot_metrics derives: per-step credit
    // units capped at AP_R_STEP_CAP; only credited steps (w > 0) take part.
    std::vector<std::size_t> credited;
    std::vector<unsigned __int128> w;
    for (std::size_t s = 0; s < n; ++s) {
        const uint64_t units = pow_v3::credit_units_for_step(credit_mass_q63[s]);
        const uint64_t weight = units < pow_v3::AP_R_STEP_CAP ? units : pow_v3::AP_R_STEP_CAP;
        if (weight > 0) {
            credited.push_back(s);
            w.push_back(weight);
        }
    }
    const std::size_t K = credited.size();
    // S_race is symmetric: evaluate each unordered pair once and count it
    // for both orders (the ordered-pair sum of the u-collision), diagonal
    // forced to 2^63.
    const unsigned __int128 Q63 = static_cast<unsigned __int128>(pow_v3::BCRED_Q_ONE);
    unsigned __int128 total = 0;
    for (std::size_t x = 0; x < K; ++x) {
        total += w[x] * w[x] * Q63;
        for (std::size_t y = x + 1; y < K; ++y) {
            const unsigned __int128 s = RaceAgreementQ63(cdf_ids[credited[x]], cdf_p[credited[x]],
                                                         cdf_ids[credited[y]], cdf_p[credited[y]]);
            total += 2 * w[x] * w[y] * s;
        }
    }
    return total;
}

void ApplyRaceCollision(pow_v3::AntiParrotMetrics& out, unsigned __int128 total_collision_race) {
    // Mirrors the derived u-arm fields of pow_v3::compute_anti_parrot_metrics
    // (n_states_eff_u, the joint HARD / GREY state tests and the kappa
    // deficit telemetry) with the u total replaced.
    out.total_collision_u = total_collision_race;
    const unsigned __int128 Q63 = static_cast<unsigned __int128>(pow_v3::BCRED_Q_ONE);
    const unsigned __int128 wsq_q63 =
        static_cast<unsigned __int128>(out.bcap) * out.bcap * Q63;
    out.n_states_eff_u = out.total_collision_u
        ? static_cast<uint64_t>(wsq_q63 / out.total_collision_u) : 0;
    const bool kappa_joint =
        pow_v3::anti_parrot_kappa_lt(out.self_collision, out.total_collision_u,
                                     pow_v3::AP_KU_HARD_NUM, pow_v3::AP_KU_HARD_DEN) &&
        pow_v3::anti_parrot_kappa_lt(out.self_collision, out.total_collision_id,
                                     pow_v3::AP_KID_HARD_NUM, pow_v3::AP_KID_HARD_DEN);
    const bool nstates_joint =
        pow_v3::anti_parrot_nstates_lt(out.bcap, out.total_collision_u, pow_v3::AP_NSTATES_U_HARD) &&
        pow_v3::anti_parrot_nstates_lt(out.bcap, out.total_collision_id, pow_v3::AP_NSTATES_ID_HARD);
    out.state_hard = kappa_joint || nstates_joint;
    const bool kappa_grey =
        pow_v3::anti_parrot_kappa_lt(out.self_collision, out.total_collision_u,
                                     pow_v3::AP_KU_GREY_NUM, pow_v3::AP_KU_GREY_DEN) &&
        pow_v3::anti_parrot_kappa_lt(out.self_collision, out.total_collision_id,
                                     pow_v3::AP_KID_GREY_NUM, pow_v3::AP_KID_GREY_DEN);
    const bool nstates_grey =
        pow_v3::anti_parrot_nstates_lt(out.bcap, out.total_collision_u, pow_v3::AP_NSTATES_U_GREY) &&
        pow_v3::anti_parrot_nstates_lt(out.bcap, out.total_collision_id, pow_v3::AP_NSTATES_ID_GREY);
    out.state_grey = kappa_grey || nstates_grey;
    const auto kappa_units = [](unsigned __int128 self, unsigned __int128 total) -> uint64_t {
        if (total == 0) return pow_v3::AP_KAPPA_UNITS_SCALE;
        return static_cast<uint64_t>((self * static_cast<unsigned __int128>(pow_v3::AP_KAPPA_UNITS_SCALE)) / total);
    };
    const uint64_t ku_ppm = kappa_units(out.self_collision, out.total_collision_u);
    const uint64_t kid_ppm = kappa_units(out.self_collision, out.total_collision_id);
    out.state_deficit = (pow_v3::AP_KU_REF_UNITS > ku_ppm ? pow_v3::AP_KU_REF_UNITS - ku_ppm : 0) +
                        (pow_v3::AP_KID_REF_UNITS > kid_ppm ? pow_v3::AP_KID_REF_UNITS - kid_ppm : 0);
}

ProofPrice ProofPriceMultiplier(ProofMode mode, int height,
                                const Consensus::Params& params,
                                const CreditMetrics& credits,
                                const StructureMetrics& structure) {
    switch (mode) {
        case ProofMode::Legacy:
            return ProofPrice{1, 1};
        case ProofMode::V3:
            // From V3PenaltyHeight (Y) until V3SunsetHeight (Z) a v3 proof owes
            // the FULL v4 price over its own validated evidence: the same
            // profile x tail x structure product, the same fail-closed
            // sentinel when the evidence is not priceable, no floor and no
            // flat surcharge. Outside the band the existing target applies.
            if (params.IsV3Penalized(height)) {
                return V4TotalPrice(credits, structure);
            }
            return ProofPrice{1, 1};
        case ProofMode::V4:
            return V4TotalPrice(credits, structure);
        case ProofMode::Invalid:
            break;
    }
    // Fail closed: PricedTarget rejects P == 0, so no hash can meet it.
    return ProofPrice{0, 0};
}

std::optional<arith_uint256> PricedTarget(const arith_uint256& T0, uint64_t P, uint64_t Q) {
    if (P == 0) return std::nullopt;                   // undefined price
    if (P == Q) return T0;                             // M == 1: existing target
    if (T0 == 0 || Q == 0) return arith_uint256{0};    // zero target (only hash 0)

    // T0 = q * P + r, 0 <= r < P  (see the exactness proof in pow_v4.h).
    const arith_uint256 divisor{P};
    const arith_uint256 q = T0 / divisor;
    const arith_uint256 r = T0 - q * divisor;          // exact, r < P < 2^64
    const uint64_t r64 = r.GetLow64();

    // low = floor((r * Q) / P), r * Q < 2^128 exact in unsigned __int128,
    // and low < Q < 2^64.
    const unsigned __int128 rq = static_cast<unsigned __int128>(r64) * static_cast<unsigned __int128>(Q);
    const uint64_t low = static_cast<uint64_t>(rq / static_cast<unsigned __int128>(P));

    // high = q * Q, guarded: overflow iff q > (2^256 - 1) / Q.
    arith_uint256 max_value{0};
    max_value -= uint64_t{1};                          // wraps to 2^256 - 1
    const arith_uint256 multiplier{Q};
    if (q > max_value / multiplier) return std::nullopt;
    const arith_uint256 high = q * multiplier;

    // result = high + low, guarded against wrap-around.
    arith_uint256 result = high;
    result += low;
    if (result < high) return std::nullopt;
    return result;
}

bool HashMeetsTarget(const arith_uint256& hash, const arith_uint256& target) {
    return hash <= target;
}

}  // namespace pow_v4
