// V3 prompt-binding / admission helpers (TIP-0003) — see pow_v3.h.
//
// Must remain semantically identical to pow_v3.py; the golden vectors in
// tests/vectors/v3_vectors.json are the contract, exercised by
// tests/test_pow_v3_cpp.cpp (C++) and tests/test_pow_v3_vectors.py (Python).

#include "pow_v3.h"
#ifdef _MSC_VER
#include <boost/multiprecision/cpp_int.hpp>
using pow_uint128 = boost::multiprecision::uint128_t;
#else
using pow_uint128 = unsigned __int128;
#endif

#include <openssl/sha.h>
#ifdef POW_V3_HAVE_ARGON2
#include <argon2.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>

namespace pow_v3 {

// ------------------------------------------------------------------------- //
// small byte helpers (mirror pow_v3._u16le / _u32le / _tok_le_bytes)
// ------------------------------------------------------------------------- //

static void append_u16le(std::vector<uint8_t>& out, uint16_t n) {
    out.push_back(static_cast<uint8_t>(n & 0xFF));
    out.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
}

static void append_u32le(std::vector<uint8_t>& out, uint32_t n) {
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>((n >> (8 * i)) & 0xFF));
}

static void append_i64le(std::vector<uint8_t>& out, int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);  // two's complement, as struct '<q'
    for (int i = 0; i < 8; ++i)
        out.push_back(static_cast<uint8_t>((u >> (8 * i)) & 0xFF));
}

// ------------------------------------------------------------------------- //
// §3 — extra_flags carrier
// ------------------------------------------------------------------------- //

bool is_valid_admission_nonce_hex(const std::string& value) {
    if (value.size() != ADMISSION_NONCE_HEX_LEN) return false;
    for (char c : value) {
        // lowercase hex only (consensus shape rule: reject uppercase)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

namespace {

// --- minimal string-escape-aware JSON scanning (NOT a parser) --------------
// Just enough structure awareness to (a) remove/locate a top-level member and
// (b) skip balanced values, mirroring the audit-marker hand-splice idiom in
// proof_processor.cpp. Anything the scan cannot walk is treated as "not a
// JSON object" and preserved under "_diff" (same terminal behaviour as
// pow_v3.merge_extra_flags_v3 on a json.loads failure).

bool is_json_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

size_t skip_ws(const std::string& s, size_t i) {
    while (i < s.size() && is_json_ws(s[i])) ++i;
    return i;
}

// s[i] must be '"'; returns index one past the closing quote, or npos.
size_t skip_string(const std::string& s, size_t i) {
    if (i >= s.size() || s[i] != '"') return std::string::npos;
    for (++i; i < s.size(); ++i) {
        if (s[i] == '\\') { ++i; continue; }  // escape-aware: skip escaped char
        if (s[i] == '"') return i + 1;
    }
    return std::string::npos;
}

// Skip one JSON value (object/array/string/primitive) starting at i; returns
// index one past it, or npos on structural failure.
size_t skip_value(const std::string& s, size_t i) {
    i = skip_ws(s, i);
    if (i >= s.size()) return std::string::npos;
    char c = s[i];
    if (c == '"') return skip_string(s, i);
    if (c == '{' || c == '[') {
        // balanced-brace scan, string-escape aware
        char open = c, close = (c == '{') ? '}' : ']';
        int depth = 0;
        for (; i < s.size(); ++i) {
            if (s[i] == '"') {
                i = skip_string(s, i);
                if (i == std::string::npos) return std::string::npos;
                --i;  // for-loop increments
            } else if (s[i] == open) {
                ++depth;
            } else if (s[i] == close) {
                if (--depth == 0) return i + 1;
            }
        }
        return std::string::npos;
    }
    // primitive: number / true / false / null — take chars up to a delimiter
    size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           !is_json_ws(s[i]))
        ++i;
    return (i > start) ? i : std::string::npos;
}

// Unescaped span [key_start, member_end) of the top-level member named `key`
// inside object `s` (s trimmed, s.front()=='{'), plus the value span. Returns
// false when absent or when the scan fails structurally (*scan_ok=false).
bool find_top_level_member(const std::string& s, const std::string& key,
                           size_t* key_start, size_t* value_start,
                           size_t* member_end, bool* scan_ok) {
    *scan_ok = false;
    size_t i = skip_ws(s, 1);  // past '{'
    if (i < s.size() && s[i] == '}') { *scan_ok = true; return false; }
    while (i < s.size()) {
        size_t k_start = i;
        size_t k_end = skip_string(s, i);
        if (k_end == std::string::npos) return false;
        // raw key text without quotes; nonce/marker keys never need unescaping
        std::string raw_key = s.substr(k_start + 1, k_end - k_start - 2);
        i = skip_ws(s, k_end);
        if (i >= s.size() || s[i] != ':') return false;
        i = skip_ws(s, i + 1);
        size_t v_start = i;
        size_t v_end = skip_value(s, i);
        if (v_end == std::string::npos) return false;
        if (raw_key == key) {
            *key_start = k_start;
            *value_start = v_start;
            *member_end = v_end;
            *scan_ok = true;
            return true;
        }
        i = skip_ws(s, v_end);
        if (i < s.size() && s[i] == ',') { i = skip_ws(s, i + 1); continue; }
        if (i < s.size() && s[i] == '}') { *scan_ok = true; return false; }
        return false;  // trailing garbage
    }
    return false;
}

// JSON string escaping for the "_diff" fallback (mirrors json.dumps for the
// ASCII range; non-ASCII bytes pass through verbatim, which is valid
// non-canonical JSON — consensus does not enforce canonical JSON, §3).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string trim_copy(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

}  // namespace

std::string merge_extra_flags_v3(const std::string& extra_flags,
                                 const std::string& nonce_hex) {
    if (!is_valid_admission_nonce_hex(nonce_hex)) {
        throw std::invalid_argument(
            "admission_nonce_hex must be exactly 64 lowercase hex chars");
    }
    const std::string v3_member =
        "\"v3\":{\"admission_nonce\":\"" + nonce_hex + "\"}";

    std::string trimmed = trim_copy(extra_flags);
    if (trimmed.empty()) {
        return "{" + v3_member + "}";
    }
    if (trimmed.front() == '{' && trimmed.back() == '}') {
        // JSON object: remove any existing top-level "v3" member (balanced,
        // string-escape-aware scan) so re-merging is idempotent, then splice
        // the fresh member at the end — proof_processor.cpp's audit-marker
        // splice at the front therefore never collides with it.
        size_t key_start = 0, value_start = 0, member_end = 0;
        bool scan_ok = false;
        std::string body = trimmed;
        if (find_top_level_member(body, "v3", &key_start, &value_start,
                                  &member_end, &scan_ok)) {
            // drop the member plus exactly one adjacent comma
            size_t cut_begin = key_start, cut_end = member_end;
            size_t after = skip_ws(body, member_end);
            if (after < body.size() && body[after] == ',') {
                cut_end = after + 1;  // "v3":{...},  -> trailing comma
            } else {
                size_t before = key_start;
                while (before > 1 && is_json_ws(body[before - 1])) --before;
                if (before > 1 && body[before - 1] == ',')
                    cut_begin = before - 1;  // ,"v3":{...} -> leading comma
            }
            body = body.substr(0, cut_begin) + body.substr(cut_end);
            body = trim_copy(body);
        }
        if (scan_ok) {
            // splice as the last member of the (possibly now empty) object
            size_t inner_end = body.find_last_not_of(" \t\r\n", body.size() - 2);
            bool empty_object =
                (inner_end == std::string::npos || body[inner_end] == '{');
            std::string head = body.substr(0, body.size() - 1);
            head = trim_copy(head);
            std::string candidate =
                head + (empty_object ? "" : ",") + v3_member + "}";
            // Self-check (load-bearing): the balanced scan only proves the
            // blob is structurally walkable, NOT valid JSON — e.g. a legacy
            // pformat blob "{'k': 'v'}" splices fine but the strict §3
            // extractor then rejects the whole string and the nonce is
            // silently LOST. Accept the splice only if the consensus
            // extractor recovers exactly this nonce; otherwise fall through
            // to the _diff wrap, which always yields valid JSON.
            std::optional<std::string> recovered =
                extract_admission_nonce_hex(candidate);
            if (recovered && *recovered == nonce_hex) {
                return candidate;
            }
        }
        // Structurally unwalkable {...} (or splice failed the extractor
        // self-check) — fall through to the _diff wrap (pow_v3.py reaches
        // the same terminal via a json.loads failure).
    }
    // Not a JSON object: preserve the original string verbatim under "_diff"
    // (matches pow_v3.merge_extra_flags_v3 / ProofWriter's completion-id merge).
    return "{\"_diff\":\"" + json_escape(extra_flags) + "\"," + v3_member + "}";
}

namespace {

// --- bounded validating JSON parser for the §3 carrier ---------------------
// Mirror of pow_v3.extract_admission_nonce: acceptance must match Python
// json.loads (strict UTF-8, \uXXXX escapes incl. surrogate pairs, strict
// number grammar, NaN/Infinity literals, no raw control chars in strings, no
// trailing garbage) PLUS the consensus bounds: duplicate object keys reject
// (any level), any node deeper than EXTRA_FLAGS_MAX_DEPTH rejects (top-level
// value = depth 1, matching pow_v3._depth_of). Nothing is materialised except
// object keys (for duplicate detection) and the one captured nonce string.
// Never throws; every violation is "no nonce claimed".

// Strict UTF-8 validation (rejects overlongs, surrogates, > U+10FFFF) —
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

class CarrierParser {
public:
    explicit CarrierParser(const std::string& s) : s_(s) {}

    // Parse the full input; on success with a well-shaped v3.admission_nonce
    // returns the hex string, else nullopt. Never throws.
    std::optional<std::string> extract() {
        i_ = skip_ws(s_, 0);
        if (i_ >= s_.size() || s_[i_] != '{') return std::nullopt;  // non-object top
        if (!parse_object(/*depth=*/1, CAPTURE_TOP)) return std::nullopt;
        i_ = skip_ws(s_, i_);
        if (i_ != s_.size()) return std::nullopt;  // trailing garbage
        if (!nonce_ || !is_valid_admission_nonce_hex(*nonce_))
            return std::nullopt;
        return nonce_;
    }

private:
    enum Capture { CAPTURE_NONE, CAPTURE_TOP, CAPTURE_V3 };

    const std::string& s_;
    size_t i_ = 0;
    std::optional<std::string> nonce_;

    bool parse_value(int depth, Capture capture) {
        // depth semantics mirror pow_v3._depth_of: this VALUE sits at
        // `depth`; any value beyond the bound rejects (an empty container AT
        // the bound is still fine because no child value exists).
        if (depth > EXTRA_FLAGS_MAX_DEPTH) return false;
        i_ = skip_ws(s_, i_);
        if (i_ >= s_.size()) return false;
        switch (s_[i_]) {
            case '{': return parse_object(depth, capture);
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

    bool parse_object(int depth, Capture capture) {
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
            if (capture == CAPTURE_TOP && key == "v3") {
                // "v3" must itself be an object; a non-object v3 still has to
                // parse (whole-input validity), it just claims no nonce.
                Capture child = (i_ < s_.size() && s_[i_] == '{')
                                    ? CAPTURE_V3
                                    : CAPTURE_NONE;
                if (!parse_value(depth + 1, child)) return false;
            } else if (capture == CAPTURE_V3 && key == "admission_nonce") {
                if (i_ < s_.size() && s_[i_] == '"') {
                    std::string value;
                    if (!parse_string(&value)) return false;
                    nonce_ = value;  // shape-checked by extract()
                } else {
                    // non-string nonce: valid JSON, no nonce claimed
                    if (!parse_value(depth + 1, CAPTURE_NONE)) return false;
                }
            } else {
                if (!parse_value(depth + 1, CAPTURE_NONE)) return false;
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
            if (!parse_value(depth + 1, CAPTURE_NONE)) return false;
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
    // duplicate-key comparison ("a" duplicates "a", as in Python) and
    // for the captured nonce value. Lone surrogates from \uXXXX are encoded
    // as their 3-byte code-unit form (Python allows them; such a nonce can
    // never pass the hex shape check anyway).
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

std::optional<std::string> extract_admission_nonce_hex(
    const std::string& extra_flags) {
    // §3 parser bounds — mirror of pow_v3.extract_admission_nonce; every
    // violation is "no nonce claimed" (nullopt), NEVER a throw.
    if (extra_flags.size() > EXTRA_FLAGS_MAX_BYTES) return std::nullopt;
    if (trim_copy(extra_flags).empty()) return std::nullopt;
    if (!is_valid_utf8(extra_flags)) return std::nullopt;
    return CarrierParser(extra_flags).extract();
}

// ------------------------------------------------------------------------- //
// §7 — v3 step message
// ------------------------------------------------------------------------- //

std::vector<uint8_t> build_step_message(
    const std::vector<uint8_t>& header_prefix,
    const std::vector<uint8_t>& vdf,
    uint32_t tick,
    uint32_t step,
    const std::vector<int64_t>& context_tokens,
    const std::string& precision,
    const uint8_t* nonce32_or_null,
    std::size_t window_size) {
    std::vector<uint8_t> msg;
    msg.reserve(header_prefix.size() + vdf.size() + 8 + window_size * 8 +
                precision.size() + (nonce32_or_null ? ADMISSION_NONCE_BYTES : 0));
    msg.insert(msg.end(), header_prefix.begin(), header_prefix.end());
    msg.insert(msg.end(), vdf.begin(), vdf.end());
    append_u32le(msg, tick);
    append_u32le(msg, step);
    // 256-slot rolling window: LAST window_size tokens, LEFT-padded with
    // zeros, 8 bytes little-endian per token (identical to pow_utils
    // tok_le_bytes and QuickVerifier::ComputeUValue).
    std::size_t ctx_len = context_tokens.size();
    std::size_t take = (ctx_len > window_size) ? window_size : ctx_len;
    std::size_t pad = window_size - take;
    for (std::size_t i = 0; i < pad; ++i) append_i64le(msg, 0);
    for (std::size_t i = ctx_len - take; i < ctx_len; ++i)
        append_i64le(msg, context_tokens[i]);
    msg.insert(msg.end(), precision.begin(), precision.end());
    if (nonce32_or_null != nullptr) {
        msg.insert(msg.end(), nonce32_or_null,
                   nonce32_or_null + ADMISSION_NONCE_BYTES);
    }
    return msg;
}

std::array<uint8_t, 32> step_digest(const std::vector<uint8_t>& message) {
    std::array<uint8_t, 32> digest{};
    // single SHA-256 (NOT double-SHA) — mirrors pow_v3.step_u_from_message
    SHA256(message.data(), message.size(), digest.data());
    return digest;
}

double step_u_from_digest(const std::array<uint8_t, 32>& digest) {
    uint32_t le = static_cast<uint32_t>(digest[0]) |
                  (static_cast<uint32_t>(digest[1]) << 8) |
                  (static_cast<uint32_t>(digest[2]) << 16) |
                  (static_cast<uint32_t>(digest[3]) << 24);
    return static_cast<double>(le) / 4294967296.0;
}

// ------------------------------------------------------------------------- //
// §6 — Argon2id admission puzzle
// ------------------------------------------------------------------------- //

std::array<uint8_t, 32> prompt_commitment(
    const std::vector<int64_t>& prompt_tokens,
    const std::vector<uint8_t>& pad_mask) {
    // SHA256(tag | u32le(n_tokens) | prompt_tokens_i64le
    // | u32le(n_mask) | pad_mask_u8) — the prefix is hashed ONCE per window
    // here; only the 32-byte digest enters the (repeatedly rehashed) Argon2id
    // message.
    if (pad_mask.size() != prompt_tokens.size()) {
        throw std::invalid_argument(
            "pad_mask length must equal prompt_tokens length");
    }
    if (prompt_tokens.size() > 0xFFFFFFFFULL) {
        throw std::invalid_argument(
            "prompt_tokens too long for u32le length prefix");
    }
    std::vector<uint8_t> buf;
    buf.reserve(PROMPT_CTX_TAG_LEN + 4 + prompt_tokens.size() * 8 + 4 +
                pad_mask.size());
    buf.insert(buf.end(), PROMPT_CTX_TAG, PROMPT_CTX_TAG + PROMPT_CTX_TAG_LEN);
    append_u32le(buf, static_cast<uint32_t>(prompt_tokens.size()));
    for (int64_t tok : prompt_tokens) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<uint8_t>((tok >> (i * 8)) & 0xFF));
        }
    }
    append_u32le(buf, static_cast<uint32_t>(pad_mask.size()));
    for (uint8_t b : pad_mask) {
        buf.push_back(b ? 1 : 0);
    }
    return step_digest(buf);
}

std::vector<uint8_t> admission_message(const std::vector<uint8_t>& msg_w,
                                       const std::string& model_identifier,
                                       const uint8_t nonce[32],
                                       const std::array<uint8_t, 32>& prompt_commitment_digest) {
    if (model_identifier.size() > 0xFFFF) {
        throw std::invalid_argument(
            "model_identifier too long for u16le length prefix");
    }
    std::vector<uint8_t> msg;
    msg.reserve(msg_w.size() + PROMPT_COMMITMENT_BYTES + 2 +
                model_identifier.size() + ADMISSION_NONCE_BYTES);
    msg.insert(msg.end(), msg_w.begin(), msg_w.end());
    msg.insert(msg.end(), prompt_commitment_digest.begin(),
               prompt_commitment_digest.end());
    append_u16le(msg, static_cast<uint16_t>(model_identifier.size()));
    msg.insert(msg.end(), model_identifier.begin(), model_identifier.end());
    msg.insert(msg.end(), nonce, nonce + ADMISSION_NONCE_BYTES);
    return msg;
}

bool argon2_compiled() noexcept {
    // Deliberately a function in THIS translation unit (not a header
    // constexpr): POW_V3_HAVE_ARGON2 may be defined per-target, and the only
    // definition that matters is the one argon2id_digest() below was compiled
    // under. Startup capability guards must see that value, not the caller's.
#ifdef POW_V3_HAVE_ARGON2
    return true;
#else
    return false;
#endif
}

std::array<uint8_t, 32> argon2id_digest(const std::vector<uint8_t>& message) {
#ifdef POW_V3_HAVE_ARGON2
    std::array<uint8_t, 32> out{};
    int rc = argon2id_hash_raw(
        ARGON2_TIME_COST, ARGON2_MEMORY_KIB, ARGON2_LANES,
        message.data(), message.size(),
        reinterpret_cast<const uint8_t*>(ARGON2_SALT), ARGON2_SALT_LEN,
        out.data(), ARGON2_HASH_LEN);
    if (rc != ARGON2_OK) {
        throw std::runtime_error(std::string("argon2id_hash_raw failed: ") +
                                 argon2_error_message(rc));
    }
    return out;
#else
    (void)message;
    throw std::runtime_error(
        "pow_v3::argon2id_digest requires libargon2 (build with "
        "-DPOW_V3_HAVE_ARGON2 and link -largon2); this binary was built "
        "without it, so v3 admission grinding/verification is unavailable");
#endif
}

uint64_t admission_expected_tries(int64_t difficulty,
                                  uint64_t normalizer,
                                  uint64_t decode_us_at_normalizer,
                                  uint64_t elig_alpha_num,
                                  uint64_t elig_alpha_den,
                                  uint64_t argon_ref_us) {
    if (difficulty <= 0) {
        throw std::invalid_argument("difficulty must be positive");
    }
    // pow_uint128 keeps the numerator exact for any plausible chain
    // constants (defaults: 4 * 1e7 * 1e6 = 4e13, far below 2^127).
    pow_uint128 numerator = static_cast<pow_uint128>(elig_alpha_num) *
                                  decode_us_at_normalizer * normalizer;
    pow_uint128 denominator = static_cast<pow_uint128>(elig_alpha_den) *
                                    argon_ref_us *
                                    static_cast<uint64_t>(difficulty);
    if (denominator == 0) {
        throw std::invalid_argument("invalid admission constants");
    }
    pow_uint128 tries = numerator / denominator;
    if (tries < 1) tries = 1;
    if (tries > static_cast<pow_uint128>(UINT64_MAX)) {
        // Unreachable with sane chain constants; refuse rather than truncate.
        throw std::overflow_error("admission expected_tries exceeds uint64");
    }
    return static_cast<uint64_t>(tries);
}

std::array<uint8_t, 32> admission_target_le(int64_t difficulty,
                                            uint64_t normalizer,
                                            uint64_t decode_us_at_normalizer,
                                            uint64_t elig_alpha_num,
                                            uint64_t elig_alpha_den,
                                            uint64_t argon_ref_us) {
    const uint64_t tries = admission_expected_tries(
        difficulty, normalizer, decode_us_at_normalizer, elig_alpha_num,
        elig_alpha_den, argon_ref_us);
    // (2^256 - 1) / tries via 256-bit-by-64-bit long division over four
    // uint64 limbs, most-significant first. Every dividend limb is
    // 0xFFFF...F; the running remainder rides in the top half of a 128-bit
    // intermediate.
    uint64_t quot[4];  // quot[0] = most significant limb
    pow_uint128 rem = 0;
    for (int i = 0; i < 4; ++i) {
        pow_uint128 cur = (rem << 64) | UINT64_MAX;
        quot[i] = static_cast<uint64_t>(cur / tries);
        rem = cur % tries;
    }
    // serialize LITTLE-endian: byte k comes from limb quot[3 - k/8]
    std::array<uint8_t, 32> target{};
    for (int k = 0; k < 32; ++k) {
        target[k] = static_cast<uint8_t>(
            (quot[3 - k / 8] >> (8 * (k % 8))) & 0xFF);
    }
    return target;
}

bool admission_valid(const std::array<uint8_t, 32>& digest,
                     const std::array<uint8_t, 32>& target_le) {
    // little-endian uint256 compare, STRICT less-than (§6): walk from the
    // most significant byte (index 31) down.
    for (int i = 31; i >= 0; --i) {
        if (digest[i] < target_le[i]) return true;
        if (digest[i] > target_le[i]) return false;
    }
    return false;  // equal is NOT valid
}

std::optional<std::array<uint8_t, 32>> admission_grind(
    const std::vector<uint8_t>& msg_w,
    const std::string& model_identifier,
    const std::array<uint8_t, 32>& target_le,
    uint64_t max_tries,
    const std::array<uint8_t, 32>& prompt_commitment_digest) {
    // Random 32-byte starting nonce; per-try increment as a little-endian
    // counter. The message is built once — the nonce occupies the trailing
    // 32 bytes of the admission message (§6 layout), so each try only
    // mutates in place. No Python in this loop (plan §9): the pybind
    // wrapper releases the GIL around this call.
    std::array<uint8_t, 32> start{};
    std::random_device rd;
    for (std::size_t i = 0; i < start.size(); i += 4) {
        uint32_t r = rd();
        std::memcpy(start.data() + i, &r, 4);
    }
    std::vector<uint8_t> msg =
        admission_message(msg_w, model_identifier, start.data(),
                          prompt_commitment_digest);
    uint8_t* nonce = msg.data() + msg.size() - ADMISSION_NONCE_BYTES;
    for (uint64_t t = 0; t < max_tries; ++t) {
        std::array<uint8_t, 32> digest = argon2id_digest(msg);
        if (admission_valid(digest, target_le)) {
            std::array<uint8_t, 32> out{};
            std::memcpy(out.data(), nonce, ADMISSION_NONCE_BYTES);
            return out;
        }
        for (std::size_t i = 0; i < ADMISSION_NONCE_BYTES; ++i) {
            if (++nonce[i] != 0) break;  // little-endian counter carry
        }
    }
    return std::nullopt;
}

// ------------------------------------------------------------------------- //
// §4 — numerically conservative B_cred
// ------------------------------------------------------------------------- //

uint64_t f64_to_q63_floor(double x) {
    // floor(x * 2^63), x finite in [0, 1], EXACT integer arithmetic (no
    // double*2^63, which would round in the 52-bit mantissa). frexp:
    // x = m * 2^e, m in [0.5, 1), so mant = m * 2^53 is an exact integer in
    // [2^52, 2^53) and x * 2^63 = mant * 2^(e+10). A right shift truncates
    // toward zero == floor for a non-negative value.
    if (x <= 0.0) return 0;
    if (x >= 1.0) return BCRED_Q_ONE;
    int e = 0;
    double m = std::frexp(x, &e);
    uint64_t mant = static_cast<uint64_t>(std::ldexp(m, 53));  // exact
    int shift = e + 10;
    if (shift >= 0) {
        // mant < 2^53 and shift <= 10 for x < 1, so mant << shift < 2^63.
        return mant << shift;
    }
    return mant >> (-shift);
}

uint64_t f64_to_q63_ceil(double x) {
    // ceil(x * 2^63), x finite in [0, 1], EXACT (see f64_to_q63_floor).
    if (x <= 0.0) return 0;
    if (x >= 1.0) return BCRED_Q_ONE;
    int e = 0;
    double m = std::frexp(x, &e);
    uint64_t mant = static_cast<uint64_t>(std::ldexp(m, 53));
    int shift = e + 10;
    if (shift >= 0) {
        return mant << shift;                     // exact, no remainder
    }
    int s = -shift;
    return (mant + ((1ULL << s) - 1)) >> s;       // round up
}

uint64_t mass_q63_for_step(double lower, double upper, uint64_t atol_q63_ceil) {
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw std::invalid_argument("invalid entropy bounds: non-finite");
    }
    if (upper < lower) {
        throw std::invalid_argument("invalid entropy bounds: upper < lower");
    }
    uint64_t hi_q = f64_to_q63_ceil(upper);       // in [0, 2^63]
    uint64_t lo_q = f64_to_q63_floor(lower);      // in [0, 2^63], <= hi_q
    // (hi_q - lo_q) can be 2^63; widen in pow_uint128 then clamp.
    pow_uint128 mass =
        static_cast<pow_uint128>(hi_q - lo_q) +
        2 * static_cast<pow_uint128>(atol_q63_ceil);
    if (mass > static_cast<pow_uint128>(BCRED_Q_ONE)) return BCRED_Q_ONE;
    return static_cast<uint64_t>(mass);
}

uint64_t credit_units_for_step(uint64_t mass_q63) {
    if (mass_q63 == 0) {
        // §4: an invalid/garbage interval never earns credit — reject the
        // proof (unreachable with atol > 0, defensive).
        throw std::invalid_argument(
            "mass_q63 == 0: invalid interval never earns credit");
    }
    // Largest n in [0, N_MAX] with BCRED_THRESHOLD_Q63[n] >= mass_q63. The
    // table is non-increasing and threshold[0] == 2^63 >= mass_q63 (clamped),
    // so n == 0 always qualifies. Binary search on the decreasing table.
    std::size_t lo = 0, hi = BCRED_N_MAX;
    while (lo < hi) {
        std::size_t mid = (lo + hi + 1) >> 1;
        if (BCRED_THRESHOLD_Q63[mid] >= mass_q63) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return static_cast<uint64_t>(lo);
}

uint64_t b_cred_units_from_bounds(const std::vector<double>& lower_bounds,
                                  const std::vector<double>& upper_bounds,
                                  uint64_t atol_q63_ceil) {
    if (lower_bounds.size() != upper_bounds.size()) {
        throw std::invalid_argument("entropy bounds size mismatch");
    }
    // Separate integer accumulator: order-independent, bounded by
    // 256 * N_MAX == 8'388'608 — no overflow.
    uint64_t total = 0;
    for (std::size_t i = 0; i < lower_bounds.size(); ++i) {
        total += credit_units_for_step(
            mass_q63_for_step(lower_bounds[i], upper_bounds[i], atol_q63_ceil));
    }
    return total;
}

// ------------------------------------------------------------------------- //
// §5 — tier rule
// ------------------------------------------------------------------------- //

const char* tier_name(Tier tier) {
    switch (tier) {
        case Tier::Invalid: return "invalid";
        case Tier::AdmissionRequired: return "admission_required";
        case Tier::Free: return "free";
    }
    return "invalid";  // unreachable
}

Tier tier_for_b_cred_units(uint64_t b_cred_units, uint64_t b_floor_units,
                           uint64_t b_free_units) {
    if (b_cred_units < b_floor_units) return Tier::Invalid;
    if (b_cred_units < b_free_units) return Tier::AdmissionRequired;
    return Tier::Free;
}

}  // namespace pow_v3
