// V3 prompt-binding / admission helpers (TIP-0003) — see pow_v3.h.
//
// Must remain semantically identical to pow_v3.py; the golden vectors in
// tests/vectors/v3_vectors.json are the contract, exercised by
// tests/test_pow_v3_cpp.cpp (C++) and tests/test_pow_v3_vectors.py (Python).

#include "pow_v3.h"

#include <openssl/sha.h>
#ifdef POW_V3_HAVE_ARGON2
#include <argon2.h>
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

// Floating-point contract for this file; see the same block in bcore's copy
// (src/verification/pow_v3.cpp). The credit helpers this file shares with the
// node feed a CONSENSUS price, so a build with excess precision or fast-math
// must fail rather than produce platform-dependent values. Enforced in source
// because this file is compiled by eleven separate pow-utils build scripts.
#if defined(__FAST_MATH__)
#error "pow_v3 must not be compiled with -ffast-math: its credit path feeds a consensus price"
#endif
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#error "pow_v3 requires FLT_EVAL_METHOD == 0 (no excess precision on the credit path)"
#endif

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

std::string merge_extra_flags_v3_fields(
    const std::string& extra_flags,
    const std::optional<std::string>& nonce_hex,
    bool stepbind) {
    if (nonce_hex && !is_valid_admission_nonce_hex(*nonce_hex)) {
        throw std::invalid_argument(
            "admission_nonce_hex must be exactly 64 lowercase hex chars");
    }
    const bool emit_v3 = nonce_hex.has_value() || stepbind;

    std::string v3_body;
    if (nonce_hex) {
        v3_body = "\"admission_nonce\":\"" + *nonce_hex + "\"";
    }
    if (stepbind) {
        if (!v3_body.empty()) v3_body += ",";
        v3_body += "\"stepbind\":1";
    }
    const std::string v3_member = "\"v3\":{" + v3_body + "}";

    std::string trimmed = trim_copy(extra_flags);
    if (trimmed.empty()) {
        return emit_v3 ? ("{" + v3_member + "}") : extra_flags;
    }
    if (trimmed.front() == '{' && trimmed.back() == '}') {
        // JSON object: remove any existing top-level "v3" member (balanced,
        // string-escape-aware scan) so re-merging is idempotent, then splice
        // the fresh member at the end — proof_processor.cpp's audit-marker
        // splice at the front therefore never collides with it.
        size_t key_start = 0, value_start = 0, member_end = 0;
        bool scan_ok = false;
        std::string body = trimmed;
        const bool found_v3 = find_top_level_member(
            body, "v3", &key_start, &value_start, &member_end, &scan_ok);
        if (found_v3) {
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
            if (!emit_v3) {
                // v3 is proof-owned. When this job has neither a nonce nor a
                // StepBind marker, remove any stale carrier inherited through
                // model_config_diff; otherwise preserve the input byte-for-byte.
                return found_v3 ? body : extra_flags;
            }
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
            const std::optional<std::string> recovered =
                extract_admission_nonce_hex(candidate);
            const bool nonce_ok = nonce_hex
                ? (recovered && *recovered == *nonce_hex)
                : !recovered;
            const bool stepbind_ok =
                extract_stepbind_flag(candidate) == stepbind;
            if (nonce_ok && stepbind_ok) {
                return candidate;
            }
        }
        // Structurally unwalkable {...} (or splice failed the extractor
        // self-check) — fall through to the _diff wrap (pow_v3.py reaches
        // the same terminal via a json.loads failure).
    }
    if (!emit_v3) return extra_flags;
    // Not a JSON object: preserve the original string verbatim under "_diff"
    // (matches pow_v3.merge_extra_flags_v3 / ProofWriter's completion-id merge).
    return "{\"_diff\":\"" + json_escape(extra_flags) + "\"," + v3_member + "}";
}

std::string merge_extra_flags_v3(const std::string& extra_flags,
                                 const std::string& nonce_hex) {
    return merge_extra_flags_v3_fields(extra_flags, nonce_hex,
                                       /*stepbind=*/false);
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

    bool extract_stepbind() {
        return extract_stepbind_state() == StepBindFlagState::CLAIMED;
    }

    StepBindFlagState extract_stepbind_state() {
        i_ = skip_ws(s_, 0);
        if (i_ >= s_.size() || s_[i_] != '{')
            return StepBindFlagState::ABSENT;
        if (!parse_object(/*depth=*/1, CAPTURE_TOP))
            return StepBindFlagState::ABSENT;
        i_ = skip_ws(s_, i_);
        if (i_ != s_.size()) return StepBindFlagState::ABSENT;
        if (!stepbind_seen_) return StepBindFlagState::ABSENT;
        return stepbind_one_ ? StepBindFlagState::CLAIMED
                             : StepBindFlagState::MALFORMED;
    }

private:
    enum Capture { CAPTURE_NONE, CAPTURE_TOP, CAPTURE_V3 };

    const std::string& s_;
    size_t i_ = 0;
    std::optional<std::string> nonce_;
    bool stepbind_seen_ = false;
    bool stepbind_one_ = false;

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
            } else if (capture == CAPTURE_V3 && key == "stepbind") {
                const size_t value_start = i_;
                stepbind_seen_ = true;
                if (!parse_value(depth + 1, CAPTURE_NONE)) return false;
                stepbind_one_ = (s_.substr(value_start, i_ - value_start) == "1");
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

bool extract_stepbind_flag(const std::string& extra_flags) {
    if (extra_flags.size() > EXTRA_FLAGS_MAX_BYTES) return false;
    if (trim_copy(extra_flags).empty()) return false;
    if (!is_valid_utf8(extra_flags)) return false;
    return extract_stepbind_flag_state(extra_flags) ==
           StepBindFlagState::CLAIMED;
}

StepBindFlagState extract_stepbind_flag_state(const std::string& extra_flags) {
    if (extra_flags.size() > EXTRA_FLAGS_MAX_BYTES)
        return StepBindFlagState::ABSENT;
    if (trim_copy(extra_flags).empty()) return StepBindFlagState::ABSENT;
    if (!is_valid_utf8(extra_flags)) return StepBindFlagState::ABSENT;
    return CarrierParser(extra_flags).extract_stepbind_state();
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

std::array<uint8_t, 32> effective_step_nonce(
    const std::array<uint8_t, 32>& prompt_commitment_digest,
    bool nonce_present,
    const uint8_t nonce[32]) {
    // E = SHA256(STEP_BIND_TAG | C(32) | present(1) | R(32)); R = nonce when
    // present else 32 zero bytes. Byte-exact mirror of pow_v3::effective_step_nonce
    // (bcore) and pow_v3.effective_step_nonce (Python).
    std::vector<uint8_t> buf;
    buf.reserve(STEP_BIND_TAG_LEN + PROMPT_COMMITMENT_BYTES + 1 +
                ADMISSION_NONCE_BYTES);
    buf.insert(buf.end(), STEP_BIND_TAG, STEP_BIND_TAG + STEP_BIND_TAG_LEN);
    buf.insert(buf.end(), prompt_commitment_digest.begin(),
               prompt_commitment_digest.end());
    buf.push_back(nonce_present ? uint8_t{1} : uint8_t{0});
    if (nonce_present && nonce != nullptr) {
        buf.insert(buf.end(), nonce, nonce + ADMISSION_NONCE_BYTES);
    } else {
        buf.insert(buf.end(), ADMISSION_NONCE_BYTES, uint8_t{0});
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
    // unsigned __int128 keeps the numerator exact for any plausible chain
    // constants (defaults: 4 * 1e7 * 1e6 = 4e13, far below 2^127).
    unsigned __int128 numerator = static_cast<unsigned __int128>(elig_alpha_num) *
                                  decode_us_at_normalizer * normalizer;
    unsigned __int128 denominator = static_cast<unsigned __int128>(elig_alpha_den) *
                                    argon_ref_us *
                                    static_cast<uint64_t>(difficulty);
    if (denominator == 0) {
        throw std::invalid_argument("invalid admission constants");
    }
    unsigned __int128 tries = numerator / denominator;
    if (tries < 1) tries = 1;
    if (tries > static_cast<unsigned __int128>(UINT64_MAX)) {
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
    unsigned __int128 rem = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 cur = (rem << 64) | UINT64_MAX;
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
    // DETERMINISM: a right shift of a uint64 by >= 64 is UNDEFINED BEHAVIOUR in
    // C++ (platform/compiler dependent), which a naive `mant >> (-shift)` hits
    // for x below ~2^-53. floor(x*2^63) is 0 there. Python's >> already yields
    // 0; guard so C++ matches bit-for-bit across x86-64/ARM64/GCC/Clang.
    if (-shift >= 64) return 0;
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
    // DETERMINISM (see f64_to_q63_floor): `1ULL << s` and `>> s` are UB for
    // s >= 64. ceil(x*2^63) for any x in (0, 2^-63) is 1 (mant > 0 here).
    if (s >= 64) return 1;
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
    // (hi_q - lo_q) can be 2^63; widen in unsigned __int128 then clamp.
    unsigned __int128 mass =
        static_cast<unsigned __int128>(hi_q - lo_q) +
        2 * static_cast<unsigned __int128>(atol_q63_ceil);
    if (mass > static_cast<unsigned __int128>(BCRED_Q_ONE)) return BCRED_Q_ONE;
    return static_cast<uint64_t>(mass);
}

uint64_t mass_lower_q63_for_step(double lower, double upper,
                                 uint64_t atol_q63_ceil) {
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        throw std::invalid_argument("invalid entropy bounds: non-finite");
    }
    if (upper < lower) {
        throw std::invalid_argument("invalid entropy bounds: upper < lower");
    }
    const uint64_t hi_q = f64_to_q63_floor(upper);   // round DOWN
    const uint64_t lo_q = f64_to_q63_ceil(lower);    // round UP
    const unsigned __int128 lo_plus =
        static_cast<unsigned __int128>(lo_q) +
        2 * static_cast<unsigned __int128>(atol_q63_ceil);
    if (static_cast<unsigned __int128>(hi_q) <= lo_plus) return 0;
    const unsigned __int128 mass = static_cast<unsigned __int128>(hi_q) - lo_plus;
    if (mass > static_cast<unsigned __int128>(BCRED_Q_ONE)) return BCRED_Q_ONE;
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

// ------------------------------------------------------------------------- //
// §8 — anti-parrot metrics. C++-ONLY for the initial consensus ship (no
// pow_v3.py mirror yet; the C++ contextual gate is the sole authority). Pure
// integer: Q63/Q32 fixed point, no float in any decision.
// ------------------------------------------------------------------------- //

uint64_t f64_to_q32_floor(double x) {
    // floor(x * 2^32), EXACT via mantissa decomposition (see f64_to_q63_floor).
    if (x <= 0.0) return 0;
    if (x >= 1.0) return AP_Q32_ONE;
    int e = 0;
    const double m = std::frexp(x, &e);
    const uint64_t mant = static_cast<uint64_t>(std::ldexp(m, 53));  // exact
    const int shift = e - 21;   // x*2^32 = mant * 2^(e-21); e <= 0 => shift < 0
    if (shift >= 0) {
        return mant << shift;
    }
    const int s = -shift;
    if (s >= 64) return 0;      // sub-2^-32 value floors to 0 (Python >> parity)
    return mant >> s;
}

BackloadSurface backload_surface(uint64_t maxrs_q32, uint64_t bcap_units) {
    // Signed delta: maxRs_real - Bcap_real in the Q32 domain (2^32/R == 2^22).
    const __int128 delta = static_cast<__int128>(maxrs_q32) -
                           (static_cast<__int128>(bcap_units) << 22);
    if (delta > (static_cast<__int128>(AP_BACKLOAD_HARD_T) << 32))
        return BackloadSurface::Hard;
    if (delta > (static_cast<__int128>(AP_BACKLOAD_GREY_T) << 32))
        return BackloadSurface::Grey;
    return BackloadSurface::Free;
}

bool anti_parrot_kappa_lt(unsigned __int128 self_collision,
                          unsigned __int128 total_collision,
                          uint64_t num, uint64_t den) {
    if (total_collision == 0) return false;
    return self_collision * den < total_collision * num;
}

bool anti_parrot_nstates_lt(uint64_t bcap_units,
                            unsigned __int128 total_collision, uint64_t n0) {
    // N_states_hi (conservative UPPER bound; total_collision is the Q63 LOWER
    // bound total_lo) < n0, division-free:  bcap^2 * 2^63 < n0 * total_lo.
    // total == 0 (no definite overlap) => NOT below (undefined). bcap^2*2^63 <=
    // 2^36*2^63 = 2^99; n0*total <= ~2^5 * 2^99 = 2^104 — inside __int128.
    if (total_collision == 0) return false;
    const unsigned __int128 lhs =
        static_cast<unsigned __int128>(bcap_units) * bcap_units *
        static_cast<unsigned __int128>(BCRED_Q_ONE);
    return lhs < static_cast<unsigned __int128>(n0) * total_collision;
}

bool anti_parrot_keff_lt(uint64_t n, uint64_t sum_c2, uint64_t k0) {
    // Keff_out == n^2 / sum_c2 < k0  <=>  n^2 < k0 * sum_c2. n^2 <= 65536,
    // sum_c2 <= 65536, k0 small — no overflow in uint64.
    if (sum_c2 == 0) return false;
    return n * n < k0 * sum_c2;
}

namespace {
// kappa in integer PPM units (floor). Monitoring only (NOT a consensus
// decision), so integer division is fine; still deterministic and identical
// across languages. Undefined kappa (no overlap) reports 1.0 (== SCALE).
uint64_t ap_kappa_units(unsigned __int128 self_collision,
                        unsigned __int128 total_collision) {
    if (total_collision == 0) return AP_KAPPA_UNITS_SCALE;
    const unsigned __int128 q =
        (self_collision * static_cast<unsigned __int128>(AP_KAPPA_UNITS_SCALE)) /
        total_collision;
    return static_cast<uint64_t>(q);
}

// Per-token CONSERVATIVE Q63 interval bound (§8), from the ATOL-expanded
// id-sorted CDF endpoints (identical widening to B_cred):
//   l_max = ceil( min(1, l+ATOL) * 2^63 )   (upper bound of the left endpoint)
//   h_min = floor(max(0, h-ATOL) * 2^63 )   (lower bound of the right endpoint)
//   m_lo  = max(0, h_min - l_max)            (DEFINITELY-present token mass)
struct ApTokBound { uint64_t l_max; uint64_t h_min; uint64_t m_lo; };

// S_id_lo cross-term: definitely-shared support == sum over shared token ids of
// min(m_lo). A LOWER bound on the true support overlap.
uint64_t ap_sid_lo(const std::map<uint32_t, ApTokBound>& a,
                   const std::map<uint32_t, ApTokBound>& b) {
    const std::map<uint32_t, ApTokBound>* small = &a;
    const std::map<uint32_t, ApTokBound>* large = &b;
    if (small->size() > large->size()) std::swap(small, large);
    uint64_t total = 0;
    for (const auto& kv : *small) {
        const auto it = large->find(kv.first);
        if (it != large->end()) {
            total += kv.second.m_lo < it->second.m_lo ? kv.second.m_lo
                                                      : it->second.m_lo;
        }
    }
    return total;
}

// S_u_lo cross-term: definitely-shared same-u interval == sum over shared token
// ids of max(0, min(h_min) - max(l_max)). A LOWER bound on the true same-u
// agreement.
uint64_t ap_su_lo(const std::map<uint32_t, ApTokBound>& a,
                  const std::map<uint32_t, ApTokBound>& b) {
    const std::map<uint32_t, ApTokBound>* small = &a;
    const std::map<uint32_t, ApTokBound>* large = &b;
    if (small->size() > large->size()) std::swap(small, large);
    uint64_t total = 0;
    for (const auto& kv : *small) {
        const auto it = large->find(kv.first);
        if (it != large->end()) {
            const uint64_t lo = kv.second.l_max > it->second.l_max
                                    ? kv.second.l_max : it->second.l_max;
            const uint64_t hi = kv.second.h_min < it->second.h_min
                                    ? kv.second.h_min : it->second.h_min;
            if (hi > lo) total += hi - lo;
        }
    }
    return total;
}
}  // namespace

AntiParrotMetrics compute_anti_parrot_metrics(
    const std::vector<uint64_t>& credit_mass_q63,
    const std::vector<uint64_t>& hard_mass_q63,
    const std::vector<std::vector<uint32_t>>& topk_ids,
    const std::vector<std::vector<double>>& topk_cdf_hi,
    const std::vector<uint32_t>& chosen_ids,
    const std::vector<uint32_t>& prompt_token_ids) {
    const std::size_t n = credit_mass_q63.size();
    if (hard_mass_q63.size() != n || topk_ids.size() != n ||
        topk_cdf_hi.size() != n) {
        throw std::invalid_argument("anti-parrot: per-step input length mismatch");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (topk_ids[i].size() != topk_cdf_hi[i].size()) {
            throw std::invalid_argument("anti-parrot: topk ids/cdf size mismatch");
        }
    }

    std::vector<uint64_t> credit_units(n), weights(n);
    for (std::size_t i = 0; i < n; ++i) {
        credit_units[i] = credit_units_for_step(credit_mass_q63[i]);
        weights[i] = credit_units[i] < AP_R_STEP_CAP ? credit_units[i]
                                                     : AP_R_STEP_CAP;
    }

    AntiParrotMetrics out;
    for (std::size_t i = 0; i < n; ++i) {
        out.braw += credit_units[i];
        out.bcap += weights[i];
    }

    // hard_run: longest consecutive run of hard-pinned steps.
    uint64_t run = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (hard_mass_q63[i] >= AP_HARD_PIN_THRESHOLD_Q63) {
            ++run;
            if (run > out.hard_run) out.hard_run = run;
        } else {
            run = 0;
        }
    }

    // maxRs: restarted conditional reuse score.
    for (std::size_t s = 0; s < n; ++s) {
        uint64_t cum = 0, total = 0;
        for (std::size_t t = s; t < n; ++t) {
            cum += credit_units[t];
            if (cum > AP_MAXRS_CUM_CAP) cum = AP_MAXRS_CUM_CAP;
            total += TERM_Q32[cum];
        }
        if (total > out.maxrs_q32) out.maxrs_q32 = total;
    }

    // reuse_cache: ROOT-ANCHORED single forward pass from step 0 (distinct from
    // maxRs, which RESTARTS at every checkpoint). The SAME per-step conservative
    // credit_units feed a running cumulative, capped per step (STEP_CAP) and in
    // total (TERM index cap); a fixed 10-bit cache allowance (CACHE_CREDIT) is
    // subtracted before the TERM_Q32 decay lookup, so the first ~2^10 cached
    // prefix states cost nothing. reuse_cache12 is the SAME accumulation with a
    // 12-bit allowance — SHADOW telemetry ONLY, it NEVER gates. Both sums are
    // <= 256*2^32 == 2^40 (each TERM_Q32 term <= 2^32), so they fit uint64.
    {
        uint64_t cumulative = 0;
        for (std::size_t t = 0; t < n; ++t) {
            uint64_t credit = credit_units[t];               // reuse existing per-step credit
            if (credit > AP_REUSE_STEP_CAP) credit = AP_REUSE_STEP_CAP;
            cumulative += credit;
            if (cumulative > AP_REUSE_TERM_C_MAX) cumulative = AP_REUSE_TERM_C_MAX;
            const uint64_t adj10 = cumulative > AP_REUSE_CACHE_CREDIT
                                       ? cumulative - AP_REUSE_CACHE_CREDIT : 0;
            out.reuse_cache_q32 += TERM_Q32[adj10];
            const uint64_t adj12 = cumulative > AP_REUSE_CACHE12_CREDIT
                                       ? cumulative - AP_REUSE_CACHE12_CREDIT : 0;
            out.reuse_cache12_q32 += TERM_Q32[adj12];        // SHADOW only
        }
        // HARD structural INVALID — admission can NEVER cure it (same envelope as
        // hardrun/backload). reuse_cache12 is telemetry and NEVER contributes.
        out.reuse_cache_hard = out.reuse_cache_q32 > AP_REUSE_CACHE_CAP_Q32;
    }

    // CONSERVATIVE Q63 interval bounds per CREDITED step (weight > 0), from the
    // ATOL-expanded id-sorted CDF endpoints — the SAME sampler-pass CDF B_cred
    // uses (never a second softmax). Token j spans [l, h] on the id-sorted CDF:
    // l == the previous token's endpoint (0 for the first), h == this token's
    // cumulative endpoint. The bound direction mirrors B_cred exactly.
    std::vector<std::size_t> credited;
    for (std::size_t s = 0; s < n; ++s) if (weights[s] > 0) credited.push_back(s);
    const std::size_t K = credited.size();
    std::vector<std::map<uint32_t, ApTokBound>> D(K);
    std::vector<unsigned __int128> W(K);
    for (std::size_t kk = 0; kk < K; ++kk) {
        const std::size_t i = credited[kk];
        double l = 0.0;
        for (std::size_t j = 0; j < topk_ids[i].size(); ++j) {
            const double h = topk_cdf_hi[i][j];
            const uint64_t l_max = f64_to_q63_ceil(std::min(1.0, l + ATOL));
            const uint64_t h_min = f64_to_q63_floor(std::max(0.0, h - ATOL));
            const uint64_t m_lo = h_min > l_max ? h_min - l_max : 0;
            D[kk][topk_ids[i][j]] = ApTokBound{l_max, h_min, m_lo};
            l = h;  // next token's left endpoint == this token's right endpoint
        }
        W[kk] = weights[i];
    }

    // Collision LOWER bounds (all integer, Q63); diagonal forced to S == 1
    // (2^63). total_*_lo <= the true totals, so self/total_lo is an UPPER bound
    // on kappa and bcap^2*2^63/total_lo an UPPER bound on the state count — the
    // gate rejects ONLY when even that conservative bound crosses the threshold.
    const unsigned __int128 Q63 = static_cast<unsigned __int128>(BCRED_Q_ONE);
    for (std::size_t x = 0; x < K; ++x) {
        const unsigned __int128 wx = W[x];
        out.self_collision += wx * wx * Q63;
        for (std::size_t y = 0; y < K; ++y) {
            const unsigned __int128 wy = W[y];
            uint64_t su, sid;
            if (x == y) {
                su = sid = BCRED_Q_ONE;      // S == 1
            } else {
                su = ap_su_lo(D[x], D[y]);
                sid = ap_sid_lo(D[x], D[y]);
            }
            const unsigned __int128 wxy = wx * wy;
            out.total_collision_u += wxy * static_cast<unsigned __int128>(su);
            out.total_collision_id += wxy * static_cast<unsigned __int128>(sid);
        }
    }

    const BackloadSurface surface = backload_surface(out.maxrs_q32, out.bcap);
    out.backload_hard = surface == BackloadSurface::Hard;
    out.backload_grey = surface == BackloadSurface::Grey;

    // N_states_hi: conservative UPPER bound on the effective whole-CDF state
    // count == (sum w)^2 * 2^63 / total_lo (>= the true count). The length-
    // INVARIANT discriminator (kappa == N_states/K_steps conflates count with
    // length — a 4-state cycle escapes a kappa threshold at shorter length).
    const unsigned __int128 wsq_q63 =
        static_cast<unsigned __int128>(out.bcap) * out.bcap * Q63;
    out.n_states_eff_u = out.total_collision_u
        ? static_cast<uint64_t>(wsq_q63 / out.total_collision_u) : 0;
    out.n_states_eff_id = out.total_collision_id
        ? static_cast<uint64_t>(wsq_q63 / out.total_collision_id) : 0;

    // state_HARD (INVALID) — CONSERVATIVE. Two discriminators, EACH joint across
    // BOTH the u and id axes; state collapse is EITHER firing:
    //   * kappa_hi joint  — recurrence DENSITY (catches long low-kappa cycles),
    //   * N_states_hi joint — length-INVARIANT COUNT (catches short cycles a
    //     kappa threshold misses, e.g. a 4-state cycle at 25 steps).
    // Each threshold sits in a visible empty gap below the nearest honest block,
    // and every historical state-parrot fires BOTH while no honest block fires
    // either — so the OR adds short-cycle coverage without any false positive.
    const bool kappa_joint =
        anti_parrot_kappa_lt(out.self_collision, out.total_collision_u,
                             AP_KU_HARD_NUM, AP_KU_HARD_DEN) &&
        anti_parrot_kappa_lt(out.self_collision, out.total_collision_id,
                             AP_KID_HARD_NUM, AP_KID_HARD_DEN);
    const bool nstates_joint =
        anti_parrot_nstates_lt(out.bcap, out.total_collision_u, AP_NSTATES_U_HARD) &&
        anti_parrot_nstates_lt(out.bcap, out.total_collision_id, AP_NSTATES_ID_HARD);
    out.state_hard = kappa_joint || nstates_joint;
    // GREY state (SHADOW telemetry; gates only under grey-enforcement): the same
    // OR at wider bands, itself needing its own empirical gap before promotion.
    const bool kappa_grey =
        anti_parrot_kappa_lt(out.self_collision, out.total_collision_u,
                             AP_KU_GREY_NUM, AP_KU_GREY_DEN) &&
        anti_parrot_kappa_lt(out.self_collision, out.total_collision_id,
                             AP_KID_GREY_NUM, AP_KID_GREY_DEN);
    const bool nstates_grey =
        anti_parrot_nstates_lt(out.bcap, out.total_collision_u, AP_NSTATES_U_GREY) &&
        anti_parrot_nstates_lt(out.bcap, out.total_collision_id, AP_NSTATES_ID_GREY);
    out.state_grey = kappa_grey || nstates_grey;

    // kappa_hi telemetry (conservative recurrence density; does NOT gate). ppm.
    const uint64_t ku_ppm =
        ap_kappa_units(out.self_collision, out.total_collision_u);
    const uint64_t kid_ppm =
        ap_kappa_units(out.self_collision, out.total_collision_id);
    out.state_deficit = (AP_KU_REF_UNITS > ku_ppm ? AP_KU_REF_UNITS - ku_ppm : 0) +
                        (AP_KID_REF_UNITS > kid_ppm ? AP_KID_REF_UNITS - kid_ppm : 0);

    // ---- Small-output-alphabet SHADOW telemetry (§8; never gates while the
    // grey guard is dormant). Output-only counts over the N chosen tokens plus
    // ONE CDF-reuse term (MassA). Left zero when chosen_ids is not supplied.
    if (!chosen_ids.empty()) {
        if (chosen_ids.size() != n) {
            throw std::invalid_argument("anti-parrot: chosen_ids length mismatch");
        }
        // c(t) occurrence counts over ALL N output tokens.
        std::map<uint32_t, uint64_t> counts;
        for (uint32_t t : chosen_ids) ++counts[t];
        out.vunique_out = counts.size();
        uint64_t sum_c2 = 0;
        std::vector<std::pair<uint64_t, uint32_t>> by_count;  // (count, id)
        by_count.reserve(counts.size());
        for (const auto& kv : counts) {
            sum_c2 += kv.second * kv.second;
            by_count.push_back({kv.second, kv.first});
        }
        out.sum_c2 = sum_c2;
        // K15_16: fewest ids (count DESC, tie -> smallest id) covering
        // >= ceil(15*N/16) positions. A == that id set.
        std::sort(by_count.begin(), by_count.end(),
                  [](const std::pair<uint64_t, uint32_t>& a,
                     const std::pair<uint64_t, uint32_t>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        const uint64_t N = static_cast<uint64_t>(n);
        const uint64_t need = (15 * N + 15) / 16;    // ceil(15*N/16)
        uint64_t cum = 0, k1516 = 0;
        std::set<uint32_t> A;
        for (const auto& p : by_count) {
            if (cum >= need) break;
            cum += p.first;
            ++k1516;
            A.insert(p.second);
        }
        out.k15_16_out = k1516;
        // Cprompt: fraction of output tokens present in the (unpadded) prompt.
        const std::set<uint32_t> prompt_set(prompt_token_ids.begin(),
                                            prompt_token_ids.end());
        uint64_t in_prompt = 0;
        for (uint32_t t : chosen_ids) if (prompt_set.count(t)) ++in_prompt;
        out.cprompt_ppm = N ? (in_prompt * 1000000ULL / N) : 0;
        // MassA: conservative LOWER-bound model mass on A per CREDITED step,
        // reusing the SAME Q63 endpoint bounds as kappa (D[kk].m_lo). Count the
        // steps whose MassA >= 7/8 (== 7 * 2^60).
        const unsigned __int128 seven_eighths =
            7 * (static_cast<unsigned __int128>(BCRED_Q_ONE) >> 3);
        uint64_t massA_hi = 0;
        for (std::size_t kk = 0; kk < K; ++kk) {
            unsigned __int128 mA = 0;
            for (const auto& kv : D[kk]) if (A.count(kv.first)) mA += kv.second.m_lo;
            if (mA >= seven_eighths) ++massA_hi;
        }
        out.massA_hi_steps = massA_hi;
        out.credited_steps = K;
        // Flags. small_output_alphabet: Keff_out (dust-robust) OR K15_16 count.
        out.small_output_alphabet =
            anti_parrot_keff_lt(N, sum_c2, AP_KEFF0) || (k1516 <= AP_K1516_0);
        // CONJUNCTIVE hit: small alphabet AND (prompt supplies it, Cprompt>=15/16,
        // OR the SUPPORT is collapsed even if probabilities vary — the case the
        // (N_states_u AND N_states_id) joint misses) AND the model is genuinely
        // concentrated on A on MOST credited steps (not coincidental). NEVER
        // standalone Keff<8 (false-positives on honest large-alphabet text).
        const bool cprompt_hi = (in_prompt * 16 >= 15 * N);           // Cprompt >= 15/16
        const bool support_collapsed = out.n_states_eff_id < AP_NSID_RELAXED;
        const bool massA_most = K > 0 && (massA_hi * 2 > K);          // most credited steps
        out.small_alphabet_hit = out.small_output_alphabet &&
                                 (cprompt_hi || support_collapsed) && massA_most;
    }
    return out;
}

const char* anti_parrot_reject_reason(const AntiParrotMetrics& m,
                                      bool grey_enforced) {
    // Anti-parrot is a STRUCTURAL validity rule: INVALID ONLY, never admission.
    // Braw tiering / the admission band are owned by VerifyV3TierAndAdmission
    // (chain params) and are NOT re-checked here.
    if (m.hard_run > AP_L_HARD) return "anti-parrot-hardrun";
    if (m.state_hard) return "anti-parrot-state";
    if (m.bcap < AP_BCAP_ABS_FLOOR_UNITS) return "anti-parrot-bcap";
    if (m.backload_hard) return "anti-parrot-backload";
    // Cache-budgeted reuse score — HARD structural INVALID (admission never cures
    // it). reuse_cache12_q32 is SHADOW telemetry and NEVER gates. Fires whenever
    // the anti-parrot height guard is active (mainnet: AntiParrotHeight = 24150,
    // see kernel/chainparams.cpp).
    if (m.reuse_cache_hard) return "anti-parrot-reusecache";
    // GREY structural state — DORMANT HOOK. grey_enforced is an optional arg
    // defaulting false, and NOTHING in the codebase passes true (no activation
    // path is wired yet): grey is currently always free. The branch exists so a
    // future calibration can promote grey to INVALID ("anti-parrot-grey", never
    // admission) once its own empirical gap is established — but that activation
    // (e.g. a chain-param sub-height) is UNBUILT. state_deficit never gates.
    if (grey_enforced && (m.state_grey || m.backload_grey)) return "anti-parrot-grey";
    // Small-output-alphabet rule — wired behind the SAME dormant grey guard, so
    // it NEVER contributes to the reject while grey_enforced is false. Emitted
    // as telemetry regardless.
    if (grey_enforced && m.small_alphabet_hit) return "anti-parrot-smallalphabet";
    return nullptr;
}

}  // namespace pow_v3
