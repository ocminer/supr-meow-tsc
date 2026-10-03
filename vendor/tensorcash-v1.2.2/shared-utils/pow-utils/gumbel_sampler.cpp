// gumbel_sampler.cpp — see gumbel_sampler.h / gumbel_sampler.py (normative).
//
// FP CONTRACTION MUST BE OFF in this translation unit: DLOG is a fixed sequence of
// IEEE basic ops whose bits every party must reproduce. GCC defaults to
// -ffp-contract=fast under -std=gnu++XX (and aarch64 has FMA), so the pragma
// below turns it off here regardless of the build's global flags; the build
// scripts also pass -ffp-contract=off, and dlog_selftest() verifies at startup.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include "gumbel_sampler.h"

#include <openssl/evp.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gumbel {

// ------------------------------------------------------------------------- //
// Reference arithmetic
// ------------------------------------------------------------------------- //

bool is_valid_head(const std::string& head) {
    return head == "h0" || head == "h1" || head == "h2";
}

std::vector<uint8_t> mode_suffix(const std::string& head, int version) {
    if (!is_valid_head(head)) {
        throw std::invalid_argument("gumbel head mode must be h0|h1|h2, got '" + head + "'");
    }
    if (version < 0 || version > 255) {
        throw std::invalid_argument("gumbel contract version must fit in one byte");
    }
    std::vector<uint8_t> out(MODE_TAG, MODE_TAG + MODE_TAG_LEN);
    out.push_back(static_cast<uint8_t>(version));
    out.insert(out.end(), head.begin(), head.end());
    return out;
}

int arm_code(const std::string& arm) {
    if (arm == "ONE") return 1;
    if (arm == "TWO") return 2;
    throw std::invalid_argument("gumbel arm must be ONE|TWO, got '" + arm + "'");
}

std::vector<uint8_t> mode_suffix_v2(const std::string& head, const std::string& arm,
                                    const std::string& canonical_policy_json) {
    if (!is_valid_head(head)) {
        throw std::invalid_argument("gumbel head mode must be h0|h1|h2, got '" + head + "'");
    }
    const int code = arm_code(arm);
    if (canonical_policy_json.empty() || canonical_policy_json.front() != '{') {
        throw std::invalid_argument("v2 declaration needs the canonical policy JSON object");
    }
    const std::string msg = std::string(POLICY_DIGEST_DOMAIN) + canonical_policy_json;
    const auto digest = sha256_bytes(reinterpret_cast<const uint8_t*>(msg.data()), msg.size());
    std::vector<uint8_t> out(MODE_TAG_V2, MODE_TAG_V2 + MODE_TAG_LEN);
    out.push_back(static_cast<uint8_t>(CONTRACT_VERSION_V2));
    out.insert(out.end(), head.begin(), head.end());
    out.push_back(static_cast<uint8_t>(code));
    out.insert(out.end(), digest.begin(), digest.begin() + 8);
    return out;
}

std::array<uint8_t, 32> sha256_bytes(const uint8_t* data, std::size_t len) {
    std::array<uint8_t, 32> out{};
    unsigned int outlen = 32;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");
    const bool ok = EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, data, len) == 1 &&
                    EVP_DigestFinal_ex(ctx, out.data(), &outlen) == 1;
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("SHA-256 failed");
    return out;
}

std::array<uint8_t, 32> seed_from_digest(const uint8_t* digest32, uint64_t draw) {
    // (LE256(D) + draw) mod 2^256 with ripple carry, little-endian bytes.
    std::array<uint8_t, 32> out{};
    uint64_t carry = draw;
    for (std::size_t k = 0; k < SEED_BYTES; ++k) {
        const uint64_t s = static_cast<uint64_t>(digest32[k]) + carry;
        out[k] = static_cast<uint8_t>(s & 0xFF);
        carry = s >> 8;
    }
    return out;
}

std::vector<uint8_t> expansion_message(const std::array<uint8_t, 32>& seed, uint32_t token_id) {
    std::vector<uint8_t> m;
    m.reserve(EXPANSION_MSG_BYTES);
    m.insert(m.end(), TAG, TAG + TAG_LEN);
    m.insert(m.end(), seed.begin(), seed.end());
    for (int i = 0; i < 4; ++i) m.push_back(static_cast<uint8_t>((token_id >> (8 * i)) & 0xFF));
    return m;
}

double uniform_from_x64(uint64_t x) {
    return (static_cast<double>(x >> 12) + 0.5) / TWO_POW_52;
}

double token_uniform(const std::array<uint8_t, 32>& seed, uint32_t token_id) {
    const auto m = expansion_message(seed, token_id);
    const auto e = sha256_bytes(m.data(), m.size());
    uint64_t x = 0;
    for (int i = 7; i >= 0; --i) x = (x << 8) | e[static_cast<std::size_t>(i)];
    return uniform_from_x64(x);
}

const int64_t DLOG_TABLE_BITS[129] = {0LL,4575622219221198729LL,4580091176815428580LL,4582334671619469595LL,4584526548160754432LL,4585683151371550048LL,4586762670936612092LL,4587834164259680703LL,4588897749810897574LL,4589560781873576673LL,4590084839412847057LL,4590605113110173759LL,4591121657214888780LL,4591634524817955617LL,4592143767884715363LL,4592649437286483779LL,4593151582831047398LL,4593650253292104379LL,4593908558177799769LL,4594154489487779744LL,4594398753454028067LL,4594641372532608884LL,4594882368728956190LL,4595121763609850936LL,4595359578315002836LL,4595595833568252445LL,4595830549688408329LL,4596063746599733521LL,4596295443842094772LL,4596525660580787555LL,4596754415616049139LL,4596981727392271574LL,4597207614006925858LL,4597432093219208081LL,4597655182458417894LL,4597876898832079158LL,4598097259133812267LL,4598245749698121793LL,4598354598358651107LL,4598462793269529301LL,4598570342236792334LL,4598677252927494651LL,4598783532872989030LL,4598889189472110243LL,4598994229994265905LL,4599098661582437717LL,4599202491256096234LL,4599305725914032100LL,4599408372337106626LL,4599510437190924449LL,4599611927028430893LL,4599712848292436588LL,4599813207318071754LL,4599913010335172501LL,4600012263470601391LL,4600110972750504420LL,4600209144102506503LL,4600306783357847458LL,4600403896253460419LL,4600500488433994519LL,4600596565453783643LL,4600692132778762951LL,4600787195788334843LL,4600881759777185931LL,4600975829957056588LL,4601069411458464522LL,4601162509332383818LL,4601255128551880811LL,4601347274013708131LL,4601438950539858171LL,4601530162879077244LL,4601620915708341581LL,4601711213634296354LL,4601801061194658795LL,4601890462859586517LL,4601979423033012030LL,4602067946053944481LL,4602156036197739558LL,4602243697677338504LL,4602330934644477120LL,4602417751190865645LL,4602504151349340332LL,4602590139094987551LL,4602675718346241176LL,4602719856069300479LL,4602762242967545554LL,4602804431331581760LL,4602846423012554017LL,4602888219835837059LL,4602929823601511564LL,4602971236084829335LL,4603012459036667836LL,4603053494183974378LL,4603094343230200225LL,4603135007855724911LL,4603175489718271013LL,4603215790453309648LL,4603255911674456942LL,4603295854973861699LL,4603335621922584523LL,4603375214070968603LL,4603414632949002379LL,4603453880066674316LL,4603492956914319976LL,4603531864962961598LL,4603570605664640369LL,4603609180452741587LL,4603647590742312886LL,4603685837930375706LL,4603723923396230179LL,4603761848501753595LL,4603799614591692610LL,4603837222993949358LL,4603874675019861607LL,4603911971964477122LL,4603949115106822367LL,4603986105710165691LL,4604022945022275130LL,4604059634275670961LL,4604096174687873131LL,4604132567461643691LL,4604168813785224344LL,4604204914832569241LL,4604240871763573129LL,4604276685724294957LL,4604312357847177061LL,4604347889251260021LL,4604383281042393305LL,4604418534313441775LL};

static inline double bits_to_double(int64_t b) {
    double d; std::memcpy(&d, &b, 8); return d;
}

static inline double dlog_log1p(double r) {
    double p = r / 9.0;
    p = p * r;
    p = p - 0.125;
    p = p * r;
    p = p + 0.14285714285714285;
    p = p * r;
    p = p - 0.16666666666666666;
    p = p * r;
    p = p + 0.2;
    p = p * r;
    p = p - 0.25;
    p = p * r;
    p = p + 0.3333333333333333;
    p = p * r;
    p = p - 0.5;
    p = p * r;
    p = p + 1.0;
    return p * r;
}

double dlog(double x) {
    if (!(x > 0.0) || std::isinf(x)) throw std::invalid_argument("dlog needs a positive finite input");
    if (x > DLOG_NEAR_LO && x < DLOG_NEAR_HI) return dlog_log1p(x - 1.0);
    uint64_t bits; std::memcpy(&bits, &x, 8);
    const int64_t e = static_cast<int64_t>((bits >> 52) & 0x7FF) - 1023;
    const uint64_t mbits = (bits & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;
    double m; std::memcpy(&m, &mbits, 8);
    const int k = static_cast<int>((m - 1.0) * 128.0);          // floor, exact
    const double c = 1.0 + static_cast<double>(k) / 128.0;
    const double r = (m - c) / c;
    const double p = dlog_log1p(r);
    double out = static_cast<double>(e) * DLOG_LN2;
    out = out + bits_to_double(DLOG_TABLE_BITS[k]);
    out = out + p;
    return out;
}

void dlog_selftest() {
    // (input bits, expected output bits) — the same vectors as gumbel_sampler.DLOG_SELFTEST,
    // expected bits produced by the Python reference (tests/gen_gumbel_vectors.py pins them too).
    static const int64_t IN[10] = {4602678819172646912LL,4600298746774613816LL,4607182418800017407LL,4368491638549381120LL,4607147234427928576LL,4607217603172106240LL,4611686018427387904LL,4630367062351886330LL,118622047889322841LL,9094988921128908188LL};
    static const int64_t EXP[10] = {-4618953502541334033LL,-4616189618054758401LL,-4854880398305394688LL,-4593004974502889478LL,-4652209596006888704LL,4575622219221198729LL,4604418534313441775LL,4615297407664951095LL,-4574084695234936907LL,4649287341619838900LL};
    for (int i = 0; i < 10; ++i) {
        const double got = dlog(bits_to_double(IN[i]));
        int64_t gb; std::memcpy(&gb, &got, 8);
        if (gb != EXP[i]) {
            throw std::runtime_error("gumbel::dlog self-test failed (FP contraction on, or non-IEEE double "
                                     "arithmetic): case " + std::to_string(i) + " got " + std::to_string(gb) +
                                     " expected " + std::to_string(EXP[i]));
        }
    }
}

double gumbel_from_uniform(double u) {
    return -dlog(-dlog(u));
}

double token_gumbel(const std::array<uint8_t, 32>& seed, uint32_t token_id) {
    return gumbel_from_uniform(token_uniform(seed, token_id));
}

RaceResult race(const std::vector<int64_t>& ids, const std::vector<float>& weights,
                const std::array<uint8_t, 32>& seed) {
    if (ids.size() != weights.size()) throw std::invalid_argument("ids/weights length mismatch");
    RaceResult r;
    r.scores.assign(ids.size(), -INFINITY);
    int64_t best_id = std::numeric_limits<int64_t>::max();
    for (std::size_t pos = 0; pos < ids.size(); ++pos) {
        const float a = weights[pos];
        if (!std::isfinite(a) || ids[pos] < 0) continue;
        const double s = static_cast<double>(a) +
                         token_gumbel(seed, static_cast<uint32_t>(ids[pos]));
        r.scores[pos] = s;
        const int64_t tid = ids[pos];
        if (s > r.best || (s == r.best && tid < best_id)) {
            if (r.winner_pos >= 0) r.second = std::max(r.second, r.best);
            r.winner_pos = static_cast<int>(pos);
            best_id = tid;
            r.best = s;
        } else {
            r.second = std::max(r.second, s);
        }
    }
    r.winner_id = (r.winner_pos >= 0) ? ids[static_cast<std::size_t>(r.winner_pos)] : -1;
    return r;
}

std::vector<double> support_probs(const std::vector<float>& weights) {
    std::vector<double> out(weights.size(), 0.0);
    double m = -INFINITY;
    for (float w : weights) if (std::isfinite(w)) m = std::max(m, static_cast<double>(w));
    if (!std::isfinite(m)) return out;
    double z = 0.0;
    for (float w : weights) if (std::isfinite(w)) z += std::exp(static_cast<double>(w) - m);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (std::isfinite(weights[i])) out[i] = std::exp(static_cast<double>(weights[i]) - m) / z;
    }
    return out;
}

// ------------------------------------------------------------------------- //
// Retry bitmap + carrier
// ------------------------------------------------------------------------- //

std::string bytes_hex(const uint8_t* data, std::size_t len) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(2 * len);
    for (std::size_t i = 0; i < len; ++i) {
        out.push_back(digits[data[i] >> 4]);
        out.push_back(digits[data[i] & 0x0F]);
    }
    return out;
}

namespace {

// --- minimal escape-aware JSON scanning (same idiom as pow_v3.cpp) ---------
bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

std::size_t skip_ws(const std::string& s, std::size_t i) {
    while (i < s.size() && is_ws(s[i])) ++i;
    return i;
}

// i at opening quote -> index past closing quote, npos on failure
std::size_t skip_string(const std::string& s, std::size_t i) {
    if (i >= s.size() || s[i] != '"') return std::string::npos;
    ++i;
    while (i < s.size()) {
        if (s[i] == '\\') { i += 2; continue; }
        if (s[i] == '"') return i + 1;
        ++i;
    }
    return std::string::npos;
}

std::size_t skip_value(const std::string& s, std::size_t i) {
    if (i >= s.size()) return std::string::npos;
    if (s[i] == '"') return skip_string(s, i);
    if (s[i] == '{' || s[i] == '[') {
        const char open = s[i], close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (i < s.size()) {
            if (s[i] == '"') { i = skip_string(s, i); if (i == std::string::npos) return i; continue; }
            if (s[i] == open) ++depth;
            else if (s[i] == close) { if (--depth == 0) return i + 1; }
            ++i;
        }
        return std::string::npos;
    }
    const std::size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' && !is_ws(s[i])) ++i;
    return (i > start) ? i : std::string::npos;
}

struct Member { std::size_t key_start, value_start, member_end; };

// Walk the members of object `s` (trimmed, s.front()=='{'). Calls fn(key,
// member) for each; returns false on structural failure or duplicate key.
template <typename Fn>
bool walk_members(const std::string& s, Fn fn) {
    std::vector<std::string> seen;
    std::size_t i = skip_ws(s, 1);
    if (i < s.size() && s[i] == '}') return true;
    while (i < s.size()) {
        const std::size_t k_start = i;
        const std::size_t k_end = skip_string(s, i);
        if (k_end == std::string::npos) return false;
        const std::string key = s.substr(k_start + 1, k_end - k_start - 2);
        if (std::find(seen.begin(), seen.end(), key) != seen.end()) return false;
        seen.push_back(key);
        i = skip_ws(s, k_end);
        if (i >= s.size() || s[i] != ':') return false;
        i = skip_ws(s, i + 1);
        const std::size_t v_start = i;
        const std::size_t v_end = skip_value(s, i);
        if (v_end == std::string::npos) return false;
        fn(key, Member{k_start, v_start, v_end});
        i = skip_ws(s, v_end);
        if (i < s.size() && s[i] == ',') { i = skip_ws(s, i + 1); continue; }
        if (i < s.size() && s[i] == '}') return true;
        return false;
    }
    return false;
}

std::string trim_copy(const std::string& s) {
    const std::size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const std::size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
                else out += static_cast<char>(c);
        }
    }
    return out;
}

std::string unquote(const std::string& s, const Member& m) {
    // value must be a plain string without escapes for our keys
    const std::string v = s.substr(m.value_start, m.member_end - m.value_start);
    if (v.size() < 2 || v.front() != '"' || v.back() != '"') return std::string("\x01");
    const std::string inner = v.substr(1, v.size() - 2);
    if (inner.find('\\') != std::string::npos) return std::string("\x01");
    return inner;
}

}  // namespace

std::string merge_extra_flags_gumbel(const std::string& extra_flags, const Decl& decl) {
    if (!is_valid_head(decl.head)) throw std::invalid_argument("invalid gumbel head");
    std::string member;
    if (decl.version == CONTRACT_VERSION_V2) {
        (void)arm_code(decl.arm);                              // throws on a bad arm
        (void)mode_suffix_v2(decl.head, decl.arm, decl.policy_json);   // validates shape
        // keys in sorted order, like pow_v3.canonical_json on the Python side
        member = "\"gumbel\":{\"arm\":\"" + decl.arm + "\",\"head\":\"" + decl.head +
                 "\",\"policy\":" + decl.policy_json + ",\"v\":" +
                 std::to_string(decl.version) + "}";
    } else {
        member = "\"gumbel\":{\"head\":\"" + decl.head + "\",\"v\":" +
                 std::to_string(decl.version) + "}";
    }
    const std::string trimmed = trim_copy(extra_flags);
    if (trimmed.empty()) return "{" + member + "}";
    if (trimmed.front() == '{' && trimmed.back() == '}') {
        std::string body = trimmed;
        bool found = false;
        Member found_m{0, 0, 0};
        const bool ok = walk_members(body, [&](const std::string& key, const Member& m) {
            if (key == "gumbel") { found = true; found_m = m; }
        });
        if (ok) {
            if (found) {
                std::size_t cut_begin = found_m.key_start, cut_end = found_m.member_end;
                const std::size_t after = skip_ws(body, found_m.member_end);
                if (after < body.size() && body[after] == ',') {
                    cut_end = after + 1;
                } else {
                    std::size_t before = found_m.key_start;
                    while (before > 1 && is_ws(body[before - 1])) --before;
                    if (before > 1 && body[before - 1] == ',') cut_begin = before - 1;
                }
                body = trim_copy(body.substr(0, cut_begin) + body.substr(cut_end));
            }
            std::string head = trim_copy(body.substr(0, body.size() - 1));
            const bool empty_object = (head == "{");
            const std::string candidate = head + (empty_object ? "" : ",") + member + "}";
            const auto recovered = extract_gumbel_decl(candidate);
            if (recovered && recovered->head == decl.head && recovered->version == decl.version &&
                recovered->arm == decl.arm && recovered->policy_json == decl.policy_json) {
                return candidate;
            }
        }
    }
    return "{\"_diff\":\"" + json_escape(extra_flags) + "\"," + member + "}";
}

std::optional<Decl> extract_gumbel_decl(const std::string& extra_flags) {
    const std::string s = trim_copy(extra_flags);
    if (s.empty() || s.size() > 4096) return std::nullopt;
    if (s.front() != '{' || s.back() != '}') return std::nullopt;
    bool found = false;
    Member gm{0, 0, 0};
    const bool ok = walk_members(s, [&](const std::string& key, const Member& m) {
        if (key == "gumbel") { found = true; gm = m; }
    });
    if (!ok || !found) return std::nullopt;
    const std::string obj = s.substr(gm.value_start, gm.member_end - gm.value_start);
    if (obj.empty() || obj.front() != '{' || obj.back() != '}') return std::nullopt;
    Decl d;
    bool has_v = false, has_head = false, has_arm = false, has_policy = false, bad = false;
    int n_keys = 0;
    const bool ok2 = walk_members(obj, [&](const std::string& key, const Member& m) {
        ++n_keys;
        const std::string raw = obj.substr(m.value_start, m.member_end - m.value_start);
        if (key == "v") {
            has_v = true;
            if (raw == std::to_string(CONTRACT_VERSION)) d.version = CONTRACT_VERSION;
            else if (raw == std::to_string(CONTRACT_VERSION_V2)) d.version = CONTRACT_VERSION_V2;
            else bad = true;
        } else if (key == "head") {
            has_head = true;
            const std::string h = unquote(obj, m);
            if (!is_valid_head(h)) bad = true; else d.head = h;
        } else if (key == "arm") {
            has_arm = true;
            d.arm = unquote(obj, m);
        } else if (key == "policy") {
            has_policy = true;
            d.policy_json = raw;      // verbatim canonical object (digest input)
        }
    });
    if (!ok2 || bad || !has_v || !has_head) return std::nullopt;
    if (d.version == CONTRACT_VERSION_V2) {
        // v2: exactly {arm, head, policy, v} with a valid arm and an object policy.
        if (!has_arm || !has_policy || n_keys != 4) return std::nullopt;
        if (d.arm != "ONE" && d.arm != "TWO") return std::nullopt;
        if (d.policy_json.empty() || d.policy_json.front() != '{' || d.policy_json.back() != '}')
            return std::nullopt;
    } else {
        d.arm.clear();
        d.policy_json.clear();
    }
    return d;
}

// ------------------------------------------------------------------------- //
// Switch policies
// ------------------------------------------------------------------------- //

std::vector<float> default_feature_vector(const SwitchFeatures& f) {
    std::vector<float> x(8, 0.0f);
    if (!f.ids || !f.probs || !f.race0) return x;
    double entropy = 0.0;
    int support = 0;
    for (std::size_t i = 0; i < f.ids->size(); ++i) {
        if ((*f.ids)[i] < 0) continue;
        ++support;
        const double p = (*f.probs)[i];
        if (p > 0.0) entropy -= p * std::log(p);
    }
    const double p_win = (f.race0->winner_pos >= 0)
        ? (*f.probs)[static_cast<std::size_t>(f.race0->winner_pos)] : 0.0;
    x[0] = static_cast<float>(f.race0->margin());
    x[1] = f.race1 ? static_cast<float>(f.race1->margin()) : 0.0f;
    x[2] = static_cast<float>(p_win);
    x[3] = static_cast<float>(entropy);
    x[4] = static_cast<float>(std::log(static_cast<double>(std::max(1, support))));
    x[5] = static_cast<float>(f.step) / static_cast<float>(WINDOW_BITS);
    x[6] = static_cast<float>(f.remaining()) / static_cast<float>(WINDOW_BITS);
    x[7] = static_cast<float>(f.retries_used);
    return x;
}

MlpSwitchPolicy::MlpSwitchPolicy(std::vector<Layer> layers, bool needs_draw1, float threshold)
    : layers_(std::move(layers)), needs_draw1_(needs_draw1), threshold_(threshold) {
    if (layers_.empty()) throw std::invalid_argument("mlp policy needs at least one layer");
    for (const auto& l : layers_) {
        if (l.w.size() != static_cast<std::size_t>(l.in) * l.out || l.b.size() != static_cast<std::size_t>(l.out)) {
            throw std::invalid_argument("mlp layer shape mismatch");
        }
    }
    if (layers_.back().out != 1) throw std::invalid_argument("mlp policy must end in one logit");
}

MlpSwitchPolicy MlpSwitchPolicy::load(const std::string& path, bool needs_draw1, float threshold) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open mlp policy file: " + path);
    std::string magic;
    int n_layers = 0;
    in >> magic >> n_layers;
    if (magic != "mlp" || n_layers <= 0) throw std::runtime_error("bad mlp policy header");
    std::vector<Layer> layers;
    for (int l = 0; l < n_layers; ++l) {
        Layer L;
        in >> L.in >> L.out;
        if (!in || L.in <= 0 || L.out <= 0) throw std::runtime_error("bad mlp layer shape");
        L.w.resize(static_cast<std::size_t>(L.in) * L.out);
        L.b.resize(static_cast<std::size_t>(L.out));
        for (auto& v : L.w) in >> v;
        for (auto& v : L.b) in >> v;
        if (!in) throw std::runtime_error("truncated mlp policy file");
        layers.push_back(std::move(L));
    }
    return MlpSwitchPolicy(std::move(layers), needs_draw1, threshold);
}

float MlpSwitchPolicy::forward(const std::vector<float>& x_in) const {
    std::vector<float> x = x_in;
    for (std::size_t li = 0; li < layers_.size(); ++li) {
        const Layer& L = layers_[li];
        if (x.size() != static_cast<std::size_t>(L.in)) throw std::invalid_argument("mlp input width mismatch");
        std::vector<float> y(static_cast<std::size_t>(L.out));
        for (int o = 0; o < L.out; ++o) {
            float acc = L.b[static_cast<std::size_t>(o)];
            for (int i = 0; i < L.in; ++i) acc += L.w[static_cast<std::size_t>(o) * L.in + i] * x[static_cast<std::size_t>(i)];
            y[static_cast<std::size_t>(o)] = (li + 1 < layers_.size()) ? std::max(0.0f, acc) : acc;
        }
        x.swap(y);
    }
    return x[0];
}

bool MlpSwitchPolicy::decide(const SwitchFeatures& f) const {
    return forward(default_feature_vector(f)) > threshold_;
}

static std::string env_or(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : def;
}

std::unique_ptr<SwitchPolicy> policy_from_spec(const std::string& spec_in) {
    const std::string spec = trim_copy(spec_in.empty() ? "never" : spec_in);
    if (spec == "never") return std::make_unique<NeverSwitchPolicy>();
    if (spec == "always") return std::make_unique<AlwaysSwitchPolicy>();
    if (spec.rfind("margin_stab:", 0) == 0) {
        // theta=<t>,ratio=<r> in either order
        double theta = 0.0, ratio = 0.0;
        bool has_theta = false, has_ratio = false;
        std::string body = spec.substr(12);
        std::size_t start = 0;
        while (start <= body.size()) {
            const std::size_t comma = body.find(',', start);
            const std::string part = body.substr(start, comma == std::string::npos ? std::string::npos
                                                                                   : comma - start);
            const std::size_t eq = part.find('=');
            if (eq == std::string::npos) throw std::invalid_argument("margin_stab needs theta=..,ratio=..");
            const std::string k = trim_copy(part.substr(0, eq));
            const double v = std::stod(part.substr(eq + 1));
            if (k == "theta" && !has_theta) { theta = v; has_theta = true; }
            else if (k == "ratio" && !has_ratio) { ratio = v; has_ratio = true; }
            else throw std::invalid_argument("margin_stab takes exactly theta and ratio");
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        if (!has_theta || !has_ratio) throw std::invalid_argument("margin_stab needs theta and ratio");
        return std::make_unique<RaceMarginStabPolicy>(theta, ratio);
    }
    if (spec.rfind("margin:", 0) == 0) return std::make_unique<RaceMarginPolicy>(std::stod(spec.substr(7)));
    if (spec.rfind("mlp:", 0) == 0) {
        const bool needs1 = env_or("POW_GUMBEL_POLICY_NEEDS_DRAW1", "0") == "1";
        return std::make_unique<MlpSwitchPolicy>(MlpSwitchPolicy::load(spec.substr(4), needs1));
    }
    throw std::invalid_argument("unknown switch policy spec '" + spec + "'");
}

std::unique_ptr<SwitchPolicy> policy_from_env() {
    return policy_from_spec(env_or("POW_GUMBEL_SWITCH_POLICY", "never"));
}

int max_retries_from_env() {
    try { return std::max(0, std::stoi(env_or("POW_GUMBEL_MAX_RETRIES", "256"))); }
    catch (...) { return static_cast<int>(WINDOW_BITS); }
}

double race_atol_from_env() {
    try { return std::stod(env_or("POW_GUMBEL_RACE_ATOL", "1e-9")); }
    catch (...) { return DEFAULT_RACE_ATOL; }
}

// ------------------------------------------------------------------------- //
// Sampler step + verifier check
// ------------------------------------------------------------------------- //

float digest_to_u_f32(const uint8_t* b) {
    const float b0 = static_cast<float>(b[0]), b1 = static_cast<float>(b[1]);
    const float b2 = static_cast<float>(b[2]), b3 = static_cast<float>(b[3]);
    return (b0 + b1 * 256.0f + b2 * 65536.0f + b3 * 16777216.0f) / 4294967296.0f;
}

SampleResult sample(const std::vector<int64_t>& ids, const std::vector<float>& weights,
                    const uint8_t* digest32, const SwitchPolicy* policy,
                    int step, int retries_used, int max_retries) {
    SampleResult out;
    const auto seed0 = seed_from_digest(digest32, 0);
    const RaceResult r0 = race(ids, weights, seed0);
    const std::vector<double> probs = support_probs(weights);
    out.margin0 = r0.margin();

    out.u_value = digest_to_u_f32(digest32);          // sampling_u == production
    auto finish = [&](const RaceResult& r, int draw) {
        out.draw = draw;
        out.token_id = r.winner_id;
        out.margin_used = r.margin();
        out.prob = (r.winner_pos >= 0)
            ? static_cast<float>(probs[static_cast<std::size_t>(r.winner_pos)]) : 0.0f;
    };

    if (policy && dynamic_cast<const NeverSwitchPolicy*>(policy) == nullptr) {
        // Both draws are evaluated (the miner can always compute both; the
        // threat model assumes it does) and the policy picks KEEP / SWITCH.
        const auto seed1 = seed_from_digest(digest32, 1);
        const RaceResult r1 = race(ids, weights, seed1);
        SwitchFeatures f;
        f.ids = &ids; f.weights = &weights; f.probs = &probs; f.race0 = &r0;
        f.race1 = policy->needs_draw1() ? &r1 : nullptr;
        f.step = step; f.retries_used = retries_used;
        out.switch_requested = policy->decide(f);
        // Only an OBSERVABLE switch (draw-1 winner differs from the draw-0
        // winner) is a SWITCH: a same-token fire stays KEEP (draw 0) and so
        // never consumes the window budget.
        if (out.switch_requested && retries_used < max_retries && r1.winner_id != r0.winner_id) {
            finish(r1, 1);
            return out;
        }
    }
    finish(r0, 0);
    return out;
}

VerifyResult verify_race(const std::vector<int64_t>& ids, const std::vector<float>& weights,
                         const uint8_t* digest32, int64_t chosen, double atol) {
    VerifyResult v;
    v.u_value = digest_to_u_f32(digest32);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == chosen && ids[i] >= 0) { v.chosen_pos = static_cast<int>(i); break; }
    }
    const std::vector<double> probs = support_probs(weights);
    if (v.chosen_pos >= 0) v.chosen_prob = probs[static_cast<std::size_t>(v.chosen_pos)];
    for (int d = 0; d < 2; ++d) {
        const auto seed = seed_from_digest(digest32, static_cast<uint64_t>(d));
        const RaceResult r = race(ids, weights, seed);
        if (d == 0) { v.winner0_id = r.winner_id; v.margin0 = r.margin(); }
        if (v.chosen_pos < 0 || r.winner_pos < 0) continue;
        const double s_chosen = r.scores[static_cast<std::size_t>(v.chosen_pos)];
        const double slack = s_chosen - r.best;
        // canonical winner (argmax, ties -> smallest id) at atol 0: a tied loser is
        // rejected; atol > 0 is the research relaxation only.
        const bool wins = (r.winner_pos == v.chosen_pos) || (atol > 0.0 && slack >= -atol);
        if (std::isfinite(s_chosen) && wins) {
            v.ok = true;
            v.draw = d;
            v.slack = slack;
            return v;
        }
        if (d == 0) v.slack = slack;
    }
    return v;
}

}  // namespace gumbel
