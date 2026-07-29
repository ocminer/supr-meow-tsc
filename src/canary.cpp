#include "canary.h"

#include <openssl/sha.h>

#include <cstring>
#include <cstdlib>

namespace meow {

bool canary_parse_seed(const std::string& hex, std::vector<uint8_t>& out) {
    if (hex.size() != 64) return false;
    out.clear();
    out.reserve(32);
    for (size_t i = 0; i < 64; i += 2) {
        auto nib = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int hi = nib(hex[i]), lo = nib(hex[i + 1]);
        if (hi < 0 || lo < 0) return false;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

std::vector<int32_t> canary_derive_prompt(const std::vector<uint8_t>& seed32,
                                          uint32_t w, int32_t n_vocab) {
    // EXPERIMENT (POW_CANARY_SUPPORT=N): narrow the sampled support to the
    // first N token ids. Qwen BPE assigns ids in roughly merge/frequency
    // order, so a low modulus approximates "the N most frequent tokens"
    // without any table to pin or version. Full-vocab soup (the spec as
    // written) yields rare multilingual/emoji tokens, which drives the
    // model into a low-entropy continuation and trips the chain's reuse
    // guard — measured ~10x the honest reject rate. 0/unset = spec §3.
    static const int32_t support = [](){
        const char* e = std::getenv("POW_CANARY_SUPPORT");
        return e ? std::atoi(e) : 0;
    }();
    // POW_CANARY_OFFSET=M skips the first M ids. Qwen's first 256 are raw
    // BYTE tokens (invalid UTF-8 on their own): at support 8192 that is 3.1%
    // per draw but 1-(1-.031)^32 = 63% of 32-token prompts containing at
    // least one — a majority-of-prompts effect, not a 3% one.
    static const int32_t offset = [](){
        const char* e = std::getenv("POW_CANARY_OFFSET");
        return e ? std::atoi(e) : 0;
    }();
    const int32_t cap = (support > 0 && support < n_vocab) ? support : n_vocab;
    const int32_t lo  = (offset > 0 && offset < cap) ? offset : 0;
    const int32_t mod = cap - lo;
    std::vector<int32_t> toks;
    toks.reserve(32);
    uint8_t msg[32 + 4 + 4];
    std::memcpy(msg, seed32.data(), 32);
    // LE32(w)
    msg[32] = uint8_t(w); msg[33] = uint8_t(w >> 8);
    msg[34] = uint8_t(w >> 16); msg[35] = uint8_t(w >> 24);
    for (uint32_t i = 0; i < 32; ++i) {
        msg[36] = uint8_t(i); msg[37] = uint8_t(i >> 8);
        msg[38] = uint8_t(i >> 16); msg[39] = uint8_t(i >> 24);
        uint8_t d[32];
        SHA256(msg, sizeof(msg), d);
        // LE_U64(d[0:8]) mod n_vocab
        uint64_t v = 0;
        for (int b = 7; b >= 0; --b) v = (v << 8) | d[b];
        toks.push_back(lo + static_cast<int32_t>(v % static_cast<uint64_t>(mod)));
    }
    return toks;
}

std::string canary_prompt_key(const std::vector<int32_t>& toks) {
    std::vector<uint8_t> msg;
    msg.reserve(toks.size() * 4);
    for (int32_t t : toks) {
        const uint32_t u = static_cast<uint32_t>(t);
        msg.push_back(uint8_t(u)); msg.push_back(uint8_t(u >> 8));
        msg.push_back(uint8_t(u >> 16)); msg.push_back(uint8_t(u >> 24));
    }
    uint8_t d[32];
    SHA256(msg.data(), msg.size(), d);
    static const char* hexd = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) { out.push_back(hexd[d[i] >> 4]); out.push_back(hexd[d[i] & 0xF]); }
    return out;
}

}  // namespace meow
