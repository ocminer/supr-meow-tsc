#include "canary.h"

#include <openssl/sha.h>

#include <cstring>

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
        toks.push_back(static_cast<int32_t>(v % static_cast<uint64_t>(n_vocab)));
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
