#include "model_profile.h"
#include <openssl/evp.h>
#include <array>
#include <fstream>
#include <memory>

namespace meow {
bool verify_q8_model(const std::string& path, std::string& error) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streampos(8709518304ULL)) {
        error = "--q8 requires the registered Qwen3-8B Q8_0 full-v2 model (8709518304 bytes)";
        return false;
    }
    file.seekg(0);
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),EVP_MD_CTX_free);
    if (!ctx || EVP_DigestInit_ex(ctx.get(),EVP_sha256(),nullptr) != 1) {
        error = "cannot initialize model SHA-256"; return false;
    }
    std::array<char,1024*1024> buffer;
    while (file.read(buffer.data(),buffer.size()) || file.gcount()) {
        if (EVP_DigestUpdate(ctx.get(),buffer.data(),size_t(file.gcount())) != 1) {
            error = "model SHA-256 update failed"; return false;
        }
    }
    if (file.bad()) { error = "model read failed during SHA-256 verification"; return false; }
    std::array<unsigned char,EVP_MAX_MD_SIZE> digest{};
    unsigned length = 0;
    if (EVP_DigestFinal_ex(ctx.get(),digest.data(),&length) != 1 || length != 32) {
        error = "model SHA-256 finalization failed"; return false;
    }
    std::string hex;
    constexpr char alphabet[] = "0123456789abcdef";
    for (unsigned i=0;i<length;++i) { hex+=alphabet[digest[i]>>4];hex+=alphabet[digest[i]&15]; }
    if (hex != q8_model_sha256) {
        error = "Q8 model SHA-256 mismatch; replace the model with the verified full-v2 artifact";
        return false;
    }
    return true;
}
}
