// =============================================================================
// canary.h — pool-derived prompts (CANARY-JOBS-SPEC).
//
// When a job carries a prompt seed (mining.notify field 9), every window's
// prompt is DERIVED, not invented: 32 token ids from SHA-256 of
// (seed ‖ LE32(w) ‖ LE32(i)). The pool can therefore precompute any prompt a
// compliant miner will use — which is what makes known-answer canary windows
// possible — and a prompt that is NOT derivable is itself a compliance
// signal. `w` is a dense per-connection counter over (session, seed): one
// atomic across all streams, reset when the seed changes (seed is per job).
//
// Encoding is normative and pinned by cross-implementation test vectors
// (spec §9/§9.1): d = SHA256(seed_raw32 ‖ LE32(w) ‖ LE32(i));
// tok[i] = LE_U64(d[0:8]) mod n_vocab; 32 ids, no BOS, no EOS, no tokenizer.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace meow {

// 64-hex → 32 raw bytes. Returns false on malformed input.
bool canary_parse_seed(const std::string& hex, std::vector<uint8_t>& out);

// The 32 derived token ids for (seed, w). n_vocab comes from mining.set_model
// — never hardcoded (spec §3).
std::vector<int32_t> canary_derive_prompt(const std::vector<uint8_t>& seed32,
                                          uint32_t w, int32_t n_vocab);

// Prompt key = SHA256(LE32(tok[0]) ‖ … ‖ LE32(tok[31])), lowercase hex — the
// reference-table join key (spec §6).
std::string canary_prompt_key(const std::vector<int32_t>& toks);

}  // namespace meow
