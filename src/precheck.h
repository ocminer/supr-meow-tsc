// =============================================================================
// precheck.h — pre-submit consensus pre-check.
//
// Every finished proof is decoded (read-only, after it has been serialized)
// and run through the verifier's own window rules, in the verifier's order,
// using the UNMODIFIED TensorCash v1.2.2 sources in vendor/tensorcash-v1.2.2:
//
//   strict        canonical kernel pass + chosen-token / chosen_probs binding
//   near-pin      proof v4 structure rule, N >= 175 near-pinned positions
//   candidate-band  V90 candidate-band condition
//   prompt-scaffold post-sunset copied-span scaffold rule
//   flat-region   post-sunset flat-region rule
//   credit-tier   B_cred tier (below B_FLOOR / below B_FREE)
//   anti-parrot   anti-parrot hard gate (hardrun, state, bcap, backload, reuse)
//
// Evaluation stops at the first rejecting gate. The proof bytes are never
// modified: the check only decides whether a proof is submitted at all.
//
//   --precheck=off     the v0.7.0 path; nothing is evaluated.
//   --precheck=shadow  evaluate and log a verdict for every proof, submit all.
//   --precheck=on      evaluate; a proof a gate rejects is not submitted.
//
// Fail-open: a proof that cannot be evaluated (unexpected shape, float
// environment refused, internal error) is always submitted.
//
// This header deliberately includes no upstream header: the upstream
// namespaces are renamed with compiler definitions that only the pre-check's
// own translation units see.
// =============================================================================
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace meow::precheck {

enum class Mode { Off, Shadow, On };

// Verifier order. The numeric value is the bit in a gate mask.
enum class Gate : int {
    Strict = 0,
    NearPin,
    CandidateBand,
    PromptScaffold,
    FlatRegion,
    CreditTier,
    AntiParrot,
    Count
};
constexpr int kGateCount = static_cast<int>(Gate::Count);
constexpr uint32_t gate_bit(Gate g) { return 1u << static_cast<int>(g); }

enum class Outcome { Pass, Reject, Unavailable };

struct Verdict {
    Outcome     outcome = Outcome::Unavailable;
    Gate        gate = Gate::Count;   // the rejecting gate when outcome == Reject
    std::string reason;               // Reject: the verifier's reason; Unavailable: why
    std::string hash_hex;             // Proof.hash, hex in wire order == verifier "Hash ID"
    bool        is_solution = false;  // Proof.is_solution: the proof met the block target
    double      ms = 0.0;             // wall time of the evaluation
};

// A decoded proof v4 window (what the gates read). Exposed for the tests.
struct Window {
    std::vector<uint32_t> row_len;    // 256 entries
    std::vector<float>    logits;     // rows concatenated, row s has row_len[s] entries
    std::vector<uint32_t> ids;        // same layout as logits
    std::vector<uint32_t> chosen;     // 256 chosen token ids
    std::vector<float>    chosen_probs;
    std::vector<uint32_t> prompt;     // full prompt_tokens
    std::vector<uint8_t>  pad;        // pad_mask as stored (may be empty)
    std::string hash_hex;
    bool        is_solution = false;
};

// True when the pre-check is compiled into this build (false under MSVC).
bool available();
// "tensorcash v1.2.2 (48d1344)" or a note that the check is compiled out.
const char* rules_version();

const char* gate_name(Gate g);
const char* mode_name(Mode m);
bool parse_mode(const std::string& s, Mode& out);
// Bits of the gates this build implements.
uint32_t implemented_gates();
// "all", "none", or a comma list of gate names; a leading '-' removes a gate
// from the set built so far ("all,-credit-tier"). Unknown names fail.
bool parse_gates(const std::string& spec, uint32_t& mask, std::string& error);
std::string gates_string(uint32_t mask);

// Sets round-to-nearest and clears flush-to-zero / denormals-are-zero on the
// calling thread if needed, then re-checks the canonical kernel's contract.
bool ensure_float_environment(std::string* why = nullptr);

// Decode a serialized MiningResponse into a Window. Read-only. False (with
// `why`) when the bytes are not a complete proof v4 window.
bool decode(const uint8_t* data, size_t n, Window& out, std::string& why) noexcept;

// Run the enabled gates in verifier order and stop at the first rejection.
Verdict evaluate_window(const Window& w, uint32_t gate_mask) noexcept;

// decode + evaluate_window. Never throws; anything unexpected is Unavailable.
Verdict evaluate(const uint8_t* data, size_t n, uint32_t gate_mask) noexcept;

}  // namespace meow::precheck
