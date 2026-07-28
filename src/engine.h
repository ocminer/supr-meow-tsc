// =============================================================================
// engine.h — embedded inference engine.
//
// This is what makes supr-meow-tsc self-contained: llama.cpp is linked IN, so
// there is no external server process, no HTTP hop and no Python. One binary
// loads the model and generates the token windows that proofs are built from.
//
// Multi-GPU is DATA parallel, not tensor parallel: each selected device gets
// its own full copy of the model and its own contexts. A miner wants N cards
// working independently — tensor-splitting a small model across cards would
// add interconnect traffic and make one slow card throttle the rest.
//
// bf16 is enforced, not chosen. The proof declares its compute precision and
// the network verifies it against the model checkpoint, so a quantised GGUF
// would run faster and produce proofs that are rejected on-chain.
// =============================================================================
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;

namespace meow {

struct EngineConfig {
    std::string       model_path;
    std::vector<int>  devices;              // CUDA ordinals, from -d
    int               slots_per_device = 8; // concurrent windows per GPU
    int               window_tokens    = 256;
    // HARD RULE: ctx per slot must exceed prompt + window, or generation stops
    // one token short of a full window and the miner emits NOTHING while
    // looking perfectly healthy. Learned the hard way; kept as a guard rail.
    int               ctx_per_slot     = 2048;
    int               threads          = 0;  // 0 = auto
};

struct DeviceEngineStats {
    int      device      = -1;
    uint64_t tokens      = 0;
    double   seconds     = 0.0;
    double   tokens_per_s = 0.0;
    size_t   vram_used   = 0;
};

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    // Loads one model instance per selected device. Returns false with a
    // human-readable reason — a miner that cannot load its model must say why,
    // not crash.
    bool load(const EngineConfig& cfg, std::string& error,
              const std::function<void(const std::string&)>& progress = nullptr);

    // Chooses the next token from the logits. The proof-of-inference sampler
    // installs itself here: it must RETURN the token it selected, because the
    // transcript and the model's own state have to advance together — feeding
    // back a different token than the one recorded makes the proof meaningless.
    // `context` is the window's tokens so far, which the sampler needs.
    // Return < 0 to abort the window.
    using TokenChooser = std::function<int(const float* logits, int n_vocab,
                                           const std::vector<int64_t>& context)>;

    // Generate `window_tokens` on one device, forcing generation to the full
    // window (no early EOS) so a proof window always closes.
    // `chooser` may be null, in which case argmax is used (benchmark path).
    // Returns tokens generated, or -1 on error.
    int generate_window(int device_slot, const std::string& prompt,
                        const TokenChooser& chooser, std::string& error);

    // Batched variant: N windows advance in lock-step on ONE device — one
    // llama_decode carries one token per stream, so the GPU amortises its
    // launch and weight-read cost across streams instead of idling between
    // single-token decodes. The chooser additionally receives the stream id,
    // which is the sampler's sequence id.
    using BatchChooser = std::function<int(int stream, const float* logits, int n_vocab,
                                           const std::vector<int64_t>& context)>;
    // Called once per step with every stream's logits, BEFORE the per-stream
    // chooser calls — the sampler batches its GPU work across streams here.
    using BatchPrepare = std::function<void(const std::vector<const float*>&, int n_vocab)>;

    // Returns windows completed (== streams on success), -1 on error.
    int generate_windows_batched(int device_slot,
                                 const std::vector<std::string>& prompts,
                                 const BatchChooser& chooser, std::string& error,
                                 const BatchPrepare& prepare = nullptr);

    // Throughput measurement across every loaded device, used by --benchmark.
    std::vector<DeviceEngineStats> benchmark(int windows_per_device, std::string& error);

    int  n_vocab() const;
    bool loaded() const { return !instances_.empty(); }
    const EngineConfig& config() const { return cfg_; }

private:
    struct Instance;                      // one model+context set, per device
    std::vector<std::unique_ptr<Instance>> instances_;
    EngineConfig cfg_;
    bool backend_ready_ = false;
};

}  // namespace meow
