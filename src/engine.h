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
    // Independent worker CONTEXTS per device. Each is its own KV cache, its
    // own mining thread, and its own sampler coordinator (separate scratch =
    // no data race), so N workers use N CPU cores for the per-token sampler
    // tail — the real wall once the sort moved to the GPU — and keep the card
    // fed while any one worker is on its CPU tail. They share the model weights
    // (loaded once per device), so the VRAM cost is only extra KV cache.
    int               workers_per_device = 1;
    int               window_tokens    = 256;
    // HARD RULE: ctx per slot must exceed prompt + window, or generation stops
    // one token short of a full window and the miner emits NOTHING while
    // looking perfectly healthy. Learned the hard way; kept as a guard rail.
    //
    // Sized JUST above one window (prompt ~24 + 256 window). A mining context
    // is cleared every window and never grows, so a large ctx only bloats the
    // KV cache — which both slows attention and caps how many slots fit in
    // VRAM. 384 keeps margin while shrinking KV ~5x vs the old 2048, which is
    // what lets the batch scale past ~48 slots.
    int               ctx_per_slot     = 384;
    int               threads          = 0;  // 0 = auto
    // Double-buffered decode (generate_windows_double): doubles n_seq_max and
    // n_ctx so two full window batches coexist in the KV cache.
    bool              double_buffer    = false;
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

    // Like the above but the sampler decides ALL streams' tokens for a step in
    // one call — which is what lets a caller run the per-stream sampler tail in
    // PARALLEL. `step` receives every stream's logits and running context and
    // must fill `out_tokens` (one per stream). The decode itself stays
    // single-threaded on this device (batched), so nothing in llama is shared
    // across threads.
    // `dev_logits` is non-null in GPU-resident-logits mode: the DEVICE base of
    // llama's output tensor (S contiguous rows of n_vocab floats, stream
    // order). The host `logits` pointers are then STALE and must not be read.
    using StepSampler = std::function<bool(const std::vector<const float*>& logits,
                                           const float* dev_logits,
                                           int n_vocab,
                                           const std::vector<std::vector<int64_t>>& ctx,
                                           std::vector<int>& out_tokens)>;
    int generate_windows_stepwise(int device_slot,
                                  const std::vector<std::string>& prompts,
                                  const StepSampler& step, std::string& error);
    // Pre-tokenized variant (canary jobs): the ids are fed VERBATIM — no
    // tokenizer, no BOS/EOS insertion. The pool derives these same ids from
    // the job's prompt seed, which is what makes its references computable.
    int generate_windows_stepwise_tok(int device_slot,
                                  const std::vector<std::vector<int32_t>>& prompt_tokens,
                                  const StepSampler& step, std::string& error);

    // Double-buffered variant: TWO window batches (A = seqs [0,S), B = seqs
    // [S,2S)) alternate on one context, so the GPU decodes batch B while the
    // CPU samples batch A from a private D2D copy of its logits — the decode
    // pipe never drains. Requires GPU-resident logits (the device pointer is
    // the thing being copied); returns -2 if that is unavailable so the
    // caller can fall back to the single-batch path. cfg.double_buffer must
    // have been set at load() time (it doubles n_ctx / n_seq_max).
    // Returns 2*S windows on success.
    int generate_windows_double(int device_slot,
                                const std::vector<std::string>& prompts_a,
                                const std::vector<std::string>& prompts_b,
                                const StepSampler& step_a,
                                const StepSampler& step_b,
                                std::string& error);
    // Pre-tokenized variant — see generate_windows_stepwise_tok.
    int generate_windows_double_tok(int device_slot,
                                const std::vector<std::vector<int32_t>>& toks_a,
                                const std::vector<std::vector<int32_t>>& toks_b,
                                const StepSampler& step_a,
                                const StepSampler& step_b,
                                std::string& error);

    // Throughput measurement across every loaded device, used by --benchmark.
    std::vector<DeviceEngineStats> benchmark(int windows_per_device, std::string& error);

    int  n_vocab() const;
    bool loaded() const { return !instances_.empty(); }
    const EngineConfig& config() const { return cfg_; }
    // Total worker contexts = devices * workers_per_device. main() runs one
    // mining thread per worker; device_slot in the calls below is a WORKER id.
    int  worker_count() const { return static_cast<int>(instances_.size()); }
    int  worker_device(int w) const;

private:
    struct Instance;                      // one model+context set, per device
    std::vector<std::unique_ptr<Instance>> instances_;
    EngineConfig cfg_;
    bool backend_ready_ = false;
};

}  // namespace meow
