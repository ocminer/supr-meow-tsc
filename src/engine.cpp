#include "engine.h"

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef POW_GPU_SORT_ENABLED
extern "C" bool pow_gpu_is_device_ptr(const void* p);
extern "C" void* pow_gpu_device_alloc(size_t bytes);
extern "C" void  pow_gpu_device_free(void* p);
extern "C" bool  pow_gpu_d2d_copy_sync(void* dst, const void* src, size_t bytes);
#else
static inline bool pow_gpu_is_device_ptr(const void*) { return false; }
static inline void* pow_gpu_device_alloc(size_t) { return nullptr; }
static inline void  pow_gpu_device_free(void*) {}
static inline bool  pow_gpu_d2d_copy_sync(void*, const void*, size_t) { return false; }
#endif

namespace meow {

struct InferenceEngine::Instance {
    int             device      = -1;
    llama_model*    model       = nullptr;   // shared; owned by owns_model==true
    llama_context*  ctx         = nullptr;
    // Second context for the double-buffered decode: each batch keeps a
    // STABLE sequence set on its own context, so llama's graph reuse
    // survives — alternating seq-id sets on ONE context forced a graph
    // rebuild every decode (measured: ~2x decode cost, a net LOSS).
    llama_context*  ctx_b       = nullptr;
    bool            owns_model  = false;      // only the first worker frees it
    void*           dbuf[2]     = {nullptr, nullptr};   // double-buffer logits copies
    size_t          dbuf_bytes  = 0;

    ~Instance() {
        pow_gpu_device_free(dbuf[0]);
        pow_gpu_device_free(dbuf[1]);
        if (ctx_b) llama_free(ctx_b);
        if (ctx)   llama_free(ctx);
        if (owns_model && model) llama_model_free(model);
    }
};

int InferenceEngine::worker_device(int w) const {
    return (w >= 0 && w < (int)instances_.size()) ? instances_[w]->device : -1;
}

InferenceEngine::InferenceEngine() = default;

InferenceEngine::~InferenceEngine() {
    instances_.clear();
    if (backend_ready_) llama_backend_free();
}

int InferenceEngine::n_vocab() const {
    if (instances_.empty() || !instances_[0]->model) return 0;
    return llama_vocab_n_tokens(llama_model_get_vocab(instances_[0]->model));
}

bool InferenceEngine::load(const EngineConfig& cfg, std::string& error,
                           const std::function<void(const std::string&)>& progress) {
    cfg_ = cfg;
    if (cfg.model_path.empty()) {
        error = "no model given — pass --model /path/to/model.gguf";
        return false;
    }
    if (cfg.devices.empty()) {
        error = "no devices selected";
        return false;
    }
    // Guard rail, not a suggestion: see EngineConfig.
    if (cfg.ctx_per_slot <= cfg.window_tokens + 64) {
        error = "ctx_per_slot (" + std::to_string(cfg.ctx_per_slot) + ") must exceed the "
                + std::to_string(cfg.window_tokens) + "-token window plus prompt headroom; "
                "otherwise generation truncates and no proof is ever produced";
        return false;
    }

    if (!backend_ready_) {
        llama_backend_init();
        backend_ready_ = true;
    }
    // llama.cpp is chatty on load; a miner's log should be its own.
    // Only real errors. Note CONT (continuation) sorts ABOVE ERROR in the enum,
    // so a naive `>= ERROR` filter lets llama's load-progress dots through.
    llama_log_set([](ggml_log_level level, const char* text, void* /*ud*/) {
        if (level == GGML_LOG_LEVEL_ERROR && text) std::fprintf(stderr, "  [llama] %s", text);
    }, nullptr);

    const int W = std::max(1, cfg.workers_per_device);
    for (int dev : cfg.devices) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999;                       // the whole model on the GPU
        mp.main_gpu     = dev;
        mp.split_mode   = LLAMA_SPLIT_MODE_NONE;     // data parallel, see engine.h
        mp.use_mmap     = true;

        if (progress) progress("loading model on GPU " + std::to_string(dev));
        llama_model* model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
        if (!model) {
            error = "failed to load model '" + cfg.model_path + "' on GPU " + std::to_string(dev) +
                    " — check the path, and that the file is a bf16 GGUF of the registered model";
            return false;
        }

        for (int w = 0; w < W; ++w) {
            auto inst = std::make_unique<Instance>();
            inst->device     = dev;
            inst->model      = model;
            inst->owns_model = (w == 0);   // one owner frees the shared weights

            llama_context_params cp = llama_context_default_params();
            cp.n_ctx     = static_cast<uint32_t>(cfg.ctx_per_slot * cfg.slots_per_device);
            // The prompt pass submits every stream's prompt in ONE batch, so
            // n_batch must cover slots * prompt_len or llama_decode fails and
            // the miner produces nothing. Sized generously from the slot count.
            cp.n_batch   = std::max(2048u, static_cast<uint32_t>(cfg.slots_per_device) * 64u);
            cp.n_ubatch  = 512;
            cp.n_seq_max = static_cast<uint32_t>(cfg.slots_per_device);
            cp.n_threads = cfg.threads > 0 ? cfg.threads : 8;
            cp.n_threads_batch = cp.n_threads;

            inst->ctx = llama_init_from_model(inst->model, cp);
            if (!inst->ctx) {
                error = "failed to create context " + std::to_string(w) + " on GPU " +
                        std::to_string(dev) + " (try fewer slots/workers or a smaller ctx)";
                return false;
            }
            if (cfg.double_buffer) {
                inst->ctx_b = llama_init_from_model(inst->model, cp);
                if (!inst->ctx_b) {
                    error = "failed to create the double-buffer context on GPU " +
                            std::to_string(dev) + " (try fewer slots or a smaller ctx)";
                    return false;
                }
            }
            instances_.push_back(std::move(inst));
        }
        if (progress) {
            progress("GPU " + std::to_string(dev) + " ready — " + std::to_string(W) +
                     " workers x " + std::to_string(cfg.slots_per_device) + " slots x " +
                     std::to_string(cfg.ctx_per_slot) + " ctx");
        }
    }
    return true;
}

int InferenceEngine::generate_window(int device_slot, const std::string& prompt,
                                     const TokenChooser& chooser,
                                     std::string& error) {
    if (device_slot < 0 || device_slot >= static_cast<int>(instances_.size())) {
        error = "invalid device slot";
        return -1;
    }
    Instance& in = *instances_[device_slot];
    const llama_vocab* vocab = llama_model_get_vocab(in.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    // tokenize
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                           toks.data(), static_cast<int32_t>(toks.size()), true, false);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                           toks.data(), static_cast<int32_t>(toks.size()), true, false);
    }
    if (n <= 0) { error = "tokenization failed"; return -1; }
    toks.resize(n);

    llama_memory_clear(llama_get_memory(in.ctx), true);

    llama_batch batch = llama_batch_get_one(toks.data(), static_cast<int32_t>(toks.size()));
    if (llama_decode(in.ctx, batch) != 0) { error = "prompt decode failed"; return -1; }

    // The window's tokens so far. The PoI sampler folds this into every
    // sampling value, so it must be the real sequence, prompt included.
    std::vector<int64_t> context;
    context.reserve(toks.size() + cfg_.window_tokens);
    for (llama_token t : toks) context.push_back(static_cast<int64_t>(t));

    int produced = 0;
    llama_token cur = 0;
    for (int i = 0; i < cfg_.window_tokens; ++i) {
        const float* logits = llama_get_logits_ith(in.ctx, -1);
        if (!logits) { error = "no logits returned"; return -1; }

        // EOS is IGNORED on purpose — a window must always reach its full
        // length or the proof never closes.
        if (chooser) {
            const int chosen = chooser(logits, n_vocab, context);
            if (chosen < 0) { error = "sampler aborted the window"; return -1; }
            cur = static_cast<llama_token>(chosen);
        } else {
            int best = 0;
            float bestv = logits[0];
            for (int t = 1; t < n_vocab; ++t) if (logits[t] > bestv) { bestv = logits[t]; best = t; }
            cur = best;
        }
        context.push_back(static_cast<int64_t>(cur));
        ++produced;

        llama_batch nb = llama_batch_get_one(&cur, 1);
        if (llama_decode(in.ctx, nb) != 0) { error = "decode failed mid-window"; return -1; }
    }
    return produced;
}

int InferenceEngine::generate_windows_batched(int device_slot,
                                              const std::vector<std::string>& prompts,
                                              const BatchChooser& chooser,
                                              std::string& error,
                                              const BatchPrepare& prepare) {
    if (device_slot < 0 || device_slot >= static_cast<int>(instances_.size())) {
        error = "invalid device slot"; return -1;
    }
    Instance& in = *instances_[device_slot];
    const llama_vocab* vocab = llama_model_get_vocab(in.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int S = static_cast<int>(prompts.size());
    if (S < 1 || S > cfg_.slots_per_device) { error = "bad stream count"; return -1; }

    // Tokenize every stream's prompt.
    std::vector<std::vector<llama_token>> toks(S);
    for (int s = 0; s < S; ++s) {
        std::vector<llama_token> t(prompts[s].size() + 8);
        int n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                               t.data(), (int32_t)t.size(), true, false);
        if (n < 0) { t.resize(-n);
            n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                               t.data(), (int32_t)t.size(), true, false); }
        if (n <= 0) { error = "tokenization failed"; return -1; }
        t.resize(n); toks[s] = std::move(t);
    }

    llama_memory_clear(llama_get_memory(in.ctx), true);

    // Prompt pass: all streams in one batch, one logits row per stream (its
    // last prompt token).
    int total_prompt = 0; for (auto& t : toks) total_prompt += (int)t.size();
    llama_batch pb = llama_batch_init(total_prompt, 0, S);
    std::vector<int32_t> last_row(S, -1);
    for (int s = 0; s < S; ++s) {
        for (size_t i = 0; i < toks[s].size(); ++i) {
            const int r = pb.n_tokens++;
            pb.token[r] = toks[s][i];
            pb.pos[r]   = (llama_pos)i;
            pb.n_seq_id[r] = 1;
            pb.seq_id[r][0] = (llama_seq_id)s;
            pb.logits[r] = (i + 1 == toks[s].size());
            if (pb.logits[r]) last_row[s] = r;
        }
    }
    const int prc = llama_decode(in.ctx, pb);
    llama_batch_free(pb);
    if (prc != 0) { error = "prompt decode failed"; return -1; }

    // Per-stream running context (prompt + generated so far).
    std::vector<std::vector<int64_t>> ctxv(S);
    for (int s = 0; s < S; ++s) {
        ctxv[s].reserve(toks[s].size() + cfg_.window_tokens);
        for (llama_token t : toks[s]) ctxv[s].push_back((int64_t)t);
    }

    llama_batch nb = llama_batch_init(S, 0, S);
    for (int step = 0; step < cfg_.window_tokens; ++step) {
        nb.n_tokens = 0;
        std::vector<const float*> all_logits(S, nullptr);
        for (int s = 0; s < S; ++s) {
            all_logits[s] = llama_get_logits_ith(in.ctx, last_row[s]);
            if (!all_logits[s]) { llama_batch_free(nb); error = "no logits returned"; return -1; }
        }
        if (prepare) prepare(all_logits, n_vocab);
        for (int s = 0; s < S; ++s) {
            const float* logits = all_logits[s];
            const int chosen = chooser(s, logits, n_vocab, ctxv[s]);
            if (chosen < 0) { llama_batch_free(nb); error = "sampler aborted the window"; return -1; }
            ctxv[s].push_back((int64_t)chosen);
            const int r = nb.n_tokens++;
            nb.token[r] = (llama_token)chosen;
            nb.pos[r]   = (llama_pos)(toks[s].size() + step);
            nb.n_seq_id[r] = 1;
            nb.seq_id[r][0] = (llama_seq_id)s;
            nb.logits[r] = true;
            last_row[s] = r;
        }
        if (llama_decode(in.ctx, nb) != 0) { llama_batch_free(nb); error = "decode failed mid-window"; return -1; }
    }
    llama_batch_free(nb);
    return S;
}

int InferenceEngine::generate_windows_stepwise(int device_slot,
                                              const std::vector<std::string>& prompts,
                                              const StepSampler& step,
                                              std::string& error) {
    if (device_slot < 0 || device_slot >= static_cast<int>(instances_.size())) {
        error = "invalid device slot"; return -1;
    }
    Instance& in = *instances_[device_slot];
    const llama_vocab* vocab = llama_model_get_vocab(in.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int S = static_cast<int>(prompts.size());
    if (S < 1 || S > cfg_.slots_per_device) { error = "bad stream count"; return -1; }

    // Tokenize every stream's prompt.
    std::vector<std::vector<llama_token>> toks(S);
    for (int s = 0; s < S; ++s) {
        std::vector<llama_token> t(prompts[s].size() + 8);
        int n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                               t.data(), (int32_t)t.size(), true, false);
        if (n < 0) { t.resize(-n);
            n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                               t.data(), (int32_t)t.size(), true, false); }
        if (n <= 0) { error = "tokenization failed"; return -1; }
        t.resize(n); toks[s] = std::move(t);
    }

    const auto tp0 = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(in.ctx), true);
    const auto tp1 = std::chrono::steady_clock::now();

    // Prompt pass: all streams in one batch, one logits row per stream (its
    // last prompt token).
    int total_prompt = 0; for (auto& t : toks) total_prompt += (int)t.size();
    llama_batch pb = llama_batch_init(total_prompt, 0, S);
    std::vector<int32_t> last_row(S, -1);
    for (int s = 0; s < S; ++s) {
        for (size_t i = 0; i < toks[s].size(); ++i) {
            const int r = pb.n_tokens++;
            pb.token[r] = toks[s][i];
            pb.pos[r]   = (llama_pos)i;
            pb.n_seq_id[r] = 1;
            pb.seq_id[r][0] = (llama_seq_id)s;
            pb.logits[r] = (i + 1 == toks[s].size());
            if (pb.logits[r]) last_row[s] = r;
        }
    }
    const int prc = llama_decode(in.ctx, pb);
    llama_batch_free(pb);
    if (prc != 0) { error = "prompt decode failed"; return -1; }
    llama_synchronize(in.ctx);   // make the prompt pass visible to the timer
    const auto tp2 = std::chrono::steady_clock::now();
    {
        static thread_local uint64_t pn = 0;
        static thread_local double p_clear = 0, p_prompt = 0;
        p_clear  += std::chrono::duration<double>(tp1 - tp0).count();
        p_prompt += std::chrono::duration<double>(tp2 - tp1).count();
        if (++pn % 8 == 0)
            std::fprintf(stderr, "[prof-pre] clear=%.0fms prompt=%.0fms (%d tok, avg of %llu)\n",
                         1000.0 * p_clear / pn, 1000.0 * p_prompt / pn, total_prompt,
                         (unsigned long long)pn);
    }

    // Per-stream running context (prompt + generated so far).
    std::vector<std::vector<int64_t>> ctxv(S);
    for (int s = 0; s < S; ++s) {
        ctxv[s].reserve(toks[s].size() + cfg_.window_tokens);
        for (llama_token t : toks[s]) ctxv[s].push_back((int64_t)t);
    }

    llama_batch nb = llama_batch_init(S, 0, S);
    double t_sample = 0.0, t_decode = 0.0;   // per-window phase timers
    // GPU-resident logits: from step 1 on, every consumed logits row comes
    // from THIS window's generation decodes (S rows, one ubatch, stream
    // order), so llama's device output tensor can be read directly and the
    // 29 MB D2H + host narrowing + H2D disappear. Step 0 consumes the PROMPT
    // pass logits (scattered rows, possibly multi-ubatch) and stays on the
    // host path. Verified once per window; POW_GPU_DEVICE_LOGITS=0 disables.
    static const bool want_dev_logits = [](){
        const char* e = std::getenv("POW_GPU_DEVICE_LOGITS");
        return !(e && *e == '0');
    }();
    bool dev_mode = false;   // becomes true once the device pointer verifies
    for (int step_i = 0; step_i < cfg_.window_tokens; ++step_i) {
        nb.n_tokens = 0;
        std::vector<const float*> all_logits(S, nullptr);
        const float* dev_logits = nullptr;
        if (want_dev_logits && step_i > 0) {
            int32_t ndev = 0;
            void* p = llama_get_logits_device(in.ctx, &ndev);   // synchronizes
            if (p && ndev == S && (dev_mode || pow_gpu_is_device_ptr(p))) {
                dev_logits = static_cast<const float*>(p);
                if (!dev_mode) {
                    dev_mode = true;
                    llama_set_skip_logits_copy(in.ctx, true);
                    static bool once = false;
                    if (!once) { once = true;
                        std::fprintf(stderr, "[engine] GPU-resident logits ACTIVE — llama D2H skipped from step 2 on\n"); }
                }
            } else if (dev_mode) {
                // The pointer verified before but is gone now — the host copy
                // was skipped, so there is nothing valid to fall back to.
                llama_batch_free(nb);
                llama_set_skip_logits_copy(in.ctx, false);
                error = "device logits vanished mid-window"; return -1;
            }
        }
        if (!dev_logits) {
            for (int s = 0; s < S; ++s) {
                all_logits[s] = llama_get_logits_ith(in.ctx, last_row[s]);
                if (!all_logits[s]) { llama_batch_free(nb); error = "no logits returned"; return -1; }
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int> chosen(S, -1);
        if (!step(all_logits, dev_logits, n_vocab, ctxv, chosen)) {
            llama_batch_free(nb);
            if (dev_mode) llama_set_skip_logits_copy(in.ctx, false);
            error = "sampler step failed"; return -1;
        }
        t_sample += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (int s = 0; s < S; ++s) {
            if (chosen[s] < 0) { llama_batch_free(nb); error = "sampler aborted a window"; return -1; }
            ctxv[s].push_back((int64_t)chosen[s]);
            const int r = nb.n_tokens++;
            nb.token[r] = (llama_token)chosen[s];
            nb.pos[r]   = (llama_pos)(toks[s].size() + step_i);
            nb.n_seq_id[r] = 1;
            nb.seq_id[r][0] = (llama_seq_id)s;
            nb.logits[r] = true;
            last_row[s] = r;
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (llama_decode(in.ctx, nb) != 0) {
            llama_batch_free(nb);
            if (dev_mode) llama_set_skip_logits_copy(in.ctx, false);
            error = "decode failed mid-window"; return -1;
        }
        t_decode += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    }
    // Restore the D2H for the next window's prompt pass (its logits rows are
    // consumed through the host path at step 0).
    if (dev_mode) llama_set_skip_logits_copy(in.ctx, false);
    // One line per completed window batch (~every 5-15 s): per-step phase cost
    // and the implied throughput. Direct measurement — immune to the job-churn
    // noise that plagues delta-counting the windows counter.
    { const double per_step_ms = 1000.0 * (t_sample + t_decode) / cfg_.window_tokens;
      std::fprintf(stderr, "[prof] batch S=%d: sample=%.1f decode-issue=%.1f ms/step -> %.2f windows/s LOOP-ONLY (see [prof-e2e])\n",
                   S, 1000.0*t_sample/cfg_.window_tokens, 1000.0*t_decode/cfg_.window_tokens,
                   per_step_ms > 0 ? 1000.0 * S / (256.0 * per_step_ms) : 0.0); }
    llama_batch_free(nb);
    return S;
}

int InferenceEngine::generate_windows_double(int device_slot,
                                             const std::vector<std::string>& prompts_a,
                                             const std::vector<std::string>& prompts_b,
                                             const StepSampler& step_a,
                                             const StepSampler& step_b,
                                             std::string& error) {
    if (!cfg_.double_buffer) { error = "engine not loaded with double_buffer"; return -1; }
    if (device_slot < 0 || device_slot >= static_cast<int>(instances_.size())) {
        error = "invalid device slot"; return -1;
    }
    Instance& in = *instances_[device_slot];
    if (!in.ctx_b) { error = "no double-buffer context"; return -2; }
    llama_context* CA = in.ctx;     // batch A — sequences 0..S-1 on ITS context
    llama_context* CB = in.ctx_b;   // batch B — sequences 0..S-1 on ITS context
    const llama_vocab* vocab = llama_model_get_vocab(in.model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int S = static_cast<int>(prompts_a.size());
    if (S < 1 || S > cfg_.slots_per_device ||
        prompts_b.size() != static_cast<size_t>(S)) { error = "bad stream count"; return -1; }

    // Private logits buffers: llama's output tensor is REUSED by the next
    // decode, so batch A's logits are copied out before batch B's decode is
    // issued (and vice versa). Host-ordered sync makes the copy race-free.
    const size_t need = sizeof(float) * static_cast<size_t>(S) * n_vocab;
    if (in.dbuf_bytes < need) {
        pow_gpu_device_free(in.dbuf[0]); pow_gpu_device_free(in.dbuf[1]);
        in.dbuf[0] = pow_gpu_device_alloc(need);
        in.dbuf[1] = pow_gpu_device_alloc(need);
        in.dbuf_bytes = (in.dbuf[0] && in.dbuf[1]) ? need : 0;
        if (!in.dbuf_bytes) { error = "cannot allocate double-buffer logits copies"; return -2; }
    }

    auto tokenize_all = [&](const std::vector<std::string>& prompts,
                            std::vector<std::vector<llama_token>>& toks) -> bool {
        toks.resize(S);
        for (int s = 0; s < S; ++s) {
            std::vector<llama_token> t(prompts[s].size() + 8);
            int n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                                   t.data(), (int32_t)t.size(), true, false);
            if (n < 0) { t.resize(-n);
                n = llama_tokenize(vocab, prompts[s].c_str(), (int32_t)prompts[s].size(),
                                   t.data(), (int32_t)t.size(), true, false); }
            if (n <= 0) return false;
            t.resize(n); toks[s] = std::move(t);
        }
        return true;
    };
    std::vector<std::vector<llama_token>> toks_a, toks_b;
    if (!tokenize_all(prompts_a, toks_a) || !tokenize_all(prompts_b, toks_b)) {
        error = "tokenization failed"; return -1;
    }

    llama_memory_clear(llama_get_memory(CA), true);
    llama_memory_clear(llama_get_memory(CB), true);
    llama_set_skip_logits_copy(CA, false);   // prompt passes use host logits
    llama_set_skip_logits_copy(CB, false);

    // Prompt pass for one batch ON ITS OWN CONTEXT; host logits are consumed
    // IMMEDIATELY (step 0 sampled here) since the next decode on that context
    // overwrites the shared host output buffer.
    std::vector<std::vector<int64_t>> ctx_a(S), ctx_b(S);
    std::vector<int> tok_a(S, -1), tok_b(S, -1);
    auto prompt_and_step0 = [&](llama_context* C,
                                const std::vector<std::vector<llama_token>>& toks,
                                const StepSampler& step,
                                std::vector<std::vector<int64_t>>& ctxv,
                                std::vector<int>& first_tok) -> bool {
        int total = 0; for (auto& t : toks) total += (int)t.size();
        llama_batch pb = llama_batch_init(total, 0, S);
        std::vector<int32_t> last_row(S, -1);
        for (int s = 0; s < S; ++s) {
            for (size_t i = 0; i < toks[s].size(); ++i) {
                const int r = pb.n_tokens++;
                pb.token[r] = toks[s][i];
                pb.pos[r]   = (llama_pos)i;
                pb.n_seq_id[r] = 1;
                pb.seq_id[r][0] = (llama_seq_id)s;
                pb.logits[r] = (i + 1 == toks[s].size());
                if (pb.logits[r]) last_row[s] = r;
            }
        }
        const int prc = llama_decode(C, pb);
        llama_batch_free(pb);
        if (prc != 0) { error = "prompt decode failed"; return false; }
        for (int s = 0; s < S; ++s) {
            ctxv[s].clear();
            ctxv[s].reserve(toks[s].size() + cfg_.window_tokens);
            for (llama_token t : toks[s]) ctxv[s].push_back((int64_t)t);
        }
        std::vector<const float*> all(S, nullptr);
        for (int s = 0; s < S; ++s) {
            all[s] = llama_get_logits_ith(C, last_row[s]);
            if (!all[s]) { error = "no logits returned"; return false; }
        }
        if (!step(all, nullptr, n_vocab, ctxv, first_tok)) { error = "sampler step failed"; return false; }
        for (int s = 0; s < S; ++s) {
            if (first_tok[s] < 0) { error = "sampler aborted a window"; return false; }
            ctxv[s].push_back((int64_t)first_tok[s]);
        }
        return true;
    };
    if (!prompt_and_step0(CA, toks_a, step_a, ctx_a, tok_a)) return -1;
    if (!prompt_and_step0(CB, toks_b, step_b, ctx_b, tok_b)) return -1;

    // Generation decode of one batch's current tokens (async, on its context).
    llama_batch nb = llama_batch_init(S, 0, S);
    auto issue_decode = [&](llama_context* C, const std::vector<int>& toks_now, int step_i,
                            const std::vector<std::vector<llama_token>>& ptoks) -> bool {
        nb.n_tokens = 0;
        for (int s = 0; s < S; ++s) {
            const int r = nb.n_tokens++;
            nb.token[r] = (llama_token)toks_now[s];
            nb.pos[r]   = (llama_pos)(ptoks[s].size() + step_i);
            nb.n_seq_id[r] = 1;
            nb.seq_id[r][0] = (llama_seq_id)s;
            nb.logits[r] = true;
        }
        return llama_decode(C, nb) == 0;
    };

    // Sync a context's pending decode, copy its logits to a private buffer.
    auto sync_and_copy = [&](llama_context* C, int which) -> const float* {
        int32_t ndev = 0;
        void* p = llama_get_logits_device(C, &ndev);   // synchronizes C
        if (!p || ndev != S) return nullptr;
        if (!pow_gpu_d2d_copy_sync(in.dbuf[which], p, need)) return nullptr;
        return static_cast<const float*>(in.dbuf[which]);
    };

    const auto t0 = std::chrono::steady_clock::now();
    // First generation decode of A; from here on host copies are skipped.
    llama_set_skip_logits_copy(CA, true);
    llama_set_skip_logits_copy(CB, true);
    bool ok = issue_decode(CA, tok_a, 0, toks_a);
    std::vector<const float*> dummy(S, nullptr);
    std::vector<int> next_a(S, -1), next_b(S, -1);
    bool first = true;

    for (int step_i = 1; step_i < cfg_.window_tokens && ok; ++step_i) {
        // --- A(step_i): sync CA, copy out, then B's decode fills the pipe.
        const float* da = sync_and_copy(CA, 0);
        if (!da) {
            if (first) {   // device path genuinely unavailable — clean fallback
                llama_set_skip_logits_copy(CA, false);
                llama_set_skip_logits_copy(CB, false);
                llama_batch_free(nb);
                return -2;
            }
            ok = false; break;
        }
        first = false;
        ok = issue_decode(CB, tok_b, step_i - 1, toks_b);
        if (!ok) break;
        if (!step_a(dummy, da, n_vocab, ctx_a, next_a)) { ok = false; break; }
        for (int s = 0; s < S; ++s) {
            if (next_a[s] < 0) { ok = false; break; }
            ctx_a[s].push_back((int64_t)next_a[s]);
        }
        if (!ok) break;
        tok_a.swap(next_a);

        // --- B(step_i-1): sync CB, copy out, then A's next decode fills the pipe.
        const float* db = sync_and_copy(CB, 1);
        if (!db) { ok = false; break; }
        ok = issue_decode(CA, tok_a, step_i, toks_a);
        if (!ok) break;
        if (!step_b(dummy, db, n_vocab, ctx_b, next_b)) { ok = false; break; }
        for (int s = 0; s < S; ++s) {
            if (next_b[s] < 0) { ok = false; break; }
            ctx_b[s].push_back((int64_t)next_b[s]);
        }
        if (!ok) break;
        tok_b.swap(next_b);
    }

    // Both batches sampled steps 0..255 (B lags A only inside an iteration).
    // The last issued decode — A's step-255 token — is waste; drain both
    // contexts so the next call starts quiet.
    if (ok) { llama_synchronize(CA); llama_synchronize(CB); }
    llama_set_skip_logits_copy(CA, false);
    llama_set_skip_logits_copy(CB, false);
    llama_batch_free(nb);
    if (!ok) { if (error.empty()) error = "double-buffered window failed"; return -1; }

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    static thread_local uint64_t dn = 0;
    static thread_local double dacc = 0;
    dacc += secs; ++dn;
    if (dn % 4 == 0)
        std::fprintf(stderr, "[prof-dbl] loop=%.2fs for 2x%d windows (%.1f w/s in-loop)\n",
                     dacc / dn, S, 2.0 * S * dn / dacc);
    return 2 * S;
}

std::vector<DeviceEngineStats> InferenceEngine::benchmark(int windows_per_device, std::string& error) {
    std::vector<DeviceEngineStats> out;
    for (size_t i = 0; i < instances_.size(); ++i) {
        DeviceEngineStats s;
        s.device = instances_[i]->device;
        const auto t0 = std::chrono::steady_clock::now();
        for (int w = 0; w < windows_per_device; ++w) {
            std::string err;
            const int n = generate_window(static_cast<int>(i),
                                          "Explain distributed consensus in detail.",
                                          nullptr, err);
            if (n < 0) { error = err; return out; }
            s.tokens += static_cast<uint64_t>(n);
        }
        const auto t1 = std::chrono::steady_clock::now();
        s.seconds = std::chrono::duration<double>(t1 - t0).count();
        s.tokens_per_s = s.seconds > 0 ? double(s.tokens) / s.seconds : 0.0;
        out.push_back(s);
    }
    return out;
}

}  // namespace meow
