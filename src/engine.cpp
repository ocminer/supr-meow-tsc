#include "engine.h"

#include "llama.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace meow {

struct InferenceEngine::Instance {
    int             device = -1;
    llama_model*    model  = nullptr;
    llama_context*  ctx    = nullptr;

    ~Instance() {
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

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

    for (int dev : cfg.devices) {
        auto inst = std::make_unique<Instance>();
        inst->device = dev;

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 999;                       // the whole model on the GPU
        mp.main_gpu     = dev;
        mp.split_mode   = LLAMA_SPLIT_MODE_NONE;     // data parallel, see engine.h
        mp.use_mmap     = true;

        if (progress) progress("loading model on GPU " + std::to_string(dev));
        inst->model = llama_model_load_from_file(cfg.model_path.c_str(), mp);
        if (!inst->model) {
            error = "failed to load model '" + cfg.model_path + "' on GPU " + std::to_string(dev) +
                    " — check the path, and that the file is a bf16 GGUF of the registered model";
            return false;
        }

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx     = static_cast<uint32_t>(cfg.ctx_per_slot * cfg.slots_per_device);
        cp.n_batch   = 2048;
        cp.n_ubatch  = 512;
        cp.n_seq_max = static_cast<uint32_t>(cfg.slots_per_device);
        cp.n_threads = cfg.threads > 0 ? cfg.threads : 8;
        cp.n_threads_batch = cp.n_threads;

        inst->ctx = llama_init_from_model(inst->model, cp);
        if (!inst->ctx) {
            error = "failed to create context on GPU " + std::to_string(dev) +
                    " (try fewer slots or a smaller --ctx)";
            return false;
        }
        if (progress) {
            progress("GPU " + std::to_string(dev) + " ready — " +
                     std::to_string(cfg.slots_per_device) + " slots x " +
                     std::to_string(cfg.ctx_per_slot) + " ctx");
        }
        instances_.push_back(std::move(inst));
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
