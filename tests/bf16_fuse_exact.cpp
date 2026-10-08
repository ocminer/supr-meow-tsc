// Deterministic forward-pass fingerprint for A/B-ing ggml-cuda graph changes
// (e.g. MEOW_BF16_FUSE=0 vs 1). S sequences, distinct fixed prompts, greedy
// decoding; prints one FNV-1a hash per step over every sequence's full-vocab
// logits, plus the greedy tokens. Two runs are bit-identical iff all lines match.
// usage: meow-bf16-fuse-exact <model.gguf> [slots=480] [steps=48]
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint64_t fnv1a(const void* p, size_t n, uint64_t h) {
    const unsigned char* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [slots] [steps]\n", argv[0]); return 2; }
    const int S     = argc > 2 ? std::atoi(argv[2]) : 480;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 48;
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = static_cast<uint32_t>(S) * 128;
    cp.n_batch   = std::max(2048u, static_cast<uint32_t>(S) * 64u);
    cp.n_ubatch  = 2048;
    cp.n_seq_max = static_cast<uint32_t>(S);
    cp.n_threads = cp.n_threads_batch = 8;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context failed\n"); return 1; }

    static const char* topics[] = {"distributed consensus", "photosynthesis", "the history of Rome", "quantum tunnelling",
                                   "sourdough baking", "plate tectonics", "compiler design", "the French revolution"};
    std::vector<std::vector<llama_token>> prompts(S);
    size_t total = 0;
    for (int s = 0; s < S; ++s) {
        const std::string text = "Explain " + std::string(topics[s % 8]) + " in detail, part " + std::to_string(s) + ".";
        std::vector<llama_token> t(256);
        const int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), t.data(), (int)t.size(), true, false);
        if (n <= 0) { std::fprintf(stderr, "tokenize failed\n"); return 1; }
        t.resize(n); total += n; prompts[s] = std::move(t);
    }

    llama_batch batch = llama_batch_init((int)std::max<size_t>(total, (size_t)S), 0, 1);
    std::vector<int> last_idx(S);
    batch.n_tokens = 0;
    for (int s = 0; s < S; ++s) {
        for (size_t j = 0; j < prompts[s].size(); ++j) {
            const int k = batch.n_tokens++;
            batch.token[k] = prompts[s][j]; batch.pos[k] = (llama_pos)j;
            batch.n_seq_id[k] = 1; batch.seq_id[k][0] = s;
            batch.logits[k] = (j + 1 == prompts[s].size());
            if (batch.logits[k]) last_idx[s] = k;
        }
    }
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "prefill decode failed\n"); return 1; }

    std::vector<int> pos(S);
    for (int s = 0; s < S; ++s) pos[s] = (int)prompts[s].size();
    for (int step = 0; step <= steps; ++step) {
        uint64_t h = 1469598103934665603ull;
        std::vector<llama_token> next(S);
        for (int s = 0; s < S; ++s) {
            const float* lg = llama_get_logits_ith(ctx, step == 0 ? last_idx[s] : s);
            if (!lg) { std::fprintf(stderr, "no logits\n"); return 1; }
            h = fnv1a(lg, sizeof(float) * (size_t)n_vocab, h);
            int best = 0;
            for (int v = 1; v < n_vocab; ++v) if (lg[v] > lg[best]) best = v;
            next[s] = best;
        }
        std::printf("step %3d hash %016llx tok0 %d tok1 %d tokLast %d\n", step, (unsigned long long)h, next[0], next[1 % S], next[S - 1]);
        std::fflush(stdout);
        if (step == steps) break;
        batch.n_tokens = S;
        for (int s = 0; s < S; ++s) {
            batch.token[s] = next[s]; batch.pos[s] = pos[s]++;
            batch.n_seq_id[s] = 1; batch.seq_id[s][0] = s; batch.logits[s] = 1;
        }
        if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed at step %d\n", step); return 1; }
    }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
