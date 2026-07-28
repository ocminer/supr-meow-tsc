#include "poi.h"

#include "stratum.h"
#include "vdf.h"

#include <zmq.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <cstdio>
#include <unordered_map>
#include <algorithm>

#include "pow_utils.h"

namespace meow {
namespace {

std::vector<uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<uint8_t> out;
    out.reserve(h.size() / 2);
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        out.push_back(static_cast<uint8_t>(std::stoul(h.substr(i, 2), nullptr, 16)));
    }
    return out;
}

std::string bytes_to_hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 0xF]); }
    return s;
}

std::string base64(const uint8_t* p, size_t n) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o; o.reserve(((n + 2) / 3) * 4);
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t a = p[i];
        const uint32_t b = (i + 1 < n) ? p[i + 1] : 0;
        const uint32_t c = (i + 2 < n) ? p[i + 2] : 0;
        const uint32_t v = (a << 16) | (b << 8) | c;
        o.push_back(t[(v >> 18) & 63]);
        o.push_back(t[(v >> 12) & 63]);
        o.push_back(i + 1 < n ? t[(v >> 6) & 63] : '=');
        o.push_back(i + 2 < n ? t[v & 63]        : '=');
    }
    return o;
}

// The sampler's egress is a ZMQ PUSH socket — that is how upstream ships it and
// patching it would mean forking the file. Instead the miner binds the matching
// PULL end on loopback IN-PROCESS, so finished proofs come back to us over a
// socket that never leaves the machine. No extra process, no upstream changes,
// and it reuses a path that is already proven in production.
constexpr int kEgressPort = 47021;

}  // namespace

struct PoiMiner::Impl {
    PowSamplingCoordinator coord{256, 1024};
    void*  zmq_ctx  = nullptr;
    void*  zmq_pull = nullptr;
    PoiJobParams job;
    std::unordered_map<std::string, std::string> params;   // re-registered per window
    std::unique_ptr<PoiShare> pending;
    std::mutex mtx;

    ~Impl() {
        if (zmq_pull) zmq_close(zmq_pull);
        if (zmq_ctx)  zmq_ctx_term(zmq_ctx);
    }
};

PoiMiner::PoiMiner() : impl_(std::make_unique<Impl>()) {}
PoiMiner::~PoiMiner() = default;

bool PoiMiner::init(int window_tokens, std::string& error) {
    window_tokens_ = window_tokens;
    if (window_tokens != 256) {
        error = "window must be 256 tokens to match the chain's PoW window";
        return false;
    }
    // Both processes must agree on the proof version or the sampler aborts on
    // its first solution check; here there is only one process, so set it.
    ::setenv("POW_PROOF_VERSION", "3", 1);
    ::setenv("POW_EGRESS_MODE", "broker", 1);      // broker mode emits shares AND solutions
    ::setenv("POW_PROXY_ENABLE", "false", 1);      // ...and carries pow_blob_hash
    ::setenv("ZMQ_PUSH_HOST", "127.0.0.1", 1);
    ::setenv("ZMQ_PUSH_PORT", std::to_string(kEgressPort).c_str(), 1);

    impl_->zmq_ctx = zmq_ctx_new();
    if (!impl_->zmq_ctx) { error = "zmq_ctx_new failed"; return false; }
    impl_->zmq_pull = zmq_socket(impl_->zmq_ctx, ZMQ_PULL);
    if (!impl_->zmq_pull) { error = "zmq_socket failed"; return false; }
    const std::string ep = "tcp://127.0.0.1:" + std::to_string(kEgressPort);
    if (zmq_bind(impl_->zmq_pull, ep.c_str()) != 0) {
        error = std::string("cannot bind proof egress on ") + ep + ": " + zmq_strerror(zmq_errno());
        return false;
    }
    int timeo = 0;   // non-blocking receive; mining must never wait on this
    zmq_setsockopt(impl_->zmq_pull, ZMQ_RCVTIMEO, &timeo, sizeof(timeo));

    try {
        impl_->coord.initialize("/tmp/supr-meow-tsc-logs", "");
    } catch (const std::exception& e) {
        error = std::string("sampler initialize failed: ") + e.what();
        return false;
    }
    return true;
}

bool PoiMiner::set_job(const PoiJobParams& p, std::string& error) {
    last_ctx_ = SIZE_MAX;   // force a fresh row on the next window
    if (!p.valid || p.header_prefix.size() != 152) {
        error = "invalid job (header_prefix must be 152 hex chars)";
        return false;
    }
    // The VDF is bound to the PARENT hash, which lives at bytes 4..36 of the
    // header prefix. Recompute only when the parent actually changed: it is a
    // sequential computation and redoing it per job would stall mining.
    // The header stores the parent hash in INTERNAL byte order; bcore's work
    // unit — and therefore the verifier — uses RPC DISPLAY order, which is the
    // reverse. Both the VDF challenge and block_hash must use display order or
    // every proof verifies RED while looking perfectly well-formed.
    const std::string parent_internal = p.header_prefix.substr(8, 64);
    auto parent_bytes = hex_to_bytes(parent_internal);
    std::reverse(parent_bytes.begin(), parent_bytes.end());
    const std::string parent = bytes_to_hex(parent_bytes.data(), parent_bytes.size());
    if (parent != parent_hash_hex_ || vdf_hex_.empty()) {
        const auto prev = parent_bytes;
        // Tick count is the VDF's difficulty; the chain reads it from the proof.
        // Kept modest here — raising it costs wall-clock per window.
        vdf_tick_ = 1000;
        const auto proof = Vdf::prove(prev, vdf_tick_);
        if (proof.empty()) { error = "VDF proof generation failed"; return false; }
        vdf_hex_ = Vdf::to_hex(proof);
        parent_hash_hex_ = parent;
    }

    std::unordered_map<std::string, std::string> params = {
        {"target",            p.block_target},
        {"share_target",      p.share_target},
        {"header_prefix",     p.header_prefix},
        // The proof serializer reads this with .at(), so it is mandatory. It is
        // the TIP hash the block builds on — the same value the VDF binds to,
        // which is why it is taken from the header prefix rather than trusted
        // as a separate field.
        {"block_hash",        parent},
        {"vdf",               vdf_hex_},
        {"tick",              std::to_string(vdf_tick_)},
        {"difficulty",        std::to_string(p.model_difficulty)},
        {"model_identifier",  p.model_identifier},
        {"compute_precision", "bf16"},
        {"proof_version",     "3"},
        {"request_id",        std::to_string(p.request_id)},
        // v3 refuses to sample without these: the sampler settings are part of
        // what the proof commits to, so they must be declared, not implied.
        // They must match exactly what on_logits() passes per token.
        {"temperature",        "1.0"},
        {"top_k",              "50"},
        {"top_p",              "1.0"},
        {"repetition_penalty", "1.0"},
        // Read with .at() during solution checking, so it MUST be present — but
        // it is only written into the proof when non-empty, so an empty value
        // satisfies the lookup without changing the proof. The pool does not
        // publish a CID; the model is identified by model_identifier.
        {"ipfs_cid",           ""},
    };
    try {
        // BOTH calls are required. The global params are the template, but
        // sample_token_complete() activates PER-SEQUENCE params and throws
        // "No PoW params registered for sequence N" if that row is absent —
        // the global call alone does not create it.
        impl_->coord.update_pow_params(params);
        impl_->coord.update_pow_params_for_sequence(0, params);
        impl_->params = params;
    } catch (const std::exception& e) {
        error = std::string("update_pow_params failed: ") + e.what();
        return false;
    }

    impl_->job = p;
    ready_ = true;
    return true;
}

int PoiMiner::on_logits(int seq_id, const float* logits, int n_vocab,
                        const std::vector<int64_t>& context,
                        float temperature, int top_k, float top_p) {
    if (!ready_) return -1;
    try {
        // A window starts when the context stops growing — the engine is back
        // to prompt-only. The coordinator needs a row allocated from those exact
        // prompt tokens, and it must be reallocated per window, not once per job.
        if (context.size() <= last_ctx_) {
            std::unordered_map<int, std::vector<int64_t>> prompts{{seq_id, context}};
            impl_->coord.ensure_sequences({seq_id}, prompts);
            // ensure_sequences() ERASES this sequence's PoW params, so they must
            // be re-registered for every window. Registering once per job leaves
            // the proof writer without request_id/difficulty and the window dies
            // at the very last step — after all 256 tokens were already spent.
            impl_->coord.update_pow_params_for_sequence(seq_id, impl_->params);
            prompt_len_ = context.size();
        }
        last_ctx_ = context.size();
        // Steps completed once this token is recorded.
        const size_t steps = context.size() - prompt_len_ + 1;

        // Labelled individually: "unordered_map::at" from any of the three is
        // indistinguishable otherwise, and they fail for different reasons.
        stage_ = "sample_token_complete";
        auto r = impl_->coord.sample_token_complete(seq_id, logits, n_vocab,
                                                    temperature, top_k, top_p,
                                                    context, "bf16");
        stage_ = "record_complete_step";
        impl_->coord.record_complete_step(seq_id, r, true);
        // ONLY at the end of a full window: check_solutions() reads
        // "chosen_tokens" out of the completed ring-buffer window, and asking
        // for it mid-window throws unordered_map::at.
        if (steps >= static_cast<size_t>(window_tokens_)) {
            stage_ = "check_solutions";
            impl_->coord.check_solutions({seq_id});
        }

        // Drain the in-process egress. Non-blocking: if nothing closed, mining
        // continues immediately.
        zmq_msg_t msg;
        zmq_msg_init(&msg);
        const int n = zmq_msg_recv(&msg, impl_->zmq_pull, ZMQ_DONTWAIT);
        if (n > 0) {
            const auto* data = static_cast<const uint8_t*>(zmq_msg_data(&msg));
            auto share = std::make_unique<PoiShare>();
            share->job_id       = impl_->job.job_id;
            // The pool dedups on (job, nonce) and rejects a repeat outright.
            // Blocks are submitted from the proof alone, so this value carries
            // no consensus meaning — it only has to be unique per job. Leaving
            // it at 0 makes every share after the first a "duplicate".
            share->nonce        = ++nonce_seq_;
            share->proof_b64    = base64(data, static_cast<size_t>(n));
            share->vdf_tick     = vdf_tick_;
            share->achieved_hex = bytes_to_hex(r.digest.data(), r.digest.size());
            {
                std::lock_guard<std::mutex> lk(impl_->mtx);
                impl_->pending = std::move(share);
            }
        }
        zmq_msg_close(&msg);
        return static_cast<int>(r.token_id);
    } catch (const std::exception& e) {
        // Report once: a silent -1 here looks like "sampler aborted the window"
        // forever with no clue why.
        static std::once_flag once;
        std::call_once(once, [&]{
            std::fprintf(stderr, "  [poi] sampler error in %s: %s\n", stage_, e.what());
        });
        return -1;
    }
}

std::unique_ptr<PoiShare> PoiMiner::take_share() {
    std::lock_guard<std::mutex> lk(impl_->mtx);
    return std::move(impl_->pending);
}

}  // namespace meow
