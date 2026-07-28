#include "poi.h"

#include "stratum.h"
#include "vdf.h"

#include <zmq.h>
#include <openssl/sha.h>

#include <cstring>
#include <mutex>
#include <deque>
#include <cstring>
#include <thread>
#include <condition_variable>
#include <utility>
#include <stdexcept>
#include <cstdio>
#include <unordered_map>
#include <algorithm>

#include "pow_utils.h"
#include "proof_generated.h"

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
    std::unordered_map<int, size_t> last_ctx;    // per sequence: last context size
    std::unordered_map<int, size_t> prompt_len;  // per sequence: window prompt length
    std::deque<std::unique_ptr<PoiShare>> pending_q;  // several streams can emit
                                                      // between two drains
    // Batched presort output, one slice per stream (idx: n_vocab each).
    std::vector<uint32_t> batch_idx;
    std::vector<float>    batch_val;
    std::vector<float>    batch_head;
    std::vector<double>   batch_stats;
    int                   batch_n_vocab = 0;
    bool                  batch_ready = false;
    // req_id → job_id for recent jobs. A proof emitted at the last token of a
    // window is drained on the NEXT on_logits call — which can already be in a
    // new job. Labelling it with the CURRENT job made the pool submit it
    // against the wrong work unit ("req_id mismatch: rpc=15 flatbuffer=14").
    // The proof names its own unit (MiningResponse.req_id); label by that.
    std::unordered_map<uint64_t, std::string> job_by_req;
    std::mutex mtx;

    ~Impl() {
        if (zmq_pull) zmq_close(zmq_pull);
        if (zmq_ctx)  zmq_ctx_term(zmq_ctx);
    }
};

PoiMiner::PoiMiner() : impl_(std::make_unique<Impl>()) {}
PoiMiner::~PoiMiner() = default;

bool PoiMiner::init(int window_tokens, std::string& error, int n_streams,
                    int egress_port) {
    // Partition the nonce space per instance: the pool dedups on (job, nonce),
    // and two devices both counting 1,2,3… collide on every share the second
    // one submits ("duplicate share"). The egress port is already unique per
    // instance, so it doubles as the partition key.
    nonce_seq_ = static_cast<uint64_t>(egress_port) << 32;
    n_streams_ = std::max(1, n_streams);
    window_tokens_ = window_tokens;
    if (window_tokens != 256) {
        error = "window must be 256 tokens to match the chain's PoW window";
        return false;
    }
    // Both processes must agree on the proof version or the sampler aborts on
    // its first solution check; here there is only one process, so set it.
    ::setenv("POW_PROOF_VERSION", "3", 1);
    // Left at "off" DELIBERATELY. Turning it on makes the step-0 digest
    // disagree with the verifier's recomputation, i.e. this validator does not
    // fold an admission nonce into the hash. Nonce-less costs the
    // [B_FLOOR, B_FREE) admission band, which is a throughput matter, not a
    // validity one — and it is what this chain actually verifies against.
    ::setenv("POW_V3_ADMISSION_MODE", "off", 1);
    ::setenv("POW_EGRESS_MODE", "broker", 1);      // broker mode emits shares AND solutions
    ::setenv("POW_PROXY_ENABLE", "false", 1);      // ...and carries pow_blob_hash
    ::setenv("ZMQ_PUSH_HOST", "127.0.0.1", 1);
    // Per-instance: the writer reads this env at initialize(), so two
    // PoiMiners must init SEQUENTIALLY, each setting its own port first.
    ::setenv("ZMQ_PUSH_PORT", std::to_string(egress_port).c_str(), 1);

    impl_->zmq_ctx = zmq_ctx_new();
    if (!impl_->zmq_ctx) { error = "zmq_ctx_new failed"; return false; }
    impl_->zmq_pull = zmq_socket(impl_->zmq_ctx, ZMQ_PULL);
    if (!impl_->zmq_pull) { error = "zmq_socket failed"; return false; }
    const std::string ep = "tcp://127.0.0.1:" + std::to_string(egress_port);
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
    impl_->last_ctx.clear();   // force fresh rows on every stream's next window
    if (!p.valid || p.header_prefix.size() != 152) {
        error = "invalid job (header_prefix must be 152 hex chars)";
        return false;
    }
    // The VDF is bound to the PARENT hash, which lives at bytes 4..36 of the
    // header prefix. Recompute only when the parent actually changed: it is a
    // sequential computation and redoing it per job would stall mining.
    // INTERNAL byte order, exactly as serialized in the header — bcore's
    // QuickVerifier reads the VDF challenge as `header_prefix.data() + 4` raw
    // ("expects the 32 bytes exactly as serialized in the header",
    // quick_verifier.cpp) and rejected every display-order proof with
    // "VDF verification failed". The share verifier checks the VDF against the
    // proof's own block_hash field, so declaring block_hash in the SAME raw
    // order keeps consensus and share verification in agreement.
    const std::string parent = p.header_prefix.substr(8, 64);
    auto parent_bytes = hex_to_bytes(parent);
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
        // The BLOCK target: bcore validates a real solution against this, and a
        // proof declaring anything else is rejected (quick_verify_failed).
        // Share-mode verification does not use it — the share threshold travels
        // separately in adjusted_bits and reaches the verifier as
        // target_override_hex. See tools/patch-staged-poi.py.
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
        for (int sq = 0; sq < n_streams_; ++sq)
            impl_->coord.update_pow_params_for_sequence(sq, params);
        impl_->params = params;
    } catch (const std::exception& e) {
        error = std::string("update_pow_params failed: ") + e.what();
        return false;
    }

    impl_->job_by_req[p.request_id] = p.job_id;
    if (impl_->job_by_req.size() > 16) {           // drop stale entries; the
        for (auto it = impl_->job_by_req.begin(); // pool forgets old jobs too
             it != impl_->job_by_req.end() && impl_->job_by_req.size() > 8;)
            it = (it->second != p.job_id) ? impl_->job_by_req.erase(it) : ++it;
    }

    impl_->job = p;
    job_id_ = p.job_id;
    ready_ = true;
    return true;
}

#ifdef POW_GPU_SORT_ENABLED
extern "C" bool pow_gpu_sort_and_stats_batched(
        const float* const* h_logits, int S, int n, float inv_temp, int snap_bf16,
        uint32_t* h_idx, float* h_val, float* h_head, double* h_stats);
extern "C" void pow_gpu_inject_presorted(const uint32_t* idx, const float* val,
                                          const double* stats, float head);
extern "C" bool pow_gpu_bind_device(int cuda_ordinal);
#endif

void PoiMiner::prepare_batch(const std::vector<const float*>& logits, int n_vocab) {
#ifdef POW_GPU_SORT_ENABLED
    impl_->batch_ready = false;
    const int S = static_cast<int>(logits.size());
    if (S < 1) return;
    impl_->batch_idx.resize(static_cast<size_t>(S) * n_vocab);
    impl_->batch_val.resize(static_cast<size_t>(S) * n_vocab);
    impl_->batch_head.resize(S);
    impl_->batch_stats.resize(static_cast<size_t>(S) * 6);
    if (pow_gpu_sort_and_stats_batched(logits.data(), S, n_vocab, 1.0f, /*snap_bf16=*/1,
                                       impl_->batch_idx.data(), impl_->batch_val.data(),
                                       impl_->batch_head.data(), impl_->batch_stats.data())) {
        impl_->batch_n_vocab = n_vocab;
        impl_->batch_ready = true;
    }
#else
    (void)logits; (void)n_vocab;
#endif
}

int PoiMiner::on_logits(int seq_id, const float* logits, int n_vocab,
                        const std::vector<int64_t>& context,
                        float temperature, int top_k, float top_p) {
    if (!ready_) return -1;
    try {
        // A window starts when the context stops growing — the engine is back
        // to prompt-only. The coordinator needs a row allocated from those exact
        // prompt tokens, and it must be reallocated per window, not once per job.
#ifdef POW_GPU_SORT_ENABLED
        if (impl_->batch_ready &&
            seq_id >= 0 && (size_t)(seq_id + 1) * impl_->batch_n_vocab <= impl_->batch_idx.size()) {
            pow_gpu_inject_presorted(
                impl_->batch_idx.data() + (size_t)seq_id * impl_->batch_n_vocab,
                impl_->batch_val.data() + (size_t)seq_id * impl_->batch_n_vocab,
                impl_->batch_stats.data() + (size_t)seq_id * 6,
                impl_->batch_head[seq_id]);
        }
#endif
        auto lc = impl_->last_ctx.find(seq_id);
        if (lc == impl_->last_ctx.end() || context.size() <= lc->second) {
            std::unordered_map<int, std::vector<int64_t>> prompts{{seq_id, context}};
            impl_->coord.ensure_sequences({seq_id}, prompts);
            // ensure_sequences() ERASES this sequence's PoW params, so they must
            // be re-registered for every window; set_prompt_tokens seeds both
            // the proof's prompt_tokens and the v3 pre-window archive (an empty
            // prompt crashes the verifier's replay with a silent timeout).
            impl_->coord.update_pow_params_for_sequence(seq_id, impl_->params);
            impl_->coord.set_prompt_tokens(seq_id,
                std::vector<int32_t>(context.begin(), context.end()));
            impl_->prompt_len[seq_id] = context.size();
        }
        impl_->last_ctx[seq_id] = context.size();
        // Steps completed once this token is recorded.
        const size_t steps = context.size() - impl_->prompt_len[seq_id] + 1;

        // Labelled individually: "unordered_map::at" from any of the three is
        // indistinguishable otherwise, and they fail for different reasons.
        // The hasher digests the context EXACTLY as handed to it — it neither
        // pads nor truncates. The verifier always replays a fixed 256-token
        // window, zero-padded on the LEFT. So the caller owns that shape: pass
        // the growing context raw and the two agree only when it happens to be
        // 256 long. That is why exactly one step per window ever verified — with
        // a 22-token prompt, step 234 gives 22+234=256, and 234 was the single
        // index missing from the verifier's failed-step list.
        std::vector<int64_t> window(window_tokens_, 0);
        const size_t take = std::min<size_t>(context.size(),
                                             static_cast<size_t>(window_tokens_));
        std::copy(context.end() - take, context.end(), window.end() - take);

        stage_ = "sample_token_complete";
        auto r = impl_->coord.sample_token_complete(seq_id, logits, n_vocab,
                                                    temperature, top_k, top_p,
                                                    window, "bf16");
        stage_ = "record_complete_step";
        impl_->coord.record_complete_step(seq_id, r, true);
        // ONLY at the end of a full window: check_solutions() reads
        // "chosen_tokens" out of the completed ring-buffer window, and asking
        // for it mid-window throws unordered_map::at.
        if (steps >= static_cast<size_t>(window_tokens_)) {
            stage_ = "check_solutions";
            impl_->coord.check_solutions({seq_id});
            // Retire the sequence so the NEXT window starts from step 0. A
            // sequence that already owns a row keeps it, and its step counter
            // just keeps climbing (16128, 16384, …) — the proof then records
            // absolute step indices while the verifier replays 0..255, so every
            // u-value disagrees. Only the step-0 digest matched, because
            // check_solutions() forces step 0 for that one.
            stage_ = "cleanup_sequence";
            impl_->coord.cleanup_sequence(seq_id);
            impl_->last_ctx.erase(seq_id);
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
            {
                const auto* mr0 = flatbuffers::GetRoot<proof::MiningResponse>(data);
                if (mr0) {
                    auto it = impl_->job_by_req.find(mr0->req_id());
                    if (it != impl_->job_by_req.end()) share->job_id = it->second;
                }
            }
            // The pool dedups on (job, nonce) and rejects a repeat outright.
            // Blocks are submitted from the proof alone, so this value carries
            // no consensus meaning — it only has to be unique per job. Leaving
            // it at 0 makes every share after the first a "duplicate".
            share->nonce        = ++nonce_seq_;
            share->proof_b64    = base64(data, static_cast<size_t>(n));
            share->vdf_tick     = vdf_tick_;
            // The claim the pool judges is the HEADER HASH —
            // SHA256d(header76 || digest[:4]) — because that is what consensus
            // compares against targets (PowHasher::check_solution and the
            // verifier's block sanity both build exactly this). Neither the
            // per-step digest nor Proof.hash is the compared value; claiming
            // either gets shares rejected on the pool's threshold checks.
            share->achieved_hex.clear();
            {
                const auto* mr = flatbuffers::GetRoot<proof::MiningResponse>(data);
                const auto* pb = mr ? mr->pow_blob() : nullptr;
                if (pb && pb->hash() && pb->hash()->size() == 32 &&
                    pb->header_prefix() && pb->header_prefix()->size() == 76) {
                    uint8_t hdr[80], h1[32], h2[32];
                    std::memcpy(hdr, pb->header_prefix()->data(), 76);
                    std::memcpy(hdr + 76, pb->hash()->data(), 4);   // nonce = digest[:4]
                    SHA256(hdr, 80, h1);
                    SHA256(h1, 32, h2);
                    // DISPLAY order (reversed): consensus interprets the raw
                    // digest as a LITTLE-endian integer (check_hash_against
                    // _target: "Interpret digest & target as Core does"), so
                    // the hex that compares correctly as a big-endian number
                    // against target hex is the byte-reversed digest — the
                    // same convention as a displayed Core block hash. Claiming
                    // raw order made the pool promote non-blocks (bcore:
                    // "does not meet target difficulty") and would drop real
                    // solutions whose raw-order hex merely LOOKS too large.
                    std::reverse(h2, h2 + 32);
                    share->achieved_hex = bytes_to_hex(h2, 32);
                }
            }
            if (share->achieved_hex.empty()) { zmq_msg_close(&msg); return static_cast<int>(r.token_id); }
            {
                std::lock_guard<std::mutex> lk(impl_->mtx);
                impl_->pending_q.push_back(std::move(share));
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
    if (impl_->pending_q.empty()) return nullptr;
    auto s = std::move(impl_->pending_q.front());
    impl_->pending_q.pop_front();
    return s;
}



// ===========================================================================
// SamplerPool — parallel per-stream sampler tail across K coordinators.
// ===========================================================================
struct SamplerPool::Impl {
    int n_streams = 0;
    int groups = 1;
    int cuda_device = 0;
    std::vector<std::unique_ptr<PoiMiner>> miners;   // one per group
    std::vector<std::pair<int,int>>        range;     // [lo,hi) global streams per group

    // Per-step handoff. The mining thread publishes the step inputs, bumps the
    // generation, and waits for every group to report the same generation done.
    std::vector<std::thread>  threads;
    std::mutex                mtx;
    std::condition_variable   cv_go, cv_done;
    uint64_t                  gen = 0;                 // step generation
    std::vector<uint64_t>     done;                    // per-group last gen finished
    bool                      stop = false;
    bool                      step_ok = true;

    // Inputs valid for the current generation (owned by the mining thread).
    const std::vector<const float*>*            in_logits = nullptr;
    int                                         in_nvocab = 0;
    const std::vector<std::vector<int64_t>>*    in_ctx = nullptr;
    std::vector<int>*                           out_tokens = nullptr;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
            ++gen;
        }
        cv_go.notify_all();
        for (auto& t : threads) if (t.joinable()) t.join();
    }

    void worker(int g) {
        // Bind THIS worker thread to the group's GPU: CUDA context is
        // per host thread, so a worker left on the default device would
        // run its sort kernels on GPU 0 regardless of which card decoded.
        pow_gpu_bind_device(cuda_device);
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_go.wait(lk, [&]{ return gen != seen || stop; });
            if (stop) return;
            seen = gen;
            const auto lo = range[g].first, hi = range[g].second;
            const auto* logits = in_logits;
            const int   nvoc   = in_nvocab;
            const auto* ctx     = in_ctx;
            auto*       out     = out_tokens;
            lk.unlock();

            bool ok = true;
            // One segmented GPU sort for this group's streams, then per-stream
            // sampling — all on THIS worker thread, so the pow_gpu thread-local
            // scratch and injection slot are naturally private to the group.
            std::vector<const float*> gl;
            gl.reserve(hi - lo);
            for (int s = lo; s < hi; ++s) gl.push_back((*logits)[s]);
            miners[g]->prepare_batch(gl, nvoc);
            for (int s = lo; s < hi && ok; ++s) {
                const int local = s - lo;
                const int tok = miners[g]->on_logits(local, (*logits)[s], nvoc,
                                                     (*ctx)[s], 1.0f, 50, 1.0f);
                if (tok < 0) ok = false; else (*out)[s] = tok;
            }

            {
                std::lock_guard<std::mutex> lk2(mtx);
                if (!ok) step_ok = false;
                done[g] = seen;
            }
            cv_done.notify_one();
        }
    }
};

SamplerPool::SamplerPool() : impl_(std::make_unique<Impl>()) {}
SamplerPool::~SamplerPool() = default;

bool SamplerPool::init(int n_streams, int groups, int base_egress_port,
                       int cuda_device, std::string& error) {
    impl_->cuda_device = cuda_device;
    impl_->n_streams = n_streams;
    impl_->groups    = std::max(1, std::min(groups, n_streams));
    const int G = impl_->groups;
    impl_->done.assign(G, 0);
    // Contiguous, balanced split of streams across groups.
    int lo = 0;
    for (int g = 0; g < G; ++g) {
        const int cnt = (n_streams - lo) / (G - g);
        impl_->range.emplace_back(lo, lo + cnt);
        auto m = std::make_unique<PoiMiner>();
        // Each group is its OWN sampler over `cnt` local streams; ports (and so
        // nonce partitions) are unique per group.
        if (!m->init(256, error, cnt, base_egress_port + g)) return false;
        impl_->miners.push_back(std::move(m));
        lo += cnt;
    }
    for (int g = 0; g < G; ++g)
        impl_->threads.emplace_back([this, g]{ impl_->worker(g); });
    return true;
}

bool SamplerPool::set_job(const PoiJobParams& p, std::string& error) {
    // Fan the job out to every group. Each group's local seq ids are 0..cnt-1,
    // so the same params apply; the job_id/request_id are identical.
    for (auto& m : impl_->miners) if (!m->set_job(p, error)) return false;
    return true;
}

bool SamplerPool::ready() const {
    return !impl_->miners.empty() && impl_->miners[0]->ready();
}
std::string SamplerPool::job_id() const {
    return impl_->miners.empty() ? std::string() : impl_->miners[0]->job_id();
}

bool SamplerPool::sample_step(const std::vector<const float*>& logits, int n_vocab,
                              const std::vector<std::vector<int64_t>>& ctx,
                              std::vector<int>& out_tokens) {
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->in_logits  = &logits;
        impl_->in_nvocab  = n_vocab;
        impl_->in_ctx     = &ctx;
        impl_->out_tokens = &out_tokens;
        impl_->step_ok    = true;
        ++impl_->gen;
    }
    impl_->cv_go.notify_all();
    const uint64_t g = impl_->gen;
    std::unique_lock<std::mutex> lk(impl_->mtx);
    impl_->cv_done.wait(lk, [&]{
        for (int i = 0; i < impl_->groups; ++i) if (impl_->done[i] != g) return false;
        return true;
    });
    return impl_->step_ok;
}

std::unique_ptr<PoiShare> SamplerPool::take_share() {
    for (auto& m : impl_->miners)
        if (auto s = m->take_share()) return s;
    return nullptr;
}

}  // namespace meow
