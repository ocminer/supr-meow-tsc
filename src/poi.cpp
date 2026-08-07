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
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <algorithm>

#include "pow_utils.h"
#include "proof_generated.h"

namespace meow {

// Early declarations: SamplerPool::Impl manages pinned receive buffers and is
// defined before the block that declares the rest of the pow_gpu interface.
#ifdef POW_GPU_SORT_ENABLED
extern "C" void* pow_gpu_host_alloc(size_t bytes);
extern "C" void  pow_gpu_host_free(void* p);
#else
static inline void* pow_gpu_host_alloc(size_t bytes) { return std::malloc(bytes); }
static inline void  pow_gpu_host_free(void* p) { std::free(p); }
#endif

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

// The chain relays blocks carrying fewer ticks than this lazily — up to a
// 9-second announcement embargo, which is orphan risk rather than rejection.
// Confirmed by the pool 2026-08-03. Was 1000, on a stale comment claiming the
// cost was per-window; it is per-block, and 315k measures 3.36 s.
constexpr uint64_t kDefaultSelfVdfTick = 315000;

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
    // Nonce partition is seeded from the port we actually bind, further down —
    // NOT from the requested one. See the bind loop for why the two must agree.
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

    impl_->zmq_ctx = zmq_ctx_new();
    if (!impl_->zmq_ctx) { error = "zmq_ctx_new failed"; return false; }
    impl_->zmq_pull = zmq_socket(impl_->zmq_ctx, ZMQ_PULL);
    if (!impl_->zmq_pull) { error = "zmq_socket failed"; return false; }

    // A busy port is NOT fatal: another miner process on this host (one per GPU
    // is a normal bare-metal layout) already owns it, so walk up until one is
    // free. Docker never hits this — each container has its own netns — which
    // is why it went unnoticed until a five-GPU rig ran five processes.
    //
    // The port must stay the single source of truth for three things that have
    // to agree, so all three are derived from the port actually BOUND:
    //   1. the PULL socket here,
    //   2. ZMQ_PUSH_PORT, which the upstream writer reads at initialize(),
    //   3. the nonce partition — the pool dedups on (job, nonce), and two
    //      instances counting 1,2,3… collide on every share the second submits.
    // Seeding the nonce from the REQUESTED port would hand the same partition
    // to two processes whenever one of them got bumped, so it is seeded below.
    // bind() is exclusive, so distinct instances necessarily end on distinct
    // ports and therefore distinct partitions.
    // Window must cover EVERY coordinator on the host, not just this one:
    // one process per GPU x n_groups each, all scanning from the same default
    // base. A 5-GPU rig at 12 groups already needs 60, so 64 would strand an
    // 8-GPU rig. 512 is far past any real rig and costs only failed binds.
    constexpr int kPortScan = 512;
    int bound_port = -1;
    std::string bind_err;
    for (int off = 0; off < kPortScan; ++off) {
        const std::string ep = "tcp://127.0.0.1:" + std::to_string(egress_port + off);
        if (zmq_bind(impl_->zmq_pull, ep.c_str()) == 0) { bound_port = egress_port + off; break; }
        bind_err = zmq_strerror(zmq_errno());
    }
    if (bound_port < 0) {
        error = "cannot bind proof egress on any port in [" + std::to_string(egress_port)
              + ", " + std::to_string(egress_port + kPortScan - 1) + "]: " + bind_err
              + " — pass --egress-base to pick a free range";
        return false;
    }
    if (bound_port != egress_port) {
        std::fprintf(stderr,
                     "notice: proof egress port %d busy, using %d instead "
                     "(another miner on this host?)\n",
                     egress_port, bound_port);
    }
    ::setenv("ZMQ_PUSH_PORT", std::to_string(bound_port).c_str(), 1);
    nonce_seq_ = static_cast<uint64_t>(bound_port) << 32;
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

static std::mutex g_vdf_mtx;   // chiavdf global state is not thread-safe
static thread_local double g_prof_wait = 0.0;
static thread_local uint64_t g_prof_n = 0;

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
    if (!p.pool_vdf.empty()) {
        // §19 pool-issued VDF: use it verbatim. The pool computed it over
        // this parent at a tick it chose, so it knows every input to every
        // u-preimage — the precondition for precomputing canary transcripts.
        // No local prove, no chiavdf lock, no miner say over `tick`.
        if (p.pool_vdf != vdf_hex_ || parent != parent_hash_hex_) {
            vdf_hex_         = p.pool_vdf;
            vdf_tick_        = p.pool_vdf_tick ? p.pool_vdf_tick : 1000;   // normative default
            parent_hash_hex_ = parent;
        }
    } else if (parent != parent_hash_hex_ || vdf_hex_.empty()) {
        std::lock_guard<std::mutex> vlk(g_vdf_mtx);   // serialize chiavdf
        const auto prev = parent_bytes;
        // Tick count is the VDF's difficulty; the chain reads it from the proof.
        // This is a PROPAGATION setting, not a throughput one: below the
        // chain's threshold a block is announced under an embargo of up to 9
        // seconds and can be orphaned. The cost is paid once per new PARENT
        // (per block), not per window — measured 3.36 s at 315k, i.e. 0.56%
        // of a 600 s block, against losing whole blocks.
        //
        // Precedence: an explicit --vdf-tick wins; else the tick the POOL
        // advertises in job field 11 (which it may send with no VDF of its
        // own, letting it raise the floor network-wide without every miner
        // rebuilding); else the chain's documented threshold.
        vdf_tick_ = p.self_vdf_tick    ? p.self_vdf_tick
                  : p.pool_vdf_tick    ? p.pool_vdf_tick
                                       : kDefaultSelfVdfTick;
        const auto proof = Vdf::prove(prev, vdf_tick_);
        if (proof.empty()) { error = "VDF proof generation failed"; return false; }
        // Say it once, plainly. The pool enforces a minimum tick and a block
        // below it is announced under an embargo, so an operator has to be able
        // to see what their rig is actually emitting — a silently low tick is
        // exactly how this went unnoticed until blocks were already at risk.
        {
            static std::atomic<bool> once{false};
            if (!once.exchange(true)) {
                std::fprintf(stderr, "  VDF self-proved at %llu ticks%s\n",
                             (unsigned long long)vdf_tick_,
                             vdf_tick_ < kDefaultSelfVdfTick
                               ? "  *** BELOW the chain's prompt-relay threshold of 315000 —"
                                 " found blocks risk a 9s announcement embargo ***" : "");
                std::fflush(stderr);
            }
        }
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
extern "C" bool pow_gpu_sort_and_stats_batched_k(
        const float* const* h_logits, int S, int n, int topk, float inv_temp, int snap_bf16,
        uint32_t* h_idx, float* h_val, float* h_head, double* h_stats);
extern "C" void pow_gpu_inject_presorted(const uint32_t* idx, const float* val,
                                          const double* stats, float head, int count);
extern "C" void pow_gpu_inject_presorted2(const uint32_t* idx, const float* val,
                                          const double* stats, float head, int count,
                                          const float* probes, int probe_n);
extern "C" bool pow_gpu_sort_and_stats_device_k(
        const float* d_logits, int S, int n, int topk, float inv_temp,
        uint32_t* h_idx, float* h_val, float* h_head, double* h_stats, float* h_probes);
extern "C" bool pow_gpu_bind_device(int cuda_ordinal);
extern "C" int  pow_gpu_device_of_ptr(const void* p);
extern "C" void pow_gpu_reset_device_scratch();
extern "C" void pow_gpu_device_sync();
extern "C" bool pow_gpu_register_host_range(const void* p, size_t bytes);
extern "C" bool pow_gpu_corrupt_device(float* d_logits, int S, int n, int mode,
                                       float sigma, uint64_t seed);
extern "C" void* pow_gpu_device_alloc(size_t bytes);
extern "C" bool pow_gpu_d2d_copy_sync(void* dst, const void* src, size_t bytes);
#else
static inline bool pow_gpu_register_host_range(const void*, size_t) { return false; }
#endif

namespace {
// Corruption test mode (POW_CORRUPT): 0 off, 1 random, 2 noise, 3 replay.
// Dishonest by design — for measuring what the pool's checks catch. See
// pow_gpu.cu. Read once, process-wide.
struct CorruptCfg {
    int   mode  = 0;
    float sigma = 0.5f;
    CorruptCfg() {
        if (const char* m = std::getenv("POW_CORRUPT")) mode = std::atoi(m);
        if (const char* s = std::getenv("POW_CORRUPT_SIGMA")) sigma = (float)std::atof(s);
        if (mode) {
            std::fprintf(stderr,
                "[corrupt] MODE %d ACTIVE (%s) — proofs are DELIBERATELY WRONG; "
                "never point this at a payable worker\n", mode,
                mode == 1 ? "random" : mode == 2 ? "noise" : mode == 3 ? "replay" : "?");
            std::fflush(stderr);
        }
    }
};
const CorruptCfg& corrupt_cfg() { static const CorruptCfg c; return c; }
}  // namespace

void PoiMiner::prepare_batch(const std::vector<const float*>& logits, int n_vocab) {
#ifdef POW_GPU_SORT_ENABLED
    impl_->batch_ready = false;
    const int S = static_cast<int>(logits.size());
    if (S < 1) return;
    impl_->batch_idx.resize(static_cast<size_t>(S) * n_vocab);
    impl_->batch_val.resize(static_cast<size_t>(S) * n_vocab);
    impl_->batch_head.resize(S);
    impl_->batch_stats.resize(static_cast<size_t>(S) * 6);
    if (pow_gpu_sort_and_stats_batched_k(logits.data(), S, n_vocab, n_vocab, 1.0f, /*snap_bf16=*/1,
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
                impl_->batch_head[seq_id], impl_->batch_n_vocab);
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

            // MEOW_DUMP_PROOFS=<dir>: write raw proof bytes locally so a miner
            // can diagnose its own output without needing the pool to capture
            // and audit for it. Off unless the variable is set; capped so a
            // forgotten setting cannot fill a rig's disk.
            //
            // Added while chasing a --split-model fault the pool could see and
            // we could not: split proofs replay RED against the real model
            // while single-GPU proofs pass, and the visible signature is in the
            // logit tail (70th logit median ~5.2 split vs ~1.1 single). Having
            // the proofs locally turns a cross-machine round trip into a diff.
            {
                static const char* dump_dir = std::getenv("MEOW_DUMP_PROOFS");
                static std::atomic<int> dumped{0};
                static const int dump_max = [](){
                    const char* m = std::getenv("MEOW_DUMP_PROOFS_MAX");
                    return m ? std::atoi(m) : 32;
                }();
                if (dump_dir && *dump_dir && dumped.load() < dump_max) {
                    const int idx = dumped.fetch_add(1);
                    if (idx < dump_max) {
                        char path[512];
                        std::snprintf(path, sizeof(path), "%s/proof-%s-%d.bin",
                                      dump_dir, share->job_id.c_str(), idx);
                        if (FILE* f = std::fopen(path, "wb")) {
                            std::fwrite(data, 1, static_cast<size_t>(n), f);
                            std::fclose(f);
                        }
                    }
                }
            }
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

            // ---- v3 B_cred gate: never submit a share that cannot be a block --
            // Consensus (TIP-0003) credits each of the 256 steps with
            // ~-log2(interval mass of the sampled token) and requires
            // B_cred >= B_FREE (70 bits) for a share to count; below B_FLOOR
            // (45 bits) it is invalid outright. We do not claim an admission
            // nonce (POW_V3_ADMISSION_MODE is deliberately "off" above), so the
            // [45,70) band is worthless to us too — the bar is 70.
            //
            // SOUNDNESS: sum(-log2 p(chosen)) is an UPPER BOUND on the credit
            // consensus computes, because ATOL widens the interval mass and a
            // larger mass earns LESS credit. So estimate < 70 implies actual
            // < 70, and dropping here can never discard a share that would have
            // been accepted. The converse is not true, which is why this only
            // ever drops — it never vouches for a share.
            //
            // Why it matters: the pool measured 6.09% of shares below the floor
            // on 2026-08-06. Those cost bandwidth and pool CPU and earn nothing,
            // and a sub-floor proof is dead even when it MEETS the difficulty
            // target — a real block was lost that way.
            {
                const auto* mrp = flatbuffers::GetRoot<proof::MiningResponse>(data);
                const auto* pf  = mrp ? mrp->pow_blob() : nullptr;
                const auto* cp  = pf ? pf->chosen_probs() : nullptr;
                if (cp && cp->size() > 0) {
                    double bits = 0.0;
                    for (flatbuffers::uoffset_t i = 0; i < cp->size(); ++i) {
                        const float p = cp->Get(i);
                        // p<=0 would be an infinite-surprisal step; treat it as
                        // the 32-bit per-step cap consensus applies rather than
                        // letting inf sail through the comparison.
                        bits += (p > 0.0f && p <= 1.0f) ? -std::log2(static_cast<double>(p)) : 32.0;
                    }
                    last_bcred_bits_.store(bits);
                    if (bits < 70.0) {
                        dropped_lowcred_.fetch_add(1);
                        zmq_msg_close(&msg);
                        return static_cast<int>(r.token_id);   // silently not submitted
                    }
                }
            }
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
    int dbg_dev = 0;
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
    const float*                                in_dev = nullptr;   // device base (GPU-resident mode)
    int                                         in_nvocab = 0;
    const std::vector<std::vector<int64_t>>*    in_ctx = nullptr;
    std::vector<int>*                           out_tokens = nullptr;
    // PINNED receive buffers (see pow_gpu_host_alloc) — pageable destinations
    // turned the "async" D2H into a 10.5 ms/step synchronous stall.
    uint32_t* sort_idx = nullptr; float* sort_val = nullptr;
    float* sort_head = nullptr; double* sort_stats = nullptr;
    float* sort_probes = nullptr;   // [S][20] pinned (GPU-resident mode)
    size_t sort_cap = 0; int sort_S = 0;
    int sort_stride = 0; bool sort_ok = false;
    bool reg_done = false;   // llama logits block page-locked (once)
    void*    corrupt_buf = nullptr;   // POW_CORRUPT: private mutable logits copy
    size_t   corrupt_cap = 0;
    uint64_t corrupt_seq = 0;         // varies the corruption per step
    void free_sort_bufs() {
        pow_gpu_host_free(sort_idx);  pow_gpu_host_free(sort_val);
        pow_gpu_host_free(sort_head); pow_gpu_host_free(sort_stats);
        pow_gpu_host_free(sort_probes);
        sort_idx = nullptr; sort_val = nullptr; sort_head = nullptr; sort_stats = nullptr;
        sort_probes = nullptr;
        sort_cap = 0; sort_S = 0;
    }
    bool ensure_sort_bufs(int S, int K) {
        const size_t need = (size_t)S * K;
        if (need <= sort_cap && S <= sort_S) return sort_idx != nullptr;
        free_sort_bufs();
        sort_idx   = (uint32_t*)pow_gpu_host_alloc(sizeof(uint32_t) * need);
        sort_val   = (float*)   pow_gpu_host_alloc(sizeof(float) * need);
        sort_head  = (float*)   pow_gpu_host_alloc(sizeof(float) * (size_t)S);
        sort_stats = (double*)  pow_gpu_host_alloc(sizeof(double) * 6 * (size_t)S);
        sort_probes= (float*)   pow_gpu_host_alloc(sizeof(float) * 20 * (size_t)S);
        if (!sort_idx || !sort_val || !sort_head || !sort_stats || !sort_probes) { free_sort_bufs(); return false; }
        sort_cap = need; sort_S = S;
        return true;
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            stop = true;
            ++gen;
        }
        cv_go.notify_all();
        for (auto& t : threads) if (t.joinable()) t.join();
        free_sort_bufs();
    }

    void worker(int g) {
        // Bind THIS worker thread to the group's GPU: CUDA context is
        // per host thread, so a worker left on the default device would
        // run its sort kernels on GPU 0 regardless of which card decoded.
        pow_gpu_bind_device(cuda_device);
        // With --split-model the logits do NOT live on that card: llama puts
        // the output tensor on whichever GPU holds the last layer. Sorting it
        // from here would be an illegal access, so the first time a device
        // pointer arrives we ask CUDA who owns it and rebind. Done once, and
        // only when it actually differs, so the single-GPU path is untouched.
        int  bound_device  = cuda_device;
        bool device_checked = false;
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mtx);
            cv_go.wait(lk, [&]{ return gen != seen || stop; });
            if (stop) return;
            seen = gen;
            const auto lo = range[g].first, hi = range[g].second;
            const auto* logits = in_logits;
            const float* dev   = in_dev;
            const int   nvoc   = in_nvocab;
            const auto* ctx     = in_ctx;
            auto*       out     = out_tokens;
            lk.unlock();

            // Per-group sort: THIS group's slice of the logits (contiguous in
            // llama's buffer) is narrowed, shipped and sorted HERE, on this
            // thread's own scratch/stream — so group A's tail runs while group
            // B's H2D+sort is still in flight, and the 0.5 ms AVX2 narrowing
            // parallelizes across all groups for free. The former central
            // sort serialized the whole step: sort(48) THEN all tails.
            // GPU-resident mode (dev != null): the slice is read straight from
            // llama's device tensor — no narrowing, no H2D at all.
            bool ok = true;
            bool gsort = false;
            const int K = sort_stride;
#ifdef POW_GPU_SORT_ENABLED
            if (sort_ok && K > 0) {
                // Rebind if the logits live on another card (--split-model). Asked
            // ONCE per worker, not per step: cudaPointerGetAttributes on every
            // step of every group would be pure overhead on the single-GPU
            // path, which is the one that matters most.
            if (dev && !device_checked) {
                    device_checked = true;
                    const int owner = pow_gpu_device_of_ptr(dev);
                    if (owner >= 0 && owner != bound_device) {
                        if (pow_gpu_bind_device(owner)) {
                            // Scratch belongs to the OLD device; drop it or the
                            // kernels write across cards and fault.
                            pow_gpu_reset_device_scratch();
                            bound_device = owner;
                            static std::atomic<bool> once{false};
                            if (!once.exchange(true))
                                std::fprintf(stderr,
                                    "[pow_gpu] split model: logits live on GPU %d, "
                                    "sampler rebound from GPU %d\n", owner, cuda_device);
                        }
                    }
                }
            // Under --split-model the logits tensor is produced on the other
            // GPU. llama's ctx->synchronize() did not fully order our direct
            // read against its producing stream, and the device-path proofs
            // came out with a noise-lifted tail that failed model replay. A
            // device barrier on the owner GPU before we read closes it. Only on
            // the split path (bound_device changed) so single-GPU pays nothing.
            // MEOW_SPLIT_SYNC=0 disables for A/B.
            if (dev && bound_device != cuda_device) {
                static const bool split_sync = [](){
                    const char* e = std::getenv("MEOW_SPLIT_SYNC");
                    return !(e && e[0] == '0');
                }();
                if (split_sync) pow_gpu_device_sync();
            }
            if (dev)
                    gsort = pow_gpu_sort_and_stats_device_k(
                        dev + (size_t)lo * nvoc, hi - lo, nvoc, K, 1.0f,
                        sort_idx  + (size_t)lo * K,
                        sort_val  + (size_t)lo * K,
                        sort_head + lo,
                        sort_stats + (size_t)lo * 6,
                        sort_probes + (size_t)lo * 20);
                else
                    gsort = pow_gpu_sort_and_stats_batched_k(
                        logits->data() + lo, hi - lo, nvoc, K, 1.0f, 1,
                        sort_idx  + (size_t)lo * K,
                        sort_val  + (size_t)lo * K,
                        sort_head + lo,
                        sort_stats + (size_t)lo * 6);
            }
#endif
            if (dev && !gsort) {
                // No host fallback exists in device mode — the host logits are
                // stale. Fail the step (aborts the window batch) rather than
                // sample garbage.
                std::lock_guard<std::mutex> lk2(mtx);
                step_ok = false;
                done[g] = seen;
                cv_done.notify_one();
                continue;
            }
            for (int s = lo; s < hi && ok; ++s) {
                const int local = s - lo;
                if (gsort)
                    pow_gpu_inject_presorted2(sort_idx + (size_t)s * K,
                                              sort_val + (size_t)s * K,
                                              sort_stats + (size_t)s * 6,
                                              sort_head[s], K,
                                              dev ? sort_probes + (size_t)s * 20 : nullptr,
                                              dev ? 20 : 0);
                const int tok = miners[g]->on_logits(local, dev ? nullptr : (*logits)[s], nvoc,
                                                     (*ctx)[s], 1.0f, 50, 1.0f);
                if (tok < 0) {
                    // The third way a window can die. Without this, a sampler
                    // refusal is indistinguishable in the log from a GPU sort
                    // failure — both surface only as "sampler step failed".
                    static std::atomic<bool> once{false};
                    if (!once.exchange(true)) {
                        std::fprintf(stderr,
                            "\n[poi] sampler rejected a step: on_logits returned %d "
                            "(group %d, stream %d, presorted=%d, device=%d, ctx=%zu tokens).\n\n",
                            tok, g, s, (int)gsort, (int)(dev != nullptr), (*ctx)[s].size());
                        std::fflush(stderr);
                    }
                    ok = false;
                } else (*out)[s] = tok;
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
    impl_->dbg_dev = cuda_device;
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

bool SamplerPool::sample_step(const std::vector<const float*>& logits,
                              const float* dev_logits, int n_vocab,
                              const std::vector<std::vector<int64_t>>& ctx,
                              std::vector<int>& out_tokens) {
    {
        const int S = impl_->n_streams;
        // The CPU tail provably reads at most rank top_k-1 (=49) of the
        // presorted prefix: the survivor scan is a strict-> prefix inside
        // [0, kpos), and every deeper statistic arrives via gpu_stats_ (see
        // pow_utils.cpp step 4/5). 2048 was legacy margin, and it is not
        // free — it is a 2 MB D2H on the step's critical path (S x 2048 x
        // 8 B) plus a 262k-entry host fill. 64 covers top_k=50 with margin.
        // MEOW_SORT_K overrides for A/B or if a future chain param raises
        // top_k; values below top_k+8 are clamped up defensively at use.
        static const int k_env = [](){
            const char* e = std::getenv("MEOW_SORT_K");
            const int v = e ? std::atoi(e) : 64;
            return v > 0 ? v : 64;
        }();
        const int K = std::min(k_env, n_vocab);
        impl_->sort_stride = K;
        if (!impl_->ensure_sort_bufs(S, K)) { impl_->sort_ok = false; return false; }
        // sort_ok now means "receive buffers ready"; the sort itself moved
        // into the worker groups (each sorts its own slice — see worker()).
        impl_->sort_ok = true;
        // ONCE (host mode only): page-lock llama's whole logits block so every
        // group's slice copy is straight DMA. Done here (not per worker)
        // because 12 threads registering overlapping slices race and lose.
        if (!dev_logits && !impl_->reg_done) {
            bool contig = true;
            for (int s = 1; s < S; ++s)
                if (logits[s] != logits[0] + (size_t)s * n_vocab) { contig = false; break; }
            if (contig)
                pow_gpu_register_host_range(logits[0], sizeof(float) * (size_t)S * n_vocab);
            impl_->reg_done = true;
        }
    }
#ifdef POW_GPU_SORT_ENABLED
    // Corruption test mode: mutate a PRIVATE copy of the device logits, then
    // let the whole real pipeline (snap, sort, stats, sampling, proof) run on
    // the corrupted values — so the emitted proof is internally consistent
    // but not what the registered model produced. Device path only; that is
    // the production path.
    if (corrupt_cfg().mode && dev_logits) {
        const size_t bytes = sizeof(float) * (size_t)impl_->n_streams * n_vocab;
        if (impl_->corrupt_cap < bytes) {
            impl_->corrupt_buf = pow_gpu_device_alloc(bytes);
            impl_->corrupt_cap = impl_->corrupt_buf ? bytes : 0;
        }
        if (impl_->corrupt_buf &&
            pow_gpu_d2d_copy_sync(impl_->corrupt_buf, dev_logits, bytes) &&
            pow_gpu_corrupt_device((float*)impl_->corrupt_buf, impl_->n_streams, n_vocab,
                                   corrupt_cfg().mode, corrupt_cfg().sigma,
                                   ++impl_->corrupt_seq)) {
            dev_logits = (const float*)impl_->corrupt_buf;
        }
    }
#endif
    {
        std::lock_guard<std::mutex> lk(impl_->mtx);
        impl_->in_logits  = &logits;
        impl_->in_dev     = dev_logits;
        impl_->in_nvocab  = n_vocab;
        impl_->in_ctx     = &ctx;
        impl_->out_tokens = &out_tokens;
        impl_->step_ok    = true;
        ++impl_->gen;
    }
    impl_->cv_go.notify_all();
    const uint64_t g = impl_->gen;
    std::unique_lock<std::mutex> lk(impl_->mtx);
    const auto tw0 = std::chrono::steady_clock::now();
    impl_->cv_done.wait(lk, [&]{
        for (int i = 0; i < impl_->groups; ++i) if (impl_->done[i] != g) return false;
        return true;
    });
    g_prof_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
    if (++g_prof_n % 512 == 0) {
        // Since the per-group restructure the wait covers each group's
        // narrow+H2D+sort AND its tail (they pipeline against each other).
        std::fprintf(stderr, "[prof2] wait(sort+tail)=%.1f ms/step\n",
                     1000.0 * g_prof_wait / g_prof_n);
        std::fflush(stderr);
    }
    return impl_->step_ok;
}

std::unique_ptr<PoiShare> SamplerPool::take_share() {
    for (auto& m : impl_->miners)
        if (auto s = m->take_share()) return s;
    return nullptr;
}

}  // namespace meow
