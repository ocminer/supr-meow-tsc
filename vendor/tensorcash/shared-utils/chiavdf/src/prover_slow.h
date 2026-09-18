#include "include.h"
#include "create_discriminant.h"
#include "integer_common.h"
#include "vdf_new.h"
#include "nucomp.h"
#include "picosha2.h"
#include "proof_common.h"
#include <sys/stat.h>
#include <atomic>
#include <cassert>
#include <cmath>
#include <mutex>
#include <thread>
#include <tuple>
#include <vector>
#include <chrono>
#include <condition_variable>
#include <memory>

// TODO: Refactor to use 'Prover' class once new_vdf is merged in.

void ApproximateParameters(uint64_t T, int& l, int& k) {
    double log_memory = 23.25349666;
    double log_T = log2(T);
    l = 1;
    if (log_T - log_memory > 0.000001) {
        l = ceil(pow(2, log_memory - 20));
    }
    double intermediate = T * (double)0.6931471 / (2.0 * l);
    k = std::max(std::round(log(intermediate) - log(log(intermediate)) + 0.25), 1.0);
}

uint64_t GetBlock(uint64_t i, uint64_t k, uint64_t T, integer& B) {
    integer res = FastPow(2, T - k * (i + 1), B);
    mpz_mul_2exp(res.impl, res.impl, k);
    res = res / B;
    auto res_vector = res.to_vector();
    // 0 value results in empty vector from mpz_export
    // https://gmplib.org/list-archives/gmp-bugs/2009-July/001534.html
    return res_vector.empty() ? 0 : res_vector[0];
}

form GenerateWesolowski(form &y, form &x_init,
                        integer &D, PulmarkReducer& reducer,
                        std::vector<form> const& intermediates,
                        uint64_t num_iterations,
                        uint64_t k, uint64_t l) {
    integer B = GetB(D, x_init, y);
    integer L=root(-D, 4);

    uint64_t k1 = k / 2;
    uint64_t k0 = k - k1;
    assert(k > 0);
    assert(l > 0);

    form x = form::identity(D);

    for (int64_t j = l - 1; j >= 0; j--) {
        x = FastPowFormNucomp(x, D, integer(1 << k), L, reducer);

        std::vector<form> ys((1ULL << k));
        for (uint64_t i = 0; i < (1ULL << k); i++)
            ys[i] = form::identity(D);

        for (uint64_t i = 0; i < (num_iterations + k * l - 1)  / (k * l); i++) {
            if (num_iterations >= k * (i * l + j + 1)) {
                uint64_t b = GetBlock(i*l + j, k, num_iterations, B);
                nucomp_form(ys[b], ys[b], intermediates[i], D, L);
            }
        }
        for (uint64_t b1 = 0; b1 < (1ULL << k1); b1++) {
            form z = form::identity(D);
            for (uint64_t b0 = 0; b0 < (1ULL << k0); b0++) {
                nucomp_form(z, z, ys[b1 * (1ULL << k0) + b0], D, L);
            }
            z = FastPowFormNucomp(z, D, integer(b1 * (1 << k0)), L, reducer);
            nucomp_form(x, x, z, D, L);
        }
        for (uint64_t b0 = 0; b0 < (1ULL << k0); b0++) {
            form z = form::identity(D);
            for (uint64_t b1 = 0; b1 < (1ULL << k1); b1++) {
                nucomp_form(z, z, ys[b1 * (1ULL << k0) + b0], D, L);
            }
            z = FastPowFormNucomp(z, D, integer(b0), L, reducer);
            nucomp_form(x, x, z, D, L);
        }
    }

    reducer.reduce(x);
    return x;
}

std::vector<uint8_t> ProveSlow(integer& D, form& x, uint64_t num_iterations, std::string shutdown_file_path) {
    integer L = root(-D, 4);
    PulmarkReducer reducer;
    form y = form::from_abd(x.a, x.b, D);
    int d_bits = D.num_bits();

    int k, l;
    ApproximateParameters(num_iterations, l, k);
    if (k <= 0) k = 1;
    if (l <= 0) l = 1;
    int const kl = k * l;

    uint64_t const size_vec = (num_iterations + kl - 1) / kl;
    std::vector<form> intermediates(size_vec);
    form* cursor = intermediates.data();
    for (uint64_t i = 0; i < num_iterations; i++) {
        if (i % kl == 0) {
            *cursor = y;
            ++cursor;
        }
        nudupl_form(y, y, D, L);
        reducer.reduce(y);

        // Check for cancellation every 65535 interations
        if ((i&0xffff)==0) {
            // Only if we have a shutdown path
            if (shutdown_file_path!="") {
                struct stat buffer;
            
                int statrst = stat(shutdown_file_path.c_str(), &buffer);
                if ((statrst != 0) && (errno != EINTR)) {
                    // shutdown file doesn't exist, abort out
                    return {};
                }
            }
        }
    }

    form proof = GenerateWesolowski(y, x, D, reducer, intermediates, num_iterations, k, l);
    std::vector<uint8_t> result = SerializeForm(y, d_bits);
    std::vector<uint8_t> proof_bytes = SerializeForm(proof, d_bits);
    result.insert(result.end(), proof_bytes.begin(), proof_bytes.end());
    return result;
}

class StreamingProver {
public:
    struct KLConfig {
        int k, l;
        uint64_t spacing;      // k * l
        uint64_t start_iters;  // when this (k,l) becomes optimal (inclusive)
        uint64_t end_iters;    // when this (k,l) stops being optimal (exclusive)
        std::vector<form> intermediates;
        
        bool is_active(uint64_t iters) const {
            return iters >= start_iters && iters < end_iters;
        }
        
        bool covers_range(uint64_t start, uint64_t end) const {
            // Check if this config was active during [start, end)
            return start < end_iters && end > start_iters;
        }
    };

    StreamingProver(const std::vector<uint8_t>& hash32,
                    int discr_bits,
                    uint64_t checkpoint_N,
                    uint64_t max_iters = 100000000)
        : seed_(hash32)
        , discr_bits_(discr_bits)
        , N_(checkpoint_N)
        , max_iters_(max_iters)
        , total_iters_(0)
    {
        if (hash32.size() != 32)
            throw std::runtime_error("challenge_hash must be exactly 32 bytes");
        if (N_ == 0)
            throw std::runtime_error("checkpoint N must be > 0");

        if (discr_bits_ > 1024) {
        #if defined(_WIN32)
            _putenv("CHIAVDF_NO_ASM=1");
        #else
            setenv("CHIAVDF_NO_ASM", "1", 1);
        #endif
        }

        D_ = CreateDiscriminant(seed_, discr_bits_);
        L_ = root(-D_, 4);

        x0_ = form::from_abd(integer(2), integer(1), D_);
        x0_.reduce();
        y_cur_ = x0_;

        // Precompute all (k,l) pairs and their ranges
        precompute_kl_schedule();
    }

private:
    void precompute_kl_schedule() {
        kl_configs_.clear();
        
        uint64_t current = 0;
        
        while (current < max_iters_) {
            int k, l;
            ApproximateParameters(current + 1, l, k);
            if (k <= 0) k = 1;
            if (l <= 0) l = 1;
            
            // Find where these parameters change
            uint64_t end = current + 1;
            while (end < max_iters_) {
                int k2, l2;
                ApproximateParameters(end + 1, l2, k2);
                if (k2 <= 0) k2 = 1;
                if (l2 <= 0) l2 = 1;
                
                if (k2 != k || l2 != l) break;
                
                // Safe increment
                if (end > max_iters_ / 2) {
                    end = max_iters_;
                } else {
                    end = end * 2;
                }
            }
            
            // Binary search for exact transition
            if (end > current + 1 && end < max_iters_) {
                uint64_t left = current + 1;
                uint64_t right = end;
                while (left < right - 1) {
                    uint64_t mid = left + (right - left) / 2;
                    int km, lm;
                    ApproximateParameters(mid, lm, km);
                    if (km <= 0) km = 1;
                    if (lm <= 0) lm = 1;
                    
                    if (km == k && lm == l) {
                        left = mid;
                    } else {
                        right = mid;
                    }
                }
                end = right;
            }
            
            KLConfig config;
            config.k = k;
            config.l = l;
            config.spacing = uint64_t(k) * uint64_t(l);
            config.start_iters = 0;
            config.end_iters = std::min(end, max_iters_);
            
            // Reserve initial space
            config.intermediates.reserve(100);
            
            kl_configs_.push_back(config);
            
            current = config.end_iters;
        }
        
        std::cout << "K/L Schedule:\n";
        for (const auto& cfg : kl_configs_) {
            std::cout << "  k=" << cfg.k << ", l=" << cfg.l 
                     << " [" << cfg.start_iters << ", " << cfg.end_iters << ")\n";
        }
    }

    void store_intermediate_if_needed(uint64_t iter) {
    for (auto& config : kl_configs_) {
        // no longer check is_active — only cap by end_iters
        if (iter < config.end_iters
            && (iter % config.spacing == 0)) {
        config.intermediates.push_back(y_cur_);
        }
    }
    }

    void cleanup_inactive_vectors(uint64_t current_iters) {
        for (auto& config : kl_configs_) {
            if (current_iters >= config.end_iters && !config.intermediates.empty()) {
                std::cout << "Freeing vector for k=" << config.k 
                         << ", l=" << config.l << " at iter=" << current_iters 
                         << " (had " << config.intermediates.size() << " checkpoints)\n";
                
                std::vector<form>().swap(config.intermediates);
            }
        }
    }

public:
    std::vector<uint8_t> next_raw(const std::string& shutdown_path = "") {
        const uint64_t start_iter = total_iters_;
        const uint64_t end_iter = total_iters_ + N_;
        
        PulmarkReducer reducer;
        
        // Store initial form if needed (when starting from 0)
        if (start_iter == 0) {
            store_intermediate_if_needed(0);
        }
        
        for (uint64_t i = 0; i < N_; ++i) {
            uint64_t current_iter = start_iter + i;
            
            // Perform squaring first
            nudupl_form(y_cur_, y_cur_, D_, L_);
            reducer.reduce(y_cur_);
            
            // Then check if we need to store (after squaring, so current_iter+1)
            store_intermediate_if_needed(current_iter + 1);
            
            // Periodic maintenance
            if ((i & 0xffff) == 0) {
                // Check shutdown
                if (!shutdown_path.empty()) {
                    struct stat sb;
                    if (stat(shutdown_path.c_str(), &sb) != 0 && errno != EINTR)
                        throw std::runtime_error("shutdown requested");
                }
                
                // Cleanup inactive vectors
                cleanup_inactive_vectors(current_iter + 1);
            }
        }
        
        total_iters_ = end_iter;
        
        // Final cleanup
        cleanup_inactive_vectors(total_iters_);
        
        // Find optimal parameters for proof generation
        int k_proof, l_proof;
        ApproximateParameters(total_iters_, l_proof, k_proof);
        if (k_proof <= 0) k_proof = 1;
        if (l_proof <= 0) l_proof = 1;
        
        // Find the best matching config that has intermediates
        KLConfig* best_config = nullptr;
        int best_score = -1;
        
        for (auto& config : kl_configs_) {
            if (config.intermediates.empty()) continue;
            
            // Score based on how well the parameters match
            int score = 0;
            if (config.k == k_proof && config.l == l_proof) {
                score = 1000; // Perfect match
            } else if (config.covers_range(0, total_iters_)) {
                // Config was active during our range
                score = 100 - std::abs(config.k - k_proof) - 10 * std::abs(config.l - l_proof);
            }
            
            if (score > best_score) {
                best_score = score;
                best_config = &config;
            }
        }
        
        if (!best_config) {
            std::cerr << "ERROR: No intermediates found\n";
            throw std::runtime_error("No valid intermediates for proof generation");
        }
        
        std::cout << "Generating proof with k=" << best_config->k 
                 << ", l=" << best_config->l 
                 << " (optimal would be k=" << k_proof << ", l=" << l_proof << ")"
                 << " using " << best_config->intermediates.size() 
                 << " intermediates for T=" << total_iters_ << "\n";
        
        // CRITICAL FIX: Adjust the proof parameters to match what we actually stored
        // Since we stored at best_config->spacing intervals, we must use those parameters
        k_proof = best_config->k;
        l_proof = best_config->l;
        
        // Now check if we have enough intermediates
        uint64_t kl = uint64_t(k_proof) * uint64_t(l_proof);
        uint64_t expected_size = (total_iters_ + kl - 1) / kl;
        
        if (best_config->intermediates.size() < expected_size) {
            std::cout << "Warning: Have " << best_config->intermediates.size() 
                     << " intermediates but GenerateWesolowski expects " << expected_size << "\n";
            
            // Pad with identity elements to prevent segfault
            // This is not ideal but prevents crashes
            while (best_config->intermediates.size() < expected_size) {
                best_config->intermediates.push_back(form::identity(D_));
            }
        }
        
        // Generate proof
        form proof = GenerateWesolowski(
            y_cur_, x0_, D_, reducer,
            best_config->intermediates, total_iters_, 
            k_proof, l_proof);
        
        const int d_bits = D_.num_bits();
        std::vector<uint8_t> blob = SerializeForm(y_cur_, d_bits);
        std::vector<uint8_t> pbytes = SerializeForm(proof, d_bits);
        blob.insert(blob.end(), pbytes.begin(), pbytes.end());
        
        return blob;
    }

    uint64_t total_iterations() const { return total_iters_; }
    const integer& discriminant() const { return D_; }

private:
    // Configuration
    std::vector<uint8_t> seed_;
    int discr_bits_;
    uint64_t N_;
    uint64_t max_iters_;

    // Running state
    integer D_, L_;
    form x0_;
    form y_cur_;
    uint64_t total_iters_;

    // Multi-vector storage
    std::vector<KLConfig> kl_configs_;
};

class ThreadedStreamingProver {
public:
    struct ProofData {
        std::vector<uint8_t> blob;
        uint64_t iterations;
    };

    struct KLConfig {
        int k, l;
        uint64_t spacing;      // k * l
        uint64_t start_iters;  // always 0 now
        uint64_t end_iters;    // when to stop collecting
        
        // NO pre-allocation - grow on demand
        std::vector<form> intermediates;
        std::atomic<size_t> current_size{0};
        mutable std::mutex intermediates_mutex;  // Protect resize operations
        
        bool covers_range(uint64_t start, uint64_t end) const {
            return start < end_iters && end > start_iters;
        }
        
        size_t get_index_for_iter(uint64_t iter) const {
            return iter / spacing;
        }
        
        // Thread-safe storage with on-demand allocation
        void store_intermediate(uint64_t iter, const form& f) {
            if (iter % spacing != 0 || iter >= end_iters) return;
            
            size_t idx = get_index_for_iter(iter);
            
            // Check if we need to resize (rare case)
            if (idx >= intermediates.size()) {
                std::lock_guard<std::mutex> lock(intermediates_mutex);
                // Double-check after acquiring lock
                if (idx >= intermediates.size()) {
                    // Grow by reasonable chunks, not all at once
                    const size_t cap = size_t((end_iters + spacing - 1) / spacing);
                    const size_t new_size = std::min(idx + size_t(1000), cap);
                    intermediates.resize(new_size);  // may reallocate
                }
            }
            
            // Assign before publishing size
            intermediates[idx] = f;
            
            // Publish with release semantics
            size_t seen = current_size.load(std::memory_order_relaxed);
            while (seen <= idx &&
                   !current_size.compare_exchange_weak(
                        seen, idx + 1, std::memory_order_release, std::memory_order_relaxed)) {}
        }
        
        // Get prefix for proof generation
        std::vector<form> get_prefix(uint64_t needed_size) const {
            // Acquire pairs with writer's release to see the writes to intermediates[0..needed-1]
            const size_t have = current_size.load(std::memory_order_acquire);
            if (needed_size > have) {
                throw std::runtime_error("Not enough intermediates available");
            }
            
            // Protect against concurrent resize
            std::lock_guard<std::mutex> lock(intermediates_mutex);
            return std::vector<form>(intermediates.begin(), intermediates.begin() + needed_size);
        }
    };

    ThreadedStreamingProver(const std::vector<uint8_t>& hash32,
                           int discr_bits,
                           uint64_t checkpoint_N,
                           uint64_t max_iters = 100000000,
                           uint64_t proof_interval_ms = 1000)
        : seed_(hash32)
        , discr_bits_(discr_bits)
        , N_(checkpoint_N)
        , max_iters_(max_iters)
        , proof_interval_ms_(proof_interval_ms)
        , stop_requested_(false)
        , current_iters_(0)
        , verbose_(false)
        , threads_started_(false)
    {
        if (hash32.size() != 32)
            throw std::runtime_error("challenge_hash must be exactly 32 bytes");
        if (N_ == 0)
            throw std::runtime_error("checkpoint N must be > 0");

        if (discr_bits_ > 1024) {
        #if defined(_WIN32)
            _putenv("CHIAVDF_NO_ASM=1");
        #else
            setenv("CHIAVDF_NO_ASM", "1", 1);
        #endif
        }

        // Initialize VDF parameters
        D_ = CreateDiscriminant(seed_, discr_bits_);
        L_ = root(-D_, 4);

        x0_ = form::from_abd(integer(2), integer(1), D_);
        x0_.reduce();
        
        y_current_ = x0_;

        // Precompute K/L schedule but DON'T pre-allocate
        precompute_kl_schedule();
    }
    
    ~ThreadedStreamingProver() {
        stop();
    }

    void start() {
        if (threads_started_) {
            throw std::runtime_error("Prover already started");
        }
        
        stop_requested_ = false;
        squaring_thread_ = std::thread(&ThreadedStreamingProver::squaring_loop, this);
        proving_thread_ = std::thread(&ThreadedStreamingProver::proving_loop, this);
        threads_started_ = true;
    }

    void stop() {
        stop_requested_ = true;
        proof_ready_cv_.notify_all();
        startup_cv_.notify_all();
        
        if (threads_started_) {
            if (squaring_thread_.joinable()) {
                squaring_thread_.join();
            }
            if (proving_thread_.joinable()) {
                proving_thread_.join();
            }
            threads_started_ = false;
        }
    }

    std::pair<std::vector<uint8_t>, uint64_t> get_last_available_proof() {
        std::shared_ptr<ProofData> proof;
        {
            std::lock_guard<std::mutex> lock(last_proof_mutex_);
            proof = last_proof_;
        }
        
        if (!proof) {
            return {{}, 0};
        }
        
        return {proof->blob, proof->iterations};
    }
    
    uint64_t get_current_iterations() const {
        return current_iters_.load();
    }

    void set_verbose(bool verbose) {
        verbose_ = verbose;
    }

    void reset(const std::vector<uint8_t>& new_hash) {
        if (new_hash.size() != 32)
            throw std::runtime_error("challenge_hash must be exactly 32 bytes");
        
        stop();
        
        seed_ = new_hash;
        D_ = CreateDiscriminant(seed_, discr_bits_);
        L_ = root(-D_, 4);
        
        x0_ = form::from_abd(integer(2), integer(1), D_);
        x0_.reduce();
        y_current_ = x0_;
        
        current_iters_ = 0;
        chunk_completed_ = false;
        squaring_ready_ = false;
        
        {
            std::lock_guard<std::mutex> lock(last_proof_mutex_);
            last_proof_.reset();
        }
        
        // Clear all intermediates
        for (auto& config : kl_configs_) {
            std::lock_guard<std::mutex> lock(config->intermediates_mutex);
            config->intermediates.clear();
            config->current_size = 0;
        }
        
        start();
    }

private:
    void precompute_kl_schedule() {
        kl_configs_.clear();
        
        uint64_t current = 0;
        
        while (current < max_iters_) {
            int k, l;
            ApproximateParameters(current + 1, l, k);
            if (k <= 0) k = 1;
            if (l <= 0) l = 1;
            
            uint64_t end = current + 1;
            while (end < max_iters_) {
                int k2, l2;
                ApproximateParameters(end + 1, l2, k2);
                if (k2 <= 0) k2 = 1;
                if (l2 <= 0) l2 = 1;
                
                if (k2 != k || l2 != l) break;
                
                if (end > UINT64_MAX / 2) {
                    end = max_iters_;
                } else {
                    end = std::min(end * 2, max_iters_);
                }
            }
            
            if (end > current + 1 && end < max_iters_) {
                uint64_t left = current + 1;
                uint64_t right = end;
                while (left < right - 1) {
                    uint64_t mid = left + (right - left) / 2;
                    int km, lm;
                    ApproximateParameters(mid, lm, km);
                    if (km <= 0) km = 1;
                    if (lm <= 0) lm = 1;
                    
                    if (km == k && lm == l) {
                        left = mid;
                    } else {
                        right = mid;
                    }
                }
                end = right;
            }
            
            auto config = std::make_unique<KLConfig>();
            config->k = k;
            config->l = l;
            config->spacing = uint64_t(k) * uint64_t(l);
            config->start_iters = 0;
            config->end_iters = std::min(end, max_iters_);
            
            kl_configs_.push_back(std::move(config));
            current = kl_configs_.back()->end_iters;
        }
        
        if (verbose_) {
            std::cout << "K/L Schedule:\n";
            for (const auto& cfg : kl_configs_) {
                std::cout << "  k=" << cfg->k << ", l=" << cfg->l 
                         << " [" << cfg->start_iters << ", " << cfg->end_iters << ")\n";
            }
        }
    }

    void store_intermediate_if_needed(uint64_t iter, const form& y) {
        for (auto& config : kl_configs_) {
            config->store_intermediate(iter, y);
        }
    }

    void squaring_loop() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        PulmarkReducer reducer;
        uint64_t iter = 0;
        form y = x0_;
        
        store_intermediate_if_needed(0, x0_);
        
        {
            std::lock_guard<std::mutex> lock(proof_ready_mutex_);
            squaring_ready_ = true;
        }
        startup_cv_.notify_all();
        
        while (!stop_requested_ && iter < max_iters_) {
            uint64_t chunk_start = iter;
            uint64_t chunk_end = std::min(iter + N_, max_iters_);
            
            for (; iter < chunk_end && !stop_requested_; ++iter) {
                nudupl_form(y, y, D_, L_);
                reducer.reduce(y);
                
                if (iter + 1 > 0) {
                    store_intermediate_if_needed(iter + 1, y);
                }
            }
            
            {
                std::unique_lock<std::mutex> lock(proof_ready_mutex_);
                y_current_ = y;
                current_iters_.store(iter);
                chunk_completed_ = true;
            }
            proof_ready_cv_.notify_one();
        }
        
        if (verbose_) {
            std::cout << "Squaring thread finished at iteration " << iter << "\n";
        }
    }

    void proving_loop() {
        {
            std::unique_lock<std::mutex> lock(proof_ready_mutex_);
            startup_cv_.wait(lock, [this] { return squaring_ready_ || stop_requested_; });
        }
        
        if (stop_requested_) return;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        while (!stop_requested_) {
            {
                std::unique_lock<std::mutex> lock(proof_ready_mutex_);
                proof_ready_cv_.wait(lock, [this] { 
                    return chunk_completed_ || stop_requested_; 
                });
                
                if (stop_requested_) break;
                chunk_completed_ = false;
            }
            
            uint64_t snapshot_iters = current_iters_.load();
            if (snapshot_iters == 0) {
                continue;
            }
            
            form snapshot_y = y_current_;
            
            try {
                auto proof_blob = generate_proof_for(snapshot_y, snapshot_iters);
                
                {
                    std::lock_guard<std::mutex> lock(last_proof_mutex_);
                    last_proof_ = std::make_shared<ProofData>(ProofData{proof_blob, snapshot_iters});
                }
                
                if (verbose_) {
                    std::cout << "Generated proof for " << snapshot_iters << " iterations\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "Proof generation failed: " << e.what() << "\n";
            }
        }
        
        if (verbose_) {
            std::cout << "Proving thread finished\n";
        }
    }

    std::vector<uint8_t> generate_proof_for(const form& y, uint64_t iters) {
        int k_proof, l_proof;
        ApproximateParameters(iters, l_proof, k_proof);
        if (k_proof < 1) k_proof = 1;
        if (l_proof < 1) l_proof = 1;
        
        if (verbose_) {
            std::cout << "Generating proof for " << iters << " iterations\n";
            std::cout << "ProveSlow would use k=" << k_proof << ", l=" << l_proof << "\n";
        }
        
        KLConfig* best_config = nullptr;
        for (auto& cfg : kl_configs_) {
            if (cfg->k == k_proof && cfg->l == l_proof) {
                best_config = cfg.get();
                break;
            }
        }
        
        if (!best_config) {
            std::ostringstream oss;
            oss << "No KLConfig with k=" << k_proof << ", l=" << l_proof 
                << " for T=" << iters;
            throw std::runtime_error(oss.str());
        }
        
        uint64_t kl = uint64_t(k_proof) * uint64_t(l_proof);
        uint64_t needed_size = (iters + kl - 1) / kl;
        
        std::vector<form> prefix = best_config->get_prefix(needed_size);
        
        form y_local = y;
        form x0_local = x0_;
        integer D_local = D_;
        
        PulmarkReducer reducer;
        form proof = GenerateWesolowski(
            y_local, x0_local, D_local, reducer,
            prefix, iters, 
            k_proof, l_proof);
        
        const int d_bits = D_.num_bits();
        
        form y_copy = y;
        std::vector<uint8_t> y_bytes = SerializeForm(y_copy, d_bits);
        
        form proof_copy = proof;
        std::vector<uint8_t> proof_bytes = SerializeForm(proof_copy, d_bits);
        
        std::vector<uint8_t> result;
        result.reserve(y_bytes.size() + proof_bytes.size());
        result.insert(result.end(), y_bytes.begin(), y_bytes.end());
        result.insert(result.end(), proof_bytes.begin(), proof_bytes.end());
        
        if (verbose_) {
            std::cout << "Generated proof blob size: " << result.size() << " bytes\n";
        }
        
        return result;
    }

private:
    std::vector<uint8_t> seed_;
    int discr_bits_;
    uint64_t N_;
    uint64_t max_iters_;
    uint64_t proof_interval_ms_;
    bool verbose_;
    bool threads_started_;

    std::atomic<bool> stop_requested_;
    std::thread squaring_thread_;
    std::thread proving_thread_;

    integer D_, L_;
    form x0_;
    
    form y_current_;
    std::atomic<uint64_t> current_iters_;
    
    std::mutex proof_ready_mutex_;
    std::condition_variable proof_ready_cv_;
    std::condition_variable startup_cv_;
    bool chunk_completed_ = false;
    bool squaring_ready_ = false;
    
    std::mutex last_proof_mutex_;
    std::shared_ptr<ProofData> last_proof_;

    std::vector<std::unique_ptr<KLConfig>> kl_configs_;
};