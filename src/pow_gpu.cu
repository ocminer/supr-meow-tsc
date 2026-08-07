// =============================================================================
// pow_gpu.cu — GPU offload for the TensorCash proof-of-inference sampler.
//
// WHY: the reference sampler does a full std::sort over the ENTIRE vocabulary
// (Qwen3 = 151,936 entries) for EVERY generated token, single-threaded. That is
// ~20 ms/token and it is why a 5090 sits at 3-18 % utilisation while mining —
// the GPU finishes the 0.6B forward pass in microseconds and then waits on a
// host-side sort. A CUB radix sort does the same work in ~0.2 ms.
//
// BIT-EXACTNESS (the whole game): a proof is only worth anything if the
// verifier can replay it. The reference comparator is a TOTAL order:
//
//     if (a.value != b.value) return a.value > b.value;  // descending value
//     return a.index < b.index;                          // ties: ascending id
//
// We reproduce it exactly by packing each entry into one 64-bit key:
//
//     hi32 = ~monotonic_u32(value)   → ascending key == descending float
//     lo32 = index                   → ascending id breaks ties
//
// and doing a plain ascending radix sort. No floating-point arithmetic is
// performed here at all — only a bijective bit remap — so the resulting order
// is identical to the CPU path for every input, including the many exact ties
// that bf16 snapping creates (bf16 has 8 mantissa bits, so duplicate values in
// a 152k vocabulary are common, and getting tie order wrong would change which
// token ids land in the top-k).
//
// Scope note: only the sort is offloaded. The sampling arithmetic (softmax,
// CDF, the draw) stays on the CPU untouched, because that math is what the
// verifier re-derives and it is not the bottleneck (k=50 elements).
// =============================================================================

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <mutex>
#ifdef __AVX2__
#include <immintrin.h>
#endif

extern "C" {
bool pow_gpu_available(void);
bool pow_gpu_sort_desc(const float* h_vals, int n, uint32_t* h_sorted_idx, float* h_sorted_vals);
bool pow_gpu_sort_and_stats(const float* h_vals, int n, float inv_temp,
                            uint32_t* h_sorted_idx, float* h_sorted_vals, double* h_stats);
void pow_gpu_shutdown(void);
}

// --------------------------------------------------------------------------
// float -> order-preserving uint32.
//   positives: flip the sign bit           (0x8000'0000 | u)
//   negatives: flip every bit              (~u)
// After this, unsigned ascending order == IEEE-754 float ascending order.
// We then complement to get descending, and place it in the high half so the
// low half (the token id, ascending) acts as the tie-breaker.
// --------------------------------------------------------------------------
__device__ __forceinline__ uint64_t pack_key(float v, uint32_t idx) {
    uint32_t u = __float_as_uint(v);
    // IEEE-754 says -0.0 == +0.0, so the CPU comparator treats them as a TIE and
    // falls through to the index. Their bit patterns differ, though, so a naive
    // bit remap would order them by sign and diverge from the reference. This is
    // not hypothetical: the bf16 snap rounds tiny negative logits to -0.0.
    // Canonicalise the sign of zero before remapping.
    if (u == 0x80000000u) u = 0x00000000u;
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<uint64_t>(~u) << 32) | static_cast<uint64_t>(idx);
}

__global__ void k_pack(const float* __restrict__ vals, uint64_t* __restrict__ keys, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = pack_key(vals[i], static_cast<uint32_t>(i));
}

// Unpack the id AND gather the value in one pass. Doing the gather here rather
// than on the host matters more than it looks: the host version is 151,936
// RANDOM reads into a 600 KB array (cache miss per element, several ms) which
// ate most of the sort win. On the GPU the scattered read is absorbed by memory
// parallelism and the host is left with two linear copies.
__global__ void k_unpack(const uint64_t* __restrict__ keys,
                         const float*    __restrict__ vals,
                         uint32_t*       __restrict__ idx_out,
                         float*          __restrict__ val_out,
                         int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const uint32_t id = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFull);
        idx_out[i] = id;
        val_out[i] = vals[id];
    }
}


// --------------------------------------------------------------------------
// Fused statistics pass. Runs on the ALREADY-SORTED device array, so it adds
// one kernel launch and six doubles of transfer — no new round trip.
//
// Computes in a single grid-stride sweep:
//   out[0] = sum_i exp(v_i*invT - max_scaled)      -> logsumexp_full on the host
//   out[1..4] = sums over sorted rank buckets [0,50) [50,500) [500,2000) [2000,n)
//   out[5] = sum over the whole vocabulary
//
// Summation order differs from the host's sequential `double` loop. That is
// SAFE here and only here: proof_verifier.py:4094-4125 feeds these bucket means
// into a Mahalanobis distance against Σ_err — the covariance the protocol
// builds to absorb cross-GPU numerical variation — rather than comparing them
// for equality. The sort ORDER, by contrast, must stay bit-exact, which is why
// that path does no arithmetic at all.
// --------------------------------------------------------------------------
__global__ void k_stats(const float* __restrict__ sorted, int n,
                        float inv_temp, double max_scaled,
                        double* __restrict__ out) {
    __shared__ double sh[6][32];
    double acc[6] = {0,0,0,0,0,0};

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const double v = static_cast<double>(sorted[i]);
        acc[0] += exp(v * static_cast<double>(inv_temp) - max_scaled);
        if      (i < 50)   acc[1] += v;
        else if (i < 500)  acc[2] += v;
        else if (i < 2000) acc[3] += v;
        else               acc[4] += v;
        acc[5] += v;
    }
    // warp reduce, then one atomic per warp per bucket
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int b = 0; b < 6; ++b) {
        double x = acc[b];
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xFFFFFFFFu, x, off);
        if (lane == 0) sh[b][warp] = x;
    }
    __syncthreads();
    if (threadIdx.x < 6) {
        const int nwarp = (blockDim.x + 31) / 32;
        double t = 0.0;
        for (int w = 0; w < nwarp; ++w) t += sh[threadIdx.x][w];
        atomicAdd(&out[threadIdx.x], t);
    }
}

// --------------------------------------------------------------------------
// Persistent scratch. Allocating 150k-element buffers per token would cost more
// than the sort itself; the sampler is called once per token forever, so the
// buffers are allocated once and reused. Sized on first use and grown only if a
// larger vocabulary shows up (model switch).
// --------------------------------------------------------------------------
namespace {
struct Scratch {
    int        cap        = 0;
    float*     d_vals     = nullptr;
    uint64_t*  d_keys_in  = nullptr;
    uint64_t*  d_keys_out = nullptr;
    uint32_t*  d_idx      = nullptr;
    float*     d_sorted   = nullptr;
    double*    d_stats    = nullptr;   // [6]
    void*      d_temp     = nullptr;
    size_t     temp_bytes = 0;
    cudaStream_t stream   = nullptr;
    bool       failed     = false;
    bool       announced  = false;

    // One line per worker thread, once. Exists because a thread whose scratch
    // allocation fails falls back to the CPU sort FOREVER and silently — which
    // is exactly the asymmetry seen in the profiler (one thread at 3.2 ms/token,
    // another at 15.2 ms in the same process). Without this you cannot tell a
    // slow GPU path from a thread that never got one.
    void announce(bool ok, const char* why) {
        if (announced) return;
        announced = true;
        const auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFFu;
        std::fprintf(stderr, "[pow-gpu] thread %04zx: %s%s%s\n",
                     static_cast<size_t>(tid),
                     ok ? "CUDA sort path ACTIVE" : "CUDA sort path UNAVAILABLE -> CPU fallback",
                     why ? " — " : "", why ? why : "");
        std::fflush(stderr);
    }

    bool ensure(int n) {
        if (failed) return false;
        if (n <= cap) return true;
        release();
        if (cudaMalloc(&d_vals,     sizeof(float)    * n) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_vals"); return false; }
        if (cudaMalloc(&d_keys_in,  sizeof(uint64_t) * n) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_keys_in"); return false; }
        if (cudaMalloc(&d_keys_out, sizeof(uint64_t) * n) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_keys_out"); return false; }
        if (cudaMalloc(&d_idx,      sizeof(uint32_t) * n) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_idx"); return false; }
        if (cudaMalloc(&d_sorted,   sizeof(float)    * n) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_sorted"); return false; }
        if (!d_stats && cudaMalloc(&d_stats, sizeof(double) * 6) != cudaSuccess) { failed = true; announce(false, "cudaMalloc d_stats"); return false; }
        if (!stream && cudaStreamCreate(&stream) != cudaSuccess)          { failed = true; announce(false, "cudaStreamCreate"); return false; }

        size_t bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, bytes, d_keys_in, d_keys_out, n, 0, 64, stream);
        if (cudaMalloc(&d_temp, bytes) != cudaSuccess) { failed = true; announce(false, "cudaMalloc cub temp"); return false; }
        temp_bytes = bytes;
        cap = n;
        announce(true, nullptr);
        return true;
    }

    void release() {
        if (d_vals)     cudaFree(d_vals);
        if (d_keys_in)  cudaFree(d_keys_in);
        if (d_keys_out) cudaFree(d_keys_out);
        if (d_idx)      cudaFree(d_idx);
        if (d_sorted)   cudaFree(d_sorted);
        if (d_stats)    cudaFree(d_stats);
        if (d_temp)     cudaFree(d_temp);
        d_vals = nullptr; d_keys_in = nullptr; d_keys_out = nullptr;
        d_idx = nullptr;  d_temp = nullptr; d_sorted = nullptr; d_stats = nullptr; temp_bytes = 0; cap = 0;
    }
};

// One scratch per host thread: llama.cpp serves slots from multiple threads and
// two concurrent sorts must not share buffers.
thread_local Scratch g_scratch;
}  // namespace

extern "C" bool pow_gpu_available(void) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

// Returns token ids in reference order, and (optionally) the matching values
// gathered on-device. The values are copies of the caller's own floats — never
// recomputed — so no logit can be altered by the round trip. Pass nullptr for
// h_sorted_vals if only the order is wanted.
extern "C" bool pow_gpu_sort_desc(const float* h_vals, int n, uint32_t* h_sorted_idx, float* h_sorted_vals) {
    if (!h_vals || !h_sorted_idx || n <= 0) return false;
    Scratch& s = g_scratch;
    if (!s.ensure(n)) return false;

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    if (cudaMemcpyAsync(s.d_vals, h_vals, sizeof(float) * n,
                        cudaMemcpyHostToDevice, s.stream) != cudaSuccess) return false;
    k_pack<<<blocks, threads, 0, s.stream>>>(s.d_vals, s.d_keys_in, n);

    size_t bytes = s.temp_bytes;
    if (cub::DeviceRadixSort::SortKeys(s.d_temp, bytes, s.d_keys_in, s.d_keys_out,
                                       n, 0, 64, s.stream) != cudaSuccess) return false;

    k_unpack<<<blocks, threads, 0, s.stream>>>(s.d_keys_out, s.d_vals, s.d_idx, s.d_sorted, n);
    if (cudaMemcpyAsync(h_sorted_idx, s.d_idx, sizeof(uint32_t) * n,
                        cudaMemcpyDeviceToHost, s.stream) != cudaSuccess) return false;
    if (h_sorted_vals &&
        cudaMemcpyAsync(h_sorted_vals, s.d_sorted, sizeof(float) * n,
                        cudaMemcpyDeviceToHost, s.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}


// Sort + fused statistics. h_stats receives 6 doubles:
//   [0] sum_exp (for logsumexp_full), [1..4] bucket sums, [5] full sum.
// The caller divides by the bucket sizes; that division is trivial and keeping
// it host-side means the kernel has no knowledge of the stats layout.
extern "C" bool pow_gpu_sort_and_stats(const float* h_vals, int n, float inv_temp,
                                       uint32_t* h_sorted_idx, float* h_sorted_vals,
                                       double* h_stats) {
    if (!pow_gpu_sort_desc(h_vals, n, h_sorted_idx, h_sorted_vals)) return false;
    if (!h_stats) return true;
    Scratch& s = g_scratch;

    // Sorted descending, so the largest scaled logit is element 0 (inv_temp > 0).
    const double max_scaled = static_cast<double>(h_sorted_vals[0]) * static_cast<double>(inv_temp);

    if (cudaMemsetAsync(s.d_stats, 0, sizeof(double) * 6, s.stream) != cudaSuccess) return false;
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;          // grid-stride; cap the atomics
    k_stats<<<blocks, threads, 0, s.stream>>>(s.d_sorted, n, inv_temp, max_scaled, s.d_stats);
    if (cudaMemcpyAsync(h_stats, s.d_stats, sizeof(double) * 6,
                        cudaMemcpyDeviceToHost, s.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return false;
    return cudaGetLastError() == cudaSuccess;
}

extern "C" void pow_gpu_shutdown(void) { g_scratch.release(); }

// ==========================================================================
// Batched segmented path — ONE launch chain samples EVERY stream's vocabulary.
//
// The per-stream entry above costs ~1 ms per call and is invoked serially per
// stream per step, so at 16 streams the GPU idles behind 16 small launch
// chains. Here the engine hands over all S logits arrays at once: snap, pack,
// one DeviceSegmentedRadixSort over S*n keys, one unpack, one stats grid with
// blockIdx.y = stream. Results land in per-stream host buffers, and the
// sampler consumes them through a thread-local injection slot instead of
// launching anything itself.
//
// The snap is the verbatim GPU twin of snap_logits_to_precision_inplace
// ("bf16"): x + (0x7FFF + ((x>>16)&1)) & 0xFFFF0000 — same bits, same
// round-to-nearest-even, same (absent) NaN handling, because the SORT ORDER
// must stay bit-exact with the CPU comparator.
// ==========================================================================
#include <vector>

// Widen wire-format bf16 (top-16 bits, already RNE-snapped on the host during
// the staging pass) back to fp32. `u16 << 16` reproduces EXACTLY the bits that
// k_snap_bf16 would produce from the fp32 input — same snap, half the PCIe
// traffic. On a x4-riser link (GPU0: Gen3 x4, ~3 GB/s) the wire size IS the
// step time, so this is worth ~2x there.
__global__ void k_widen_bf16(const uint16_t* __restrict__ w, float* __restrict__ v, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) v[i] = __uint_as_float(static_cast<uint32_t>(w[i]) << 16);
}

__global__ void k_snap_bf16(float* __restrict__ v, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        uint32_t x = __float_as_uint(v[i]);
        x = (x + (0x00007FFFu + ((x >> 16) & 1u))) & 0xFFFF0000u;
        v[i] = __uint_as_float(x);
    }
}

__global__ void k_pack_seg(const float* __restrict__ vals, uint64_t* __restrict__ keys,
                           int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) keys[i] = pack_key(vals[i], static_cast<uint32_t>(i % n));
}

// Combined-key variant: segment id rides in the TOP bits, so ONE flat radix
// sort of all S*n keys produces segment-major / value-descending / id-ascending
// order — no DeviceSegmentedRadixSort, which is 5-10x slower than a flat sort
// for few-large-segments (measured 17-22 ms vs a target of ~2 ms).
// Layout: [55..50]=seg (S<=64), [49..18]=transformed value, [17..0]=id
// (n<=262144). Sorted ascending over bits [0,56).
__device__ __forceinline__ uint32_t transform_val(float v) {
    uint32_t u = __float_as_uint(v);
    if (u == 0x80000000u) u = 0x00000000u;      // -0.0 == +0.0, same rank
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ~u;                                   // ascending sort => value DESC
}

// ---------------------------------------------------------------------------
// TOP-K RADIX SELECT (opt-in: MEOW_TOPK_SELECT=1)
//
// nsys on a 5090 at 128 slots: the full-vocabulary radix sort is 40.7% of ALL
// GPU time, plus 6.0% histogram and 1.2% exclusive-sum — while poi.cpp only
// ever reads the top K=2048 columns back. We were sorting 151,936 keys per
// stream and discarding 98.7% of the result.
//
// Instead: histogram the top HIST_BITS of the transformed value, walk buckets
// ascending until the cumulative count covers K, then compact only the keys at
// or above that bucket and sort THOSE. Typically ~4k candidates instead of
// 152k — a ~35x smaller sort.
//
// WHY THIS IS PROOF-SAFE. The key is [55..50]=seg | [49..18]=value | [17..0]=id
// and sorts ascending, i.e. value DESC with id ASC on ties — the chain's
// normative rule (pow_v3 §4.1). Selection is by the SAME key, so the surviving
// candidate set is exactly the true top-|cand| and its sorted order is
// bit-identical to the full sort's first |cand| entries. Ties are never split:
// a bucket is taken whole, so two equal values always land on the same side.
// The verifier replays from the top-50 support (pow_utils.py:139 caps top_k at
// 50 when top_p<1.0), which is far inside K.
//
// If a degenerate distribution puts more than the candidate capacity in range,
// the caller falls back to the full sort — correctness never depends on the
// histogram being well behaved.
// 16 bits, not 11: the field is a TRANSFORMED float, so the high bits are
// sign+exponent and logits sit in a narrow exponent range — at 11 bits the
// first bucket reaching K already held >8192 keys and every step fell back to
// the full sort (measured: ratio 1.023, i.e. no gain at all). 16 bits gives
// sign+exponent+7 mantissa bits, which actually separates the head of the
// distribution. 65536 bins x 4 B exceeds shared memory, so the histogram is
// global — it is a streaming pass and was only ~6% of GPU time anyway.
#define TOPK_HIST_BITS 16
#define TOPK_HIST_BINS (1 << TOPK_HIST_BITS)

// Bucket of a packed key = the top TOPK_HIST_BITS of its 32-bit value field.
__device__ __forceinline__ uint32_t topk_bin_of(uint64_t key) {
    return static_cast<uint32_t>((key >> (18 + 32 - TOPK_HIST_BITS)) & (TOPK_HIST_BINS - 1));
}

__global__ void k_topk_hist(const uint64_t* __restrict__ keys, int n, int total,
                            uint32_t* __restrict__ hist /* [S][BINS] */) {
    const int seg = blockIdx.y;
    const size_t base = static_cast<size_t>(seg) * n;
    uint32_t* g = hist + static_cast<size_t>(seg) * TOPK_HIST_BINS;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        atomicAdd(&g[topk_bin_of(keys[base + i])], 1u);
    (void)total;
}

// One block per segment: walk buckets ascending (= value descending) until the
// running count covers K, and publish that bucket as the cutoff.
__global__ void k_topk_cutoff(const uint32_t* __restrict__ hist, int K, int cap,
                              uint32_t* __restrict__ cutoff /* [S] */,
                              int* __restrict__ overflow /* [S] */) {
    const int seg = blockIdx.x;
    if (threadIdx.x != 0) return;
    const uint32_t* g = hist + static_cast<size_t>(seg) * TOPK_HIST_BINS;
    uint32_t acc = 0, b = 0;
    for (; b < TOPK_HIST_BINS; ++b) {
        acc += g[b];
        if (acc >= static_cast<uint32_t>(K)) break;
    }
    if (b >= TOPK_HIST_BINS) b = TOPK_HIST_BINS - 1;
    cutoff[seg]   = b;
    // acc is exactly how many keys the compaction will emit for this segment.
    // Publish it: a fallback must be able to say HOW badly it overflowed,
    // otherwise widening the histogram is guesswork.
    overflow[seg] = (acc > static_cast<uint32_t>(cap)) ? static_cast<int>(acc) : 0;
}

// Compact keys whose bucket <= cutoff into a per-segment slot of fixed stride
// `cap`. Order within the slot is arbitrary; the sort that follows fixes it.
__global__ void k_topk_compact(const uint64_t* __restrict__ keys, int n, int cap,
                               const uint32_t* __restrict__ cutoff,
                               uint64_t* __restrict__ out, int* __restrict__ cnt) {
    const int seg = blockIdx.y;
    const size_t base = static_cast<size_t>(seg) * n;
    const uint32_t cut = cutoff[seg];
    uint64_t* dst = out + static_cast<size_t>(seg) * cap;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const uint64_t k = keys[base + i];
        if (topk_bin_of(k) <= cut) {
            const int slot = atomicAdd(&cnt[seg], 1);
            if (slot < cap) dst[slot] = k;
        }
    }
}

// Segment offsets for the compacted array: [seg*cap, seg*cap + cnt[seg]).
__global__ void k_topk_offsets(const int* __restrict__ cnt, int S, int cap,
                               int* __restrict__ beg, int* __restrict__ end) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < S) {
        beg[s] = s * cap;
        int c = cnt[s]; if (c > cap) c = cap;
        end[s] = s * cap + c;
    }
}

// Stats for the select path. The rank buckets [0,50) [50,500) [500,2000) all
// live inside K, so they come from the sorted top-K. The remaining terms are
// ORDER-INDEPENDENT and are taken over the raw unsorted values:
//   out[0] logsumexp accumulator, out[5] whole-vocabulary sum,
//   out[4] = out[5] - sum(top 2000)   (ranks [2000,n))
// Summation order differs from the full-sort path, which is safe for exactly
// the reason documented at k_stats: these bucket means feed a Mahalanobis
// distance against the protocol's Sigma_err, not an equality test.
__global__ void k_stats_topk(const float* __restrict__ topv, int K,
                             const float* __restrict__ raw, int n, float inv_temp,
                             const float* __restrict__ heads,
                             double* __restrict__ out /* [S][6] */) {
    const int seg = blockIdx.y;
    const double max_scaled = static_cast<double>(heads[seg]) * static_cast<double>(inv_temp);
    const float* tv = topv + static_cast<size_t>(seg) * K;
    const float* rv = raw  + static_cast<size_t>(seg) * n;
    __shared__ double sh[6][32];
    double acc[6] = {0,0,0,0,0,0};
    // full-vocabulary, order-independent terms
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const double v = static_cast<double>(rv[i]);
        acc[0] += exp(v * static_cast<double>(inv_temp) - max_scaled);
        acc[5] += v;
    }
    // rank buckets from the sorted top-K
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < K; r += blockDim.x * gridDim.x) {
        const double v = static_cast<double>(tv[r]);
        if      (r < 50)   acc[1] += v;
        else if (r < 500)  acc[2] += v;
        else if (r < 2000) acc[3] += v;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int j = 0; j < 6; ++j) {
        double x = acc[j];
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xFFFFFFFFu, x, off);
        if (lane == 0) sh[j][warp] = x;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nw = (blockDim.x + 31) >> 5;
        for (int j = 0; j < 6; ++j) {
            double x = (threadIdx.x < nw) ? sh[j][threadIdx.x] : 0.0;
            for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xFFFFFFFFu, x, off);
            if (threadIdx.x == 0 && x != 0.0) atomicAdd(&out[seg*6 + j], x);
        }
    }
}

// Unpack the first K of each compacted+sorted segment into the [S][K] outputs.
__global__ void k_topk_unpack(const uint64_t* __restrict__ sortedk, int cap, int K,
                              const float* __restrict__ vals, int n,
                              uint32_t* __restrict__ idx_out, float* __restrict__ val_out) {
    const int seg = blockIdx.y;
    const uint64_t* src = sortedk + static_cast<size_t>(seg) * cap;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < K; r += blockDim.x * gridDim.x) {
        const uint64_t k = src[r];
        const uint32_t id = static_cast<uint32_t>(k & 0x3FFFFull);
        idx_out[static_cast<size_t>(seg) * K + r] = id;
        val_out[static_cast<size_t>(seg) * K + r] = vals[static_cast<size_t>(seg) * n + id];
    }
}

__global__ void k_pack_flat(const float* __restrict__ vals, uint64_t* __restrict__ keys,
                            int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const uint64_t seg = static_cast<uint64_t>(i / n);
        const uint64_t idx = static_cast<uint64_t>(i % n);
        keys[i] = (seg << 50) | (static_cast<uint64_t>(transform_val(vals[i])) << 18) | idx;
    }
}

__global__ void k_unpack_flat(const uint64_t* __restrict__ keys,
                              const float*    __restrict__ vals,
                              uint32_t*       __restrict__ idx_out,
                              float*          __restrict__ val_out,
                              int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const int seg = static_cast<int>(keys[i] >> 50);
        const uint32_t id = static_cast<uint32_t>(keys[i] & 0x3FFFFULL);
        idx_out[i] = id;
        val_out[i] = vals[static_cast<size_t>(seg) * n + id];
    }
}

// ---------------------------------------------------------------------------
// GPU-resident-logits variants: the input is llama's own DEVICE output tensor
// (read-only — the next decode reuses that memory, and nothing may write it).
// The bf16 snap therefore happens IN REGISTERS, fused into pack and the
// unpack gather. Bit-identical to snapping a copy first: same RNE formula,
// and transform_val() of the snapped value gives the same key bits.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float snap_bf16_reg(float v) {
    uint32_t x = __float_as_uint(v);
    x = (x + (0x00007FFFu + ((x >> 16) & 1u))) & 0xFFFF0000u;
    return __uint_as_float(x);
}

__global__ void k_pack_flat_snap(const float* __restrict__ vals, uint64_t* __restrict__ keys,
                                 int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const uint64_t seg = static_cast<uint64_t>(i / n);
        const uint64_t idx = static_cast<uint64_t>(i % n);
        const float sv = snap_bf16_reg(vals[i]);
        keys[i] = (seg << 50) | (static_cast<uint64_t>(transform_val(sv)) << 18) | idx;
    }
}

// Snapping variant for the DEVICE path: values are read straight from llama's
// output tensor, so the bf16 snap must be applied in-register exactly as
// k_unpack_flat_snap does — the snapped value is what the proof commits to.
__global__ void k_topk_unpack_snap(const uint64_t* __restrict__ sortedk, int cap, int K,
                                   const float* __restrict__ vals, int n,
                                   uint32_t* __restrict__ idx_out, float* __restrict__ val_out) {
    const int seg = blockIdx.y;
    const uint64_t* src = sortedk + static_cast<size_t>(seg) * cap;
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < K; r += blockDim.x * gridDim.x) {
        const uint32_t id = static_cast<uint32_t>(src[r] & 0x3FFFFull);
        idx_out[static_cast<size_t>(seg) * K + r] = id;
        val_out[static_cast<size_t>(seg) * K + r] =
            snap_bf16_reg(vals[static_cast<size_t>(seg) * n + id]);
    }
}

// ---------------------------------------------------------------------------
// 32-BIT SORTPAIRS PATH (MEOW_SORT32=1).
//
// The 56-bit flat key costs 7 radix passes; the sort is bandwidth-bound at 85%
// of peak, so passes ARE the cost. A 32-bit key (transform_val of the SNAPPED
// value) is 4 passes = 57% of the traffic. The token id rides along as the
// 32-bit VALUE of a SortPairs, and the segment field is not needed at all
// because DeviceSegmentedRadixSort scopes each segment via b.d_offs.
//
// CORRECTNESS RESTS ON STABILITY: CUB's radix sort is stable, input order
// within a segment is ascending id (we pack i = seg*n + id in order), so equal
// keys keep ascending id — which IS the normative tie rule (value DESC via
// transform_val, id ASC). The snap must happen BEFORE the transform: the
// snapped value is what the proof commits to, and two floats that snap to the
// same bf16 MUST compare equal here or the tiebreak never engages.
__global__ void k_pack32_snap(const float* __restrict__ vals,
                              uint32_t* __restrict__ key32,
                              uint32_t* __restrict__ val32,
                              int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        key32[i] = transform_val(snap_bf16_reg(vals[i]));
        val32[i] = static_cast<uint32_t>(i % n);          // token id in segment
    }
}

__global__ void k_unpack32_snap(const uint32_t* __restrict__ val32_sorted,
                                const float*    __restrict__ vals,
                                uint32_t*       __restrict__ idx_out,
                                float*          __restrict__ val_out,
                                int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const int seg = i / n;
        const uint32_t id = val32_sorted[i];
        idx_out[i] = id;
        val_out[i] = snap_bf16_reg(vals[static_cast<size_t>(seg) * n + id]);
    }
}

__global__ void k_unpack_flat_snap(const uint64_t* __restrict__ keys,
                                   const float*    __restrict__ vals,
                                   uint32_t*       __restrict__ idx_out,
                                   float*          __restrict__ val_out,
                                   int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const int seg = static_cast<int>(keys[i] >> 50);
        const uint32_t id = static_cast<uint32_t>(keys[i] & 0x3FFFFULL);
        idx_out[i] = id;
        val_out[i] = snap_bf16_reg(vals[static_cast<size_t>(seg) * n + id]);
    }
}

// The 20 telemetry probes (reference: working_logits[i*(n/20)], snapped) —
// gathered here because in device mode no host copy of the logits exists.
__global__ void k_probes_snap(const float* __restrict__ vals, int S, int n,
                              float* __restrict__ out /* [S][20] */) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // S*20 threads
    if (i < S * 20) {
        const int s = i / 20;
        const int p = i % 20;
        const int step = n / 20 > 0 ? n / 20 : 1;
        const size_t idx = static_cast<size_t>(p) * step;
        out[i] = (idx < (size_t)n) ? snap_bf16_reg(vals[(size_t)s * n + idx]) : 0.0f;
    }
}

__global__ void k_unpack_seg(const uint64_t* __restrict__ keys,
                             const float*    __restrict__ vals,
                             uint32_t*       __restrict__ idx_out,
                             float*          __restrict__ val_out,
                             int n, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const int seg = i / n;
        const uint32_t id = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFull);
        idx_out[i] = id;
        val_out[i] = vals[seg * n + id];
    }
}

// One block per segment writes its stats directly (see k_stats_seg), so the
// zeroing pass is unnecessary. That memset was 309,010 calls and 6% of ALL
// CUDA API time in the nsys trace. MEOW_STATS_1BLK=0 restores the old
// 64-block + memset path for A/B.
// MEASURED WORSE — default OFF. Removing the 309k cudaMemsetAsync calls (6% of
// CUDA API time) by giving each segment ONE block that writes instead of
// accumulates costs 10%: one block cannot saturate the GPU for a 151,936-entry
// reduction, and the sampler tail goes 3.4 -> 6.6 ms. Bracketed on a solo 5090:
//   64-block+memset 18.60 | 1-block 16.70 | 1-block 16.66
// Kept as an opt-in (MEOW_STATS_1BLK=1) only because it may win on a card where
// the sampler is not the parallelism-limited path. Launch-count reduction is
// not free when it trades away occupancy.
static inline bool meow_stats_1blk() {
    static const bool v = [](){
        const char* e = std::getenv("MEOW_STATS_1BLK");
        return e && *e == '1';
    }();
    return v;
}

__global__ void k_stats_seg(const float* __restrict__ sorted, int n, float inv_temp,
                            const float* __restrict__ heads,
                            double* __restrict__ out /* [S][6] */) {
    const int seg = blockIdx.y;
    const float* v = sorted + static_cast<size_t>(seg) * n;
    const double max_scaled = static_cast<double>(heads[seg]) * static_cast<double>(inv_temp);
    __shared__ double sh[6][32];
    double acc[6] = {0,0,0,0,0,0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const double x = static_cast<double>(v[i]);
        acc[0] += exp(x * static_cast<double>(inv_temp) - max_scaled);
        if      (i < 50)   acc[1] += x;
        else if (i < 500)  acc[2] += x;
        else if (i < 2000) acc[3] += x;
        else               acc[4] += x;
        acc[5] += x;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = 0; b < 6; ++b) {
        double x = acc[b];
        for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xFFFFFFFFu, x, off);
        if (lane == 0) sh[b][warp] = x;
    }
    __syncthreads();
    if (threadIdx.x < 6) {
        const int nwarp = (blockDim.x + 31) / 32;
        double t = 0.0;
        for (int w = 0; w < nwarp; ++w) t += sh[threadIdx.x][w];
        // gridDim.x == 1 means this block owns the whole segment, so it can
        // WRITE instead of accumulate — which lets the caller drop the
        // per-call cudaMemsetAsync(d_stats). That memset was 309,010 calls and
        // 6% of ALL CUDA API time in the nsys trace; at ~23 us of host time
        // each it cost more than the work it prepared. Multi-block grids keep
        // the atomic path and still need the memset.
        if (gridDim.x == 1) out[seg * 6 + threadIdx.x] = t;
        else                atomicAdd(&out[seg * 6 + threadIdx.x], t);
    }
}

namespace {
struct BatchScratch {
    int cap_total = 0, cap_seg = 0;
    float*    d_vals = nullptr;
    uint16_t* d_wire = nullptr;   // bf16 wire format (contiguous fast path)
    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_out = nullptr;
    uint32_t* d_idx = nullptr;
    float*    d_sorted = nullptr;
    float*    d_heads = nullptr;
    double*   d_stats = nullptr;
    float*    d_probes = nullptr;   // [S][20] telemetry probes (device-logits mode)
    int*      d_offs = nullptr;
    // top-K select scratch (MEOW_TOPK_SELECT=1)
    uint32_t* d_hist = nullptr;     // [S][BINS]
    uint32_t* d_cut  = nullptr;     // [S]
    int*      d_cnt  = nullptr;     // [S]
    int*      d_ovf  = nullptr;     // [S]
    uint64_t* d_cand = nullptr;     // [S][cand_cap]
    uint64_t* d_cand_sorted = nullptr;
    int*      d_kbeg = nullptr;     // [S]
    int*      d_kend = nullptr;     // [S]
    int       cand_cap = 0;
    bool      topk_ready = false;
    void*     d_temp = nullptr;
    size_t    temp_bytes = 0;
    cudaStream_t stream = nullptr;
    bool failed = false;

    // Report the first allocation failure with the numbers needed to act on it.
    // Without this a starved GPU looks identical to a broken kernel: every
    // window fails, nothing says why, and the miner sits at 0.00 PoI/s
    // indefinitely (observed: 7h on a rented rig).
    void report_alloc_failure(int S, int n) {
        static std::atomic<bool> once{false};
        if (once.exchange(true)) return;
        size_t freeb = 0, totb = 0;
        cudaMemGetInfo(&freeb, &totb);
        const double need_mb = (double)S * n * (4+2+8+8+4+4) / (1024.0*1024.0);
        std::fprintf(stderr,
            "\n[pow_gpu] FATAL: sampler scratch allocation failed on the GPU.\n"
            "          wanted ~%.0f MB per worker group (%d streams x %d vocab); "
            "VRAM free %.2f GB of %.2f GB.\n"
            "          The model and its KV cache have taken the card. Lower "
            "--slots (or --ctx), or use a GPU with more VRAM.\n"
            "          Every window will fail until this is fixed.\n\n",
            need_mb, S, n, freeb / 1073741824.0, totb / 1073741824.0);
        std::fflush(stderr);
    }

    bool ensure(int S, int n) {
        if (failed) return false;
        const int total = S * n;
        if (total <= cap_total && S <= cap_seg) return true;
        release();
        bool ok = cudaMalloc(&d_vals, sizeof(float)*total) == cudaSuccess
               && cudaMalloc(&d_wire, sizeof(uint16_t)*total) == cudaSuccess
               && cudaMalloc(&d_keys_in, sizeof(uint64_t)*total) == cudaSuccess
               && cudaMalloc(&d_keys_out, sizeof(uint64_t)*total) == cudaSuccess
               && cudaMalloc(&d_idx, sizeof(uint32_t)*total) == cudaSuccess
               && cudaMalloc(&d_sorted, sizeof(float)*total) == cudaSuccess
               && cudaMalloc(&d_heads, sizeof(float)*S) == cudaSuccess
               && cudaMalloc(&d_stats, sizeof(double)*6*S) == cudaSuccess
               && cudaMalloc(&d_probes, sizeof(float)*20*S) == cudaSuccess
               && cudaMalloc(&d_offs, sizeof(int)*(S+1)) == cudaSuccess
               && (stream || cudaStreamCreate(&stream) == cudaSuccess);
        if (ok) {
            std::vector<int> offs(S+1);
            for (int i = 0; i <= S; ++i) offs[i] = i * n;
            ok = cudaMemcpyAsync(d_offs, offs.data(), sizeof(int)*(S+1),
                                 cudaMemcpyHostToDevice, stream) == cudaSuccess
              && cudaStreamSynchronize(stream) == cudaSuccess;
        }
        // top-K select scratch. cand_cap bounds the compaction; a distribution
        // that overflows it falls back to the full sort rather than truncating.
        if (ok) {
            // 65536, not a guess: the instrumented fallback reported
            // worst_candidates=39303 for a real Qwen3-8B logit distribution at
            // 16-bit buckets. Logits cluster tightly, so even fine buckets pull
            // ~39k candidates — still a ~3.9x smaller sort than 151,936, which
            // is the whole point. Sized with headroom above the measurement.
            cand_cap = 65536;
            topk_ready =
                   cudaMalloc(&d_hist, sizeof(uint32_t)*(size_t)S*TOPK_HIST_BINS) == cudaSuccess
                && cudaMalloc(&d_cut,  sizeof(uint32_t)*S) == cudaSuccess
                && cudaMalloc(&d_cnt,  sizeof(int)*S) == cudaSuccess
                && cudaMalloc(&d_ovf,  sizeof(int)*S) == cudaSuccess
                && cudaMalloc(&d_cand, sizeof(uint64_t)*(size_t)S*cand_cap) == cudaSuccess
                && cudaMalloc(&d_cand_sorted, sizeof(uint64_t)*(size_t)S*cand_cap) == cudaSuccess
                && cudaMalloc(&d_kbeg, sizeof(int)*S) == cudaSuccess
                && cudaMalloc(&d_kend, sizeof(int)*S) == cudaSuccess;
            if (!topk_ready) cudaGetLastError();   // optional path; full sort still works
        }
        if (ok) {
            size_t b1 = 0, b2 = 0, b3 = 0;
            cub::DeviceSegmentedRadixSort::SortKeys(nullptr, b1, d_keys_in, d_keys_out,
                                                    total, S, d_offs, d_offs+1, 0, 64, stream);
            cub::DeviceRadixSort::SortKeys(nullptr, b2, d_keys_in, d_keys_out,
                                           total, 0, 56, stream);
            // 32-bit SortPairs path (MEOW_SORT32): key32/val32 alias the halves
            // of the existing 8 B/elem key buffers, so no extra VRAM.
            cub::DeviceSegmentedRadixSort::SortPairs(
                nullptr, b3,
                reinterpret_cast<uint32_t*>(d_keys_in),  reinterpret_cast<uint32_t*>(d_keys_out),
                reinterpret_cast<uint32_t*>(d_keys_in)+total, reinterpret_cast<uint32_t*>(d_keys_out)+total,
                total, S, d_offs, d_offs+1, 0, 32, stream);
            size_t bytes = b1 > b2 ? b1 : b2; if (b3 > bytes) bytes = b3;
            ok = cudaMalloc(&d_temp, bytes) == cudaSuccess;
            temp_bytes = bytes;
        }
        if (!ok) { report_alloc_failure(S, n); cudaGetLastError(); failed = true;
            std::fprintf(stderr, "[pow-gpu] batched scratch alloc FAILED — per-stream path stays\n");
            return false; }
        cap_total = total; cap_seg = S;
        return true;
    }
    void release() {
        for (void* p : {(void*)d_vals,(void*)d_wire,(void*)d_keys_in,(void*)d_keys_out,(void*)d_idx,
                        (void*)d_sorted,(void*)d_heads,(void*)d_stats,(void*)d_probes,(void*)d_offs,(void*)d_temp,
                        (void*)d_hist,(void*)d_cut,(void*)d_cnt,(void*)d_ovf,(void*)d_cand,
                        (void*)d_cand_sorted,(void*)d_kbeg,(void*)d_kend})
            if (p) cudaFree(p);
        d_hist=nullptr; d_cut=nullptr; d_cnt=nullptr; d_ovf=nullptr;
        d_cand=nullptr; d_cand_sorted=nullptr; d_kbeg=nullptr; d_kend=nullptr;
        cand_cap=0; topk_ready=false;
        d_vals=nullptr; d_wire=nullptr; d_keys_in=nullptr; d_keys_out=nullptr; d_idx=nullptr;
        d_sorted=nullptr; d_heads=nullptr; d_stats=nullptr; d_probes=nullptr; d_offs=nullptr; d_temp=nullptr;
        temp_bytes=0; cap_total=0; cap_seg=0;
    }
};
thread_local BatchScratch g_batch;

// Thread-local injection: the sampler's patched sort block consumes ONE
// pre-sorted result instead of launching, then the slot auto-clears.
// `probes` is non-null only in device-logits mode: the sampler then has NO
// valid host copy of the logits and must take telemetry probes from here.
struct Injected { const uint32_t* idx = nullptr; const float* val = nullptr;
                  const double* stats = nullptr; float head = 0.0f; int count = 0;
                  const float* probes = nullptr; int probe_n = 0; };
thread_local Injected g_inj;
thread_local double p_stage = 0, p_issue = 0, p_gpu = 0;
thread_local uint64_t p_n = 0;
}  // namespace

extern "C" bool pow_gpu_sort_and_stats_batched(
        const float* const* h_logits, int S, int n, float inv_temp, int snap_bf16,
        uint32_t* h_idx /* [S*n] */, float* h_val /* [S*n] */,
        float* h_head /* [S] */, double* h_stats /* [S*6] */) {
    if (S < 1 || n < 1) return false;
    if (!g_batch.ensure(S, n)) return false;
    auto& b = g_batch;
    const int total = S * n;
    for (int s = 0; s < S; ++s)
        if (cudaMemcpyAsync(b.d_vals + (size_t)s*n, h_logits[s], sizeof(float)*n,
                            cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
    const int T = 256, G = (total + T - 1) / T;
    if (snap_bf16) k_snap_bf16<<<G, T, 0, b.stream>>>(b.d_vals, total);
    k_pack_seg<<<G, T, 0, b.stream>>>(b.d_vals, b.d_keys_in, n, total);
    cub::DeviceSegmentedRadixSort::SortKeys(b.d_temp, b.temp_bytes, b.d_keys_in, b.d_keys_out,
                                            total, S, b.d_offs, b.d_offs + 1, 0, 64, b.stream);
    k_unpack_seg<<<G, T, 0, b.stream>>>(b.d_keys_out, b.d_vals, b.d_idx, b.d_sorted, n, total);
    if (cudaMemcpy2DAsync(b.d_heads, sizeof(float), b.d_sorted, sizeof(float)*(size_t)n,
                          sizeof(float), S, cudaMemcpyDeviceToDevice, b.stream) != cudaSuccess)
        return false;
    if (!meow_stats_1blk() && cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream) != cudaSuccess) return false;
    dim3 grid(meow_stats_1blk() ? 1 : 64, S);
    k_stats_seg<<<grid, 256, 0, b.stream>>>(b.d_sorted, n, inv_temp, b.d_heads, b.d_stats);
    bool ok = cudaMemcpyAsync(h_idx, b.d_idx, sizeof(uint32_t)*total, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           // Sorted VALUES ride back too: filling pretemp_desc_ from these is
           // two linear reads; gathering working_logits[idx[r]] on the host is
           // 151,936 random reads per token — the exact cache-miss storm the
           // per-stream kernel's fused gather was built to avoid.
           && cudaMemcpyAsync(h_val, b.d_sorted, sizeof(float)*total, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_head, b.d_heads, sizeof(float)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_stats, b.d_stats, sizeof(double)*6*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess;
    return ok && cudaStreamSynchronize(b.stream) == cudaSuccess;
}

extern "C" void pow_gpu_inject_presorted(const uint32_t* idx, const float* val,
                                          const double* stats, float head, int count) {
    g_inj.idx = idx; g_inj.val = val; g_inj.stats = stats; g_inj.head = head; g_inj.count = count;
    g_inj.probes = nullptr; g_inj.probe_n = 0;
}
extern "C" void pow_gpu_inject_presorted2(const uint32_t* idx, const float* val,
                                          const double* stats, float head, int count,
                                          const float* probes, int probe_n) {
    g_inj.idx = idx; g_inj.val = val; g_inj.stats = stats; g_inj.head = head; g_inj.count = count;
    g_inj.probes = probes; g_inj.probe_n = probe_n;
}
// The sampler asks this BEFORE its host-side snap: with probes injected there
// is no valid host logits array and the snap must be skipped entirely.
extern "C" bool pow_gpu_peek_presorted_probes(void) {
    return g_inj.idx != nullptr && g_inj.probes != nullptr;
}
extern "C" bool pow_gpu_take_presorted(const uint32_t** idx, const float** val,
                                       const double** stats, float* head, int* count) {
    if (!g_inj.idx) return false;
    *idx = g_inj.idx; *val = g_inj.val; *stats = g_inj.stats; *head = g_inj.head; *count = g_inj.count;
    g_inj.idx = nullptr; g_inj.val = nullptr; g_inj.stats = nullptr;
    g_inj.probes = nullptr; g_inj.probe_n = 0;
    return true;
}
extern "C" bool pow_gpu_take_presorted2(const uint32_t** idx, const float** val,
                                        const double** stats, float* head, int* count,
                                        const float** probes, int* probe_n) {
    if (!g_inj.idx) return false;
    *idx = g_inj.idx; *val = g_inj.val; *stats = g_inj.stats; *head = g_inj.head; *count = g_inj.count;
    *probes = g_inj.probes; *probe_n = g_inj.probe_n;
    g_inj.idx = nullptr; g_inj.val = nullptr; g_inj.stats = nullptr;
    g_inj.probes = nullptr; g_inj.probe_n = 0;
    return true;
}

// True iff p is a CUDA device pointer (the engine verifies llama's tensor
// really lives on the GPU before enabling device-logits mode).
extern "C" bool pow_gpu_is_device_ptr(const void* p) {
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) { cudaGetLastError(); return false; }
    return attr.type == cudaMemoryTypeDevice;
}

// Which GPU actually owns this allocation. Needed for --split-model: llama
// puts the output tensor on whichever card holds the last layer, which is not
// necessarily the card a sampler thread was bound to. Sorting a pointer that
// lives on another device is an illegal access, so the caller rebinds to the
// answer. Returns -1 if the pointer is not device memory.
extern "C" int pow_gpu_device_of_ptr(const void* p) {
    cudaPointerAttributes attr{};
    if (!p) return -1;
    if (cudaPointerGetAttributes(&attr, p) != cudaSuccess) { cudaGetLastError(); return -1; }
    if (attr.type != cudaMemoryTypeDevice) return -1;
    return attr.device;
}

// ==========================================================================
// Device-input sort: identical outputs to pow_gpu_sort_and_stats_batched_k,
// but the logits never touch the host — d_logits is llama's own output
// tensor (S contiguous rows of n floats, read-only). No narrow, no H2D; the
// bf16 snap runs in registers inside pack/unpack. Additionally returns the
// 20 telemetry probes per stream, since no host copy exists to read them.
// ==========================================================================
extern "C" bool pow_gpu_sort_and_stats_device_k(
        const float* d_logits, int S, int n, int topk, float inv_temp,
        uint32_t* h_idx /* [S*topk] */, float* h_val /* [S*topk] */,
        float* h_head /* [S] */, double* h_stats /* [S*6] */,
        float* h_probes /* [S*20] */) {
    if (!d_logits || S < 1 || n < 1) return false;
    if (topk < 1 || topk > n) topk = n;
    if (S > 64 || n > (1 << 18)) {
        // Not a transient error: this configuration can never sort. Say so —
        // silently returning false here fails every window forever.
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                "\n[pow_gpu] FATAL: sampler slice out of range (%d streams, %d vocab).\n"
                "          A worker group may sort at most 64 streams and 262144 vocab.\n"
                "          Raise --groups so that slots/groups <= 64 (e.g. --slots %d "
                "needs --groups >= %d).\n\n",
                S, n, S, (S + 63) / 64);
            std::fflush(stderr);
        }
        return false;
    }
    if (!g_batch.ensure(S, n)) return false;
    auto& b = g_batch;
    const int total = S * n;
    const int T = 256, G = (total + T - 1) / T;
    // CEILING PROBE (MEOW_SAMPLER_STUB=1): skip ALL sampler GPU work and return
    // structurally-valid garbage. Shares are invalid by construction — this is
    // ONLY for measuring the upper bound of any sampler optimisation. If the
    // step rate does not rise with the sampler removed entirely, then the
    // sampler is fully hidden behind decode and NO amount of sort/stats work
    // will ever help, which closes that line of investigation for good.
    // Never enable on a real pool.
    // RELEASE GUARD: this returns garbage, so a public binary must not let a
    // user turn it on against a real pool by accident — the shares would be
    // rejected and look like a miner bug. Requires the extra confirmation
    // token, which is documented nowhere a miner would find casually.
    static const bool stub = [](){
        const char* e = std::getenv("MEOW_SAMPLER_STUB");
        const char* c = std::getenv("MEOW_I_UNDERSTAND_THIS_MINES_INVALID_SHARES");
        return e && *e == '1' && c && *c == '1';
    }();
    if (stub) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true))
            std::fprintf(stderr,
                "\n[pow_gpu] *** SAMPLER STUB ACTIVE — EVERY SHARE IS INVALID ***\n"
                "          This is a benchmark instrument. Unset MEOW_SAMPLER_STUB to mine.\n\n");
        for (int sg = 0; sg < S; ++sg) {
            for (int r = 0; r < topk; ++r) {
                h_idx[(size_t)sg*topk + r] = (uint32_t)r;
                h_val[(size_t)sg*topk + r] = 100.0f - (float)r * 0.01f;
            }
            h_head[sg] = 100.0f;
            for (int j = 0; j < 6; ++j) h_stats[(size_t)sg*6 + j] = 1.0;
            for (int j = 0; j < 20; ++j) h_probes[(size_t)sg*20 + j] = 0.0f;
        }
        return true;
    }

    // Per-stage GPU timers for the path that ACTUALLY runs. The existing
    // [evprof] probes are in the host-logits function; device-logits mode uses
    // this one, so the sampler's internal breakdown was never visible — which
    // is why five optimisation attempts had to guess at which stage dominates.
    // POW_GPU_EVPROF=1 enables; costs two event records per stage otherwise.
    static thread_local cudaEvent_t dev[8] = {};
    static thread_local bool dev_ready = false;
    const bool dprof = [](){ const char* e = std::getenv("POW_GPU_EVPROF"); return e && *e == '1'; }();
    if (dprof && !dev_ready) { for (auto& e : dev) cudaEventCreate(&e); dev_ready = true; }
    #define DEVR(i) if (dprof) cudaEventRecord(dev[i], b.stream)
    // 32-bit SortPairs path: 4 radix passes instead of 7. See kernel comments
    // for the stability-based correctness argument. Mutually exclusive with the
    // top-K select (both are alternative sorts).
    static const bool sort32 = [](){
        const char* e = std::getenv("MEOW_SORT32");
        return e && *e == '1';
    }();
    uint32_t* k32i = reinterpret_cast<uint32_t*>(b.d_keys_in);
    uint32_t* k32o = reinterpret_cast<uint32_t*>(b.d_keys_out);
    DEVR(0);
    if (sort32)
        k_pack32_snap<<<G, T, 0, b.stream>>>(d_logits, k32i, k32i + total, n, total);
    else
        k_pack_flat_snap<<<G, T, 0, b.stream>>>(d_logits, b.d_keys_in, n, total);
    DEVR(1);

    // ---- top-K radix select on the DEVICE path -------------------------
    // The sort is 1.47 of 2.07 ms of sampler GPU time (71%, POW_GPU_EVPROF).
    // Select ~39k candidates instead of sorting all 151,936 keys per stream.
    // This was previously implemented ONLY in the host-logits function, which
    // device-logits mode never calls — so MEOW_TOPK_SELECT measured nothing.
    // Proof-safety is unchanged from that implementation: selection uses the
    // SAME packed key, buckets are taken whole so ties never split, and the
    // top-K ordering is bit-identical to the full sort's prefix.
    static const bool dev_select = [](){
        const char* e = std::getenv("MEOW_TOPK_SELECT");
        return e && *e == '1';
    }();
    bool dsel = false;
    if (!sort32 && dev_select && b.topk_ready && topk > 0 && topk <= b.cand_cap && n > 4 * b.cand_cap) {
        cudaMemsetAsync(b.d_hist, 0, sizeof(uint32_t)*(size_t)S*TOPK_HIST_BINS, b.stream);
        cudaMemsetAsync(b.d_cnt,  0, sizeof(int)*S, b.stream);
        dim3 hg(64, S);
        k_topk_hist<<<hg, 256, 0, b.stream>>>(b.d_keys_in, n, total, b.d_hist);
        k_topk_cutoff<<<S, 32, 0, b.stream>>>(b.d_hist, topk, b.cand_cap, b.d_cut, b.d_ovf);
        k_topk_compact<<<hg, 256, 0, b.stream>>>(b.d_keys_in, n, b.cand_cap, b.d_cut, b.d_cand, b.d_cnt);
        k_topk_offsets<<<(S + 63) / 64, 64, 0, b.stream>>>(b.d_cnt, S, b.cand_cap, b.d_kbeg, b.d_kend);
        static thread_local int* ho = nullptr; static thread_local int ho_cap = 0;
        if (ho_cap < S) {
            if (ho) cudaFreeHost(ho);
            ho = nullptr; ho_cap = 0;
            if (cudaMallocHost(&ho, sizeof(int)*S) == cudaSuccess) ho_cap = S; else cudaGetLastError();
        }
        bool ovf = true;
        if (ho_cap >= S &&
            cudaMemcpyAsync(ho, b.d_ovf, sizeof(int)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess &&
            cudaStreamSynchronize(b.stream) == cudaSuccess) {
            ovf = false;
            for (int i = 0; i < S; ++i) if (ho[i]) { ovf = true; break; }
        }
        if (!ovf) {
            size_t tb = b.temp_bytes;
            if (cub::DeviceSegmentedRadixSort::SortKeys(
                    b.d_temp, tb, b.d_cand, b.d_cand_sorted,
                    S * b.cand_cap, S, b.d_kbeg, b.d_kend, 0, 56, b.stream) == cudaSuccess) {
                dim3 ug((topk + 255) / 256, S);
                k_topk_unpack_snap<<<ug, 256, 0, b.stream>>>(b.d_cand_sorted, b.cand_cap, topk,
                                                             d_logits, n, b.d_idx, b.d_sorted);
                dsel = true;
            }
        }
        if (!dsel) {
            static std::atomic<bool> once{false};
            if (!once.exchange(true))
                std::fprintf(stderr, "[pow_gpu] device top-K select fell back to full sort\n");
        }
    }
    if (sort32) {
        size_t tb = b.temp_bytes;
        cub::DeviceSegmentedRadixSort::SortPairs(
            b.d_temp, tb, k32i, k32o, k32i + total, k32o + total,
            total, S, b.d_offs, b.d_offs + 1, 0, 32, b.stream);
    } else if (!dsel)
        cub::DeviceRadixSort::SortKeys(b.d_temp, b.temp_bytes, b.d_keys_in, b.d_keys_out,
                                       total, 0, 56, b.stream);
    DEVR(2);
    if (sort32)
        k_unpack32_snap<<<G, T, 0, b.stream>>>(k32o + total, d_logits, b.d_idx, b.d_sorted, n, total);
    else if (!dsel)
        k_unpack_flat_snap<<<G, T, 0, b.stream>>>(b.d_keys_out, d_logits, b.d_idx, b.d_sorted, n, total);
    DEVR(3);
    const size_t dstride = dsel ? (size_t)topk : (size_t)n;
    if (cudaMemcpy2DAsync(b.d_heads, sizeof(float), b.d_sorted, sizeof(float)*dstride,
                          sizeof(float), S, cudaMemcpyDeviceToDevice, b.stream) != cudaSuccess)
        return false;
    if (!meow_stats_1blk() && cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream) != cudaSuccess) return false;
    dim3 grid(meow_stats_1blk() ? 1 : 64, S);
    DEVR(4);
    // With the select engaged the sorted array holds only topk ranks, so the
    // rank buckets come from it while the order-independent full-vocabulary
    // terms are taken from llama's raw tensor (k_stats_topk), exactly as on
    // the host path.
    if (dsel) {
        cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream);
        dim3 sg(64, S);
        k_stats_topk<<<sg, 256, 0, b.stream>>>(b.d_sorted, topk, d_logits, n,
                                               inv_temp, b.d_heads, b.d_stats);
    } else
    k_stats_seg<<<grid, 256, 0, b.stream>>>(b.d_sorted, n, inv_temp, b.d_heads, b.d_stats);
    DEVR(5);
    k_probes_snap<<<(S*20 + 63)/64, 64, 0, b.stream>>>(d_logits, S, n, b.d_probes);
    DEVR(6);
    bool ok = cudaMemcpy2DAsync(h_idx, sizeof(uint32_t)*topk, b.d_idx, sizeof(uint32_t)*dstride,
                                sizeof(uint32_t)*topk, S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpy2DAsync(h_val, sizeof(float)*topk, b.d_sorted, sizeof(float)*dstride,
                                sizeof(float)*topk, S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_head, b.d_heads, sizeof(float)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_stats, b.d_stats, sizeof(double)*6*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_probes, b.d_probes, sizeof(float)*20*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess;
    if (dprof) cudaEventRecord(dev[7], b.stream);
    ok = ok && cudaStreamSynchronize(b.stream) == cudaSuccess;
    if (dprof && ok) {
        static thread_local uint64_t dn = 0; static thread_local float da[7] = {};
        float dt = 0;
        for (int i = 0; i < 7; ++i) { cudaEventElapsedTime(&dt, dev[i], dev[i+1]); da[i] += dt; }
        if (++dn % 256 == 0) {
            std::fprintf(stderr,
                "[devprof] pack=%.3f sort=%.3f unpack=%.3f heads+memset=%.3f "
                "stats=%.3f probes=%.3f d2h=%.3f ms/step\n",
                da[0]/dn, da[1]/dn, da[2]/dn, da[3]/dn, da[4]/dn, da[5]/dn, da[6]/dn);
            std::fflush(stderr);
        }
    }
    const cudaError_t ce = cudaGetLastError();
    if (!ok || ce != cudaSuccess) {
        // The device path has no host fallback (llama's D2H is skipped), so
        // this kills the window. Name the CUDA error once — "sampler step
        // failed" on its own is not actionable.
        static std::atomic<bool> once{false};
        if (!once.exchange(true)) {
            std::fprintf(stderr,
                "\n[pow_gpu] device sort failed: %s (%d streams, %d vocab, topk %d)\n\n",
                cudaGetErrorString(ce == cudaSuccess ? cudaErrorUnknown : ce), S, n, topk);
            std::fflush(stderr);
        }
        return false;
    }
    return true;
}

// Bind the calling thread's CUDA context to a device — each mining thread
// must run its sampler kernels on ITS OWN GPU, not on device 0 by default.
extern "C" bool pow_gpu_bind_device(int cuda_ordinal) {
    return cudaSetDevice(cuda_ordinal) == cudaSuccess;
}

// Full-device barrier on the CURRENTLY bound device. Under --split-model the
// logits tensor is produced on the last-layer GPU; llama_get_logits_device()
// calls ctx->synchronize(), but that did NOT fully order our direct read of
// that tensor against llama's producing stream on the other device — the
// device-path proofs came out with a noise-lifted logit tail (70th ~2.6 vs
// ~0.8 single) and failed model replay. A device sync on the owner GPU before
// we read closes that gap.
extern "C" void pow_gpu_device_sync() {
    cudaDeviceSynchronize();
}

// Drop this thread's device scratch so the next ensure() reallocates on
// whatever GPU is bound NOW. Required when --split-model makes a sampler
// thread rebind: the buffers were allocated on the old card, and launching
// kernels against them from the new one is an illegal access — which is
// precisely how the first rebind attempt failed, after the rebind itself had
// already succeeded. Host-pinned buffers are untouched; page-locked memory is
// reachable from any device in the process.
extern "C" void pow_gpu_reset_device_scratch() {
    g_batch.release();
    g_batch.failed     = false;
    g_batch.cap_total  = 0;
    g_batch.cap_seg    = 0;
    g_batch.temp_bytes = 0;
    g_batch.stream     = nullptr;
    g_scratch.release();
}

// Process-global pinned registration of the (single) llama logits buffer.
// With per-group sorting, 12 worker threads DMA from slices of the SAME
// buffer; the per-thread cudaHostRegister below would race — one thread wins,
// the rest see AlreadyRegistered, mark themselves failed and fall back to a
// pointless staging memcpy forever. The pool registers the full range ONCE up
// front; threads whose source lies inside it skip registration entirely.
namespace {
std::mutex     g_hostreg_mtx;
const uint8_t* g_hostreg_base = nullptr;
size_t         g_hostreg_size = 0;
}
extern "C" bool pow_gpu_register_host_range(const void* p, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_hostreg_mtx);
    if (g_hostreg_base == static_cast<const uint8_t*>(p) && g_hostreg_size >= bytes) return true;
    if (g_hostreg_base) {
        cudaHostUnregister(const_cast<uint8_t*>(g_hostreg_base));
        g_hostreg_base = nullptr; g_hostreg_size = 0;
    }
    if (cudaHostRegister(const_cast<void*>(p), bytes, cudaHostRegisterPortable) == cudaSuccess) {
        g_hostreg_base = static_cast<const uint8_t*>(p); g_hostreg_size = bytes;
        std::fprintf(stderr, "[pow-gpu] logits page-locked globally (%zu MB) — DMA H2D\n", bytes >> 20);
        std::fflush(stderr);
        return true;
    }
    cudaGetLastError();
    return false;
}
static bool pow_gpu_host_range_covered(const void* p, size_t bytes) {
    std::lock_guard<std::mutex> lk(g_hostreg_mtx);
    const auto* q = static_cast<const uint8_t*>(p);
    return g_hostreg_base && q >= g_hostreg_base && q + bytes <= g_hostreg_base + g_hostreg_size;
}

// Truncated-copy variant under debug: identical sort/stats, but only the top
// `topk` ranks of idx+val ride back to the host (the CPU tail reads no further
// when GPU stats are valid). This is the version that hung in the miner.
extern "C" bool pow_gpu_sort_and_stats_batched_k(
        const float* const* h_logits, int S, int n, int topk, float inv_temp, int snap_bf16_in,
        uint32_t* h_idx /* [S*topk] */, float* h_val /* [S*topk] */,
        float* h_head /* [S] */, double* h_stats /* [S*6] */) {
    int snap_bf16 = snap_bf16_in;
    if (S < 1 || n < 1) return false;
    if (topk < 1 || topk > n) topk = n;
    if (!g_batch.ensure(S, n)) return false;
    auto& b = g_batch;
    const int total = S * n;
    const auto tA = std::chrono::steady_clock::now();
    static thread_local cudaEvent_t ev[9] = {};
    static thread_local bool ev_ready = false;
    const bool ev_prof = [](){ const char* e = std::getenv("POW_GPU_EVPROF"); return e && *e == '1'; }();
    if (ev_prof && !ev_ready) { for (auto& e : ev) cudaEventCreate(&e); ev_ready = true; }
    #define EVR(i) if (ev_prof) cudaEventRecord(ev[i], b.stream)
    EVR(8);   // FIRST thing on the stream this call: 8→0 spans H2D + any queue delay

    // The measured wall at 48 streams was HOST-side: 48 cudaMemcpyAsync calls
    // per step (~150k driver calls/s) serialize in the driver and contend
    // across processes. llama.cpp writes one output row per batch slot into a
    // single buffer, and the stepwise loop assigns rows in stream order — so
    // the 48 "separate" logits arrays are almost always ONE contiguous block.
    // Detect that and make it ONE driver call.
    bool contiguous = true;
    { static const bool allow = [](){ const char* e = std::getenv("POW_GPU_CONTIG");
        return !(e && *e == '0'); }();
      if (!allow) contiguous = false; }
    if (contiguous)
        for (int s = 1; s < S; ++s)
            if (h_logits[s] != h_logits[0] + (size_t)s * n) { contiguous = false; break; }
    { static bool once = false;
      if (!once) { once = true;
        std::fprintf(stderr, "[pow-gpu] batched H2D path: %s (S=%d)\n",
                     contiguous ? "CONTIGUOUS (1 copy/step)" : "per-stream (S copies/step)", S);
        std::fflush(stderr); } }
    if (contiguous) {
        // Page-lock llama's logits buffer ONCE per thread: a pageable 29 MB
        // H2D goes through the driver's staging path (~8-10 ms/step measured);
        // registered memory is straight DMA (~1 ms). The buffer address is
        // stable for the life of the context, so one registration serves every
        // step. Failure is non-fatal — the copy still works, just slower.
        static thread_local const void* reg_base = nullptr;
        static thread_local size_t      reg_size = 0;
        static thread_local int         reg_state = 0;   // 0=untried 1=ok -1=failed
        // Slice already inside the globally registered llama buffer? Then it
        // is DMA-ready — no per-thread registration, no staging fallback.
        const bool glob_reg = pow_gpu_host_range_covered(h_logits[0], sizeof(float)*(size_t)total);
        if (glob_reg) reg_state = 1;
        // Try registration ONCE. The failed branch used to re-fire every step
        // (reg_base stays null, so `reg_base != h_logits[0]` was always true):
        // 1687 retries, each walking 29 MB of pages inside the driver before
        // failing — the hidden ~9 ms/step that no GPU event could see.
        if (!glob_reg &&
            (reg_state == 0 || (reg_state > 0 && reg_base != (const void*)h_logits[0]))) {
            if (reg_base) { cudaHostUnregister(const_cast<void*>(reg_base)); reg_base = nullptr; }
            const cudaError_t rc = cudaHostRegister(const_cast<float*>(h_logits[0]),
                                                    sizeof(float)*(size_t)total,
                                                    cudaHostRegisterPortable);
            if (rc == cudaSuccess) {
                reg_base = h_logits[0]; reg_size = sizeof(float)*(size_t)total; reg_state = 1;
                std::fprintf(stderr, "[pow-gpu] logits page-locked (%zu MB) — DMA H2D\n", reg_size >> 20);
            } else {
                reg_state = -1;
                cudaGetLastError();
                std::fprintf(stderr, "[pow-gpu] cudaHostRegister failed (%s) — pinned STAGING instead\n",
                             cudaGetErrorName(rc));
            }
            std::fflush(stderr);
        }
        // bf16 wire: narrow to the snapped top-16 bits DURING the staging pass
        // we pay anyway, ship half the bytes, widen on device (bit-identical
        // to snapping the fp32 on device). POW_GPU_BF16_WIRE=0 restores fp32.
        static thread_local uint16_t* stage16 = nullptr;
        static thread_local size_t stage16_cap = 0;
        static const bool bf16_wire = [](){ const char* e = std::getenv("POW_GPU_BF16_WIRE");
            return !(e && *e == '0'); }();
        bool shipped16 = false;
        if (bf16_wire && snap_bf16) {
            if (stage16_cap < (size_t)total) {
                if (stage16) cudaFreeHost(stage16);
                stage16 = nullptr; stage16_cap = 0;
                if (cudaMallocHost(&stage16, sizeof(uint16_t)*(size_t)total) == cudaSuccess)
                    stage16_cap = (size_t)total;
                else cudaGetLastError();
            }
            if (stage16) {
                const uint32_t* in = reinterpret_cast<const uint32_t*>(h_logits[0]);
                int i = 0;
#ifdef __AVX2__
                // 8 lanes per iteration: same RNE-to-bf16 narrowing as the
                // scalar tail below, ~4x faster (~2 ms -> ~0.5 ms for 7.3M
                // values). packus interleaves 128-bit lanes; the permute
                // restores element order.
                const __m256i bias = _mm256_set1_epi32(0x7FFF);
                const __m256i one  = _mm256_set1_epi32(1);
                for (; i + 16 <= total; i += 16) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
                    __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i + 8));
                    a = _mm256_srli_epi32(_mm256_add_epi32(a, _mm256_add_epi32(bias,
                            _mm256_and_si256(_mm256_srli_epi32(a, 16), one))), 16);
                    c = _mm256_srli_epi32(_mm256_add_epi32(c, _mm256_add_epi32(bias,
                            _mm256_and_si256(_mm256_srli_epi32(c, 16), one))), 16);
                    __m256i pk = _mm256_packus_epi32(a, c);
                    pk = _mm256_permute4x64_epi64(pk, 0xD8);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(stage16 + i), pk);
                }
#endif
                for (; i < total; ++i) {
                    const uint32_t x = in[i];
                    stage16[i] = static_cast<uint16_t>((x + (0x00007FFFu + ((x >> 16) & 1u))) >> 16);
                }
                if (cudaMemcpyAsync(b.d_wire, stage16, sizeof(uint16_t)*(size_t)total,
                                    cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
                shipped16 = true;
            }
        }
        if (!shipped16) {
            const float* src = h_logits[0];
            if (reg_state < 0) {
                static thread_local float* stage = nullptr;
                static thread_local size_t stage_cap = 0;
                if (stage_cap < (size_t)total) {
                    if (stage) cudaFreeHost(stage);
                    stage = nullptr; stage_cap = 0;
                    if (cudaMallocHost(&stage, sizeof(float)*(size_t)total) == cudaSuccess)
                        stage_cap = (size_t)total;
                    else cudaGetLastError();
                }
                if (stage) { std::memcpy(stage, h_logits[0], sizeof(float)*(size_t)total); src = stage; }
            }
            if (cudaMemcpyAsync(b.d_vals, src, sizeof(float)*total,
                                cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
        }
        p_stage += std::chrono::duration<double>(std::chrono::steady_clock::now() - tA).count();
        EVR(0);
        if (shipped16) {
            k_widen_bf16<<<(total + 255) / 256, 256, 0, b.stream>>>(b.d_wire, b.d_vals, total);
            snap_bf16 = 0;   // values are already snapped; skip k_snap_bf16
        }
    } else {
        for (int s = 0; s < S; ++s)
            if (cudaMemcpyAsync(b.d_vals + (size_t)s*n, h_logits[s], sizeof(float)*n,
                                cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
    }
    const int T = 256, G = (total + T - 1) / T;
    if (snap_bf16) k_snap_bf16<<<G, T, 0, b.stream>>>(b.d_vals, total);
    EVR(1);

    // ---- top-K radix select (opt-in) -------------------------------------
    // Sorts ~4k candidates instead of 151,936 per stream. Falls through to the
    // full sort on ANY doubt: not enabled, no scratch, or a segment whose
    // candidate set overflowed cand_cap.
    static const bool topk_select = [](){
        const char* e = std::getenv("MEOW_TOPK_SELECT");
        return e && *e == '1';
    }();
    bool selected = false;
    if (topk_select && b.topk_ready && topk > 0 && topk <= b.cand_cap && n > 4 * b.cand_cap) {
        k_pack_seg<<<G, T, 0, b.stream>>>(b.d_vals, b.d_keys_in, n, total);
        cudaMemsetAsync(b.d_hist, 0, sizeof(uint32_t)*(size_t)S*TOPK_HIST_BINS, b.stream);
        cudaMemsetAsync(b.d_cnt,  0, sizeof(int)*S, b.stream);
        dim3 hg(64, S);
        k_topk_hist<<<hg, 256, 0, b.stream>>>(
            b.d_keys_in, n, total, b.d_hist);
        k_topk_cutoff<<<S, 32, 0, b.stream>>>(b.d_hist, topk, b.cand_cap, b.d_cut, b.d_ovf);
        k_topk_compact<<<hg, 256, 0, b.stream>>>(b.d_keys_in, n, b.cand_cap, b.d_cut,
                                                 b.d_cand, b.d_cnt);
        k_topk_offsets<<<(S + 63) / 64, 64, 0, b.stream>>>(b.d_cnt, S, b.cand_cap,
                                                           b.d_kbeg, b.d_kend);
        // One tiny D2H: did any segment overflow? Cheaper than being wrong.
        static thread_local int* h_ovf = nullptr;
        static thread_local int  h_ovf_cap = 0;
        if (h_ovf_cap < S) {
            if (h_ovf) cudaFreeHost(h_ovf);
            h_ovf = nullptr; h_ovf_cap = 0;
            if (cudaMallocHost(&h_ovf, sizeof(int)*S) == cudaSuccess) h_ovf_cap = S;
            else cudaGetLastError();
        }
        bool overflowed = true;
        if (h_ovf_cap >= S &&
            cudaMemcpyAsync(h_ovf, b.d_ovf, sizeof(int)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess &&
            cudaStreamSynchronize(b.stream) == cudaSuccess) {
            overflowed = false;
            for (int i = 0; i < S; ++i) if (h_ovf[i]) { overflowed = true; break; }
        }
        if (!overflowed) {
            size_t tb = b.temp_bytes;
            if (cub::DeviceSegmentedRadixSort::SortKeys(
                    b.d_temp, tb, b.d_cand, b.d_cand_sorted,
                    S * b.cand_cap, S, b.d_kbeg, b.d_kend, 0, 56, b.stream) == cudaSuccess) {
                dim3 ug((topk + 255) / 256, S);
                k_topk_unpack<<<ug, 256, 0, b.stream>>>(b.d_cand_sorted, b.cand_cap, topk,
                                                        b.d_vals, n, b.d_idx, b.d_sorted);
                // head = rank-0 value of each segment, now at stride topk
                cudaMemcpy2DAsync(b.d_heads, sizeof(float), b.d_sorted, sizeof(float)*(size_t)topk,
                                  sizeof(float), S, cudaMemcpyDeviceToDevice, b.stream);
                if (!meow_stats_1blk()) cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream);
                dim3 sg(meow_stats_1blk() ? 1 : 64, S);
                k_stats_topk<<<sg, 256, 0, b.stream>>>(b.d_sorted, topk, b.d_vals, n,
                                                       inv_temp, b.d_heads, b.d_stats);
                selected = true;
            }
        }
        if (!selected) {
            static std::atomic<bool> once{false};
            if (!once.exchange(true)) {
                int worst = 0;
                if (h_ovf_cap >= S) for (int i = 0; i < S; ++i) if (h_ovf[i] > worst) worst = h_ovf[i];
                std::fprintf(stderr,
                    "[pow_gpu] top-K select fell back: ready=%d topk=%d cap=%d n=%d "
                    "worst_candidates=%d (0 => not an overflow)\n",
                    (int)b.topk_ready, topk, b.cand_cap, n, worst);
            }
        }
    }
    if (!selected)
    if (S <= 64 && n <= (1 << 18)) {
        // Flat combined-key sort: one radix pass over all S*n keys.
        k_pack_flat<<<G, T, 0, b.stream>>>(b.d_vals, b.d_keys_in, n, total);
        EVR(2);
        cub::DeviceRadixSort::SortKeys(b.d_temp, b.temp_bytes, b.d_keys_in, b.d_keys_out,
                                       total, 0, 56, b.stream);
        EVR(3);
        k_unpack_flat<<<G, T, 0, b.stream>>>(b.d_keys_out, b.d_vals, b.d_idx, b.d_sorted, n, total);
        EVR(4);
    } else {
        k_pack_seg<<<G, T, 0, b.stream>>>(b.d_vals, b.d_keys_in, n, total);
        cub::DeviceSegmentedRadixSort::SortKeys(b.d_temp, b.temp_bytes, b.d_keys_in, b.d_keys_out,
                                                total, S, b.d_offs, b.d_offs + 1, 0, 64, b.stream);
        k_unpack_seg<<<G, T, 0, b.stream>>>(b.d_keys_out, b.d_vals, b.d_idx, b.d_sorted, n, total);
    }
    // The select path already produced heads/stats and packs at stride topk;
    // the full-sort path packs at stride n and needs them computed here.
    if (!selected) {
        if (cudaMemcpy2DAsync(b.d_heads, sizeof(float), b.d_sorted, sizeof(float)*(size_t)n,
                              sizeof(float), S, cudaMemcpyDeviceToDevice, b.stream) != cudaSuccess)
            return false;
        if (!meow_stats_1blk() && cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream) != cudaSuccess) return false;
        dim3 grid(meow_stats_1blk() ? 1 : 64, S);
        EVR(5);
        k_stats_seg<<<grid, 256, 0, b.stream>>>(b.d_sorted, n, inv_temp, b.d_heads, b.d_stats);
        EVR(6);
    }
    const size_t src_stride = selected ? (size_t)topk : (size_t)n;
    // Top-`topk` of each stream only: src stride src_stride, dst stride topk.
    const auto tB = std::chrono::steady_clock::now();
    bool ok = cudaMemcpy2DAsync(h_idx, sizeof(uint32_t)*topk, b.d_idx, sizeof(uint32_t)*src_stride,
                                sizeof(uint32_t)*topk, S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpy2DAsync(h_val, sizeof(float)*topk, b.d_sorted, sizeof(float)*src_stride,
                                sizeof(float)*topk, S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_head, b.d_heads, sizeof(float)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_stats, b.d_stats, sizeof(double)*6*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess;
    p_issue += std::chrono::duration<double>(std::chrono::steady_clock::now() - tB).count();
    const auto tC = std::chrono::steady_clock::now();
    ok = ok && cudaStreamSynchronize(b.stream) == cudaSuccess;
    p_gpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - tC).count();
    if (ev_prof && ok) {
        EVR(7);   // after everything (recorded post-sync; timeline query below)
        static thread_local uint64_t evn = 0;
        static thread_local float evacc[7] = {};
        float dt = 0;
        static thread_local float ev_h2d = 0;
        cudaEventElapsedTime(&dt, ev[8], ev[0]); ev_h2d += dt;
        for (int i = 0; i < 7; ++i) {
            if (i == 4 && !(S <= 64 && n <= (1 << 18))) continue;
            cudaEventElapsedTime(&dt, ev[i], ev[i+1]); evacc[i] += dt;
        }
        if (++evn % 256 == 0) {
            std::fprintf(stderr, "[evprof] h2d+delay=%.2f snap=%.2f pack=%.2f sort=%.2f unpack=%.2f gap=%.2f stats=%.2f d2h=%.2f ms\n",
                         ev_h2d/evn, evacc[0]/evn, evacc[1]/evn, evacc[2]/evn, evacc[3]/evn,
                         evacc[4]/evn, evacc[5]/evn, evacc[6]/evn);
            std::fflush(stderr);
        }
    }
    if (++p_n % 512 == 0) {
        std::fprintf(stderr, "[prof3] stage+h2d=%.1f issue=%.1f gpuwait=%.1f ms/step\n",
                     1000.0*p_stage/p_n, 1000.0*p_issue/p_n, 1000.0*p_gpu/p_n);
        std::fflush(stderr);
    }
    return ok;
}

// Device-buffer helpers for the engine's double-buffered decode: batch A's
// logits are copied OUT of llama's (reused) output tensor before batch B's
// decode is issued. Host-ordered: the copy is synchronized before the next
// llama_decode call, so llama's kernels can never overwrite data still being
// read. 29 MB D2D on a 5090 is ~40 us — noise next to a 3.8 ms decode.
extern "C" void* pow_gpu_device_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    return p;
}
extern "C" void pow_gpu_device_free(void* p) { if (p) cudaFree(p); }
extern "C" bool pow_gpu_d2d_copy_sync(void* dst, const void* src, size_t bytes) {
    static thread_local cudaStream_t s = nullptr;
    if (!s && cudaStreamCreate(&s) != cudaSuccess) { cudaGetLastError(); return false; }
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s) == cudaSuccess
        && cudaStreamSynchronize(s) == cudaSuccess;
}

// ==========================================================================
// CORRUPTOR (test-only, POW_CORRUPT=<mode>). Deliberately produces proofs
// that are NOT what the registered model computed, so the pool can measure
// what its fingerprint/canary checks actually catch. Modes:
//   1 random   — logits replaced by uniform noise in a plausible range
//   2 noise    — real logits + Gaussian noise (POW_CORRUPT_SIGMA, default 0.5)
//   3 replay   — logits of stream (s+1)%S used for stream s (context mismatch)
// NEVER enable on a payable worker: these are dishonest shares by design.
// ==========================================================================
__global__ void k_corrupt_random(float* __restrict__ v, int total, uint64_t seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint64_t x = seed ^ (uint64_t)i * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    // uniform in [-8, 24) — the rough range real logits occupy
    v[i] = -8.0f + 32.0f * ((float)(x >> 40) / 16777216.0f);
}
__global__ void k_corrupt_noise(float* __restrict__ v, int total, uint64_t seed, float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint64_t x = seed ^ (uint64_t)i * 0x9E3779B97F4A7C15ull;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    const float u1 = fmaxf(1e-7f, (float)((x >> 40) & 0xFFFFFF) / 16777216.0f);
    const float u2 = (float)((x >> 16) & 0xFFFFFF) / 16777216.0f;
    v[i] += sigma * sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
extern "C" bool pow_gpu_corrupt_device(float* d_logits, int S, int n, int mode,
                                       float sigma, uint64_t seed) {
    const int total = S * n;
    const int T = 256, G = (total + T - 1) / T;
    static thread_local cudaStream_t cs = nullptr;
    if (!cs && cudaStreamCreate(&cs) != cudaSuccess) { cudaGetLastError(); return false; }
    if (mode == 1) {
        k_corrupt_random<<<G, T, 0, cs>>>(d_logits, total, seed);
    } else if (mode == 2) {
        k_corrupt_noise<<<G, T, 0, cs>>>(d_logits, total, seed, sigma);
    } else if (mode == 3) {
        // Rotate rows by one stream: s reads (s+1)%S. Needs a temp copy of
        // row 0 so the rotation is not destructive.
        static thread_local float* tmp = nullptr;
        static thread_local size_t tmp_n = 0;
        if (tmp_n < (size_t)n) {
            if (tmp) cudaFree(tmp);
            if (cudaMalloc(&tmp, sizeof(float) * n) != cudaSuccess) { cudaGetLastError(); return false; }
            tmp_n = n;
        }
        cudaMemcpyAsync(tmp, d_logits, sizeof(float) * n, cudaMemcpyDeviceToDevice, cs);
        for (int s = 0; s + 1 < S; ++s)
            cudaMemcpyAsync(d_logits + (size_t)s * n, d_logits + (size_t)(s + 1) * n,
                            sizeof(float) * n, cudaMemcpyDeviceToDevice, cs);
        cudaMemcpyAsync(d_logits + (size_t)(S - 1) * n, tmp, sizeof(float) * n,
                        cudaMemcpyDeviceToDevice, cs);
    } else {
        return false;
    }
    return cudaStreamSynchronize(cs) == cudaSuccess && cudaGetLastError() == cudaSuccess;
}

// Pinned host allocation for the sampler's receive buffers. An "async" D2H
// into pageable memory is synchronous in effect — the driver stages it and
// blocks; with a PITCHED 2D copy it degrades to row-by-row staging, which
// measured 10.5 ms/step. Pinned destinations make the same copies true DMA.
extern "C" void* pow_gpu_host_alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    return p;
}
extern "C" void pow_gpu_host_free(void* p) {
    if (p) cudaFreeHost(p);
}
