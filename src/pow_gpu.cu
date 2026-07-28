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
#include <cstdio>
#include <thread>

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
        atomicAdd(&out[seg * 6 + threadIdx.x], t);
    }
}

namespace {
struct BatchScratch {
    int cap_total = 0, cap_seg = 0;
    float*    d_vals = nullptr;
    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_out = nullptr;
    uint32_t* d_idx = nullptr;
    float*    d_sorted = nullptr;
    float*    d_heads = nullptr;
    double*   d_stats = nullptr;
    int*      d_offs = nullptr;
    void*     d_temp = nullptr;
    size_t    temp_bytes = 0;
    cudaStream_t stream = nullptr;
    bool failed = false;

    bool ensure(int S, int n) {
        if (failed) return false;
        const int total = S * n;
        if (total <= cap_total && S <= cap_seg) return true;
        release();
        bool ok = cudaMalloc(&d_vals, sizeof(float)*total) == cudaSuccess
               && cudaMalloc(&d_keys_in, sizeof(uint64_t)*total) == cudaSuccess
               && cudaMalloc(&d_keys_out, sizeof(uint64_t)*total) == cudaSuccess
               && cudaMalloc(&d_idx, sizeof(uint32_t)*total) == cudaSuccess
               && cudaMalloc(&d_sorted, sizeof(float)*total) == cudaSuccess
               && cudaMalloc(&d_heads, sizeof(float)*S) == cudaSuccess
               && cudaMalloc(&d_stats, sizeof(double)*6*S) == cudaSuccess
               && cudaMalloc(&d_offs, sizeof(int)*(S+1)) == cudaSuccess
               && (stream || cudaStreamCreate(&stream) == cudaSuccess);
        if (ok) {
            std::vector<int> offs(S+1);
            for (int i = 0; i <= S; ++i) offs[i] = i * n;
            ok = cudaMemcpyAsync(d_offs, offs.data(), sizeof(int)*(S+1),
                                 cudaMemcpyHostToDevice, stream) == cudaSuccess
              && cudaStreamSynchronize(stream) == cudaSuccess;
        }
        if (ok) {
            size_t bytes = 0;
            cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, d_keys_in, d_keys_out,
                                                    total, S, d_offs, d_offs+1, 0, 64, stream);
            ok = cudaMalloc(&d_temp, bytes) == cudaSuccess;
            temp_bytes = bytes;
        }
        if (!ok) { failed = true;
            std::fprintf(stderr, "[pow-gpu] batched scratch alloc FAILED — per-stream path stays\n");
            return false; }
        cap_total = total; cap_seg = S;
        return true;
    }
    void release() {
        for (void* p : {(void*)d_vals,(void*)d_keys_in,(void*)d_keys_out,(void*)d_idx,
                        (void*)d_sorted,(void*)d_heads,(void*)d_stats,(void*)d_offs,(void*)d_temp})
            if (p) cudaFree(p);
        d_vals=nullptr; d_keys_in=nullptr; d_keys_out=nullptr; d_idx=nullptr;
        d_sorted=nullptr; d_heads=nullptr; d_stats=nullptr; d_offs=nullptr; d_temp=nullptr;
        temp_bytes=0; cap_total=0; cap_seg=0;
    }
};
thread_local BatchScratch g_batch;

// Thread-local injection: the sampler's patched sort block consumes ONE
// pre-sorted result instead of launching, then the slot auto-clears.
struct Injected { const uint32_t* idx = nullptr; const double* stats = nullptr; float head = 0.0f; };
thread_local Injected g_inj;
}  // namespace

extern "C" bool pow_gpu_sort_and_stats_batched(
        const float* const* h_logits, int S, int n, float inv_temp, int snap_bf16,
        uint32_t* h_idx /* [S*n] */, float* h_head /* [S] */, double* h_stats /* [S*6] */) {
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
    if (cudaMemsetAsync(b.d_stats, 0, sizeof(double)*6*S, b.stream) != cudaSuccess) return false;
    dim3 grid(64, S);
    k_stats_seg<<<grid, 256, 0, b.stream>>>(b.d_sorted, n, inv_temp, b.d_heads, b.d_stats);
    bool ok = cudaMemcpyAsync(h_idx, b.d_idx, sizeof(uint32_t)*total, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_head, b.d_heads, sizeof(float)*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess
           && cudaMemcpyAsync(h_stats, b.d_stats, sizeof(double)*6*S, cudaMemcpyDeviceToHost, b.stream) == cudaSuccess;
    return ok && cudaStreamSynchronize(b.stream) == cudaSuccess;
}

extern "C" void pow_gpu_inject_presorted(const uint32_t* idx, const double* stats, float head) {
    g_inj.idx = idx; g_inj.stats = stats; g_inj.head = head;
}
extern "C" bool pow_gpu_take_presorted(const uint32_t** idx, const double** stats, float* head) {
    if (!g_inj.idx) return false;
    *idx = g_inj.idx; *stats = g_inj.stats; *head = g_inj.head;
    g_inj.idx = nullptr; g_inj.stats = nullptr;
    return true;
}

// Bind the calling thread's CUDA context to a device — each mining thread
// must run its sampler kernels on ITS OWN GPU, not on device 0 by default.
extern "C" bool pow_gpu_bind_device(int cuda_ordinal) {
    return cudaSetDevice(cuda_ordinal) == cudaSuccess;
}
