// Exact BF16 histogram: derive rank buckets from counts rather than sorting
// every vocabulary entry. Token-id insertion only touches the requested head.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace {
constexpr int bins = 65536, threads = 256, bins_per_thread = bins / threads;
__device__ float snap(float f) {
    uint32_t u = __float_as_uint(f);
    u = (u + 0x7fffU + ((u >> 16) & 1U)) & 0xffff0000U;
    return __uint_as_float(u);
}
__device__ uint32_t key(float f) {
    uint32_t u = __float_as_uint(snap(f)) >> 16;
    if (u == 0x8000U) u = 0; // signed zeros share a rank
    return (u & 0x8000U) ? u : ((~u) & 0x7fffU);
}
__device__ float value(uint32_t k) {
    return __uint_as_float(((k & 0x8000U) ? k : ((~k) & 0x7fffU)) << 16);
}
__global__ void count_values(const float* values, int n, uint32_t* counts) {
    const int s = blockIdx.y;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        atomicAdd(counts + size_t(s) * bins + key(values[size_t(s) * n + i]), 1U);
}
__device__ uint32_t overlap(uint32_t start, uint32_t count, uint32_t lo, uint32_t hi) {
    const uint32_t a = max(start, lo), b = min(start + count, hi);
    return b > a ? b - a : 0;
}
__global__ void summarize(uint32_t* hist, int n, int k, float inv_temp, double* stats) {
    const int s = blockIdx.x, t = threadIdx.x, begin = t * bins_per_thread;
    uint32_t* row = hist + size_t(s) * bins;
    using Scan = cub::BlockScan<uint32_t, threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ float head;
    __shared__ double partial[6][threads];
    uint32_t total = 0, prefix = 0;
    for (int j = 0; j < bins_per_thread; ++j) total += row[begin + j];
    Scan(scan).ExclusiveSum(total, prefix);
    uint32_t rank = prefix;
    for (int j = 0; j < bins_per_thread; ++j) {
        const auto count = row[begin + j];
        if (count && rank == 0) head = value(begin + j);
        rank += count;
    }
    __syncthreads();
    rank = prefix;
    for (int j = 0; j < bins_per_thread; ++j) {
        const uint32_t count = row[begin + j];
        row[begin + j] = rank;
        rank += count;
    }
    __syncthreads();
    // Adjacent BF16 bins usually contain nearly all the mass. Distribute
    // those bins over all lanes instead of making one lane evaluate their
    // exponentials serially. Keep full ranks until statistics are complete.
    double sums[6] = {};
    for (int bin = t; bin < bins; bin += threads) {
        const uint32_t start = row[bin];
        const uint32_t count = (bin + 1 < bins ? row[bin + 1] : uint32_t(n)) - start;
        if (count) {
            const double v = value(bin);
            sums[0] += double(count) * exp(v * double(inv_temp) - double(head) * double(inv_temp));
            sums[1] += v * overlap(start, count, 0, 50);
            sums[2] += v * overlap(start, count, 50, 500);
            sums[3] += v * overlap(start, count, 500, 2000);
            sums[4] += v * overlap(start, count, 2000, uint32_t(n));
            sums[5] += v * count;
        }
    }
    for (int j = 0; j < 6; ++j) partial[j][t] = sums[j];
    __syncthreads();
    for (int stride = threads / 2; stride; stride /= 2) {
        if (t < stride)
            for (int j = 0; j < 6; ++j) partial[j][t] += partial[j][t + stride];
        __syncthreads();
    }
    if (t < 6) stats[s * 6 + t] = partial[t][0];
}
__global__ void select_ids(const float* values, int n, int k,
                           const uint32_t* ranks, uint32_t* ids) {
    const int s = blockIdx.y;
    const uint32_t* row = ranks + size_t(s) * bins;
    uint32_t* output = ids + size_t(s) * k;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < uint32_t(n);
         i += blockDim.x * gridDim.x) {
        const auto bin = key(values[size_t(s) * n + i]);
        const uint32_t start = min(row[bin], uint32_t(k));
        const uint32_t end = bin + 1 < bins ? min(row[bin + 1], uint32_t(k)) : uint32_t(k);
        uint32_t candidate = i;
        // Each equal-value bucket owns a disjoint range. Atomic insertion
        // retains its smallest token IDs, including a tie crossing rank k.
        for (uint32_t r = start; r < end && candidate != UINT32_MAX; ++r) {
            const uint32_t old = atomicMin(output + r, candidate);
            candidate = max(old, candidate);
        }
    }
}
__global__ void gather(const float* values, int n, int k, const uint32_t* ids,
                      float* selected, float* heads, float* probes) {
    const int s = blockIdx.x;
    for (int r = threadIdx.x; r < k; r += blockDim.x) {
        const auto id = ids[size_t(s) * k + r];
        selected[size_t(s) * k + r] = id < uint32_t(n) ? snap(values[size_t(s) * n + id]) : NAN;
    }
    if (threadIdx.x == 0) {
        const auto id = ids[size_t(s) * k];
        heads[s] = id < uint32_t(n) ? snap(values[size_t(s) * n + id]) : NAN;
    }
    if (threadIdx.x < 20)
        probes[s * 20 + threadIdx.x] = snap(values[size_t(s) * n + (n / 20) * threadIdx.x]);
}
struct Scratch {
    cudaStream_t stream = nullptr;
    uint32_t* hist = nullptr;
    uint32_t* ids = nullptr;
    float *values = nullptr, *heads = nullptr, *probes = nullptr;
    double* stats = nullptr;
    int rows = 0, k = 0, device = -1;
    void clear() {
        cudaFree(hist); cudaFree(ids); cudaFree(values); cudaFree(heads); cudaFree(probes); cudaFree(stats);
        hist = ids = nullptr; values = heads = probes = nullptr; stats = nullptr; rows = k = 0;
    }
    ~Scratch() {
        int current = -1; cudaGetDevice(&current);
        if (device >= 0 && current != device) cudaSetDevice(device);
        clear(); if (stream) cudaStreamDestroy(stream);
        if (current >= 0 && current != device) cudaSetDevice(current);
    }
    bool reserve(int s, int requested_k) {
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess) return false;
        if (device != current) {
            if (device >= 0) {
                if (cudaSetDevice(device) != cudaSuccess) return false;
                clear(); if (stream) cudaStreamDestroy(stream);
                if (cudaSetDevice(current) != cudaSuccess) return false;
            }
            stream = nullptr; device = current;
        }
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (rows >= s && k == requested_k) return true;
        clear();
        if (cudaMalloc(&hist, size_t(s) * bins * sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&ids, size_t(s) * requested_k * sizeof(uint32_t)) != cudaSuccess ||
            cudaMalloc(&values, size_t(s) * requested_k * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&heads, s * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&probes, s * 20 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&stats, s * 6 * sizeof(double)) != cudaSuccess) { clear(); return false; }
        rows = s; k = requested_k; return true;
    }
};
thread_local Scratch scratch;
}

extern "C" bool pow_gpu_histogram_device_k(const float* logits, int s, int n, int k, float inv_temp,
        uint32_t* ids, float* values, float* heads, double* stats, float* probes) {
    if (!logits || !ids || !values || !heads || !stats || s < 1 || n < 20 || k < 1 || k > n || k > 64) return false;
    auto& b = scratch;
    if (!b.reserve(s, k)) return false;
    if (cudaMemsetAsync(b.hist, 0, size_t(s) * bins * sizeof(uint32_t), b.stream) != cudaSuccess ||
        cudaMemsetAsync(b.ids, 0xff, size_t(s) * k * sizeof(uint32_t), b.stream) != cudaSuccess) return false;
    count_values<<<dim3(32, s), threads, 0, b.stream>>>(logits, n, b.hist);
    summarize<<<s, threads, 0, b.stream>>>(b.hist, n, k, inv_temp, b.stats);
    select_ids<<<dim3(32, s), threads, 0, b.stream>>>(logits, n, k, b.hist, b.ids);
    gather<<<s, threads, 0, b.stream>>>(logits, n, k, b.ids, b.values, b.heads, b.probes);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (cudaMemcpyAsync(ids, b.ids, size_t(s) * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, b.stream) != cudaSuccess ||
        cudaMemcpyAsync(values, b.values, size_t(s) * k * sizeof(float), cudaMemcpyDeviceToHost, b.stream) != cudaSuccess ||
        cudaMemcpyAsync(heads, b.heads, s * sizeof(float), cudaMemcpyDeviceToHost, b.stream) != cudaSuccess ||
        cudaMemcpyAsync(stats, b.stats, s * 6 * sizeof(double), cudaMemcpyDeviceToHost, b.stream) != cudaSuccess) return false;
    if (probes && cudaMemcpyAsync(probes, b.probes, s * 20 * sizeof(float), cudaMemcpyDeviceToHost, b.stream) != cudaSuccess) return false;
    if (cudaStreamSynchronize(b.stream) != cudaSuccess) return false;
    for (int i = 0; i < s * 6; ++i)
        if (!std::isfinite(stats[i])) return false;
    return true;
}

// Prompt rows are host-resident. Keep this staging allocation separate from
// the histogram workspace so resizing either cannot invalidate the other.
extern "C" bool pow_gpu_histogram_host_k(const float* const* logits, int s, int n, int k,
        float inv_temp, uint32_t* ids, float* values, float* heads, double* stats) {
    if (!logits || s < 1 || n < 20 || k < 1 || k > n || k > 64) return false;
    struct Input {
        float* data = nullptr;
        size_t capacity = 0;
        int device = -1;
        void clear() {
            int current = -1; cudaGetDevice(&current);
            if (device >= 0 && current != device) cudaSetDevice(device);
            cudaFree(data); data = nullptr; capacity = 0;
            if (current >= 0 && current != device) cudaSetDevice(current);
        }
        ~Input() { clear(); }
    };
    static thread_local Input input;
    const size_t bytes = size_t(s) * n * sizeof(float);
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess) return false;
    if (current != input.device) { input.clear(); input.device = current; }
    if (bytes > input.capacity) {
        input.clear();
        if (cudaMalloc(&input.data, bytes) != cudaSuccess) return false;
        input.capacity = bytes;
    }
    auto& b = scratch;
    if (!b.reserve(s, k)) return false;
    bool contiguous = true;
    for (int row = 0; row < s; ++row) {
        if (!logits[row]) return false;
        if (logits[row] != logits[0] + size_t(row) * n) contiguous = false;
    }
    if (contiguous) {
        if (cudaMemcpyAsync(input.data, logits[0], bytes, cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
    } else {
        for (int row = 0; row < s; ++row)
            if (cudaMemcpyAsync(input.data + size_t(row) * n, logits[row], size_t(n) * sizeof(float),
                                cudaMemcpyHostToDevice, b.stream) != cudaSuccess) return false;
    }
    return pow_gpu_histogram_device_k(input.data, s, n, k, inv_temp, ids, values, heads, stats, nullptr);
}
