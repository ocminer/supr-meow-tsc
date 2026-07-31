// =============================================================================
// tuning.h — per-GPU optimal settings, chosen automatically.
//
// The best configuration is NOT a single number: it depends on how the card's
// memory bandwidth, compute and VRAM interact with the model being mined, and
// the winning setting on one card can be the losing one on another. Two
// measured examples, both counter-intuitive:
//
//   * RTX 5090 (32 GB): double-buffering LOSES. 8B decode is bandwidth-bound,
//     so throughput is sequences-per-weight-read; a second context pays that
//     read twice, and VRAM forces 64 slots/context instead of 128. 12.8 w/s
//     double vs 19.5 single.
//   * H100 80GB: double-buffering is NEUTRAL (37.4 vs 36.4) because VRAM does
//     NOT force smaller contexts — but llama.cpp caps n_seq_max at 256, so a
//     single context cannot grow past 256 slots however much VRAM is free.
//
// So the table is empirical, per (architecture, VRAM) class, and every entry
// carries the measured number that justifies it. Anything unknown falls back
// to a conservative profile derived from VRAM alone.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace meow {

struct TuningProfile {
    const char* name;        // human label for the log line
    int   slots;             // concurrent windows per GPU
    int   groups;            // sampler threads per GPU
    bool  double_buffer;     // two alternating contexts
    const char* evidence;    // what was measured, so nobody "optimises" it back
};

// `sm` is compute capability as major*10+minor (89 = Ada, 90 = Hopper,
// 120 = Blackwell). `vram_bytes` is the card's total memory. `model_bytes` is
// the on-disk model size, or 0 if not yet known.
TuningProfile tuning_for(int sm, size_t vram_bytes, size_t model_bytes);

}  // namespace meow
