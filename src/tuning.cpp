#include "tuning.h"

namespace meow {

TuningProfile tuning_for(int sm, size_t vram_bytes, size_t model_bytes) {
    const double vram_gb = double(vram_bytes) / (1024.0 * 1024.0 * 1024.0);

    // ---- measured profiles -------------------------------------------
    // H100 80GB (sm_90). llama.cpp's n_seq_max cap of 256 is the ceiling,
    // not VRAM — 512 slots fails to create a context however much memory is
    // free. Double-buffering measured neutral (37.4 vs 36.4 w/s) at 2x the
    // VRAM, so it is not worth the memory.
    if (sm == 90 && vram_gb > 60.0)
        return { "H100 80GB", 256, 12, false,
                 "37.6-38.0 w/s at 256 slots (128->28.9, 192->33.8); groups flat (12:37.8 24:37.6 32:37.5 48:37.3); 2x256 double 37.4 at 56GB; "
                 "512 slots impossible (llama n_seq_max<=256)" };

    // RTX 5090 32GB (sm_120). Bandwidth-bound at 8B: one big batch beats two
    // alternating ones because a second context pays the 15.3 GB weight read
    // a second time to hide a ~2.4 ms CPU tail.
    if (sm == 120 && vram_gb > 24.0)
        return { "RTX 5090 32GB", 128, 12, false,
                 "19.5 w/s single 128 slots; 64x2 double 12.8; 96x2 17.0; 192 single 17.4 (KV traffic)" };

    // ---- conservative fallbacks, derived from VRAM -------------------
    // Budget: model weights + KV (~57 MB/slot for an 8B at ctx 384) + ~4 GB
    // of compute buffers must fit, with headroom. Keep well clear of the
    // ceiling: an OOM mid-window costs more than a few percent of throughput.
    const double model_gb = model_bytes ? double(model_bytes) / (1024.0*1024.0*1024.0) : 15.3;
    const double free_gb  = vram_gb - model_gb - 4.0;
    int slots = int(free_gb * 1024.0 / 57.0 * 0.80);   // 80% of what fits
    if (slots > 256) slots = 256;                       // llama n_seq_max cap
    if (slots < 8)   slots = 8;
    const int groups = slots >= 128 ? 12 : (slots >= 32 ? 8 : 4);
    return { "auto (from VRAM)", slots, groups, false,
             "no measured profile for this card — sized from VRAM, verify with [prof-e2e]" };
}

}  // namespace meow
