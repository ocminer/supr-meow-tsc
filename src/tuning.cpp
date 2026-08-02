#include "tuning.h"

namespace meow {

TuningProfile tuning_for(int sm, size_t vram_bytes, size_t model_bytes) {
    const double vram_gb = double(vram_bytes) / (1024.0 * 1024.0 * 1024.0);

    // ---- measured profiles -------------------------------------------
    // H100 80GB (sm_90). llama.cpp's n_seq_max cap of 256 is the ceiling,
    // not VRAM — 512 slots fails to create a context however much memory is
    // free. Double-buffering measured neutral (37.4 vs 36.4 w/s) at 2x the
    // VRAM, so it is not worth the memory.
    // H200 141GB (sm_90). Measured 2026-08-01: SAME 40.5 w/s as the H100 at
    // the same config, despite 43% more memory bandwidth (4.8 vs 3.35 TB/s) —
    // so at 512 slots this workload is NOT bandwidth-bound any more, and the
    // extra VRAM buys nothing. Everything that spends the extra memory lost:
    // 768 slots 24.0, 1024 slots 29.6, and 512x2 double-buffered 36.6 vs 39.9
    // on a concurrent control. Raising LLAMA_MAX_SEQ to reach those slot
    // counts costs 11% on its own (512 slots measured 35.9 on a MAX_SEQ=1024
    // build vs 40.4 on the standard one) because the constant sizes per-batch
    // bitsets whatever you actually use. Do not "optimise" this back.
    if (sm == 90 && vram_gb > 100.0)
        return { "H200 141GB", 512, 12, false,
                 "40.5 w/s at 512 slots, 0 rejects — identical to the H100 despite 43% more "
                 "bandwidth, so 512 is a workload ceiling here, not a memory one. Measured "
                 "against a concurrent control: 768->24.0, 1024->29.6, 512x2 double->36.6 vs "
                 "39.9, ubatch 4096->40.9 (noise). MAX_SEQ=1024 build costs 11% even at 512." };

    if (sm == 90 && vram_gb > 60.0)
        return { "H100 80GB", 512, 12, false,
                 "40.5 w/s at 512 slots, 0 rejects (128->28.9 192->33.8 256->37.6 384->38.1 512->40.5); "
                 "REQUIRES tools/llama-max-seq-512.patch — stock llama caps n_seq_max at 256; "
                 "groups flat (12:37.8 24:37.6 32:37.5 48:37.3); double-buffering neutral" };

    // RTX PRO 6000 Blackwell 96GB (sm_120). Same silicon and roughly the same
    // bandwidth as a 5090, but 3x the VRAM — and that is the whole story: the
    // 5090 is capped at 128 slots by memory, not by bandwidth. Measured
    // 2026-08-01 across 8 cards in parallel: 128->18.7 (the 5090's config),
    // 192->21.6, 256->23.9/23.8, 320->24.3, 384->25.5, 512->25.6. Unlocking
    // slots is worth +37%. MUST come before the 5090 branch below, which keys
    // only on sm_120 and would hand this card 32 GB settings.
    if (sm == 120 && vram_gb > 60.0)
        return { "RTX PRO 6000 96GB", 512, 12, false,
                 "25.6 w/s at 512 slots (llama's seq cap, not VRAM — 56 GB of 96 used). "
                 "128->18.7 192->21.6 256->23.9/23.8 320->24.3 384->25.5 448->23.5 512->25.6; "
                 "duplicate 256 on a second card agreed to 0.4%, so the parallel sweep is sound." };

    // RTX 5090 32GB (sm_120). Bandwidth-bound at 8B: one big batch beats two
    // alternating ones because a second context pays the 15.3 GB weight read
    // a second time to hide a ~2.4 ms CPU tail. 128 slots is a VRAM limit, not
    // an optimum — the 96 GB PRO 6000 above reaches 512 and gains 37%.
    if (sm == 120 && vram_gb > 24.0)
        return { "RTX 5090 32GB", 128, 12, false,
                 "19.5 w/s single 128 slots; 64x2 double 12.8; 96x2 17.0; 192 single 17.4 (KV traffic)" };

    // A100 80GB SXM (sm_80). Measured 2026-08-01: 20.7 w/s at 512 slots, and
    // the curve is nearly FLAT (256->19.9, 384->19.6, 512->20.8/20.7 on two
    // runs) where the H100 climbed 28.9->40.5 over the same range. So the A100
    // is already saturated at low batch — it is bandwidth-bound in the regime
    // the H100 only reaches much later, which is consistent with 2.04 vs
    // 3.35 TB/s. It also sat AT its 400 W limit throughout (413 W draw), unlike
    // the H200 which idled at 476 W of 700 W. Nothing here is worth retuning:
    // a 25% slot change moves throughput by ~1 w/s.
    if (sm == 80 && vram_gb > 60.0)
        return { "A100 80GB", 512, 12, false,
                 "20.7 w/s at 512 slots, 0 rejects; curve nearly flat (256->19.9, 384->19.6, "
                 "512->20.8/20.7) so the card saturates early and slot count barely matters. "
                 "Power-limited: 413 W against a 400 W cap." };

    // RTX 6000 Ada 48GB (sm_89). Same VRAM class as the A6000 but the curve is
    // FLAT rather than rising — 320 -> 12.4, 403 -> 12.5/12.6 — so unlike the
    // A6000 it is not gaining from slots at the top; it saturates by ~320.
    // 403 is what the VRAM heuristic picks and it is fine, but it leaves only
    // ~1.3 GB spare: MEOW_UBATCH=4096 OOMs at this slot count. Do not enable
    // the CPU-tail tuning here — util is 92-98%, so there is nothing to
    // reclaim (see docs/BENCHMARKS.md), and the larger ubatch does not fit.
    if (sm == 89 && vram_gb > 40.0)
        return { "RTX 6000 Ada 48GB", 403, 12, false,
                 "12.5 w/s at 403 slots (47.8 GB of 49.1 — only ~1.3 GB spare, and "
                 "MEOW_UBATCH=4096 OOMs here). Curve is flat: 320->12.4, 403->12.5/12.6, so "
                 "320 is equally good with 6 GB more headroom. Util 92-98%, sampler tail 20 ms." };

    // RTX A6000 48GB (sm_86, also A40). The cheapest card tested and the best
    // value of any of them. Unlike the big cards this one is VRAM-bound, and
    // in that regime slots help MONOTONICALLY — no KV-traffic regression at
    // all: 128->7.8, 192->8.3, 256->8.8/8.8, 320->9.0, 361->9.5. 361 is the
    // ceiling: 400 slots OOMs, and trading context for slots does not rescue
    // it (420 and 460 at ctx 336 both OOM) because compute buffers grow with
    // batch too. Groups are flat even on 10 CPUs (5->8.7, 8->8.8, 12->8.8).
    // Power-capped at 287 W of 300 W.
    if (sm == 86 && vram_gb > 40.0)
        return { "RTX A6000 48GB", 361, 12, false,
                 "9.5 w/s at 361 slots — the VRAM ceiling (400 OOMs; 420/460 at ctx 336 also OOM). "
                 "Slots help monotonically here: 128->7.8 192->8.3 256->8.8 320->9.0 361->9.5. "
                 "Groups flat on 10 CPUs (5:8.7 8:8.8 12:8.8). 287 W of a 300 W cap." };

    // ---- conservative fallbacks, derived from VRAM -------------------
    // Budget: model weights + KV (~57 MB/slot for an 8B at ctx 384) + ~4 GB
    // of compute buffers must fit, with headroom. Keep well clear of the
    // ceiling: an OOM mid-window costs more than a few percent of throughput.
    const double model_gb = model_bytes ? double(model_bytes) / (1024.0*1024.0*1024.0) : 15.3;
    const double free_gb  = vram_gb - model_gb - 4.0;
    int slots = int(free_gb * 1024.0 / 57.0 * 0.80);   // 80% of what fits
    if (slots > 512) slots = 512;                       // llama n_seq_max cap (patched to 512)
    if (slots < 8)   slots = 8;
    const int groups = slots >= 128 ? 12 : (slots >= 32 ? 8 : 4);
    return { "auto (from VRAM)", slots, groups, false,
             "no measured profile for this card — sized from VRAM, verify with [prof-e2e]" };
}

}  // namespace meow
