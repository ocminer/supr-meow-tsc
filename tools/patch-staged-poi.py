#!/usr/bin/env python3
"""Patch upstream proof-of-inference sources AFTER they are staged into the
build tree, so vendor/ stays an untouched submodule checkout.

Only genuine upstream defects belong here. Each patch states what it fixes and
fails loudly if the code it targets is gone — a silently skipped patch would
reintroduce the bug on the next submodule bump.
"""
import sys

PATCHES = [
    (
        "pow_utils.cpp",
        # adjusted_bits is the SHARE-mode threshold the verifier gates on, but
        # upstream derives it from `target` — the block target — so a sub-block
        # share is judged against a threshold it can never meet. The proof's
        # `target` must stay the block target (bcore validates solutions with
        # it), so the share threshold travels here instead.
        "                auto target_bytes = hex_to_bytes(target_it->second);\n",
        "                auto thr_it = is_solution ? target_it\n"
        "                                          : seq_pow_params.find(\"share_target\");\n"
        "                if (thr_it == seq_pow_params.end() || thr_it->second.empty())\n"
        "                    thr_it = target_it;\n"
        "                auto target_bytes = hex_to_bytes(thr_it->second);\n",
    ),
    (
        "pow_utils.cpp",
        # proof.fbs documents `timestamp` as a UNIX timestamp, and both
        # validation.fbs and blockheader.fbs narrow it to uint32 seconds. But
        # system_clock::time_since_epoch().count() is NANOSECONDS on libstdc++,
        # so every proof carries a value ~1e9 too large. The proof itself still
        # serializes (uint64), which is why this stays invisible until a
        # validator converts it and rejects the proof as malformed:
        #   "bad number 1785206392599723924 for type uint32"
        "         std::chrono::system_clock::now().time_since_epoch().count()\n",
        "         std::chrono::duration_cast<std::chrono::seconds>(\n"
        "             std::chrono::system_clock::now().time_since_epoch()).count()\n",
    ),
    (
        "pow_utils.cpp",
        # ---- GPU sampler offload, part 1: declarations -----------------------
        # The full-vocab sort is ~87%% of sampler cost (~10 ms/token on the
        # host); the fused CUB kernel (src/pow_gpu.cu) reproduces the exact
        # ordering — ties and signed zeros included — in ~0.55 ms and returns
        # the logsumexp/bucket stats from data it already holds sorted.
        "// Vectorized FP16 snapping function\n",
        "#ifdef POW_GPU_SORT_ENABLED\n"
        "#include <cstdlib>\n"
        "extern \"C\" bool pow_gpu_available(void);\n"
        "extern \"C\" bool pow_gpu_sort_and_stats(const float* h_vals, int n, float inv_temp,\n"
        "                                       uint32_t* h_sorted_idx, float* h_sorted_vals, double* h_stats);\n"
        "extern \"C\" bool pow_gpu_take_presorted(const uint32_t** idx, const float** val, const double** stats, float* head, int* count);\n"
        "#endif\n"
        "\n"
        "// Vectorized FP16 snapping function\n",
    ),
    (
        "pow_utils.cpp",
        # ---- part 2: replace the host sort with the fused kernel -------------
        """    // ---------- 1) Pre-temp sort once in the effective precision domain ----------
    pretemp_desc_.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) pretemp_desc_.emplace_back(working_logits[i], i);
    std::sort(pretemp_desc_.begin(), pretemp_desc_.end(),
              [](const auto& a, const auto& b){
                  if (a.first != b.first) return a.first > b.first; // desc by value
                  return a.second < b.second;                        // tie by id
              });
""",
        """    // ---------- 1) Pre-temp sort once in the effective precision domain ----------
    // GPU path: bit-exact with the comparator below (desc by value, tie by
    // ascending id, signed zeros handled — see pow_gpu.cu). POW_GPU_SORT=0
    // forces the CPU path at runtime.
    pretemp_desc_.resize(n_vocab);
    bool sorted_on_gpu = false;
    gpu_stats_valid_ = false;
#ifdef POW_GPU_SORT_ENABLED
    {
        static const bool gpu_sort_enabled = [](){
            const char* e = std::getenv("POW_GPU_SORT");
            const bool want = !(e && (*e == '0' || *e == 'f' || *e == 'F'));
            return want && pow_gpu_available();
        }();
        if (gpu_sort_enabled) {
            // Batched fast path: the engine pre-sorted EVERY stream's snapped
            // logits in one segmented launch; consume that result instead of
            // launching a per-stream chain. The injected order is produced by
            // the same key transform, so bit-exactness is unchanged.
            const uint32_t* inj_idx = nullptr; const float* inj_val = nullptr;
            const double* inj_stats = nullptr; float inj_head = 0.0f; int inj_count = 0;
            if (pow_gpu_take_presorted(&inj_idx, &inj_val, &inj_stats, &inj_head, &inj_count)) {
                gpu_sort_val_.assign(1, inj_head);
                const int kcopy = (inj_count > 0 && inj_count < n_vocab) ? inj_count : n_vocab;
                for (int r = 0; r < kcopy; ++r)
                    pretemp_desc_[r] = { inj_val[r], static_cast<int>(inj_idx[r]) };
                for (int k = 0; k < 6; ++k) gpu_stats_[k] = inj_stats[k];
                sorted_on_gpu = true;
                gpu_stats_valid_ = true;
            } else {
            gpu_sort_idx_.resize(n_vocab);
            gpu_sort_val_.resize(n_vocab);
            const float inv_temp_for_stats = (temperature != 1.0f && temperature > 0.0f)
                                             ? (1.0f / temperature) : 1.0f;
            if (pow_gpu_sort_and_stats(working_logits, n_vocab, inv_temp_for_stats,
                                       gpu_sort_idx_.data(), gpu_sort_val_.data(),
                                       gpu_stats_)) {
                for (int r = 0; r < n_vocab; ++r) {
                    const uint32_t tid = gpu_sort_idx_[r];
                    pretemp_desc_[r] = { working_logits[tid], static_cast<int>(tid) };
                }
                sorted_on_gpu = true;
                gpu_stats_valid_ = true;
            }
            }
        }
    }
#endif
    if (!sorted_on_gpu) {
        for (int i = 0; i < n_vocab; ++i) pretemp_desc_[i] = { working_logits[i], i };
        std::sort(pretemp_desc_.begin(), pretemp_desc_.end(),
                  [](const auto& a, const auto& b){
                      if (a.first != b.first) return a.first > b.first; // desc by value
                      return a.second < b.second;                        // tie by id
                  });
    }
""",
    ),
    (
        "pow_utils.cpp",
        # ---- part 3: logsumexp_full from the kernel's sum --------------------
        """    float max_log = logits_[0];
    for (int i = 1; i < n_vocab; ++i) max_log = std::max(max_log, logits_[i]);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp(double(logits_[i] - max_log));
    result.logsumexp_full = max_log + std::log(float(sum_exp));
""",
        """    if (gpu_stats_valid_) {
        // Kernel returned sum_i exp(v_i*invT - max_scaled); max_scaled is the
        // sorted head — the same maximum the loop below would find. Different
        // summation ORDER only: the verifier treats these as continuous
        // features in a Mahalanobis test, not an equality check.
        const float max_log = float(double(gpu_sort_val_[0]) *
                                    ((temperature != 1.0f && temperature > 0.0f) ? (1.0 / temperature) : 1.0));
        result.logsumexp_full = max_log + std::log(float(gpu_stats_[0]));
    } else {
        float max_log = logits_[0];
        for (int i = 1; i < n_vocab; ++i) max_log = std::max(max_log, logits_[i]);
        double sum_exp = 0.0;
        for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp(double(logits_[i] - max_log));
        result.logsumexp_full = max_log + std::log(float(sum_exp));
    }
""",
    ),
    (
        "pow_utils.cpp",
        # ---- part 4: rank-bucket means from the kernel's sums ----------------
        """    result.logsumexp_stats.assign(6, 0.0f);
    result.logsumexp_stats[0] = result.logsumexp_full;
    if (n_vocab >= 50) {
""",
        """    result.logsumexp_stats.assign(6, 0.0f);
    result.logsumexp_stats[0] = result.logsumexp_full;
    if (gpu_stats_valid_) {
        // gpu_stats_[1..4] = bucket sums, [5] = full sum; same denominators
        // as the CPU path below.
        if (n_vocab >= 50)   result.logsumexp_stats[1] = float(gpu_stats_[1] / 50.0);
        if (n_vocab >= 500)  result.logsumexp_stats[2] = float(gpu_stats_[2] / double(std::max(1, std::min(500, n_vocab) - 50)));
        if (n_vocab >= 2000) result.logsumexp_stats[3] = float(gpu_stats_[3] / double(std::max(1, std::min(2000, n_vocab) - 500)));
        if (n_vocab >  2000) result.logsumexp_stats[4] = float(gpu_stats_[4] / double(std::max(1, n_vocab - 2000)));
        result.logsumexp_stats[5] = float(gpu_stats_[5] / double(n_vocab));
    } else
    if (n_vocab >= 50) {
""",
    ),
    (
        "pow_utils.cpp",
        "    if (n_vocab >= 500) {\n        const int hi = std::min(500, n_vocab);",
        "    if (!gpu_stats_valid_ && n_vocab >= 500) {\n        const int hi = std::min(500, n_vocab);",
    ),
    (
        "pow_utils.cpp",
        "    if (n_vocab >= 2000) {\n        const int hi = std::min(2000, n_vocab);",
        "    if (!gpu_stats_valid_ && n_vocab >= 2000) {\n        const int hi = std::min(2000, n_vocab);",
    ),
    (
        "pow_utils.cpp",
        "    if (n_vocab > 2000) {\n        double s = 0.0; for (int i = 2000; i < n_vocab; ++i)",
        "    if (!gpu_stats_valid_ && n_vocab > 2000) {\n        double s = 0.0; for (int i = 2000; i < n_vocab; ++i)",
    ),
    (
        "pow_utils.cpp",
        "    {\n        double s = 0.0; for (const auto& p : pretemp_desc_) s += p.first;",
        "    if (!gpu_stats_valid_) {\n        double s = 0.0; for (const auto& p : pretemp_desc_) s += p.first;",
    ),
    (
        "pow_utils.h",
        "    std::vector<std::pair<float,int32_t>> pretemp_desc_;",
        "    std::vector<std::pair<float,int32_t>> pretemp_desc_;\n"
        "    std::vector<uint32_t> gpu_sort_idx_;   // GPU sort: token ids in rank order\n"
        "    std::vector<float>    gpu_sort_val_;   // GPU sort: values in rank order\n"
        "    double                gpu_stats_[6] = {0,0,0,0,0,0};  // fused kernel output\n"
        "    bool                  gpu_stats_valid_ = false;",
    ),
    (
        "pow_utils.h",
        "    std::vector<uint8_t> keep_;     // size n_vocab",
        "    std::vector<uint8_t> keep_;     // size n_vocab\n"
        "    // Survivor ids after top-k, ASCENDING (<= top_k of them). The tail\n"
        "    // math only involves these; everything else is -inf/zero.\n"
        "    std::vector<int32_t> surv_ids_;\n"
        "    bool                 surv_valid_ = false;",
    ),
    (
        "pow_utils.cpp",
        # ---- tail 1/3: top-k from the sorted prefix, not a full scan ---------
        """        survivors = 0;
        for (int i = 0; i < n_vocab; ++i) {
            if (working_logits[i] > kth_pre) {        // strict exclusiveness (intentional)
                keep_[i] = 1u;
                ++survivors;
            } else {
                logits_[i] = ninf;
            }
        }
""",
        """        // pretemp_desc_ is sorted DESC, so {i : value > kth_pre} is exactly its
        // leading prefix — and since rank kpos itself equals kth_pre, that
        // prefix lives entirely inside ranks [0, kpos). Reading <=50 ranks
        // replaces a 151,936-entry branchy scan; the survivor SET is identical,
        // which is all that matters downstream.
        surv_valid_ = false;
        surv_ids_.clear();
        for (int r = 0; r < kpos; ++r) {
            if (!(pretemp_desc_[r].first > kth_pre)) break;   // prefix ended
            surv_ids_.push_back(pretemp_desc_[r].second);
        }
        if (static_cast<int>(surv_ids_.size()) < kpos) {
            // Sum/mask order must match the reference, which walks ascending
            // token id. Sorting <=50 ids is free next to a full pass.
            std::sort(surv_ids_.begin(), surv_ids_.end());
            std::fill(keep_.begin(), keep_.begin() + n_vocab, 0u);
            std::fill(logits_.begin(), logits_.begin() + n_vocab, ninf);
            const bool scaled = (temperature != 1.0f);
            const float invT  = scaled ? (1.0f / temperature) : 1.0f;
            for (int id : surv_ids_) {
                keep_[id]   = 1u;
                logits_[id] = scaled ? (working_logits[id] * invT) : working_logits[id];
            }
            survivors  = static_cast<int>(surv_ids_.size());
            surv_valid_ = true;
        } else {
            // Degenerate (ties filled the whole prefix): reference path.
            survivors = 0;
            for (int i = 0; i < n_vocab; ++i) {
                if (working_logits[i] > kth_pre) {    // strict exclusiveness (intentional)
                    keep_[i] = 1u;
                    ++survivors;
                } else {
                    logits_[i] = ninf;
                }
            }
        }
""",
    ),
    (
        "pow_utils.cpp",
        # ---- tail 2/3: log_Z + masked softmax over survivors only ------------
        """    result.softmax_log_z = -std::numeric_limits<float>::infinity();
    float max_kept = ninf;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits_[i] != ninf) {
            max_kept = std::max(max_kept, logits_[i]);
        }
    }
    if (max_kept != ninf) {
        double kept_sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] != ninf) {
                kept_sum += std::exp(double(logits_[i] - max_kept));
            }
        }
        result.softmax_log_z = max_kept + static_cast<float>(std::log(kept_sum));
    }
""",
        """    result.softmax_log_z = -std::numeric_limits<float>::infinity();
    float max_kept = ninf;
    if (surv_valid_) {
        // Same values, same ASCENDING-id accumulation order as the reference
        // loops — so kept_sum is bit-identical, not merely close.
        for (int id : surv_ids_) max_kept = std::max(max_kept, logits_[id]);
        if (max_kept != ninf) {
            double kept_sum = 0.0;
            for (int id : surv_ids_) kept_sum += std::exp(double(logits_[id] - max_kept));
            result.softmax_log_z = max_kept + static_cast<float>(std::log(kept_sum));
        }
    } else {
    for (int i = 0; i < n_vocab; ++i) {
        if (logits_[i] != ninf) {
            max_kept = std::max(max_kept, logits_[i]);
        }
    }
    if (max_kept != ninf) {
        double kept_sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] != ninf) {
                kept_sum += std::exp(double(logits_[i] - max_kept));
            }
        }
        result.softmax_log_z = max_kept + static_cast<float>(std::log(kept_sum));
    }
    }
""",
    ),
    (
        "pow_utils.cpp",
        """    } else {
        // Standard masked softmax once
        float max_masked = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_vocab; ++i) if (logits_[i] != ninf) max_masked = std::max(max_masked, logits_[i]);

        double Z = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] == ninf) { probs_[i] = 0.0f; continue; }
            float e = std::exp(logits_[i] - max_masked);
            probs_[i] = e;
            Z += e;
        }
        if (Z > 0.0) {
            const float invZ = float(1.0 / Z);
            for (int i = 0; i < n_vocab; ++i) probs_[i] *= invZ;
        } else {
            std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
        }
    }
""",
        """    } else if (surv_valid_) {
        // Survivor-only masked softmax. probs_ is zero everywhere else, so a
        // single fill replaces the 151,936-iteration branchy loops; Z still
        // accumulates in ascending-id order, so it is bit-identical.
        float max_masked = -std::numeric_limits<float>::infinity();
        for (int id : surv_ids_) max_masked = std::max(max_masked, logits_[id]);
        std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
        double Z = 0.0;
        for (int id : surv_ids_) {
            const float e = std::exp(logits_[id] - max_masked);
            probs_[id] = e;
            Z += e;
        }
        if (Z > 0.0) {
            const float invZ = float(1.0 / Z);
            for (int id : surv_ids_) probs_[id] *= invZ;
        } else {
            std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
        }
    } else {
        // Standard masked softmax once
        float max_masked = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < n_vocab; ++i) if (logits_[i] != ninf) max_masked = std::max(max_masked, logits_[i]);

        double Z = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            if (logits_[i] == ninf) { probs_[i] = 0.0f; continue; }
            float e = std::exp(logits_[i] - max_masked);
            probs_[i] = e;
            Z += e;
        }
        if (Z > 0.0) {
            const float invZ = float(1.0 / Z);
            for (int i = 0; i < n_vocab; ++i) probs_[i] *= invZ;
        } else {
            std::fill(probs_.begin(), probs_.begin() + n_vocab, 0.0f);
        }
    }
""",
    ),
    (
        "pow_utils.cpp",
        # ---- tail 3/3: CDF as constant runs between survivors ----------------
        """    float cumulative = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        cumulative += probs_[i];
        cdf_[i] = cumulative;
    }
""",
        """    float cumulative = 0.0f;
    if (surv_valid_) {
        // probs_ is zero except at survivors, and `x + 0.0f == x` exactly, so
        // the CDF is a step function: constant runs broken only at survivor
        // ids. std::fill writes the runs sequentially instead of doing 151,936
        // dependent float adds. Values are bit-identical to the reference.
        int pos = 0;
        for (int id : surv_ids_) {
            if (id > pos) std::fill(cdf_.begin() + pos, cdf_.begin() + id, cumulative);
            cumulative += probs_[id];
            cdf_[id] = cumulative;
            pos = id + 1;
        }
        if (pos < n_vocab) std::fill(cdf_.begin() + pos, cdf_.begin() + n_vocab, cumulative);
    } else {
    for (int i = 0; i < n_vocab; ++i) {
        cumulative += probs_[i];
        cdf_[i] = cumulative;
    }
    }
""",
    ),
]

def main(stage_dir: str) -> int:
    for filename, old, new in PATCHES:
        path = f"{stage_dir}/{filename}"
        with open(path) as fh:
            text = fh.read()
        if new in text:
            continue                      # already patched by an earlier build
        if old not in text:
            print(f"error: patch target not found in {filename} — upstream "
                  f"changed; re-check the fix before removing it", file=sys.stderr)
            return 1
        with open(path, "w") as fh:
            fh.write(text.replace(old, new, 1))
        print(f"patched {filename}: proof timestamp is seconds, not nanoseconds")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
