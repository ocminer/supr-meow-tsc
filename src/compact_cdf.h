#pragma once
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace meow {
// Exact top-k-exclusive, temperature=1, top-p=1 distribution. Every addition
// follows ascending token-id order, matching the dense reference's float CDF.
struct CompactCdf {
    std::vector<int> ids;
    std::vector<float> cdf;
    CompactCdf(const std::vector<std::pair<float,int>>& ranked, int k) {
        std::vector<std::pair<int,float>> survivors;
        for (int r = 0; r < k-1 && ranked[r].first > ranked[k-1].first; ++r)
            survivors.emplace_back(ranked[r].second,ranked[r].first);
        if (survivors.empty()) survivors.emplace_back(ranked[0].second,ranked[0].first);
        std::sort(survivors.begin(),survivors.end());
        float maximum = survivors.front().second;
        for (const auto& item : survivors) maximum = std::max(maximum,item.second);
        double z = 0;
        for (const auto& item : survivors) {
            ids.push_back(item.first);
            cdf.push_back(std::exp(item.second-maximum));
            z += cdf.back();
        }
        const float inv = z > 0 ? float(1.0/z) : 0.0f;
        float cumulative = 0;
        for (float& v : cdf) { v *= inv; cumulative += v; v = cumulative; }
    }
    std::pair<int,float> map_rank(size_t rank, float u, int vocab) const {
        // Dense lower_bound returns token zero at u=0, even if its mass is zero.
        if (u == 0) return {0, ids.front() == 0 ? cdf.front() : 0.0f};
        if (rank == ids.size()) return {vocab,0.0f};
        return {ids[rank],cdf[rank]};
    }
};
}
