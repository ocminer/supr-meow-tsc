#!/usr/bin/env python3
"""Apply opt-in miner performance changes to staged PoI sources, not vendor."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'pow_utils.cpp'
s = p.read_text(encoding="utf-8")
marker = '// MEOW compact distribution experiment'
if marker in s:
    sys.exit(0)
start = '    // ---------- 5) TOP-K (strict exclusive <=) ----------'
end = '    // ---------- 10) Telemetry: pre-temp top-50 + 20 probes ----------'
assert s.count(start) == 1 and s.count(end) == 1, 'PoI sampler changed; review compact patch'
# The dense buffer reset remains on all fallback/check paths. The compact
# branch never reads keep_, so it needs no per-token clear after initial sizing.
old = '        keep_.assign(n_vocab, 0u);\n    }\n\n    const int m_cap'
new = '''        const char* compact = std::getenv("MEOW_COMPACT_CDF");
        if (!(compact && compact[0] == '1' && pow_gpu_peek_presorted_probes()))
            keep_.assign(n_vocab, 0u);
    }

    const int m_cap'''
assert old in s, 'PoI capacity reset changed'
s = s.replace(old,new,1)
s = '#include "compact_cdf.h"\n' + s
s = s.replace(start, '''    // MEOW compact distribution experiment
    static const bool compact_enabled = [] {
        const char* e = std::getenv("MEOW_COMPACT_CDF"); return e && e[0] == '1';
    }();
    static const bool compact_check = [] {
        const char* e = std::getenv("MEOW_COMPACT_CHECK"); return e && e[0] == '1';
    }();
    const bool compact = compact_enabled && proof_version_ < pow_v4::V4_PROOF_VERSION && gpu_probes_valid_ && gpu_stats_valid_ &&
        temperature == 1.0f && top_p == 1.0f && top_k > 0 && top_k < n_vocab &&
        pretemp_desc_.size() >= size_t(top_k);
    SamplingResult compact_result;
    if (compact) {
        meow::CompactCdf distribution(pretemp_desc_,top_k);
        auto [rank,u,digest] = hasher->sample_token(context,step,distribution.cdf,admission_nonce_ptr);
        auto [token,probability] = distribution.map_rank(size_t(rank),u,n_vocab);
        result.token_id = token; result.u_value = u; result.digest = std::move(digest);
        result.token_prob = probability;
        result.softmax_log_z = distribution.log_normalizer;
        if (compact_check) compact_result = result;
    }
    if (!compact || compact_check) {
'''+start,1)
s = s.replace(end, '''    }
    if (compact && compact_check && (result.token_id != compact_result.token_id ||
        result.token_prob != compact_result.token_prob || result.u_value != compact_result.u_value ||
        result.digest != compact_result.digest ||
        result.softmax_log_z != compact_result.softmax_log_z))
        throw std::runtime_error("compact CDF differs from dense reference");

'''+end,1)
p.write_text(s, encoding="utf-8")
