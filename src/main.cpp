// =============================================================================
// supr-meow-tsc — TensorCash (TSC) miner.
//
// STATUS: foundation. Device selection, tuning and telemetry are complete and
// usable today (`--list-devices`, `--dry-run`). The mining pipeline —
// inference, proof-of-inference sampling, VDF, pool client — is being brought
// in from the work in nepl-tsc; see README.md for exactly what is and is not
// wired yet. The binary refuses to pretend it is mining when it is not.
// =============================================================================

#include "device.h"
#include "stratum.h"
#include "engine.h"
#include "vdf.h"
#include "poi.h"
#include "api.h"
#include "canary.h"
#include <algorithm>
#include "../vendor/nlohmann/json.hpp"
#include <random>
#include <atomic>
#include <thread>
extern "C" bool pow_gpu_bind_device(int cuda_ordinal);
#include <mutex>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

const char* kVersion = "0.2.0";

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ---------------------------------------------------------------- colors
// ANSI when stdout is a terminal and --no-color was not given. Rig logs and
// flight sheets get plain text automatically.
bool g_color = false;
const char* C_G() { return g_color ? "\033[32m" : ""; }   // green
const char* C_R() { return g_color ? "\033[31m" : ""; }   // red
const char* C_Y() { return g_color ? "\033[33m" : ""; }   // yellow
const char* C_C() { return g_color ? "\033[36m" : ""; }   // cyan
const char* C_B() { return g_color ? "\033[1m"  : ""; }   // bold
const char* C_0() { return g_color ? "\033[0m"  : ""; }   // reset

std::string timestamp_now() {
    char buf[16];
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;
}

// Difficulty from a 64-hex big-endian target: 2^256 / target, approximated
// from the leading 64 bits — display only, per STRATUM-TSC §4.4.
double target_to_difficulty(const std::string& hex) {
    if (hex.size() < 16) return 0.0;
    uint64_t hi = 0;
    for (int i = 0; i < 16; ++i) {
        const char c = hex[i];
        const int v = (c >= '0' && c <= '9') ? c - '0'
                    : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                    : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
        if (v < 0) return 0.0;
        hi = (hi << 4) | static_cast<uint64_t>(v);
    }
    if (hi == 0) return 0.0;                       // effectively impossible target
    return 18446744073709551616.0 / static_cast<double>(hi);   // 2^64 / hi64
}

std::string human_difficulty(double d) {
    char buf[32];
    if      (d >= 1e12) std::snprintf(buf, sizeof(buf), "%.2fT", d / 1e12);
    else if (d >= 1e9)  std::snprintf(buf, sizeof(buf), "%.2fG", d / 1e9);
    else if (d >= 1e6)  std::snprintf(buf, sizeof(buf), "%.2fM", d / 1e6);
    else if (d >= 1e3)  std::snprintf(buf, sizeof(buf), "%.2fK", d / 1e3);
    else                std::snprintf(buf, sizeof(buf), "%.2f", d);
    return buf;
}

std::string human_uptime(double secs) {
    char buf[32];
    const int h = int(secs) / 3600, m = (int(secs) % 3600) / 60, s = int(secs) % 60;
    if (h > 0) std::snprintf(buf, sizeof(buf), "%dh %02dm", h, m);
    else       std::snprintf(buf, sizeof(buf), "%dm %02ds", m, s);
    return buf;
}

struct Options {
    std::string pool;            // -o
    std::string user;            // -u
    std::string pass = "x";      // -p
    std::string devices;         // -d   ("" = all)
    std::string model_path;      // --model
    int         log_interval = 30;
    bool        list_devices = false;
    bool        dry_run      = false;
    bool        no_color     = false;
    bool        protocol_test = false;
    bool        benchmark    = false;
    bool        vdf_test     = false;
    std::string vdf_vector;          // --vdf-vector <64-hex parent> -> print exact vdf bytes
    std::string canary_test;         // --canary-test <seed64hex>:<w> -> print step-0 top-10
    std::string canary_ref;          // --canary-ref <seed64hex>:<wmax> -> emit reference table
    std::string canary_out;          // --canary-out <path> (default canary-ref-<seed8>.jsonl)
    int         slots = 8;
    int         workers = 1;   // independent contexts per GPU (--workers)
    int         groups  = 8;   // parallel sampler threads per GPU (--groups)
    int         egress_base = 47021;  // base loopback port for proof egress (--egress-base)
    std::string api_bind = "127.0.0.1:21550";  // --api-bind ("off" disables)
    std::vector<std::string> pools;   // repeated -o = failover order
    meow::DeviceTuning tuning;   // --cclock/--mclock/--pl/--fan/--lock-core
};

void print_usage() {
    std::printf(
"supr-meow-tsc %s — TensorCash (TSC) GPU miner\n"
"\n"
"USAGE\n"
"  supr-meow-tsc -o <pool> -u <wallet>[.<worker>] [-p <pass>] [options]\n"
"\n"
"REQUIRED\n"
"  -o, --pool <url>        stratum+tcp://host:port (or stratum+ssl://).\n"
"                          Repeat -o for failover pools, tried in order.\n"
"                          e.g. stratum+tcp://tsc.suprnova.cc:3307\n"
"  -u, --user <wallet>     TSC payout address, optionally .workername\n"
"                          e.g. tc1qexample....rig01   (testnet: tct1...)\n"
"\n"
"OPTIONAL\n"
"  -p, --pass <x>          Pool password. Unused by TSC pools; accepted so\n"
"                          existing flight sheets do not need editing.\n"
"  -d, --devices <list>    Devices to mine on: -d 0 or -d 0,1. Omit for ALL.\n"
"      --model <path>      Path to the chain-registered model (GGUF).\n"
"      --log-interval <s>  Seconds between status lines (default 30).\n"
"      --list-devices      Print every detected GPU and exit.\n"
"      --dry-run           Set up devices, show telemetry, do not mine.\n"
"      --benchmark         Load the model and measure per-GPU throughput.\n"
"      --vdf-test          Self-test the VDF (prove, verify, and confirm a\n"
"                          proof does NOT verify against another challenge).\n"
"      --slots <n>         Concurrent windows per GPU (default 8).\n"
"      --protocol-test     Connect to the pool and print jobs/targets/model\n"
"                          without mining. Verifies pool reachability and\n"
"                          that both sides speak TSC Stratum.\n"
"      --api-bind <h:p>    Local JSON stats API (default 127.0.0.1:21550);\n"
"                          rig managers (HiveOS h-stats.sh) read this.\n"
"                          --api-bind off disables it.\n"
"      --no-color          Plain output for logs and flight sheets.\n"
"  -h, --help              This text.\n"
"  -V, --version           Version.\n"
"\n"
"TUNING  (applied per selected device; omit to leave the card untouched)\n"
"      --cclock <mhz>      Core clock OFFSET, may be negative (e.g. --cclock -150)\n"
"      --mclock <mhz>      Memory clock OFFSET\n"
"      --lock-core <mhz>   Lock the core clock to an absolute value\n"
"      --pl <watts>        Power limit\n"
"      --fan <percent>     Fixed fan speed; omit to leave the driver's curve\n"
"\n"
"NOTES\n"
"  Tuning needs permission the driver only grants to root or a persistence-mode\n"
"  session. Failures are reported and mining continues — a miner should never\n"
"  refuse to work because a fan could not be set.\n"
"  On exit, fans are handed back to the driver and locked clocks released.\n"
"\n"
"EXAMPLES\n"
"  supr-meow-tsc -o stratum+tcp://tsc.suprnova.cc:3307 -u tc1qexample.rig01 -p x\n"
"  supr-meow-tsc -o stratum+tcp://pool-a:3307 -o stratum+tcp://pool-b:3307 -u tc1q…\n"
"  supr-meow-tsc -o stratum+tcp://tsc.suprnova.cc:3307 -u tc1qexample --protocol-test\n"
"  supr-meow-tsc --list-devices\n"
"\n", kVersion);
}

bool need_value(int i, int argc, const char* flag) {
    if (i + 1 >= argc) { std::fprintf(stderr, "error: %s needs a value\n", flag); return false; }
    return true;
}

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto eq = [&](const char* s, const char* l) { return a == s || a == l; };

        if (eq("-h", "--help"))            { print_usage(); std::exit(0); }
        else if (eq("-V", "--version"))    { std::printf("supr-meow-tsc %s\n", kVersion); std::exit(0); }
        else if (a == "--list-devices")    o.list_devices = true;
        else if (a == "--dry-run")         o.dry_run = true;
        else if (a == "--protocol-test")   o.protocol_test = true;
        else if (a == "--benchmark")       o.benchmark = true;
        else if (a == "--vdf-test")        o.vdf_test = true;
        else if (a == "--vdf-vector")      { if (!need_value(i, argc, "--vdf-vector")) return false; o.vdf_vector = argv[++i]; }
        else if (a == "--canary-test")     { if (!need_value(i, argc, "--canary-test")) return false; o.canary_test = argv[++i]; }
        else if (a == "--canary-ref")      { if (!need_value(i, argc, "--canary-ref")) return false; o.canary_ref = argv[++i]; }
        else if (a == "--canary-out")      { if (!need_value(i, argc, "--canary-out")) return false; o.canary_out = argv[++i]; }
        else if (a == "--slots")           { if (!need_value(i, argc, "--slots")) return false; o.slots = std::atoi(argv[++i]); }
        else if (a == "--workers")         { if (!need_value(i, argc, "--workers")) return false; o.workers = std::atoi(argv[++i]); }
        else if (a == "--groups")          { if (!need_value(i, argc, "--groups")) return false; o.groups = std::atoi(argv[++i]); }
        else if (a == "--egress-base")     { if (!need_value(i, argc, "--egress-base")) return false; o.egress_base = std::atoi(argv[++i]); }
        else if (a == "--api-bind")        { if (!need_value(i, argc, "--api-bind")) return false; o.api_bind = argv[++i]; }
        else if (a == "--no-color")        o.no_color = true;
        else if (eq("-o", "--pool"))       { if (!need_value(i, argc, "-o")) return false; o.pool = argv[++i]; o.pools.push_back(o.pool); }
        else if (eq("-u", "--user"))       { if (!need_value(i, argc, "-u")) return false; o.user = argv[++i]; }
        else if (eq("-p", "--pass"))       { if (!need_value(i, argc, "-p")) return false; o.pass = argv[++i]; }
        else if (eq("-d", "--devices"))    { if (!need_value(i, argc, "-d")) return false; o.devices = argv[++i]; }
        else if (a == "--model")           { if (!need_value(i, argc, "--model")) return false; o.model_path = argv[++i]; }
        else if (a == "--log-interval")    { if (!need_value(i, argc, "--log-interval")) return false; o.log_interval = std::atoi(argv[++i]); }
        else if (a == "--cclock")          { if (!need_value(i, argc, "--cclock")) return false; o.tuning.core_offset_mhz = std::atoi(argv[++i]); }
        else if (a == "--mclock")          { if (!need_value(i, argc, "--mclock")) return false; o.tuning.mem_offset_mhz = std::atoi(argv[++i]); }
        else if (a == "--lock-core")       { if (!need_value(i, argc, "--lock-core")) return false; o.tuning.lock_core_mhz = std::atoi(argv[++i]); }
        else if (a == "--pl")              { if (!need_value(i, argc, "--pl")) return false; o.tuning.power_limit_w = std::atoi(argv[++i]); }
        else if (a == "--fan")             { if (!need_value(i, argc, "--fan")) return false; o.tuning.fan_pct = std::atoi(argv[++i]); }
        else {
            std::fprintf(stderr, "error: unknown option '%s' (try --help)\n", a.c_str());
            return false;
        }
    }
    return true;
}

std::string human_bytes(size_t b) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", double(b) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

// SRBMiner-style launch table: everything a rig operator checks first —
// what was recognized, and what state each card is in RIGHT NOW.
void print_device_table(const meow::DeviceManager& dm) {
    std::printf("%s--------------------------------------------------------------------------------%s\n", C_C(), C_0());
    std::printf("  %-3s %-24s %-18s %8s %6s %5s %6s %10s %10s\n",
                "GPU", "Name", "Architecture", "VRAM", "Temp", "Fan", "Power", "Core", "Memory");
    for (const auto& d : dm.devices()) {
        const auto t = dm.telemetry(d.index);
        char temp[16] = "-", fan[16] = "-", pw[16] = "-", core[16] = "-", mem[16] = "-";
        if (t.valid) {
            if (t.temp_c     >= 0) std::snprintf(temp, sizeof(temp), "%dC",   t.temp_c);
            if (t.fan_pct    >= 0) std::snprintf(fan,  sizeof(fan),  "%d%%",  t.fan_pct);
            if (t.power_w    >= 0) std::snprintf(pw,   sizeof(pw),   "%dW",   t.power_w);
            if (t.clock_core >= 0) std::snprintf(core, sizeof(core), "%dMHz", t.clock_core);
            if (t.clock_mem  >= 0) std::snprintf(mem,  sizeof(mem),  "%dMHz", t.clock_mem);
        }
        std::printf("  %-3d %s%-24s%s %-18s %8s %6s %5s %6s %10s %10s\n",
                    d.index, C_B(), d.name.c_str(), C_0(),
                    meow::DeviceManager::sm_arch_name(d.sm_major, d.sm_minor).c_str(),
                    human_bytes(d.vram_total).c_str(), temp, fan, pw, core, mem);
    }
    std::printf("%s--------------------------------------------------------------------------------%s\n\n", C_C(), C_0());
}

void print_telemetry(const meow::DeviceManager& dm) {
    std::printf("  %-3s %-6s %-6s %-6s %-9s %-9s %-6s %s\n",
                "GPU", "temp", "fan", "power", "core", "mem", "util", "vram");
    for (const auto& d : dm.devices()) {
        const auto t = dm.telemetry(d.index);
        if (!t.valid) {
            std::printf("  %-3d %s\n", d.index, "telemetry unavailable (no NVML)");
            continue;
        }
        char temp[16], fan[16], pw[16], core[16], mem[16], util[16], vram[24];
        std::snprintf(temp, sizeof(temp), t.temp_c >= 0 ? "%dC" : "-", t.temp_c);
        std::snprintf(fan,  sizeof(fan),  t.fan_pct >= 0 ? "%d%%" : "-", t.fan_pct);
        std::snprintf(pw,   sizeof(pw),   t.power_w >= 0 ? "%dW" : "-", t.power_w);
        std::snprintf(core, sizeof(core), t.clock_core >= 0 ? "%dMHz" : "-", t.clock_core);
        std::snprintf(mem,  sizeof(mem),  t.clock_mem >= 0 ? "%dMHz" : "-", t.clock_mem);
        std::snprintf(util, sizeof(util), t.util_gpu >= 0 ? "%d%%" : "-", t.util_gpu);
        std::snprintf(vram, sizeof(vram), "%s", human_bytes(t.vram_used).c_str());
        std::printf("  %-3d %-6s %-6s %-6s %-9s %-9s %-6s %s\n",
                    d.index, temp, fan, pw, core, mem, util, vram);
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    Options o;
    if (!parse_args(argc, argv, o)) return 2;
    g_color = !o.no_color && ::isatty(1);

    std::printf("%s%s+------------------------------------------------------------+%s\n", C_C(), C_B(), C_0());
    std::printf("%s%s|  supr-meow-tsc %-8s        TensorCash (TSC) GPU miner  |%s\n", C_C(), C_B(), kVersion, C_0());
    std::printf("%s%s|  algorithm: proof-of-inference (LLM transcript + VDF)      |%s\n", C_C(), C_B(), C_0());
    std::printf("%s%s+------------------------------------------------------------+%s\n", C_C(), C_B(), C_0());

    meow::DeviceManager dm;
    std::string err;
    if (!dm.init(o.devices, err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    print_device_table(dm);

    if (o.list_devices) return 0;

    // Tuning before anything else, so the cards are in their intended state
    // before load arrives.
    const bool want_tuning =
        o.tuning.core_offset_mhz != meow::DeviceTuning::UNSET ||
        o.tuning.mem_offset_mhz  != meow::DeviceTuning::UNSET ||
        o.tuning.power_limit_w   != meow::DeviceTuning::UNSET ||
        o.tuning.fan_pct         != meow::DeviceTuning::UNSET ||
        o.tuning.lock_core_mhz   != meow::DeviceTuning::UNSET;
    if (want_tuning) {
        for (const auto& d : dm.devices()) {
            std::string terr;
            if (!dm.apply_tuning(d.index, o.tuning, terr)) {
                // Not fatal: a rig that cannot set a fan must still mine.
                std::fprintf(stderr, "warning: GPU %d tuning: %s\n", d.index, terr.c_str());
            } else {
                std::printf("  GPU %d tuned\n", d.index);
            }
        }
        std::printf("\n");
    }

    if (!o.vdf_vector.empty()) {
        // Cross-implementation pinning vector (canary-jobs stage 2): given a
        // parent hash, print the EXACT serialized VDF bytes this miner folds
        // into every u-value at the normative tick (1000). Another chiavdf
        // build must reproduce this byte-for-byte or transcript precompute
        // is impossible — Wesolowski y is deterministic, but proof
        // serialization is an implementation detail until pinned.
        if (o.vdf_vector.size() != 64) {
            std::fprintf(stderr, "error: --vdf-vector needs a 64-hex parent hash\n");
            return 2;
        }
        std::vector<uint8_t> parent;
        for (size_t i = 0; i + 1 < o.vdf_vector.size(); i += 2)
            parent.push_back((uint8_t)std::stoul(o.vdf_vector.substr(i, 2), nullptr, 16));
        const uint64_t tick = 1000;   // normative (CANARY-JOBS-SPEC §10)
        const auto proof = meow::Vdf::prove(parent, tick);
        if (proof.empty()) { std::fprintf(stderr, "error: VDF prove failed\n"); return 1; }
        std::printf("parent  = %s\ntick    = %llu\nvdf_len = %zu bytes\nvdf     = %s\n",
                    o.vdf_vector.c_str(), (unsigned long long)tick, proof.size(),
                    meow::Vdf::to_hex(proof).c_str());
        return 0;
    }

    if (!o.canary_ref.empty()) {
        // Reference-table generator (CANARY-JOBS-SPEC, pool-side automation):
        // for every w in [0, wmax) derive the prompt, run the prompt pass in
        // BATCHES, and emit one JSONL record per w with the step-0 top-64
        // (ids + bf16-snapped values, §4.1 tie rule). This binary IS the
        // reference implementation — the same engine, snap and sort that
        // mine — so on the same GPU arch the table matches compliant miners
        // bit-for-bit, and the chain's Σ_err envelope covers the rest.
        const auto colon = o.canary_ref.rfind(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "error: --canary-ref wants <seed64hex>:<wmax>\n"); return 2;
        }
        std::vector<uint8_t> seed;
        const std::string seed_hex = o.canary_ref.substr(0, colon);
        if (!meow::canary_parse_seed(seed_hex, seed)) {
            std::fprintf(stderr, "error: bad seed hex\n"); return 2;
        }
        const uint32_t wmax = (uint32_t)std::strtoul(o.canary_ref.c_str() + colon + 1, nullptr, 10);
        if (wmax == 0) { std::fprintf(stderr, "error: wmax must be > 0\n"); return 2; }

        meow::EngineConfig ec;
        ec.model_path = o.model_path;
        ec.slots_per_device = std::max(1, o.slots);
        ec.devices.push_back(dm.devices()[0].index);
        meow::InferenceEngine engine;
        std::string eerr;
        if (!engine.load(ec, eerr, nullptr)) { std::fprintf(stderr, "error: %s\n", eerr.c_str()); return 1; }
        const int n_vocab = engine.n_vocab();
        constexpr int kTopK = 64;

        const std::string out_path = !o.canary_out.empty() ? o.canary_out
            : "canary-ref-" + seed_hex.substr(0, 8) + ".jsonl";
        FILE* out = std::fopen(out_path.c_str(), "w");
        if (!out) { std::fprintf(stderr, "error: cannot open %s\n", out_path.c_str()); return 1; }
        {
            nlohmann::json hdr;
            hdr["format"]    = "supr-meow-tsc canary-ref v1";
            hdr["generator"] = std::string("supr-meow-tsc/") + kVersion;
            hdr["model"]     = o.model_path;
            hdr["n_vocab"]   = n_vocab;
            hdr["seed"]      = seed_hex;
            hdr["wmax"]      = wmax;
            hdr["top_k"]     = kTopK;
            hdr["snap"]      = "bf16";
            hdr["tie_rule"]  = "desc value, ties asc token id (spec 4.1)";
            std::fprintf(out, "%s\n", hdr.dump().c_str());
        }

        const int S = ec.slots_per_device;
        uint32_t done = 0;
        const auto tgen0 = std::chrono::steady_clock::now();
        for (uint32_t w0 = 0; w0 < wmax; w0 += (uint32_t)S) {
            const int batch = (int)std::min<uint32_t>((uint32_t)S, wmax - w0);
            std::vector<std::vector<int32_t>> prompts;
            prompts.reserve(batch);
            for (int b = 0; b < batch; ++b)
                prompts.push_back(meow::canary_derive_prompt(seed, w0 + (uint32_t)b, n_vocab));
            bool captured = false;
            engine.generate_windows_stepwise_tok(0, prompts,
                [&](const std::vector<const float*>& all, const float*, int nv,
                    const std::vector<std::vector<int64_t>>&, std::vector<int>&) -> bool {
                    for (int b = 0; b < batch; ++b) {
                        std::vector<std::pair<float,int>> v(nv);
                        for (int i = 0; i < nv; ++i) {
                            uint32_t x; std::memcpy(&x, &all[b][i], 4);
                            x = (x + (0x00007FFFu + ((x >> 16) & 1u))) & 0xFFFF0000u;
                            float sv; std::memcpy(&sv, &x, 4);
                            v[i] = { sv, i };
                        }
                        std::partial_sort(v.begin(), v.begin() + kTopK, v.end(),
                            [](const auto& a, const auto& c){
                                if (a.first != c.first) return a.first > c.first;
                                return a.second < c.second;
                            });
                        nlohmann::json rec;
                        rec["w"]   = w0 + (uint32_t)b;
                        rec["key"] = meow::canary_prompt_key(prompts[b]);
                        nlohmann::json idx = nlohmann::json::array();
                        nlohmann::json val = nlohmann::json::array();
                        for (int k = 0; k < kTopK; ++k) { idx.push_back(v[k].second); val.push_back(v[k].first); }
                        rec["idx"] = std::move(idx);
                        rec["val"] = std::move(val);
                        std::fprintf(out, "%s\n", rec.dump().c_str());
                    }
                    captured = true;
                    return false;   // step 0 is all we need
                }, eerr);
            if (!captured) {
                std::fprintf(stderr, "error at w=%u: %s\n", w0, eerr.c_str());
                std::fclose(out);
                return 1;
            }
            done += (uint32_t)batch;
            if ((w0 / (uint32_t)S) % 50 == 0) {
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - tgen0).count();
                std::fprintf(stderr, "  %u/%u (%.0f w/s)\n", done, wmax, el > 0 ? done / el : 0.0);
            }
        }
        std::fclose(out);
        const double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tgen0).count();
        std::printf("wrote %u records to %s in %.1fs\n", done, out_path.c_str(), el);
        return 0;
    }

    if (!o.canary_test.empty()) {
        // Validate the canary derivation + model + bf16 snap + tie-rule sort
        // end-to-end against the spec's test vectors (§9 / §9.1): derive the
        // prompt for (seed, w), run the prompt pass, print step-0 top-10.
        const auto colon = o.canary_test.rfind(':');
        if (colon == std::string::npos) {
            std::fprintf(stderr, "error: --canary-test wants <seed64hex>:<w>\n"); return 2;
        }
        std::vector<uint8_t> seed;
        if (!meow::canary_parse_seed(o.canary_test.substr(0, colon), seed)) {
            std::fprintf(stderr, "error: bad seed hex\n"); return 2;
        }
        const uint32_t w = (uint32_t)std::strtoul(o.canary_test.c_str() + colon + 1, nullptr, 10);
        meow::EngineConfig ec;
        ec.model_path = o.model_path;
        ec.slots_per_device = 1;
        ec.devices.push_back(dm.devices()[0].index);
        meow::InferenceEngine engine;
        std::string eerr;
        if (!engine.load(ec, eerr, nullptr)) { std::fprintf(stderr, "error: %s\n", eerr.c_str()); return 1; }
        const auto toks = meow::canary_derive_prompt(seed, w, engine.n_vocab());
        std::printf("w=%u tokens[0:8] = [", w);
        for (int i = 0; i < 8; ++i) std::printf("%s%d", i ? ", " : "", toks[i]);
        std::printf("]\n");
        bool printed = false;
        engine.generate_windows_stepwise_tok(0, {toks},
            [&](const std::vector<const float*>& all, const float*, int n_vocab,
                const std::vector<std::vector<int64_t>>&, std::vector<int>&) -> bool {
                // Snap to bf16 and sort by the NORMATIVE rule (§4.1):
                // descending value, ties by ascending token id.
                std::vector<std::pair<float,int>> v(n_vocab);
                for (int i = 0; i < n_vocab; ++i) {
                    uint32_t x; std::memcpy(&x, &all[0][i], 4);
                    x = (x + (0x00007FFFu + ((x >> 16) & 1u))) & 0xFFFF0000u;
                    float sv; std::memcpy(&sv, &x, 4);
                    v[i] = { sv, i };
                }
                std::partial_sort(v.begin(), v.begin() + 10, v.end(),
                    [](const auto& a, const auto& b){
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                    });
                std::printf("step0 idx[0:10] = [");
                for (int i = 0; i < 10; ++i) std::printf("%s%d", i ? ", " : "", v[i].second);
                std::printf("]\nstep0 val[0:10] = [");
                for (int i = 0; i < 10; ++i) std::printf("%s%g", i ? ", " : "", v[i].first);
                std::printf("]\n");
                printed = true;
                return false;   // one step is all we need — abort the window
            }, eerr);
        return printed ? 0 : 1;
    }

    if (o.vdf_test) {
        std::printf("VDF self-test (chiavdf, 1024-bit discriminant)\n");
        const auto t0 = std::chrono::steady_clock::now();
        std::string verr;
        const bool ok = meow::Vdf::self_test(verr, 1000);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!ok) { std::fprintf(stderr, "  FAILED: %s\n", verr.c_str()); return 1; }
        std::printf("  prove+verify of 1000 iterations in %.0f ms\n", ms);
        std::printf("  wrong-challenge rejection: ok\n  VDF ready\n");
        return 0;
    }

    if (o.benchmark) {
        meow::EngineConfig ec;
        ec.model_path       = o.model_path;
        ec.slots_per_device = o.slots;
        ec.workers_per_device = std::max(1, o.workers);
        for (const auto& d : dm.devices()) ec.devices.push_back(d.index);

        meow::InferenceEngine engine;
        std::string eerr;
        std::printf("loading %s\n", ec.model_path.c_str());
        if (!engine.load(ec, eerr, [](const std::string& m){ std::printf("  %s\n", m.c_str()); std::fflush(stdout); })) {
            std::fprintf(stderr, "error: %s\n", eerr.c_str());
            return 1;
        }
        std::printf("  vocabulary: %d tokens\n\nrunning %d window(s) per GPU…\n", engine.n_vocab(), 2);
        const auto st = engine.benchmark(2, eerr);
        if (st.empty()) { std::fprintf(stderr, "error: %s\n", eerr.c_str()); return 1; }
        std::printf("\n  %-5s %10s %10s %12s\n", "GPU", "tokens", "seconds", "tok/s");
        double total = 0;
        for (const auto& s2 : st) {
            std::printf("  %-5d %10llu %10.1f %12.1f\n", s2.device,
                        (unsigned long long)s2.tokens, s2.seconds, s2.tokens_per_s);
            total += s2.tokens_per_s;
        }
        std::printf("  %-5s %10s %10s %12.1f  (aggregate)\n\n", "all", "", "", total);
        dm.restore_all();
        return 0;
    }

    if (o.protocol_test || !o.dry_run) {
        if (o.pools.empty() || o.user.empty()) {
            std::fprintf(stderr,
                "error: -o and -u are required (use --list-devices or --dry-run otherwise)\n");
            return 2;
        }
        std::vector<meow::PoolUrl> urls;
        for (const auto& raw : o.pools) {
            meow::PoolUrl u; std::string perr;
            if (!meow::PoolUrl::parse(raw, u, perr)) {
                std::fprintf(stderr, "error: %s\n", perr.c_str());
                return 2;
            }
            urls.push_back(u);
        }

        meow::StratumClient client;
        std::atomic<int> jobs{0};
        // Job state must exist BEFORE the client starts: the pool sends the
        // first job during the handshake, and a callback installed afterwards
        // would miss it — leaving the miner idle until the next block.
        std::mutex jm;
        meow::PoolJob   cur_job;
        meow::PoolModel cur_model;
        std::atomic<bool> job_dirty{false};

        // Per-GPU share/window counters, indexed by position in dm.devices().
        // The stratum `tag` carries the CUDA device index through submit().
        constexpr int kMaxGpus = 16;
        struct GpuCounters {
            std::atomic<uint64_t> windows{0}, accepted{0}, rejected{0}, stale{0};
            std::atomic<uint64_t> rate_milli{0};    // rolling PoI/s * 1000 (reporter-fed)
            std::atomic<uint64_t> acc_pending{0};   // accepts not yet printed
            std::atomic<int64_t>  acc_last_print{0};// steady ms of last accept line
        };
        static GpuCounters gpu_stats[kMaxGpus];
        auto gpu_slot = [&](int cuda_index) -> int {
            const auto& devs = dm.devices();
            for (size_t i = 0; i < devs.size() && i < kMaxGpus; ++i)
                if (devs[i].index == cuda_index) return static_cast<int>(i);
            return 0;
        };

        meow::StratumCallbacks cb;
        cb.on_log = [](const std::string& m) {
            std::printf("[%s] %s\n", timestamp_now().c_str(), m.c_str()); std::fflush(stdout);
        };
        cb.on_model = [&](const meow::PoolModel& m) {
            { std::lock_guard<std::mutex> lk(jm); cur_model = m; }
            std::printf("[%s] %spool model:%s %s@%.12s  difficulty %llu  precision %s\n",
                        timestamp_now().c_str(), C_C(), C_0(),
                        m.name.c_str(), m.commit.c_str(),
                        (unsigned long long)m.difficulty, m.precision.c_str());
            std::fflush(stdout);
        };
        cb.on_target = [](const std::string& t) {
            // The number a human debugs vardiff with, plus the raw target for
            // when the number is not enough.
            const double diff = target_to_difficulty(t);
            std::printf("[%s] %spool sets new difficulty: %s%s (share target %.16s…)\n",
                        timestamp_now().c_str(), C_Y(),
                        human_difficulty(diff).c_str(), C_0(), t.c_str());
            std::fflush(stdout);
        };
        cb.on_job = [&](const meow::PoolJob& j) {
            { std::lock_guard<std::mutex> lk(jm); cur_job = j; job_dirty = true; }
            if (!j.pool_vdf.empty()) {
                static std::atomic<bool> said{false};
                if (!said.exchange(true))
                    std::printf("[%s] %spool-issued VDF ACTIVE%s — using the job's proof "
                                "(tick %llu), not proving locally\n",
                                timestamp_now().c_str(), C_C(), C_0(),
                                (unsigned long long)(j.pool_vdf_tick ? j.pool_vdf_tick : 1000));
            }
            std::printf("[%s] new job %s  height %llu  %s\n",
                        timestamp_now().c_str(), j.job_id.c_str(),
                        (unsigned long long)j.height,
                        j.clean ? "(clean — parent changed)" : "(continue)");
            std::fflush(stdout);
            jobs++;
        };
        cb.on_submit_result = [&](bool ok, int code, const std::string& msg, int tag) {
            auto& gs = gpu_stats[gpu_slot(tag)];
            if (ok) {
                gs.accepted++;
                // Coalesce: at high share rates (easy targets) one line per
                // share floods the log — fold bursts into "x N", max one
                // accept line per GPU per second. Rejections always print.
                const int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                const uint64_t pending = gs.acc_pending.fetch_add(1) + 1;
                int64_t last = gs.acc_last_print.load();
                if (now_ms - last >= 1000 &&
                    gs.acc_last_print.compare_exchange_strong(last, now_ms)) {
                    gs.acc_pending.fetch_sub(pending);
                    if (pending > 1)
                        std::printf("[%s] %sshare accepted%s x%llu [GPU %d]\n",
                                    timestamp_now().c_str(), C_G(), C_0(),
                                    (unsigned long long)pending, tag);
                    else
                        std::printf("[%s] %sshare accepted%s [GPU %d]\n",
                                    timestamp_now().c_str(), C_G(), C_0(), tag);
                }
            } else if (code == 21) {
                gs.stale++;
                std::printf("[%s] %sstale share%s [GPU %d] — job not found (normal at block change)\n",
                            timestamp_now().c_str(), C_Y(), C_0(), tag);
            } else {
                gs.rejected++;
                std::printf("[%s] %sshare REJECTED%s [GPU %d] — %s\n",
                            timestamp_now().c_str(), C_R(), C_0(), tag, msg.c_str());
            }
            std::fflush(stdout);
        };

        client.configure(urls, o.user, o.pass, std::string("supr-meow-tsc/") + kVersion, cb);
        std::printf("connecting to %s as %s\n\n", urls[0].raw.c_str(), o.user.c_str());
        if (!client.start()) { std::fprintf(stderr, "error: cannot start pool client\n"); return 1; }

        if (o.protocol_test) {
            std::printf("protocol test — Ctrl-C to stop\n\n");
            while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const auto st = client.stats();
            std::printf("\n%d job(s) received, %llu reconnect(s)\n",
                        jobs.load(), (unsigned long long)st.reconnects);
            client.stop();
            dm.restore_all();
            return 0;
        }

        // ---- real mining -------------------------------------------------
        // Double-buffered decode is the default: two window batches alternate
        // so the GPU never drains between sample steps. MEOW_DOUBLE_BUFFER=0
        // reverts to the single-batch path (also the automatic fallback when
        // GPU-resident logits are unavailable).
        const bool want_double = [](){
            const char* e = std::getenv("MEOW_DOUBLE_BUFFER");
            return !(e && *e == '0');
        }();
        meow::EngineConfig ec;
        ec.model_path       = o.model_path;
        ec.slots_per_device = o.slots;
        ec.double_buffer    = want_double;
        for (const auto& d : dm.devices()) ec.devices.push_back(d.index);

        meow::InferenceEngine engine;
        std::string eerr;
        if (!engine.load(ec, eerr, [](const std::string& m){ std::printf("  %s\n", m.c_str()); std::fflush(stdout); })) {
            std::fprintf(stderr, "error: %s\n", eerr.c_str());
            client.stop();
            return 1;
        }
        {
            // The line an operator checks against the pool's announced model.
            const auto slash = o.model_path.rfind('/');
            std::printf("[%s] %smodel loaded:%s %s  (vocabulary %d, precision bf16)\n",
                        timestamp_now().c_str(), C_G(), C_0(),
                        slash == std::string::npos ? o.model_path.c_str()
                                                   : o.model_path.c_str() + slash + 1,
                        engine.n_vocab());
            std::fflush(stdout);
        }

        const int n_streams = std::max(1, o.slots);
        const int n_groups  = std::max(1, std::min(o.groups, n_streams));

        // One SamplerPool per worker context (one llama context per device).
        // Each pool runs the per-stream sampler tail across n_groups threads,
        // each its own coordinator — this is what uses the idle cores. Egress
        // ports are partitioned so no two coordinators collide (they also key
        // the nonce partition): worker w, group g -> egress_base + w*64 + g.
        // --egress-base lets a SECOND process (one per GPU) claim a disjoint
        // range, e.g. GPU 0 at 47021 and GPU 1 at 48021, so the two processes'
        // loopback egress and nonce partitions never overlap.
        const int n_workers = engine.worker_count();
        std::vector<std::unique_ptr<meow::SamplerPool>> pools;    // batch A
        std::vector<std::unique_ptr<meow::SamplerPool>> pools_b;  // batch B (double-buffer)
        for (int w = 0; w < n_workers; ++w) {
            auto sp = std::make_unique<meow::SamplerPool>();
            if (!sp->init(n_streams, n_groups, o.egress_base + w * 64, engine.worker_device(w), eerr)) {
                std::fprintf(stderr, "error: %s\n", eerr.c_str());
                client.stop();
                return 1;
            }
            pools.push_back(std::move(sp));
            if (want_double) {
                // Its own coordinators, egress ports and nonce partition —
                // batch B is a fully independent set of windows.
                auto sb = std::make_unique<meow::SamplerPool>();
                if (!sb->init(n_streams, n_groups, o.egress_base + w * 64 + 32,
                              engine.worker_device(w), eerr)) {
                    std::fprintf(stderr, "error: %s\n", eerr.c_str());
                    client.stop();
                    return 1;
                }
                pools_b.push_back(std::move(sb));
            }
        }
        std::printf("  proof-of-inference ready — %d context(s) x %d sampler group(s) across %zu device(s)\n\n",
                    n_workers, n_groups, engine.config().devices.size());

        std::atomic<uint64_t> windows{0}, shares{0};
        const auto t_start = std::chrono::steady_clock::now();
        const uint64_t prompt_salt = std::random_device{}();

        // Canary jobs (CANARY-JOBS-SPEC): w is a DENSE counter over
        // (session, seed) — ONE atomic across all streams and workers of this
        // connection, reset when the job's seed changes. Density from 0 is
        // what keeps the pool's precomputed range reachable; the single
        // counter is what prevents (seed, w) reuse across streams.
        std::atomic<uint32_t> canary_w{0};
        std::mutex            canary_mx;
        std::string           canary_seed_hex;   // seed the counter belongs to
        static std::atomic<bool> canary_announced{false};

        // Dedicated submitter: base64/JSON-serializing ~120 KB proofs was
        // eating 0.7 s per window batch ON THE MINING THREAD. Shares are
        // handed off here and the GPU goes straight back to work.
        struct PendingShare { std::unique_ptr<meow::PoiShare> sh; int device; };
        std::mutex submit_mx;
        std::condition_variable submit_cv;
        std::deque<PendingShare> submit_q;
        std::thread submitter([&]{
            for (;;) {
                PendingShare ps;
                {
                    std::unique_lock<std::mutex> lk(submit_mx);
                    submit_cv.wait(lk, [&]{ return g_stop || !submit_q.empty(); });
                    if (submit_q.empty()) { if (g_stop) return; else continue; }
                    ps = std::move(submit_q.front());
                    submit_q.pop_front();
                }
                client.submit(ps.sh->job_id, ps.sh->nonce, ps.sh->proof_b64,
                              ps.sh->achieved_hex, ps.sh->vdf_tick, ps.device);
                ++shares;
            }
        });

        auto mine_device = [&](int worker) {
            meow::SamplerPool& pool = *pools[worker];
            meow::SamplerPool* poolB = want_double ? pools_b[worker].get() : nullptr;
            pow_gpu_bind_device(engine.worker_device(worker));
            uint64_t my_windows = 0;
            bool double_ok = want_double;   // falls false if -2 (no device logits)
            while (!g_stop) {
                meow::PoolJob   j;
                meow::PoolModel m;
                { std::lock_guard<std::mutex> lk(jm); j = cur_job; m = cur_model; }
                if (!j.valid || !m.valid || j.share_target.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    continue;
                }
                if (!pool.ready() || pool.job_id() != j.job_id) {
                    meow::PoiJobParams p;
                    p.header_prefix    = j.header_prefix;
                    p.block_target     = j.block_target;
                    p.share_target     = j.share_target;
                    p.model_identifier = m.name + "@" + m.commit;
                    p.model_difficulty = m.difficulty;
                    p.normalizer       = m.normalizer;
                    p.job_id           = j.job_id;
                    p.request_id       = j.request_id;
                    p.pool_vdf         = j.pool_vdf;        // §19, empty = prove locally
                    p.pool_vdf_tick    = j.pool_vdf_tick;
                    p.valid            = true;
                    std::string perr;
                    if (!pool.set_job(p, perr) || (poolB && !poolB->set_job(p, perr))) {
                        std::fprintf(stderr, "  job rejected: %s\n", perr.c_str());
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        continue;
                    }
                }

                // Seed-derived prompts when the job carries field 9; legacy
                // salted strings otherwise (also the automatic behavior
                // against pools that predate the canary spec).
                std::vector<uint8_t> seed;
                const bool seeded = !j.prompt_seed.empty() &&
                                    meow::canary_parse_seed(j.prompt_seed, seed);
                if (seeded) {
                    std::lock_guard<std::mutex> lk(canary_mx);
                    if (canary_seed_hex != j.prompt_seed) {
                        canary_seed_hex = j.prompt_seed;
                        canary_w.store(0);          // dense from 0 per (session, seed)
                    }
                    if (!canary_announced.exchange(true))
                        std::printf("[%s] %sprompt-seed mode ACTIVE%s — prompts derived per CANARY-JOBS-SPEC\n",
                                    timestamp_now().c_str(), C_C(), C_0());
                }
                auto make_prompts = [&](const char* tag) {
                    std::vector<std::string> prompts;
                    prompts.reserve(n_streams);
                    for (int st = 0; st < n_streams; ++st)
                        prompts.push_back("Explain distributed consensus in detail. [" +
                            std::to_string(prompt_salt) + "-" + std::to_string(worker) + tag +
                            std::to_string(my_windows) + "-" + std::to_string(st) + "]");
                    return prompts;
                };
                auto make_tok_prompts = [&]() {
                    std::vector<std::vector<int32_t>> tp;
                    tp.reserve(n_streams);
                    for (int st = 0; st < n_streams; ++st)
                        tp.push_back(meow::canary_derive_prompt(
                            seed, canary_w.fetch_add(1), engine.n_vocab()));
                    return tp;
                };

                std::string gerr;
                const auto tb0 = std::chrono::steady_clock::now();
                int n = -1;
                auto samp_a = [&](const std::vector<const float*>& all, const float* dev,
                                  int n_vocab, const std::vector<std::vector<int64_t>>& ctx,
                                  std::vector<int>& out) -> bool {
                    return pool.sample_step(all, dev, n_vocab, ctx, out);
                };
                auto samp_b = [&](const std::vector<const float*>& all, const float* dev,
                                  int n_vocab, const std::vector<std::vector<int64_t>>& ctx,
                                  std::vector<int>& out) -> bool {
                    return poolB->sample_step(all, dev, n_vocab, ctx, out);
                };
                if (double_ok) {
                    n = seeded
                        ? engine.generate_windows_double_tok(worker,
                              make_tok_prompts(), make_tok_prompts(), samp_a, samp_b, gerr)
                        : engine.generate_windows_double(worker,
                              make_prompts("A"), make_prompts("B"), samp_a, samp_b, gerr);
                    if (n == -2) {
                        double_ok = false;
                        std::fprintf(stderr, "  device logits unavailable — single-batch fallback\n");
                        continue;
                    }
                } else {
                    n = seeded
                        ? engine.generate_windows_stepwise_tok(worker, make_tok_prompts(), samp_a, gerr)
                        : engine.generate_windows_stepwise(worker, make_prompts("-"), samp_a, gerr);
                }
                const auto tb1 = std::chrono::steady_clock::now();
                if (n < 0) {
                    std::fprintf(stderr, "  window failed (worker %d): %s\n", worker, gerr.c_str());
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    continue;
                }
                my_windows += n;
                windows += n;
                gpu_stats[gpu_slot(engine.worker_device(worker))].windows += n;

                {
                    // Hand shares to the submitter thread; never serialize
                    // 120 KB proofs on the mining thread.
                    std::lock_guard<std::mutex> lk(submit_mx);
                    while (auto sh = pool.take_share())
                        submit_q.push_back({std::move(sh), engine.worker_device(worker)});
                    if (poolB)
                        while (auto sh = poolB->take_share())
                            submit_q.push_back({std::move(sh), engine.worker_device(worker)});
                }
                submit_cv.notify_one();
                // End-to-end batch accounting — [prof] times only the
                // generation loop and once hid 2 s/batch of dead time.
                const auto tb2 = std::chrono::steady_clock::now();
                const double gen_s = std::chrono::duration<double>(tb1 - tb0).count();
                const double sub_s = std::chrono::duration<double>(tb2 - tb1).count();
                std::fprintf(stderr, "[prof-e2e] gen=%.2fs submit=%.2fs -> %.1f windows/s end-to-end\n",
                             gen_s, sub_s,
                             (gen_s + sub_s) > 0 ? n / (gen_s + sub_s) : 0.0);
            }
        };

        // ---- local JSON stats API (HiveOS h-stats.sh reads this) ---------
        auto api_snapshot = [&]() -> std::string {
            nlohmann::json j;
            const double up = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            const auto st = client.stats();
            meow::PoolModel m;
            { std::lock_guard<std::mutex> lk(jm); m = cur_model; }
            j["name"]     = "supr-meow-tsc";
            j["ver"]      = kVersion;
            j["algo"]     = "tsc-poi";
            j["uptime"]   = static_cast<uint64_t>(up);
            j["model"]    = m.valid ? m.name + "@" + m.commit : "";
            j["pool"]     = urls[0].raw;
            j["user"]     = o.user;
            j["connected"]= client.connected();
            j["accepted"] = st.accepted;
            j["rejected"] = st.rejected;
            j["stale"]    = st.stale;
            double total_rate = 0.0;
            nlohmann::json gpus = nlohmann::json::array();
            const auto& devs = dm.devices();
            for (size_t i = 0; i < devs.size() && i < kMaxGpus; ++i) {
                const auto& gs = gpu_stats[i];
                const auto t = dm.telemetry(devs[i].index);
                // Rolling rate from the reporter once it has one interval;
                // lifetime average during the first seconds after start.
                const uint64_t rm = gs.rate_milli.load();
                const double rate = rm > 0 ? double(rm) / 1000.0
                                           : (up > 0 ? double(gs.windows.load()) / up : 0.0);
                total_rate += rate;
                nlohmann::json g;
                g["id"]       = devs[i].index;
                g["name"]     = devs[i].name;
                g["bus_id"]   = devs[i].pci_bus;
                g["poi_s"]    = rate;
                g["windows"]  = gs.windows.load();
                g["accepted"] = gs.accepted.load();
                g["rejected"] = gs.rejected.load();
                g["stale"]    = gs.stale.load();
                g["temp"]     = t.valid ? t.temp_c     : -1;
                g["fan"]      = t.valid ? t.fan_pct    : -1;
                g["power"]    = t.valid ? t.power_w    : -1;
                g["core_mhz"] = t.valid ? t.clock_core : -1;
                g["mem_mhz"]  = t.valid ? t.clock_mem  : -1;
                g["util"]     = t.valid ? t.util_gpu   : -1;
                gpus.push_back(std::move(g));
            }
            j["total_poi_s"] = total_rate;
            j["gpus"] = std::move(gpus);
            return j.dump();
        };
        meow::StatsHttpServer api;
        if (o.api_bind != "off" && o.api_bind != "0") {
            std::string aerr;
            if (api.start(o.api_bind, api_snapshot, aerr))
                std::printf("[%s] stats API on http://%s/\n", timestamp_now().c_str(), o.api_bind.c_str());
            else
                std::printf("[%s] %swarning:%s %s — continuing without stats API\n",
                            timestamp_now().c_str(), C_Y(), C_0(), aerr.c_str());
        }

        std::vector<std::thread> workers;
        for (int w = 0; w < n_workers; ++w)
            workers.emplace_back(mine_device, w);

        // ---- periodic report (SRBMiner-style), every --log-interval s ----
        std::vector<uint64_t> last_windows(kMaxGpus, 0);
        auto last_report = std::chrono::steady_clock::now();
        const int interval = std::max(5, o.log_interval);
        while (!g_stop) {
            for (int i = 0; i < interval * 10 && !g_stop; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (g_stop) break;
            const auto now = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(now - last_report).count();
            last_report = now;
            const double up = std::chrono::duration<double>(now - t_start).count();
            const auto st = client.stats();

            std::printf("%s================================================================================%s\n", C_C(), C_0());
            std::printf(" %ssupr-meow-tsc %s%s | uptime %s | pool %s%s%s\n",
                        C_B(), kVersion, C_0(), human_uptime(up).c_str(),
                        client.connected() ? C_G() : C_R(),
                        client.connected() ? "connected" : "DISCONNECTED", C_0());
            double total_rate = 0.0;
            const auto& devs = dm.devices();
            for (size_t i = 0; i < devs.size() && i < kMaxGpus; ++i) {
                auto& gs = gpu_stats[i];
                const uint64_t w_now = gs.windows.load();
                const double rate = dt > 0 ? double(w_now - last_windows[i]) / dt : 0.0;
                last_windows[i] = w_now;
                gs.rate_milli.store(static_cast<uint64_t>(rate * 1000.0));  // API reads this
                total_rate += rate;
                const auto t = dm.telemetry(devs[i].index);
                char temp[16] = "-", fan[16] = "-", pw[16] = "-", clk[32] = "-";
                if (t.valid) {
                    if (t.temp_c  >= 0) std::snprintf(temp, sizeof(temp), "%dC",  t.temp_c);
                    if (t.fan_pct >= 0) std::snprintf(fan,  sizeof(fan),  "%d%%", t.fan_pct);
                    if (t.power_w >= 0) std::snprintf(pw,   sizeof(pw),   "%dW",  t.power_w);
                    if (t.clock_core >= 0 && t.clock_mem >= 0)
                        std::snprintf(clk, sizeof(clk), "%d/%d MHz", t.clock_core, t.clock_mem);
                }
                std::printf(" GPU %d: %s%6.2f PoI/s%s | %4s | fan %4s | %5s | %14s | %sacc %llu%s rej %llu stale %llu\n",
                            devs[i].index, C_B(), rate, C_0(), temp, fan, pw, clk,
                            C_G(), (unsigned long long)gs.accepted.load(), C_0(),
                            (unsigned long long)gs.rejected.load(),
                            (unsigned long long)gs.stale.load());
            }
            std::printf(" %sTOTAL: %6.2f PoI/s%s | shares: %s%llu accepted%s, %s%llu rejected%s, %llu stale\n",
                        C_B(), total_rate, C_0(),
                        C_G(), (unsigned long long)st.accepted, C_0(),
                        st.rejected ? C_R() : "", (unsigned long long)st.rejected, st.rejected ? C_0() : "",
                        (unsigned long long)st.stale);
            std::printf("%s================================================================================%s\n", C_C(), C_0());
            std::fflush(stdout);
        }
        for (auto& w : workers) w.join();
        submit_cv.notify_all();
        if (submitter.joinable()) submitter.join();

        api.stop();
        client.stop();
        dm.restore_all();
        return 0;
    }

    std::printf("dry run — telemetry every %ds, Ctrl-C to stop\n\n", o.log_interval);
    while (!g_stop) {
        print_telemetry(dm);
        std::printf("\n");
        std::fflush(stdout);
        for (int i = 0; i < o.log_interval * 10 && !g_stop; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    std::printf("shutting down — restoring fan control and clocks\n");
    dm.restore_all();
    return 0;
}
