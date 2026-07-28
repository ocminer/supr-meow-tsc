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
#include <random>
#include <mutex>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

const char* kVersion = "0.1.0-foundation";

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

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
    int         slots = 8;
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
        else if (a == "--slots")           { if (!need_value(i, argc, "--slots")) return false; o.slots = std::atoi(argv[++i]); }
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

void print_device_table(const meow::DeviceManager& dm) {
    std::printf("\n  ID  %-28s %-22s %10s  %s\n", "GPU", "Architecture", "VRAM", "NVML");
    std::printf("  --  %-28s %-22s %10s  %s\n", "---------------------------",
                "---------------------", "---------", "----");
    for (const auto& d : dm.devices()) {
        std::printf("  %-2d  %-28s %-22s %10s  %s\n",
                    d.index, d.name.c_str(),
                    meow::DeviceManager::sm_arch_name(d.sm_major, d.sm_minor).c_str(),
                    human_bytes(d.vram_total).c_str(),
                    d.nvml_ok ? "yes" : "no");
    }
    std::printf("\n");
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

    std::printf("supr-meow-tsc %s — TensorCash miner\n", kVersion);

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

        meow::StratumCallbacks cb;
        cb.on_log = [](const std::string& m) { std::printf("  [pool] %s\n", m.c_str()); std::fflush(stdout); };
        cb.on_model = [&](const meow::PoolModel& m) {
            { std::lock_guard<std::mutex> lk(jm); cur_model = m; }
            std::printf("  [pool] model %s@%.12s  difficulty %llu  precision %s\n",
                        m.name.c_str(), m.commit.c_str(),
                        (unsigned long long)m.difficulty, m.precision.c_str());
            std::fflush(stdout);
        };
        cb.on_target = [](const std::string& t) {
            std::printf("  [pool] share target %.16s…\n", t.c_str()); std::fflush(stdout);
        };
        cb.on_job = [&](const meow::PoolJob& j) {
            { std::lock_guard<std::mutex> lk(jm); cur_job = j; job_dirty = true; }
            std::printf("  [pool] job %s  height %llu  %s\n",
                        j.job_id.c_str(), (unsigned long long)j.height,
                        j.clean ? "(clean — parent changed)" : "(continue)");
            std::fflush(stdout);
            jobs++;
        };
        cb.on_submit_result = [](bool ok, int code, const std::string& msg) {
            std::printf("  [pool] share %s%s%s\n", ok ? "accepted" : "REJECTED",
                        ok ? "" : " — ", ok ? "" : msg.c_str());
            std::fflush(stdout);
            (void)code;
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
        meow::EngineConfig ec;
        ec.model_path       = o.model_path;
        ec.slots_per_device = o.slots;
        for (const auto& d : dm.devices()) ec.devices.push_back(d.index);

        meow::InferenceEngine engine;
        std::string eerr;
        if (!engine.load(ec, eerr, [](const std::string& m){ std::printf("  %s\n", m.c_str()); std::fflush(stdout); })) {
            std::fprintf(stderr, "error: %s\n", eerr.c_str());
            client.stop();
            return 1;
        }

        meow::PoiMiner poi;
        if (!poi.init(256, eerr)) {
            std::fprintf(stderr, "error: %s\n", eerr.c_str());
            client.stop();
            return 1;
        }
        std::printf("  proof-of-inference ready (VDF %s…)\n\n", poi.vdf_hex().substr(0, 12).c_str());


        uint64_t windows = 0, shares = 0;
        const uint64_t prompt_salt = std::random_device{}();
        const auto t_start = std::chrono::steady_clock::now();

        while (!g_stop) {
            meow::PoolJob   j;
            meow::PoolModel m;
            { std::lock_guard<std::mutex> lk(jm); j = cur_job; m = cur_model; }
            if (!j.valid || !m.valid || j.share_target.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                continue;
            }
            if (job_dirty.exchange(false)) {
                meow::PoiJobParams p;
                p.header_prefix    = j.header_prefix;
                p.block_target     = j.block_target;
                p.share_target     = j.share_target;
                p.model_identifier = m.name + "@" + m.commit;
                p.model_difficulty = m.difficulty;
                p.normalizer       = m.normalizer;
                p.job_id           = j.job_id;
                p.request_id       = j.request_id;
                p.valid            = true;
                std::string perr;
                if (!poi.set_job(p, perr)) {
                    std::fprintf(stderr, "  job rejected: %s\n", perr.c_str());
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    continue;
                }
            }

            // Every window MUST use a different prompt. The v3 admission grind
            // folds the prompt into its preimage and its commitment, so an
            // identical prompt yields an identical nonce, an identical
            // transcript and a byte-identical proof — which the pool rejects as
            // a duplicate share. The salt is random per run and includes the
            // device, so two rigs (or two restarts) never grind the same window.
            const std::string prompt =
                "Explain distributed consensus in detail. [" +
                std::to_string(prompt_salt) + "-" + std::to_string(windows) + "]";

            // One window. The sampler chooses every token and records the
            // transcript; the engine advances on the sampler's choice.
            std::string gerr;
            const int n = engine.generate_window(0, prompt,
                [&](const float* logits, int n_vocab, const std::vector<int64_t>& ctx) -> int {
                    return poi.on_logits(0, logits, n_vocab, ctx, 1.0f, 50, 1.0f);
                }, gerr);
            if (n < 0) {
                std::fprintf(stderr, "  window failed: %s\n", gerr.c_str());
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                continue;
            }
            ++windows;

            if (auto sh = poi.take_share()) {
                client.submit(sh->job_id, sh->nonce, sh->proof_b64, sh->achieved_hex, sh->vdf_tick);
                ++shares;
            }

            if (windows % 5 == 0) {
                const double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
                const auto st = client.stats();
                std::printf("  %llu windows, %llu submitted | %.2f PoI/s | pool: %llu acc %llu rej %llu stale\n",
                            (unsigned long long)windows, (unsigned long long)shares,
                            secs > 0 ? double(windows) / secs : 0.0,
                            (unsigned long long)st.accepted, (unsigned long long)st.rejected,
                            (unsigned long long)st.stale);
                std::fflush(stdout);
            }
        }

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
