// =============================================================================
// stratum.h — TSC Stratum client (see docs/STRATUM-TSC.md).
//
// Spec-driven and pool-agnostic: this client knows the protocol, never a
// particular operator. It runs its own thread, reconnects with backoff and
// fails over across the pools given with repeated -o.
// =============================================================================
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace meow {

struct PoolUrl {
    std::string host;
    int         port = 0;
    bool        tls  = false;
    std::string raw;
    // Accepts stratum+tcp://host:port, stratum+ssl://host:port, host:port.
    // Rejects ws:// and wss:// with an explanation — that is the upstream
    // broker dialect, a different protocol entirely.
    static bool parse(const std::string& url, PoolUrl& out, std::string& error);
};

struct PoolModel {
    std::string name;         // "Qwen/Qwen3-0.6B"
    std::string commit;       // pinned revision — mining any other is rejected on-chain
    std::string model_hash;
    std::string precision;    // "bf16"; consensus, not a tuning knob
    uint64_t    difficulty = 0;
    uint64_t    normalizer = 1000000;
    bool        valid = false;
};

struct PoolJob {
    std::string job_id;
    std::string header_prefix;   // 152 hex chars (76 bytes)
    std::string block_target;    // 64 hex
    std::string share_target;    // 64 hex, from the most recent set_target
    uint32_t    nbits = 0;
    uint64_t    height = 0;
    uint64_t    expires_at = 0;
    uint64_t    request_id = 0;  // work unit the proof must commit to
    bool        clean = false;   // true ⇒ parent changed, abandon in-flight work
    bool        valid = false;
};

struct StratumCallbacks {
    std::function<void(const PoolModel&)>  on_model;
    std::function<void(const std::string&)> on_target;      // 64-hex share target
    std::function<void(const PoolJob&)>    on_job;
    std::function<void(bool, int, const std::string&)> on_submit_result;  // ok, code, message
    std::function<void(const std::string&)> on_log;         // human-readable status
};

class StratumClient {
public:
    StratumClient();
    ~StratumClient();

    // `pools` is tried in order; on failure the client moves to the next and
    // wraps around, so a rig survives one pool going down.
    void configure(std::vector<PoolUrl> pools, std::string user, std::string pass,
                   std::string user_agent, StratumCallbacks cbs);

    bool start();
    void stop();

    bool connected() const { return connected_.load(); }

    // Queue a share. Non-blocking: the network thread owns the socket, so a
    // slow pool can never stall the mining loop.
    void submit(const std::string& job_id, uint64_t nonce, const std::string& proof_b64,
                const std::string& achieved_hex, uint64_t vdf_tick);

    struct Stats {
        uint64_t accepted = 0, rejected = 0, stale = 0, submitted = 0;
        uint64_t reconnects = 0;
        std::string pool;
    };
    Stats stats() const;

private:
    void run();
    bool connect_once(const PoolUrl& p, std::string& error);
    void close_socket();
    bool send_line(const std::string& s);
    void handle_line(const std::string& line);
    void handshake();

    std::vector<PoolUrl> pools_;
    size_t               pool_idx_ = 0;
    std::string          user_, pass_, ua_;
    StratumCallbacks     cb_;

    int                  fd_ = -1;
    void*                tls_ = nullptr;      // opaque; TLS lands with the mining path
    std::thread          thread_;
    std::atomic<bool>    running_{false};
    std::atomic<bool>    connected_{false};

    mutable std::mutex   mtx_;
    std::string          rxbuf_;
    std::string          share_target_;
    uint64_t             next_id_ = 1;
    std::vector<std::string> outq_;           // lines waiting for the socket
    Stats                stats_;
};

}  // namespace meow
