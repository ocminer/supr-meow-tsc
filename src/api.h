// =============================================================================
// api.h — local JSON stats API.
//
// One tiny HTTP/1.1 endpoint on loopback: any GET returns a JSON snapshot of
// the miner (hashrate, per-GPU telemetry, share counts). This is how rig
// managers integrate — HiveOS's h-stats.sh, for example, curls this and maps
// the fields into its own schema (see docs/hiveos/). Dependency-free by
// design: a rig API that needs a web framework is a liability.
// =============================================================================
#pragma once

#include <functional>
#include <string>
#include <thread>
#include <atomic>

namespace meow {

class StatsHttpServer {
public:
    // `snapshot` is called per request from the server thread and must return
    // the full JSON body. It must be thread-safe.
    using Snapshot = std::function<std::string()>;

    ~StatsHttpServer();

    // bind is "host:port" ("127.0.0.1:21550"). Returns false with `error` set
    // if the socket cannot be bound (port taken, bad address).
    bool start(const std::string& bind, Snapshot snapshot, std::string& error);
    void stop();

private:
    void run();

    int               fd_ = -1;
    std::thread       thread_;
    std::atomic<bool> running_{false};
    Snapshot          snapshot_;
};

}  // namespace meow
