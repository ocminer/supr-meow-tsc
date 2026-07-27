#include "stratum.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>

#include "../vendor/nlohmann/json.hpp"

using json = nlohmann::json;

namespace meow {

// ---------------------------------------------------------------- URL

bool PoolUrl::parse(const std::string& url, PoolUrl& out, std::string& error) {
    out.raw = url;
    std::string rest = url;
    out.tls = false;

    auto starts = [&](const char* p) { return rest.rfind(p, 0) == 0; };
    if (starts("ws://") || starts("wss://")) {
        error = "ws:// and wss:// are the upstream broker protocol, not TSC Stratum.\n"
                "       Use stratum+tcp://host:port (or stratum+ssl://host:port).";
        return false;
    }
    if      (starts("stratum+tcp://")) rest = rest.substr(14);
    else if (starts("stratum+ssl://")) { rest = rest.substr(14); out.tls = true; }
    else if (starts("stratum+tls://")) { rest = rest.substr(14); out.tls = true; }
    else if (rest.find("://") != std::string::npos) {
        error = "unsupported URL scheme in '" + url + "'";
        return false;
    }

    // strip any trailing path — stratum has no path component
    const size_t slash = rest.find('/');
    if (slash != std::string::npos) rest = rest.substr(0, slash);

    const size_t colon = rest.rfind(':');
    if (colon == std::string::npos) { error = "missing port in '" + url + "'"; return false; }
    out.host = rest.substr(0, colon);
    const std::string ps = rest.substr(colon + 1);
    if (out.host.empty() || ps.empty()) { error = "malformed pool address '" + url + "'"; return false; }
    for (char c : ps) if (!isdigit(static_cast<unsigned char>(c))) {
        error = "port is not a number in '" + url + "'"; return false;
    }
    out.port = std::stoi(ps);
    if (out.port <= 0 || out.port > 65535) { error = "port out of range in '" + url + "'"; return false; }
    return true;
}

// ---------------------------------------------------------------- lifecycle

StratumClient::StratumClient() = default;
StratumClient::~StratumClient() { stop(); }

void StratumClient::configure(std::vector<PoolUrl> pools, std::string user, std::string pass,
                              std::string ua, StratumCallbacks cbs) {
    pools_ = std::move(pools);
    user_  = std::move(user);
    pass_  = std::move(pass);
    ua_    = std::move(ua);
    cb_    = std::move(cbs);
}

bool StratumClient::start() {
    if (pools_.empty()) return false;
    running_ = true;
    thread_ = std::thread([this] { run(); });
    return true;
}

void StratumClient::stop() {
    running_ = false;
    close_socket();
    if (thread_.joinable()) thread_.join();
}

void StratumClient::close_socket() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (fd_ >= 0) { ::shutdown(fd_, SHUT_RDWR); ::close(fd_); fd_ = -1; }
    connected_ = false;
}

StratumClient::Stats StratumClient::stats() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return stats_;
}

// ---------------------------------------------------------------- net

bool StratumClient::connect_once(const PoolUrl& p, std::string& error) {
    if (p.tls) {
        // Deliberate: shipping a half-done TLS path would be worse than saying
        // so. Plain TCP is the same protocol; TLS lands with the mining code.
        error = "stratum+ssl:// is not implemented in this build yet — use stratum+tcp://";
        return false;
    }

    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    const std::string port = std::to_string(p.port);
    const int rc = ::getaddrinfo(p.host.c_str(), port.c_str(), &hints, &res);
    if (rc != 0 || !res) { error = std::string("cannot resolve ") + p.host + ": " + gai_strerror(rc); return false; }

    int fd = -1;
    for (addrinfo* a = res; a; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        ::close(fd); fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) { error = "connection refused by " + p.host + ":" + port; return false; }

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));

    std::lock_guard<std::mutex> lk(mtx_);
    fd_ = fd;
    rxbuf_.clear();
    connected_ = true;
    stats_.pool = p.raw;
    return true;
}

bool StratumClient::send_line(const std::string& s) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (fd_ < 0) { outq_.push_back(s); return false; }
    const std::string line = s + "\n";
    size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::send(fd_, line.data() + off, line.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

void StratumClient::handshake() {
    json sub = {{"id", 1}, {"method", "mining.subscribe"},
                {"params", json::array({ua_, nullptr, json{{"protocol", "tsc/1.0"}}})}};
    send_line(sub.dump());
    json auth = {{"id", 2}, {"method", "mining.authorize"},
                 {"params", json::array({user_, pass_})}};
    send_line(auth.dump());
    {
        std::lock_guard<std::mutex> lk(mtx_);
        next_id_ = 3;
    }
}

void StratumClient::submit(const std::string& job_id, uint64_t nonce, const std::string& proof_b64,
                           const std::string& achieved_hex, uint64_t vdf_tick) {
    uint64_t id;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        id = next_id_++;
        stats_.submitted++;
    }
    json m = {{"id", id}, {"method", "mining.submit"},
              {"params", json::array({user_, job_id, nonce, proof_b64, achieved_hex, vdf_tick})}};
    send_line(m.dump());
}

// ---------------------------------------------------------------- protocol

void StratumClient::handle_line(const std::string& line) {
    json m;
    try { m = json::parse(line); }
    catch (const std::exception& e) {
        if (cb_.on_log) cb_.on_log(std::string("bad JSON from pool: ") + e.what());
        return;
    }

    // Notifications
    if (m.contains("method") && !m["method"].is_null()) {
        const std::string method = m["method"].get<std::string>();
        const json& p = m.contains("params") ? m["params"] : json::array();

        if (method == "mining.notify" && p.is_array() && p.size() >= 7) {
            PoolJob j;
            j.job_id        = p[0].get<std::string>();
            j.header_prefix = p[1].get<std::string>();
            j.block_target  = p[2].get<std::string>();
            j.nbits         = p[3].is_number() ? p[3].get<uint32_t>() : 0;
            j.height        = p[4].is_number() ? p[4].get<uint64_t>() : 0;
            j.expires_at    = p[5].is_number() ? p[5].get<uint64_t>() : 0;
            j.clean         = p[6].is_boolean() ? p[6].get<bool>() : true;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                j.share_target = share_target_;
            }
            // Spec §4.5: header_prefix is exactly 76 bytes. A pool that breaks
            // this would silently produce invalid proofs, so reject loudly.
            if (j.header_prefix.size() != 152) {
                if (cb_.on_log) cb_.on_log("pool sent a malformed job (header_prefix must be 152 hex chars) — ignoring");
                return;
            }
            j.valid = true;
            if (cb_.on_job) cb_.on_job(j);
            return;
        }
        if (method == "mining.set_target" && p.is_array() && !p.empty()) {
            const std::string t = p[0].get<std::string>();
            { std::lock_guard<std::mutex> lk(mtx_); share_target_ = t; }
            if (cb_.on_target) cb_.on_target(t);
            return;
        }
        if (method == "mining.set_model" && p.is_array() && !p.empty() && p[0].is_object()) {
            const json& o = p[0];
            PoolModel md;
            md.name       = o.value("name", "");
            md.commit     = o.value("commit", "");
            md.model_hash = o.value("model_hash", "");
            md.precision  = o.value("precision", "bf16");
            md.difficulty = o.value("difficulty", uint64_t{0});
            md.normalizer = o.value("normalizer", uint64_t{1000000});
            md.valid      = !md.name.empty();
            if (cb_.on_model) cb_.on_model(md);
            return;
        }
        if (method == "mining.ping") {
            json pong = {{"id", m.value("id", json())}, {"method", "mining.pong"}, {"params", json::array()}};
            send_line(pong.dump());
            return;
        }
        if (method == "mining.set_extranonce" || method == "mining.pong") return;
        if (cb_.on_log) cb_.on_log("unhandled notification: " + method);
        return;
    }

    // Responses
    const bool has_err = m.contains("error") && !m["error"].is_null();
    const uint64_t id  = m.value("id", uint64_t{0});

    if (id == 1) {
        if (has_err) { if (cb_.on_log) cb_.on_log("subscribe rejected by pool"); return; }
        if (cb_.on_log) cb_.on_log("subscribed");
        return;
    }
    if (id == 2) {
        if (has_err || (m.contains("result") && m["result"].is_boolean() && !m["result"].get<bool>())) {
            if (cb_.on_log) cb_.on_log("AUTHORIZATION FAILED — check -u (wallet address)");
        } else if (cb_.on_log) {
            cb_.on_log("authorized as " + user_);
        }
        return;
    }

    // Share result
    int code = 0;
    std::string msg;
    if (has_err) {
        const json& e = m["error"];
        if (e.is_object()) { code = e.value("code", 20); msg = e.value("message", ""); }
        else if (e.is_array() && e.size() >= 2) { code = e[0].get<int>(); msg = e[1].get<std::string>(); }
    }
    const bool ok = !has_err;
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (ok) stats_.accepted++;
        else if (code == 21) stats_.stale++;   // stale is normal at block edges
        else stats_.rejected++;
    }
    if (cb_.on_submit_result) cb_.on_submit_result(ok, code, msg);
}

// ---------------------------------------------------------------- loop

void StratumClient::run() {
    int backoff = 1;
    while (running_) {
        const PoolUrl& p = pools_[pool_idx_ % pools_.size()];
        std::string err;
        if (!connect_once(p, err)) {
            if (cb_.on_log) cb_.on_log("pool " + p.raw + ": " + err);
            pool_idx_++;                              // failover to the next -o
            for (int i = 0; i < backoff * 10 && running_; ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            backoff = std::min(backoff * 2, 30);
            continue;
        }
        if (cb_.on_log) cb_.on_log("connected to " + p.raw);
        backoff = 1;
        handshake();

        // flush anything queued while disconnected
        {
            std::vector<std::string> q;
            { std::lock_guard<std::mutex> lk(mtx_); q.swap(outq_); }
            for (auto& l : q) send_line(l);
        }

        while (running_) {
            pollfd pfd{};
            { std::lock_guard<std::mutex> lk(mtx_); pfd.fd = fd_; }
            if (pfd.fd < 0) break;
            pfd.events = POLLIN;
            const int pr = ::poll(&pfd, 1, 1000);
            if (pr < 0) break;
            if (pr == 0) continue;
            if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) break;

            char buf[65536];
            const ssize_t n = ::recv(pfd.fd, buf, sizeof(buf), 0);
            if (n <= 0) break;

            std::string chunk(buf, static_cast<size_t>(n));
            std::string work;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                rxbuf_ += chunk;
                // Spec: submits are ~160 KB, so lines are big by design. Guard
                // only against an unbounded run-on line.
                if (rxbuf_.size() > 4u * 1024 * 1024) { rxbuf_.clear(); break; }
                work.swap(rxbuf_);
            }
            size_t start = 0, nl;
            while ((nl = work.find('\n', start)) != std::string::npos) {
                const std::string line = work.substr(start, nl - start);
                start = nl + 1;
                if (!line.empty()) handle_line(line);
            }
            if (start < work.size()) {
                std::lock_guard<std::mutex> lk(mtx_);
                rxbuf_ = work.substr(start) + rxbuf_;
            }
        }

        close_socket();
        if (running_) {
            std::lock_guard<std::mutex> lk(mtx_);
            stats_.reconnects++;
        }
        if (cb_.on_log && running_) cb_.on_log("disconnected — reconnecting");
        for (int i = 0; i < 20 && running_; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

}  // namespace meow
