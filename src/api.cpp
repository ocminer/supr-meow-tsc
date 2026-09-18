#include "api.h"


#include <cstring>
#include <cstdio>

namespace meow {

StatsHttpServer::~StatsHttpServer() { stop(); }

bool StatsHttpServer::start(const std::string& bind_addr, Snapshot snapshot, std::string& error) {
    if (!net::init()) { error = "Winsock initialization failed"; return false; }
    const auto colon = bind_addr.rfind(':');
    if (colon == std::string::npos) { error = "api bind must be host:port"; return false; }
    const std::string host = bind_addr.substr(0, colon);
    const int port = std::atoi(bind_addr.c_str() + colon + 1);
    if (port <= 0 || port > 65535) { error = "bad api port"; return false; }

    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ == net::invalid) { error = "api socket() failed"; return false; }
    int one = 1;
    net::option(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &sa.sin_addr) != 1) {
        error = "bad api host '" + host + "'";
        net::close(fd_); fd_ = net::invalid; return false;
    }
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::listen(fd_, 8) != 0) {
        error = "cannot bind api on " + bind_addr + " (port in use?)";
        net::close(fd_); fd_ = net::invalid; return false;
    }

    snapshot_ = std::move(snapshot);
    running_  = true;
    thread_   = std::thread([this]{ run(); });
    return true;
}

void StatsHttpServer::stop() {
    running_ = false;
    if (fd_ != net::invalid) { ::shutdown(fd_, net::shutdown_both); net::close(fd_); fd_ = net::invalid; }
    if (thread_.joinable()) thread_.join();
}

void StatsHttpServer::run() {
    while (running_) {
        net::poll_fd p{fd_, POLLIN, 0};
        const int r = net::poll(&p, 1, 250);
        if (!running_) break;
        if (r <= 0) continue;

        const net::socket_type c = ::accept(fd_, nullptr, nullptr);
        if (c == net::invalid) continue;

        // Read (and ignore) the request line + headers; a stats API answers
        // the same thing to every GET. Bounded read with a short timeout so a
        // stuck client cannot pin the thread.
        net::receive_timeout(c, 1000);
        char buf[2048];
        (void)::recv(c, buf, sizeof(buf), 0);

        std::string body;
        try { body = snapshot_ ? snapshot_() : "{}"; }
        catch (...) { body = "{}"; }

        char hdr[160];
        const int n = std::snprintf(hdr, sizeof(hdr),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Content-Length: %zu\r\n"
            "Connection: close\r\n\r\n", body.size());
        (void)net::send(c, hdr, static_cast<size_t>(n));
        (void)net::send(c, body.data(), body.size());
        net::close(c);
    }
}

}  // namespace meow
