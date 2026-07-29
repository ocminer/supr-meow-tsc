#include "api.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <cstdio>

namespace meow {

StatsHttpServer::~StatsHttpServer() { stop(); }

bool StatsHttpServer::start(const std::string& bind_addr, Snapshot snapshot, std::string& error) {
    const auto colon = bind_addr.rfind(':');
    if (colon == std::string::npos) { error = "api bind must be host:port"; return false; }
    const std::string host = bind_addr.substr(0, colon);
    const int port = std::atoi(bind_addr.c_str() + colon + 1);
    if (port <= 0 || port > 65535) { error = "bad api port"; return false; }

    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) { error = "api socket() failed"; return false; }
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(static_cast<uint16_t>(port));
    if (::inet_pton(AF_INET, host.c_str(), &sa.sin_addr) != 1) {
        error = "bad api host '" + host + "'";
        ::close(fd_); fd_ = -1; return false;
    }
    if (::bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0 ||
        ::listen(fd_, 8) != 0) {
        error = "cannot bind api on " + bind_addr + " (port in use?)";
        ::close(fd_); fd_ = -1; return false;
    }

    snapshot_ = std::move(snapshot);
    running_  = true;
    thread_   = std::thread([this]{ run(); });
    return true;
}

void StatsHttpServer::stop() {
    running_ = false;
    if (fd_ >= 0) { ::shutdown(fd_, SHUT_RDWR); ::close(fd_); fd_ = -1; }
    if (thread_.joinable()) thread_.join();
}

void StatsHttpServer::run() {
    while (running_) {
        pollfd p{fd_, POLLIN, 0};
        const int r = ::poll(&p, 1, 250);
        if (!running_) break;
        if (r <= 0) continue;

        const int c = ::accept(fd_, nullptr, nullptr);
        if (c < 0) continue;

        // Read (and ignore) the request line + headers; a stats API answers
        // the same thing to every GET. Bounded read with a short timeout so a
        // stuck client cannot pin the thread.
        timeval tv{1, 0};
        ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
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
        (void)::send(c, hdr, static_cast<size_t>(n), MSG_NOSIGNAL);
        (void)::send(c, body.data(), body.size(), MSG_NOSIGNAL);
        ::close(c);
    }
}

}  // namespace meow
