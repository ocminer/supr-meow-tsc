#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace meow::net {
#ifdef _WIN32
using socket_type = SOCKET;
using poll_fd = WSAPOLLFD;
constexpr socket_type invalid = INVALID_SOCKET;
constexpr int shutdown_both = SD_BOTH;
inline bool init() {
    static const bool ok = [] { WSADATA data{}; return WSAStartup(MAKEWORD(2,2), &data) == 0; }();
    return ok;
}
inline int close(socket_type fd) { return ::closesocket(fd); }
inline int poll(poll_fd* fds, unsigned long count, int ms) { return WSAPoll(fds, count, ms); }
#else
using socket_type = int;
using poll_fd = pollfd;
constexpr socket_type invalid = -1;
constexpr int shutdown_both = SHUT_RDWR;
inline bool init() { return true; }
inline int close(socket_type fd) { return ::close(fd); }
inline int poll(poll_fd* fds, unsigned long count, int ms) { return ::poll(fds, count, ms); }
#endif
inline int option(socket_type fd, int level, int key, const void* value, int size) {
    return ::setsockopt(fd, level, key, static_cast<const char*>(value), size);
}
inline std::ptrdiff_t send(socket_type fd, const char* data, size_t size) {
#ifdef _WIN32
    return ::send(fd, data, static_cast<int>(std::min(size, size_t(INT_MAX))), 0);
#else
    return ::send(fd, data, size, MSG_NOSIGNAL);
#endif
}
inline void receive_timeout(socket_type fd, int ms) {
#ifdef _WIN32
    const DWORD timeout = static_cast<DWORD>(ms);
    option(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#else
    timeval timeout{ms / 1000, (ms % 1000) * 1000};
    option(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif
}
} // namespace meow::net
