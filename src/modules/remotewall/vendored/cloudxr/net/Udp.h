// Minimal Winsock UDP wrapper for the low-latency transport demo.
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cstdint>
#pragma comment(lib, "ws2_32.lib")

namespace net {

inline bool init() { WSADATA w; return WSAStartup(MAKEWORD(2, 2), &w) == 0; }
inline void cleanup() { WSACleanup(); }

inline sockaddr_in addr(const char* ip, uint16_t port) {
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
    inet_pton(AF_INET, ip, &a.sin_addr); return a;
}

struct UdpSocket {
    SOCKET s = INVALID_SOCKET;
    bool open() { s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP); return s != INVALID_SOCKET; }
    bool bindPort(uint16_t port) {
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_ANY); a.sin_port = htons(port);
        return bind(s, (sockaddr*)&a, sizeof(a)) == 0;
    }
    void setRecvTimeout(int ms) { DWORD t = (DWORD)ms; setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (char*)&t, sizeof(t)); }
    void setSendBuf(int bytes) { setsockopt(s, SOL_SOCKET, SO_SNDBUF, (char*)&bytes, sizeof(bytes)); }
    void setRecvBuf(int bytes) { setsockopt(s, SOL_SOCKET, SO_RCVBUF, (char*)&bytes, sizeof(bytes)); }
    int sendTo(const void* d, int n, const sockaddr_in& a) { return sendto(s, (const char*)d, n, 0, (const sockaddr*)&a, sizeof(a)); }
    int recv(void* d, int n) { return recvfrom(s, (char*)d, n, 0, nullptr, nullptr); }
    void close() { if (s != INVALID_SOCKET) { closesocket(s); s = INVALID_SOCKET; } }
};

} // namespace net
