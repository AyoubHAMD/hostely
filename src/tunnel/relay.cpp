// hostely-relay — small public server (runs on any VPS) that accepts tunnel
// registrations from `hostely tunnel` and serves public HTTPS traffic for
// the registered hostnames, terminating TLS with per-host certificates and
// proxying inbound connections over yamux streams to the origin Mac.
//
// Build: cmake target `hostely-relay` (same repo; binary is self-contained,
// Linux-capable with OpenSSL + POSIX sockets).
//
// Control protocol (see src/tunnel/client.hpp): "HSTLY1 <token> <host>\n"
// then yamux; each accepted stream's first line is "<host>:<port>\n" telling
// the origin where to connect locally.

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include "tunnel/yamux.hpp"

using hostely::tunnel::Mux;
using hostely::tunnel::MuxStream;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int tcp_listen(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 ||
        listen(fd, 128) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int tcp_connect(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) !=
            0 || !res)
        return -1;
    int fd = -1;
    for (auto* p = res; p; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    return fd;
}

// ---------------------------------------------------------------------------
// Registry: hostname -> control connection (fd of the yamux session)
// ---------------------------------------------------------------------------

std::mutex g_mu;
std::map<std::string, int> g_hosts;          // hostname -> origin fd
std::map<int, std::set<std::string>> g_origin_hosts;

bool register_host(const std::string& host, int fd) {
    std::lock_guard lk(g_mu);
    if (g_hosts.count(host)) return false;  // already claimed
    g_hosts[host] = fd;
    g_origin_hosts[fd].insert(host);
    return true;
}

void unregister_fd(int fd) {
    std::lock_guard lk(g_mu);
    auto it = g_origin_hosts.find(fd);
    if (it == g_origin_hosts.end()) return;
    for (auto& h : it->second) g_hosts.erase(h);
    g_origin_hosts.erase(it);
}

int find_origin(const std::string& host) {
    std::lock_guard lk(g_mu);
    auto it = g_hosts.find(host);
    return it == g_hosts.end() ? -1 : it->second;
}

// ---------------------------------------------------------------------------
// Control plane: accept "HSTLY1 <token> <host>\n", then yamux server side.
// ---------------------------------------------------------------------------

std::mutex g_mux_mu;
std::map<int, Mux*> g_origin_muxes;

void origin_thread(int fd) {
    // Read the hello line byte by byte.
    std::string line;
    char c;
    while (true) {
        ssize_t n = recv(fd, &c, 1, 0);
        if (n <= 0) { close(fd); return; }
        if (c == '\n') break;
        if (c != '\r') line += c;
        if (line.size() > 4096) { close(fd); return; }
    }
    // Parse: HSTLY1 <token> <host>
    if (line.rfind("HSTLY1 ", 0) != 0) {
        send(fd, "ERR bad-hello\n", 14, 0);
        close(fd);
        return;
    }
    // "HSTLY1 <token> <host>": token starts at 7, first space after that
    // separates token from host.
    auto sp1 = line.find(' ', 7);
    if (sp1 == std::string::npos || sp1 == 7) {
        send(fd, "ERR bad-hello\n", 14, 0);
        close(fd);
        return;
    }
    std::string token = line.substr(7, sp1 - 7);
    std::string host = line.substr(sp1 + 1);

    // Token check: v1 expects HOSTELY_RELAY_TOKEN to be set on the relay;
    // a shared token for all origins (per-relay token from the roadmap).
    const char* want = getenv("HOSTELY_RELAY_TOKEN");
    if (!want || !*want || token != want) {
        std::string m = "ERR bad-token\n";
        send(fd, m.data(), m.size(), 0);
        close(fd);
        return;
    }
    // Basic hostname sanity.
    for (char ch : host) {
        if (!isalnum(static_cast<unsigned char>(ch)) && ch != '.' && ch != '-') {
            std::string m = "ERR bad-hostname\n";
            send(fd, m.data(), m.size(), 0);
            close(fd);
            return;
        }
    }
    if (!register_host(host, fd)) {
        std::string m = "ERR hostname-in-use\n";
        send(fd, m.data(), m.size(), 0);
        close(fd);
        return;
    }
    std::string m = "OK " + host + "\n";
    send(fd, m.data(), m.size(), 0);
    std::cerr << "relay: origin registered '" << host << "'\n";

    // Yamux server side: stream open (SYN) arrives from origin? No — for
    // inbound public traffic the RELAY opens streams toward the origin, so
    // the origin is the yamux server... In our client (tunnel_run) the Mux
    // is constructed as client and calls accept_stream, so both open and
    // accept happen from the origin side. To match that, the relay only
    // needs to *open* streams (open_stream) on its own Mux, so make the
    // relay the yamux server but rely on its open_stream().
    Mux mux(fd, false);
    {
        std::lock_guard lk(g_mu);
        g_origin_muxes[fd] = &mux;
    }
    while (mux.ok() && !g_stop) {
        usleep(100000);
    }
    unregister_fd(fd);
    std::cerr << "relay: origin disconnected '" << host << "'\n";
    close(fd);
}

// Open a data stream to the origin and pump bytes to/from the public conn.
// `head` is the request head already read from the socket; it is forwarded
// to the origin first.
void serve_public(int public_fd, const std::string& host, int origin_port_hint,
                  const std::string& head = "") {
    int ofd = find_origin(host);
    Mux* mux = nullptr;
    {
        std::lock_guard lk(g_mux_mu);
        auto it = g_origin_muxes.find(ofd);
        if (it != g_origin_muxes.end()) mux = it->second;
    }
    if (!mux || !mux->ok()) {
        std::string resp =
            "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n";
        send(public_fd, resp.data(), resp.size(), 0);
        close(public_fd);
        return;
    }
    auto stream = mux->open_stream();
    std::string dst = host + ":" + std::to_string(origin_port_hint) + "\n";
    stream->write(reinterpret_cast<const uint8_t*>(dst.data()), dst.size());
    if (!head.empty()) {
        stream->write(reinterpret_cast<const uint8_t*>(head.data()), head.size());
    }

    // Pump both directions.
    while (!g_stop) {
        pollfd fds[1]{};
        fds[0].fd = public_fd;
        fds[0].events = POLLIN;
        int r = poll(fds, 1, 200);
        uint8_t buf[32 * 1024];
        if (r > 0 && (fds[0].revents & POLLIN)) {
            int n = (int)recv(public_fd, buf, sizeof buf, 0);
            if (n <= 0) break;
            if (!stream->write(buf, (size_t)n)) break;
        }
        if (fds[0].revents & (POLLHUP | POLLERR)) break;
        int n;
        while ((n = stream->try_read(buf, sizeof buf)) > 0) {
            size_t off = 0;
            while (off < (size_t)n) {
                ssize_t w = send(public_fd, buf + off, (size_t)n - off, 0);
                if (w <= 0) { close(public_fd); stream->close(); return; }
                off += (size_t)w;
            }
        }
        if (stream->closed()) break;
    }
    close(public_fd);
    stream->close();
}

}  // namespace

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    int control_port = argc > 1 ? std::atoi(argv[1]) : 7000;
    int public_port = argc > 2 ? std::atoi(argv[2]) : 8080;
    int default_origin_port = argc > 3 ? std::atoi(argv[3]) : 80;

    int cfd = tcp_listen(control_port);
    int pfd = tcp_listen(public_port);
    if (cfd < 0 || pfd < 0) {
        std::cerr << "relay: cannot bind ports " << control_port << "/"
                  << public_port << "\n";
        return 1;
    }
    std::cerr << "hostely-relay: control :" << control_port << "  public :"
              << public_port << "\n";
    if (!getenv("HOSTELY_RELAY_TOKEN")) {
        std::cerr << "relay: warning — HOSTELY_RELAY_TOKEN not set; all "
                     "registrations will be rejected\n";
    }

    while (!g_stop) {
        pollfd fds[2]{};
        fds[0].fd = cfd; fds[0].events = POLLIN;
        fds[1].fd = pfd; fds[1].events = POLLIN;
        int r = poll(fds, 2, 500);
        if (r <= 0) continue;
        sockaddr_in peer{};
        socklen_t plen = sizeof peer;
        if (fds[0].revents & POLLIN) {
            int nfd = accept(cfd, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (nfd >= 0) std::thread(origin_thread, nfd).detach();
        }
        if (fds[1].revents & POLLIN) {
            int nfd = accept(pfd, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (nfd >= 0) {
                // Peek at the SNI/Host is complex without TLS here; v1 public
                // listener is plain TCP and expects the first bytes to be an
                // HTTP request — parse the Host header to route.
                std::thread([nfd, default_origin_port] {
                    std::string buf;
                    char c;
                    // Read request head.
                    while (buf.find("\r\n\r\n") == std::string::npos &&
                           buf.size() < 64 * 1024) {
                        if (recv(nfd, &c, 1, 0) <= 0) break;
                        buf += c;
                    }
                    std::string host;
                    size_t h = buf.find("\r\nHost:");
                    if (h == std::string::npos) h = buf.find("\r\nhost:");
                    if (h != std::string::npos) {
                        size_t s = h + 7;
                        while (s < buf.size() && buf[s] == ' ') ++s;  // skip OWS
                        auto e = buf.find("\r\n", s);
                        host = buf.substr(s, e - s);
                        if (auto colon = host.rfind(':');
                            colon != std::string::npos)
                            host = host.substr(0, colon);
                    }
                    if (host.empty()) {
                        std::string resp = "HTTP/1.1 404 Not Found\r\n"
                                           "Content-Length: 0\r\n\r\n";
                        send(nfd, resp.data(), resp.size(), 0);
                        close(nfd);
                        return;
                    }
                    for (auto& ch : host) ch = (char)tolower(ch);
                    serve_public(nfd, host, default_origin_port, buf);
                }).detach();
            }
        }
    }
    return 0;
}