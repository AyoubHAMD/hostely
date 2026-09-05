#include "tunnel/client.hpp"

#include "tunnel/yamux.hpp"

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
#include <string>
#include <thread>

namespace hostely::tunnel {

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

int tcp_connect(const std::string& host, int port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string port_s = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0 || !res)
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

SSL* tls_wrap(int fd, const std::string& host) {
    static SSL_CTX* ctx = [] {
        auto* c = SSL_CTX_new(TLS_client_method());
        SSL_CTX_set_default_verify_paths(c);
        return c;
    }();
    SSL* ssl = SSL_new(ctx);
    SSL_set_fd(ssl, fd);
    SSL_set_tlsext_host_name(ssl, host.c_str());
    SSL_set1_host(ssl, host.c_str());
    if (SSL_connect(ssl) != 1) {
        SSL_free(ssl);
        return nullptr;
    }
    return ssl;
}

// Read one CRLF-terminated control line from the relay.
bool read_line_plain(int fd, std::string& out) {
    out.clear();
    char c;
    while (true) {
        ssize_t n = recv(fd, &c, 1, 0);
        if (n <= 0) return false;
        if (c == '\n') break;
        if (c != '\r') out += c;
        if (out.size() > 4096) return false;
    }
    return true;
}

bool send_all_plain(int fd, const std::string& data) {
    size_t off = 0;
    while (off < data.size()) {
        ssize_t n = send(fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

int fd_from_ssl(SSL* ssl) { return SSL_get_fd(ssl); }

// Bridge one yamux stream to a local TCP connection.
void bridge_stream(std::shared_ptr<MuxStream> stream, const TunnelOptions& opts) {
    // First line of the stream: "<host>:<port>\n" (v1: honor port, else default).
    std::string dst;
    uint8_t b;
    while (stream->read(&b, 1) == 1) {
        if (b == '\n') break;
        dst += static_cast<char>(b);
        if (dst.size() > 512) break;
    }
    int host_port = opts.local_port;
    std::string host = opts.local_host;
    if (auto colon = dst.rfind(':'); colon != std::string::npos) {
        if (!opts.local_port_set) {
            try { host_port = std::stoi(dst.substr(colon + 1)); } catch (...) {}
        }
    }

    int upstream = tcp_connect(host, host_port);
    if (upstream < 0) {
        stream->close();
        return;
    }
    int one = 1;
    setsockopt(upstream, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);

    // Pump both directions. The mux reader thread owns the relay socket, so
    // we poll the upstream fd and drain buffered stream data on wakeup.
    while (!g_stop && !stream->closed()) {
        pollfd fds[1]{};
        fds[0].fd = upstream;
        fds[0].events = POLLIN;
        int r = poll(fds, 1, 200);
        uint8_t buf[32 * 1024];
        if (r > 0 && (fds[0].revents & POLLIN)) {
            int n = (int)recv(upstream, buf, sizeof buf, 0);
            if (n <= 0) break;
            if (!stream->write(buf, (size_t)n)) break;
        }
        if (fds[0].revents & (POLLHUP | POLLERR)) break;
        // Relay -> local: drain whatever the mux has buffered.
        int n;
        while ((n = stream->try_read(buf, sizeof buf)) > 0) {
            size_t off = 0;
            while (off < (size_t)n) {
                ssize_t w = send(upstream, buf + off, (size_t)n - off, 0);
                if (w <= 0) { close(upstream); stream->close(); return; }
                off += (size_t)w;
            }
        }
    }
    close(upstream);
    stream->close();
}

}  // namespace

int tunnel_run(const TunnelOptions& opts) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    while (!g_stop) {
        int fd = tcp_connect(opts.relay_host, opts.relay_port);
        if (fd < 0) {
            std::cerr << "tunnel: cannot reach relay " << opts.relay_host
                      << ":" << opts.relay_port << "; retrying in 5s\n";
            sleep(5);
            continue;
        }
        // Control handshake over plain TCP (v1; TLS backhaul planned).
        std::string hello =
            "HSTLY1 " + opts.token + " " + opts.hostname + "\n";
        if (!send_all_plain(fd, hello)) {
            close(fd);
            sleep(5);
            continue;
        }
        std::string reply;
        if (!read_line_plain(fd, reply) || reply.rfind("OK", 0) != 0) {
            std::cerr << "tunnel: relay rejected registration: " << reply << "\n";
            close(fd);
            sleep(10);
            continue;
        }
        std::cerr << "tunnel: registered '" << opts.hostname
                  << "' at relay " << opts.relay_host << "\n";

        // Yamux session over the same connection.
        {
            Mux mux(fd, true);
            while (mux.ok() && !g_stop) {
                auto stream = mux.accept_stream();
                if (!stream) break;
                std::thread(bridge_stream, stream, opts).detach();
            }
        }
        close(fd);
        if (!g_stop) {
            std::cerr << "tunnel: connection lost; reconnecting in 3s\n";
            sleep(3);
        }
    }
    return 0;
}

}  // namespace hostely::tunnel