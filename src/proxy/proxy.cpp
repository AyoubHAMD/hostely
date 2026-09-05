#include "proxy/proxy.hpp"

#include "exposure/certs.hpp"
#include "exposure/expose.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/err.h>
#include <openssl/ssl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>

namespace hostely::exposure {

namespace {

std::atomic<bool> g_stop{false};

void on_signal(int) { g_stop = true; }

// ---------------------------------------------------------------------------
// Route table — rebuilt wholesale (atomic swap), never mutated in place.
// ---------------------------------------------------------------------------

struct RouteTable {
    // host -> (target, port)
    std::map<std::string, std::pair<std::string, int>> by_host;
    std::map<std::string, std::string> certs;  // host -> pem (loaded lazily)
};

std::shared_mutex g_routes_mu;
RouteTable g_routes;

void rebuild_routes() {
    RouteTable next;
    for (const auto& r : routes_load()) {
        std::string host = r.host;
        // lowercase
        for (auto& c : host) c = static_cast<char>(std::tolower(c));
        next.by_host[host] = {r.target, r.port};
    }
    std::unique_lock lock(g_routes_mu);
    g_routes = std::move(next);
}

bool lookup_route(const std::string& host, std::string& target, int& port) {
    std::shared_lock lock(g_routes_mu);
    auto it = g_routes.by_host.find(host);
    if (it == g_routes.by_host.end()) return false;
    target = it->second.first;
    port = it->second.second;
    return true;
}

// Resolve a container name -> 127.0.0.1 published port is out of reach from
// the host in general; Apple `container` publishes ports on the host, so we
// target 127.0.0.1:<published>. For unpublished ports we try the container's
// IP via `container ls` output cached here. v1: container IPs are reachable
// from the host (vmnet bridge network), use `container inspect` when needed.
// For simplicity we parse `container ls --format json` on reload.
struct ContainerNet {
    std::string name;
    std::string ip;
    std::map<int, int> published;  // container port -> host port
};
std::map<std::string, ContainerNet> g_container_nets;

void refresh_container_nets() {
    std::string cmd = "container ls --format json 2>/dev/null";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return;
    std::string buf;
    char chunk[8192];
    size_t n;
    while ((n = fread(chunk, 1, sizeof chunk, fp)) > 0) buf.append(chunk, n);
    pclose(fp);
    try {
        auto j = nlohmann::json::parse(buf);
        std::map<std::string, ContainerNet> next;
        for (const auto& c : j) {
            ContainerNet net;
            net.name = c["configuration"].value("id", "");
            if (c.value("status", "") != "running") continue;
            if (auto networks = c["status"].value("networks", nlohmann::json::array());
                !networks.empty()) {
                net.ip = networks[0].value("ipv4Address", "");
            }
            for (const auto& [k, v] :
                 c["configuration"].value("ports", nlohmann::json::object())
                     .items()) {
                // key is container port, v is host port (number or string)
                try {
                    net.published[std::stoi(k)] = std::stoi(v.is_string() ? v.get<std::string>() : v.dump());
                } catch (...) {}
            }
            if (!net.name.empty()) next[net.name] = net;
        }
        std::unique_lock lock(g_routes_mu);
        g_container_nets = std::move(next);
    } catch (...) {}
}

bool container_target(const std::string& name, int port, std::string& host_out) {
    std::shared_lock lock(g_routes_mu);
    auto it = g_container_nets.find(name);
    if (it == g_container_nets.end()) return false;
    // Prefer the container IP + target port (no published-port guessing).
    if (!it->second.ip.empty()) {
        host_out = it->second.ip;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Upstream request — httplib client, streaming body pass-through.
// ---------------------------------------------------------------------------

bool proxy_upstream(const std::string& target_host, int target_port,
                    const httplib::Request& req, httplib::Response& res) {
    httplib::Client cli(target_host, target_port);
    cli.set_connection_timeout(10);
    cli.set_read_timeout(300);
    httplib::Headers hdrs;
    for (auto& [k, v] : req.headers) {
        std::string kl;
        for (auto c : k) kl += static_cast<char>(std::tolower(c));
        if (kl == "host" || kl == "connection" || kl == "upgrade" ||
            kl == "transfer-encoding" || kl == "content-length") {
            continue;
        }
        hdrs.emplace(k, v);
    }
    hdrs.emplace("X-Forwarded-For", req.remote_addr);
    hdrs.emplace("X-Forwarded-Proto", "https");
    hdrs.emplace("X-Forwarded-Host", req.get_header_value("Host"));

    httplib::Result r;
    if (req.method == "GET") {
        r = cli.Get(req.path.c_str(), hdrs);
    } else if (req.method == "POST") {
        r = cli.Post(req.path.c_str(), hdrs, req.body,
                     req.get_header_value("Content-Type"));
    } else if (req.method == "PUT") {
        r = cli.Put(req.path.c_str(), hdrs, req.body,
                    req.get_header_value("Content-Type"));
    } else if (req.method == "PATCH") {
        r = cli.Patch(req.path.c_str(), hdrs, req.body,
                      req.get_header_value("Content-Type"));
    } else if (req.method == "DELETE") {
        r = cli.Delete(req.path.c_str(), hdrs, req.body,
                       req.get_header_value("Content-Type"));
    } else if (req.method == "HEAD") {
        r = cli.Head(req.path.c_str(), hdrs);
    } else if (req.method == "OPTIONS") {
        r = cli.Options(req.path.c_str(), hdrs);
    } else {
        res.status = 405;
        return false;
    }
    if (!r) {
        res.status = 502;
        res.set_content("hostely proxy: upstream unreachable", "text/plain");
        return false;
    }
    res.status = r->status;
    for (auto& [k, v] : r->headers) {
        std::string kl;
        for (auto c : k) kl += static_cast<char>(std::tolower(c));
        if (kl == "transfer-encoding" || kl == "connection") continue;
        res.set_header(k, v);
    }
    res.body = r->body;
    return true;
}

// ---------------------------------------------------------------------------
// Connection handlers
// ---------------------------------------------------------------------------

void serve_plain(int fd) {
    httplib::Server srv;  // reused only as a Request/Response holder — we
                          // parse manually below instead.
    (void)srv;
    // Minimal HTTP/1.1 handling: read request head, proxy, write response.
    FILE* fr = fdopen(dup(fd), "r");
    if (!fr) { close(fd); return; }
    std::string line, method, path, host;
    auto read_line = [&]() -> bool {
        line.clear();
        int c;
        while ((c = fgetc(fr)) != EOF) {
            if (c == '\n') break;
            if (c != '\r') line += static_cast<char>(c);
        }
        return !line.empty() || c != EOF;
    };
    if (!read_line()) { fclose(fr); close(fd); return; }
    {
        std::istringstream ss(line);
        ss >> method >> path;
    }
    httplib::Request req;
    req.method = method;
    req.path = path;
    req.remote_addr = "0.0.0.0";
    while (read_line()) {
        if (line.empty()) break;
        if (auto colon = line.find(':'); colon != std::string::npos) {
            std::string k = line.substr(0, colon);
            std::string v = line.substr(colon + 1);
            // trim
            while (!v.empty() && (v.front() == ' ')) v.erase(v.begin());
            req.headers.emplace(k, v);
            if (k == "Host") {
                host = v;
                if (auto p = host.rfind(':'); p != std::string::npos)
                    host = host.substr(0, p);
            }
        }
    }
    // Body (best-effort content-length).
    size_t clen = 0;
    if (auto v = req.get_header_value("Content-Length"); !v.empty())
        clen = static_cast<size_t>(std::stoull(v));
    if (clen > 0 && clen < 64 * 1024 * 1024) {
        std::string body(clen, '\0');
        fread(body.data(), 1, clen, fr);
        req.body = body;
    }
    fclose(fr);  // closes dup'd fd; original still open

    httplib::Response res;
    std::string target;
    int port = 0;
    std::string thost;
    for (auto& c : host) c = static_cast<char>(std::tolower(c));
    if (!host.empty() && lookup_route(host, target, port)) {
        if (container_target(target, port, thost)) {
            proxy_upstream(thost, port, req, res);
        } else {
            res.status = 502;
            res.set_content("hostely proxy: container '" + target + "' not running",
                            "text/plain");
        }
    } else {
        res.status = 404;
        res.set_content("hostely proxy: no route for host '" + host + "'",
                        "text/plain");
    }
    std::ostringstream head;
    head << "HTTP/1.1 " << res.status << " " << httplib::status_message(res.status)
         << "\r\n";
    for (auto& [k, v] : res.headers) head << k << ": " << v << "\r\n";
    head << "Content-Length: " << res.body.size() << "\r\nConnection: close\r\n\r\n";
    std::string out = head.str() + res.body;
    send(fd, out.data(), out.size(), 0);
    close(fd);
}

// TLS: SNI callback picks the right cert from the store.
SSL_CTX* g_base_ssl_ctx = nullptr;

int sni_callback(SSL* ssl, int*, const char*) {
    const char* name = SSL_get_servername(ssl, TLSEXT_NAMETYPE_host_name);
    if (!name) return SSL_TLSEXT_ERR_NOACK;
    std::string host = name;
    for (auto& c : host) c = static_cast<char>(std::tolower(c));
    std::string cert_pem, key_pem;
    if (!cert_store_load(host, cert_pem, key_pem) || key_pem.empty())
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    // Build (and cache) a per-host SSL_CTX.
    static std::mutex mu;
    static std::map<std::string, SSL_CTX*> cache;
    SSL_CTX* ctx = nullptr;
    {
        std::lock_guard lk(mu);
        auto it = cache.find(host);
        if (it != cache.end()) ctx = it->second;
    }
    if (!ctx) {
        ctx = SSL_CTX_new(TLS_server_method());
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        BIO* cb = BIO_new_mem_buf(cert_pem.data(), (int)cert_pem.size());
        BIO* kb = BIO_new_mem_buf(key_pem.data(), (int)key_pem.size());
        X509* x = PEM_read_bio_X509(cb, nullptr, nullptr, nullptr);
        EVP_PKEY* k = PEM_read_bio_PrivateKey(kb, nullptr, nullptr, nullptr);
        if (x && k) {
            SSL_CTX_use_certificate(ctx, x);
            SSL_CTX_use_PrivateKey(ctx, k);
        }
        X509_free(x);
        EVP_PKEY_free(k);
        BIO_free(cb);
        BIO_free(kb);
        std::lock_guard lk(mu);
        cache[host] = ctx;
    }
    SSL_set_SSL_CTX(ssl, ctx);
    return SSL_TLSEXT_ERR_OK;
}

void serve_tls(int fd) {
    SSL* ssl = SSL_new(g_base_ssl_ctx);
    if (!ssl) { close(fd); return; }
    SSL_set_fd(ssl, fd);
    if (SSL_accept(ssl) <= 0) {
        SSL_free(ssl);
        close(fd);
        return;
    }
    // Reuse the plain handler by wrapping the SSL BIO in a FILE* is messy;
    // do a simplified direct loop: read until \r\n\r\n, parse, proxy, write.
    std::string reqbuf;
    char chunk[8192];
    while (reqbuf.find("\r\n\r\n") == std::string::npos) {
        int n = SSL_read(ssl, chunk, sizeof chunk);
        if (n <= 0) { SSL_free(ssl); close(fd); return; }
        reqbuf.append(chunk, n);
        if (reqbuf.size() > 128 * 1024) break;
    }
    auto head_end = reqbuf.find("\r\n\r\n");
    std::string head = reqbuf.substr(0, head_end);
    std::string body = reqbuf.substr(head_end + 4);

    std::istringstream ss(head);
    std::string method, path;
    ss >> method >> path;
    httplib::Request req;
    req.method = method;
    req.path = path;
    req.remote_addr = "0.0.0.0";
    std::string host;
    std::string line;
    while (std::getline(ss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (auto colon = line.find(':'); colon != std::string::npos) {
            std::string k = line.substr(0, colon);
            std::string v = line.substr(colon + 1);
            while (!v.empty() && v.front() == ' ') v.erase(v.begin());
            req.headers.emplace(k, v);
            if (k == "Host") {
                host = v;
                if (auto p = host.rfind(':'); p != std::string::npos)
                    host = host.substr(0, p);
            }
        }
    }
    // Remaining body bytes per content-length (buffered, capped).
    if (auto v = req.get_header_value("Content-Length"); !v.empty()) {
        size_t clen = std::stoull(v);
        while (body.size() < clen && body.size() < 64 * 1024 * 1024) {
            int n = SSL_read(ssl, chunk, sizeof chunk);
            if (n <= 0) break;
            body.append(chunk, n);
        }
        body.resize(std::min(body.size(), clen));
        req.body = body;
    }

    httplib::Response res;
    std::string target;
    int port = 0;
    std::string thost;
    for (auto& c : host) c = static_cast<char>(std::tolower(c));
    bool ok = false;
    if (!host.empty() && lookup_route(host, target, port)) {
        if (container_target(target, port, thost)) {
            ok = proxy_upstream(thost, port, req, res);
        } else {
            res.status = 502;
            res.set_content("hostely proxy: container '" + target +
                                "' not running",
                            "text/plain");
        }
    } else {
        res.status = 404;
        res.set_content("hostely proxy: no route for host '" + host + "'",
                        "text/plain");
    }
    (void)ok;
    std::ostringstream head_out;
    head_out << "HTTP/1.1 " << res.status << " "
             << httplib::status_message(res.status) << "\r\n";
    for (auto& [k, v] : res.headers) head_out << k << ": " << v << "\r\n";
    head_out << "Content-Length: " << res.body.size() << "\r\nConnection: close\r\n\r\n";
    std::string out = head_out.str() + res.body;
    SSL_write(ssl, out.data(), (int)out.size());
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(fd);
}

// ---------------------------------------------------------------------------
// Listeners
// ---------------------------------------------------------------------------

int make_listener(int port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) < 0 ||
        listen(fd, 64) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

}  // namespace

int proxy_serve(const ProxyOptions& opts) {
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    if (opts.https_port > 0) {
        g_base_ssl_ctx = SSL_CTX_new(TLS_server_method());
        SSL_CTX_set_min_proto_version(g_base_ssl_ctx, TLS1_2_VERSION);
        SSL_CTX_set_tlsext_servername_callback(g_base_ssl_ctx, sni_callback);
    }

    rebuild_routes();
    refresh_container_nets();

    int http_fd = opts.http_port > 0 ? make_listener(opts.http_port) : -1;
    int https_fd = opts.https_port > 0 ? make_listener(opts.https_port) : -1;
    if (http_fd < 0 && https_fd < 0) {
        std::cerr << "proxy: cannot bind listeners (are ports 80/443 in use?)\n";
        return 1;
    }

    std::cout << "hostely proxy listening"
              << (http_fd >= 0 ? " http=:" + std::to_string(opts.http_port) : "")
              << (https_fd >= 0 ? " https=:" + std::to_string(opts.https_port) : "")
              << "  (ctrl-c to stop)\n";

    // Reload routes every 3 s.
    std::thread reloader([&] {
        while (!g_stop) {
            for (int i = 0; i < 30 && !g_stop; ++i)
                usleep(100000);
            rebuild_routes();
            refresh_container_nets();
        }
    });

    while (!g_stop) {
        fd_set fds;
        FD_ZERO(&fds);
        int maxfd = -1;
        if (http_fd >= 0) { FD_SET(http_fd, &fds); maxfd = std::max(maxfd, http_fd); }
        if (https_fd >= 0) { FD_SET(https_fd, &fds); maxfd = std::max(maxfd, https_fd); }
        timeval tv{0, 200 * 1000};
        int ready = select(maxfd + 1, &fds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;
        if (http_fd >= 0 && FD_ISSET(http_fd, &fds)) {
            sockaddr_in peer{};
            socklen_t plen = sizeof peer;
            int cfd = accept(http_fd, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (cfd >= 0)
                std::thread(serve_plain, cfd).detach();
        }
        if (https_fd >= 0 && FD_ISSET(https_fd, &fds)) {
            sockaddr_in peer{};
            socklen_t plen = sizeof peer;
            int cfd = accept(https_fd, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (cfd >= 0)
                std::thread(serve_tls, cfd).detach();
        }
    }
    if (reloader.joinable()) reloader.join();
    std::cout << "hostely proxy stopped\n";
    return 0;
}

}  // namespace hostely::exposure