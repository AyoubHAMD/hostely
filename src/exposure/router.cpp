#include "exposure/router.hpp"

#include "exposure/dns.hpp"
#include "log/logger.hpp"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

using json = nlohmann::json;
namespace lg = hostely::log;

namespace hostely::exposure {

namespace {

// ---------------------------------------------------------------------------
// NAT-PMP (RFC 6886) / PCP (RFC 6887) over UDP to the default gateway.
// ---------------------------------------------------------------------------

uint32_t default_gateway_v4() {
    // Parse `route -n get default` output (no /proc on macOS).
    FILE* fp = popen("route -n get default 2>/dev/null", "r");
    if (!fp) return 0;
    char buf[4096];
    std::string out;
    while (fgets(buf, sizeof buf, fp)) out += buf;
    pclose(fp);
    auto pos = out.find("gateway:");
    if (pos == std::string::npos) return 0;
    std::string gw = out.substr(pos + 8);
    // trim
    while (!gw.empty() && (gw.front() == ' ' || gw.front() == '\t'))
        gw.erase(gw.begin());
    auto eol = gw.find_first_of("\n\r");
    if (eol != std::string::npos) gw = gw.substr(0, eol);
    in_addr a{};
    if (inet_pton(AF_INET, gw.c_str(), &a) != 1) return 0;
    return a.s_addr;
}


// One NAT-PMP/PCP UDP exchange with timeout. Returns response bytes.
bool udp_exchange(uint32_t gw, const uint8_t* req, size_t reqlen,
                  uint8_t* resp, size_t& resplen, int timeout_ms = 3000) {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return false;
    timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(5351);  // NAT-PMP and PCP share the port
    dst.sin_addr.s_addr = gw;
    bool ok = sendto(fd, req, reqlen, 0,
                     reinterpret_cast<sockaddr*>(&dst), sizeof dst) ==
              (ssize_t)reqlen;
    if (ok) {
        ssize_t n = recv(fd, resp, resplen, 0);
        if (n > 0) resplen = static_cast<size_t>(n);
        else ok = false;
    }
    close(fd);
    return ok;
}

// NAT-PMP: opcode 0 (public IP), opcode 1 (TCP) / 2 (UDP) map.
bool natpmp_public_ip(uint32_t gw, uint32_t& ip_out) {
    uint8_t req[2] = {0, 0};  // version 0, opcode 0
    uint8_t resp[16];
    size_t rl = sizeof resp;
    if (!udp_exchange(gw, req, sizeof req, resp, rl) || rl < 12) return false;
    if (resp[1] != 128 || resp[2] != 0) return false;  // server resp + success
    ip_out = *reinterpret_cast<uint32_t*>(resp + 8);
    return true;
}

bool natpmp_map(uint32_t gw, int ext, int internal, const char* proto,
                uint32_t& ip_out) {
    uint8_t req[12] = {};
    req[0] = 0;
    req[1] = (strcmp(proto, "udp") == 0) ? 2 : 1;
    req[2] = 0; req[3] = 0;  // reserved
    uint16_t priv = htons(static_cast<uint16_t>(internal));
    uint16_t pub = htons(static_cast<uint16_t>(ext));
    memcpy(req + 4, &priv, 2);
    memcpy(req + 6, &pub, 2);
    uint32_t lifetime = htonl(7200);  // 2h; refreshed by the watcher
    memcpy(req + 8, &lifetime, 4);

    for (int attempt = 0; attempt < 4; ++attempt) {  // RFC retransmit
        uint8_t resp[16];
        size_t rl = sizeof resp;
        if (udp_exchange(gw, req, sizeof req, resp, rl) && rl >= 16 &&
            resp[1] == req[1] + 128 && resp[2] == 0) {
            memcpy(&ip_out, resp + 8, 4);
            return true;
        }
        usleep(250000 << attempt);
    }
    return false;
}

// PCP MAP opcode 1 (RFC 6887). Preferred over NAT-PMP when available.
bool pcp_map(uint32_t gw, int ext, int internal, const char* proto,
             uint32_t& ip_out) {
    uint8_t req[60] = {};
    req[0] = 2;  // version
    req[1] = 1;  // MAP opcode
    uint32_t lifetime = htonl(7200);
    memcpy(req + 4, &lifetime, 4);
    std::mt19937 rng(static_cast<unsigned>(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    for (int i = 0; i < 12; ++i) req[8 + i] = static_cast<uint8_t>(rng());
    // protocol (6 = TCP, 17 = UDP) at offset 24
    req[24] = (strcmp(proto, "udp") == 0) ? 17 : 6;
    // internal port at 28, suggested external port at 30
    uint16_t ip_ = htons(static_cast<uint16_t>(internal));
    uint16_t ep_ = htons(static_cast<uint16_t>(ext));
    memcpy(req + 28, &ip_, 2);
    memcpy(req + 30, &ep_, 2);
    // IPv4 external address 36..39 = 0 (wildcard), v6 prefix 40..55 zero,
    // remote peer 56..59 zero.

    uint8_t resp[110];
    size_t rl = sizeof resp;
    if (!udp_exchange(gw, req, sizeof req, resp, rl) || rl < 60) return false;
    if (resp[0] != 2 || resp[1] != 129) return false;  // MAP reply
    uint8_t code = resp[3];
    if (code != 0) return false;
    uint16_t got = 0;
    memcpy(&got, resp + 32, 2);   // assigned external port
    (void)got;
    return true;
}

// ---------------------------------------------------------------------------
// UPnP IGD fallback — plain HTTP SOAP over TCP to the gateway.
// ---------------------------------------------------------------------------

std::string gateway_http_base() {
    uint32_t gw = default_gateway_v4();
    if (!gw) return "";
    in_addr a{};
    a.s_addr = gw;
    char ip[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &a, ip, sizeof ip);
    return std::string("http://") + ip + ":5000";  // common IGD port
}

bool upnp_map(int ext, int internal, const char* proto, uint32_t& ip_out) {
    std::string base = gateway_http_base();
    if (base.empty()) return false;
    httplib::Client cli(base);
    cli.set_connection_timeout(5);

    std::string proto_s = (strcmp(proto, "udp") == 0) ? "UDP" : "TCP";
    std::ostringstream body;
    body << "<?xml version=\"1.0\"?>"
         << "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\" "
            "s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
            "<s:Body><u:AddPortMapping xmlns:u=\"urn:schemas-upnp-org:"
            "service:WANIPConnection:1\">"
            "<NewRemoteHost></NewRemoteHost>"
            "<NewExternalPort>" << ext << "</NewExternalPort>"
            "<NewProtocol>" << proto_s << "</NewProtocol>"
            "<NewInternalPort>" << internal << "</NewInternalPort>"
            "<NewInternalClient>127.0.0.1</NewInternalClient>"
            "<NewEnabled>1</NewEnabled>"
            "<NewPortMappingDescription>hostely</NewPortMappingDescription>"
            "<NewLeaseDuration>0</NewLeaseDuration>"
            "</u:AddPortMapping></s:Body></s:Envelope>";
    auto res = cli.Post("/control?WANIPConn1", body.str(),
                        "text/xml; charset=\"utf-8\"");
    if (!res || res->status != 200) return false;
    // Best-effort: parse NewExternalIPAddress from the IGD status endpoint
    // is skipped; return success without IP.
    return true;
}

}  // namespace

RouterMapResult router_map_port(int ext_port, int int_port, const char* proto) {
    RouterMapResult out;
    uint32_t gw = default_gateway_v4();
    if (!gw) {
        out.error = "no default IPv4 gateway found";
        return out;
    }
    uint32_t ip = 0;
    if (pcp_map(gw, ext_port, int_port, proto, ip)) {
        out.ok = true;
        out.method = "pcp";
        in_addr a{};
        a.s_addr = ip;
        if (ip) {
            char s[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &a, s, sizeof s);
            out.public_ip = s;
        }
        return out;
    }
    if (natpmp_map(gw, ext_port, int_port, proto, ip)) {
        out.ok = true;
        out.method = "natpmp";
        in_addr a{};
        a.s_addr = ip;
        if (ip) {
            char s[INET_ADDRSTRLEN] = {0};
            inet_ntop(AF_INET, &a, s, sizeof s);
            out.public_ip = s;
        }
        return out;
    }
    if (upnp_map(ext_port, int_port, proto, ip)) {
        out.ok = true;
        out.method = "upnp";
        return out;
    }
    out.error = "router refused mapping (PCP, NAT-PMP and UPnP all failed)";
    return out;
}

bool router_unmap_port(int ext_port, const char* proto) {
    // NAT-PMP delete = request with lifetime 0.
    uint32_t gw = default_gateway_v4();
    if (!gw) return false;
    uint8_t req[12] = {};
    req[1] = (strcmp(proto, "udp") == 0) ? 2 : 1;
    uint16_t ext = htons(static_cast<uint16_t>(ext_port));
    memcpy(req + 6, &ext, 2);
    uint8_t resp[16];
    size_t rl = sizeof resp;
    return udp_exchange(gw, req, sizeof req, resp, rl, 2000);
}

std::string router_public_ip() {
    uint32_t gw = default_gateway_v4();
    uint32_t ip = 0;
    if (gw && natpmp_public_ip(gw, ip)) {
        in_addr a{};
        a.s_addr = ip;
        char s[INET_ADDRSTRLEN] = {0};
        inet_ntop(AF_INET, &a, s, sizeof s);
        return s;
    }
    return "";
}

int router_watch(const std::string& domain, const std::string& zone,
                 int interval_s) {
    auto provider = make_dns_provider();
    if (!provider) {
        lg::error("no DNS provider configured (set HOSTELY_CF_API_TOKEN)");
        return 1;
    }
    std::string z = zone;
    if (z.empty()) {
        size_t last = domain.rfind('.');
        size_t prev = last == std::string::npos ? std::string::npos
                                                : domain.rfind('.', last - 1);
        z = prev == std::string::npos ? domain : domain.substr(prev + 1);
    }
    lg::info("watching public IP; DNS A record: " + domain + " (zone " + z + ")");
    std::string last_ip;
    while (true) {
        std::string ip = router_public_ip();
        if (!ip.empty() && ip != last_ip) {
            lg::info("public IP: " + ip +
                     (last_ip.empty() ? "" : " (changed from " + last_ip + ")"));
            if (provider->set_a(z, domain, ip)) {
                lg::info("DNS updated: " + domain + " -> " + ip);
            } else {
                lg::error("DNS update failed for " + domain);
            }
            last_ip = ip;
        } else if (ip.empty()) {
            lg::warn("could not determine public IP this cycle");
        }
        std::this_thread::sleep_for(
            std::chrono::seconds(interval_s > 0 ? interval_s : 300));
    }
    return 0;
}

}  // namespace hostely::exposure