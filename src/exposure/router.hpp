#pragma once
// Router automation (roadmap Phase 2).
//
// NAT-PMP/PCP is hand-rolled (RFC 6886/6887 — a couple of small UDP
// request/response packets), UPnP IGD is a plain-HTTP SOAP call. Ladder:
// NAT-PMP -> PCP -> UPnP. Never drop a mapping because a refresh failed.
//
// Commands:
//   hostely router map <ext-port> <int-port> [--proto tcp|udp]
//   hostely router unmap <ext-port> [--proto tcp|udp]
//   hostely router status
//   hostely router watch --domain <dns.name> [--zone <zone>] [--interval S]
//       (poll public IP; update the A record when it changes)
#include <string>

namespace hostely::exposure {

struct RouterMapResult {
    bool ok = false;
    std::string method;     // "natpmp" | "pcp" | "upnp"
    std::string public_ip;  // when discoverable
    std::string error;
};

RouterMapResult router_map_port(int ext_port, int int_port, const char* proto);
bool router_unmap_port(int ext_port, const char* proto);
std::string router_public_ip();          // NAT-PMP, then fallbacks
int router_watch(const std::string& domain, const std::string& zone,
                 int interval_s);

}  // namespace hostely::exposure