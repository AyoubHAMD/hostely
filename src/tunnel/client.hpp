#pragma once
// hostely tunnel — outbound client that connects to a hostely-relay, then
// bridges every incoming multiplexed stream to a local container port.
//
// Wire protocol on the control connection (TLS to relay):
//   client -> relay: "HSTLY1 <token> <hostname>\n"
//   relay  -> client: "OK <hostname>\n" or "ERR <msg>\n"
//   then yamux session; each accepted stream = one proxied TCP conn, first
//   bytes of stream are the client's original dst: "<host>:<port>\n".
#include <string>

namespace hostely::tunnel {

struct TunnelOptions {
    std::string relay_host;     // e.g. relay.example.com
    int relay_port = 443;
    std::string token;
    std::string hostname;       // public hostname this tunnel serves
    std::string local_host;     // container IP or 127.0.0.1
    int local_port = 80;
    bool local_port_set = false;  // when true, overrides relay-provided port
};

// Blocks and runs reconnect loop until SIGINT/SIGTERM. Returns exit code.
int tunnel_run(const TunnelOptions& opts);

}  // namespace hostely::tunnel