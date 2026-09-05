#pragma once
// hostely proxy — 80/443 reverse proxy with SNI routing and WebSocket
// passthrough. Routes come from the exposure route table (routes.json)
// plus running containers that claim hostnames; hot-reloaded on change.
#include <string>

namespace hostely::exposure {

struct ProxyOptions {
    int http_port = 80;        // 0 disables the plain listener
    int https_port = 443;      // 0 disables the TLS listener
    bool foreground = true;
};

int proxy_serve(const ProxyOptions& opts);   // blocks until SIGINT/SIGTERM

}  // namespace hostely::exposure