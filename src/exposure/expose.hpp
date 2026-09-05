#pragma once
// Route table + expose orchestration.
//
// routes.json (under state_dir()/exposure/) is the single source of truth
// for the proxy:
//   { "routes": [ { "host": "app.example.com",
//                   "target": "container-name",
//                   "port": 8080,
//                   "tls": true,
//                   "created_by": "expose" } ] }
// Full-rebuild + atomic swap semantics (never deltas).
#include <string>
#include <vector>

namespace hostely::exposure {

struct Route {
    std::string host;
    std::string target;   // container name
    int port = 80;
    bool tls = true;
};

std::vector<Route> routes_load();
bool routes_save_atomic(const std::vector<Route>& routes);
bool route_add(const Route& r);        // replaces same-host entry
bool route_remove(const std::string& host);
std::string routes_json_string(const std::vector<Route>& routes);

}  // namespace hostely::exposure