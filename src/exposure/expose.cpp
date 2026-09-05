#include "exposure/expose.hpp"

#include "paths.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace hostely::exposure {

std::vector<Route> routes_load() {
    std::vector<Route> out;
    auto path = paths::routes_file();
    std::ifstream in(path);
    if (!in) return out;
    try {
        json j = json::parse(in);
        for (const auto& r : j.value("routes", json::array())) {
            Route route;
            route.host = r.value("host", "");
            route.target = r.value("target", "");
            route.port = r.value("port", 80);
            route.tls = r.value("tls", true);
            if (!route.host.empty() && !route.target.empty()) {
                out.push_back(route);
            }
        }
    } catch (...) {}
    return out;
}

std::string routes_json_string(const std::vector<Route>& routes) {
    json arr = json::array();
    for (const auto& r : routes) {
        arr.push_back({{"host", r.host},
                       {"target", r.target},
                       {"port", r.port},
                       {"tls", r.tls}});
    }
    return json{{"routes", arr}}.dump(2) + "\n";
}

bool routes_save_atomic(const std::vector<Route>& routes) {
    auto path = paths::routes_file();
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec) return false;
    auto tmp = path;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out) return false;
        out << routes_json_string(routes);
    }
    fs::rename(tmp, path, ec);
    return !ec;
}

bool route_add(const Route& r) {
    auto routes = routes_load();
    bool replaced = false;
    for (auto& existing : routes) {
        if (existing.host == r.host) {
            existing = r;
            replaced = true;
        }
    }
    if (!replaced) routes.push_back(r);
    return routes_save_atomic(routes);
}

bool route_remove(const std::string& host) {
    auto routes = routes_load();
    std::vector<Route> kept;
    bool removed = false;
    for (auto& r : routes) {
        if (r.host == host) {
            removed = true;
        } else {
            kept.push_back(r);
        }
    }
    if (!removed) return false;
    return routes_save_atomic(kept);
}

}  // namespace hostely::exposure