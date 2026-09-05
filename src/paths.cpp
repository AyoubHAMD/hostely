#include "paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <system_error>

namespace fs = std::filesystem;

namespace hostely::paths {

namespace {

constexpr const char* kAppName = "hostely";

}  // namespace

fs::path home_dir() {
    if (const char* home = std::getenv("HOME"); home && *home) {
        return fs::path(home);
    }
    return {};
}

fs::path state_dir() {
    if (auto h = home_dir(); !h.empty()) {
        return h / "Library" / "Application Support" / kAppName;
    }
    return fs::path();
}

fs::path log_dir() {
    if (auto h = home_dir(); !h.empty()) {
        return h / "Library" / "Logs" / kAppName;
    }
    return fs::path();
}

fs::path config_file() {
    auto d = state_dir();
    if (d.empty()) return d;
    return d / "config.toml";
}

fs::path models_dir() {
    auto d = state_dir();
    if (d.empty()) return d;
    return d / "models";
}

fs::path serve_lock_file() {
    auto d = state_dir();
    if (d.empty()) return d;
    return d / "serve.lock.json";
}

fs::path exposure_dir() {
    auto d = state_dir();
    if (d.empty()) return d;
    return d / "exposure";
}

fs::path certs_dir() {
    auto d = exposure_dir();
    if (d.empty()) return d;
    return d / "certs";
}

fs::path routes_file() {
    auto d = exposure_dir();
    if (d.empty()) return d;
    return d / "routes.json";
}

bool ensure_state_dir() {
    std::error_code ec;
    auto d = state_dir();
    if (d.empty()) return false;
    fs::create_directories(d, ec);
    return !ec;
}

bool ensure_log_dir() {
    std::error_code ec;
    auto d = log_dir();
    if (d.empty()) return false;
    fs::create_directories(d, ec);
    return !ec;
}

bool ensure_models_dir() {
    std::error_code ec;
    auto d = models_dir();
    if (d.empty()) return false;
    fs::create_directories(d, ec);
    return !ec;
}

}  // namespace hostely::paths
