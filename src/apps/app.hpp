#pragma once

// `hostely app` — run a compose-defined stack as a managed app.
//
// One app = one compose file = a set of named containers plus a state file
// (state_dir/apps/<app>.json) recording what was launched and with which
// configuration. Commands:
//
//   hostely app up <dir>     find compose file in <dir>, translate, reconcile
//   hostely app stop <app>   stop every container the app owns
//   hostely app ps           list known apps and their container status
//   hostely app logs <app> [service]
//   hostely app rm <app>     stop + forget the app (containers are removed)
//
// Translation rules (Apple container runtime on macOS 15 constraints):
//   - bind mounts of stateful dirs are unusable (virtiofs forbids chown), so
//     named volumes keep their name and *anonymous/relative* sources become
//     named volumes "<app>-<svc>-<dst-basename>"; absolute host paths stay
//     bind-mounted with a warning.
//   - service-name hostnames inside env values (e.g. PG_DATABASE_HOST=db)
//     are rewritten to the Mac's LAN IP, because macOS 15 has no
//     container-to-container DNS.

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "apps/compose.hpp"
#include "services/manager.hpp"

namespace hostely::apps {

struct AppServiceState {
    std::string service;          // compose service name (e.g. "db")
    std::string container;        // container name (e.g. "twenty-db")
    std::string image;
    std::string config_hash;      // change here => recreate on next `up`
    std::vector<std::pair<std::string, std::string>> ports;   // host:container
};

struct AppState {
    std::string name;
    std::string compose_path;
    std::vector<AppServiceState> services;   // topo order
};

struct AppError {
    std::string message;
};

// Translate + launch (or reconcile) an app from a directory containing a
// compose file. `messages` receives human-readable progress lines (already
// suitable for stderr/stdout printing). On failure returns nullopt + error.
std::optional<bool> app_up(const std::string& dir,
                           const std::optional<std::string>& name_override,
                           const std::vector<std::pair<std::string, std::string>>& extra_env,
                           services::Manager& mgr,
                           std::vector<std::string>& messages,
                           AppError& err);

// Stop every container of an app (reverse topo order). Idempotent.
std::optional<bool> app_stop(const std::string& app, services::Manager& mgr,
                             std::vector<std::string>& messages, AppError& err);

// All known apps with per-service status lines.
std::optional<std::vector<std::pair<std::string, std::vector<std::string>>>>
app_ps(services::Manager& mgr, AppError& err);

// Logs: one service or all. Returns the captured text.
std::optional<std::string> app_logs(const std::string& app,
                                    const std::optional<std::string>& service,
                                    bool follow, services::Manager& mgr,
                                    AppError& err);

// Stop + remove containers + delete the state file.
std::optional<bool> app_rm(const std::string& app, services::Manager& mgr,
                           std::vector<std::string>& messages, AppError& err);

// The app name `up` would derive from a directory (last path component).
std::string app_name_from_dir(const std::string& dir);

}  // namespace hostely::apps