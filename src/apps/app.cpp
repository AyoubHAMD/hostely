#include "apps/app.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <stdlib.h>

#include "paths.hpp"
#include "services/cli.hpp"

namespace hostely::apps {

using json = nlohmann::json;

namespace {

std::filesystem::path apps_state_dir() {
    auto dir = paths::state_dir() / "apps";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

std::filesystem::path app_state_file(const std::string& app) {
    return apps_state_dir() / (app + ".json");
}

bool load_state(const std::string& app, AppState& out) {
    std::ifstream in(app_state_file(app));
    if (!in) return false;
    try {
        json j = json::parse(in);
        out.name = j.value("name", app);
        out.compose_path = j.value("compose_path", "");
        out.services.clear();
        for (const auto& s : j.value("services", json::array())) {
            AppServiceState st;
            st.service = s.value("service", "");
            st.container = s.value("container", "");
            st.image = s.value("image", "");
            st.config_hash = s.value("config_hash", "");
            for (const auto& p : s.value("ports", json::array())) {
                if (p.is_array() && p.size() == 2)
                    st.ports.emplace_back(p[0].get<std::string>(),
                                          p[1].get<std::string>());
            }
            out.services.push_back(std::move(st));
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool save_state(const AppState& st, AppError& err) {
    std::error_code ec;
    std::filesystem::create_directories(apps_state_dir(), ec);
    json j;
    j["name"] = st.name;
    j["compose_path"] = st.compose_path;
    j["services"] = json::array();
    for (const auto& s : st.services) {
        json sj;
        sj["service"] = s.service;
        sj["container"] = s.container;
        sj["image"] = s.image;
        sj["config_hash"] = s.config_hash;
        sj["ports"] = json::array();
        for (const auto& [h, c] : s.ports) sj["ports"].push_back(json::array({h, c}));
        j["services"].push_back(sj);
    }
    std::ofstream out(app_state_file(st.name));
    if (!out) {
        err = {"cannot write " + app_state_file(st.name).string()};
        return false;
    }
    out << j.dump(2) << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

std::string trim_slash(std::string p) {
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    return p;
}

// Best-effort primary LAN IPv4 (en0 first), e.g. "192.168.1.110".
std::string lan_ip() {
    ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return "";
    std::string fallback;
    for (auto* ifa = ifs; ifa; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
        std::string name = ifa->ifa_name ? ifa->ifa_name : "";
        if (name == "lo0") continue;
        char buf[INET_ADDRSTRLEN] = {0};
        auto* sin = reinterpret_cast<sockaddr_in*>(ifa->ifa_addr);
        inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof buf);
        std::string ip = buf;
        if (ip.rfind("169.254.", 0) == 0) continue;
        if (name == "en0") { fallback = ip; break; }   // en0 wins
        if (fallback.empty()) fallback = ip;
    }
    freeifaddrs(ifs);
    return fallback;
}

bool port_in_use(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    int rc = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr);
    close(fd);
    return rc == 0;
}

int pick_free_host_port(int wanted) {
    int p = wanted;
    while (p < 65535 && port_in_use(p)) ++p;
    return p;
}

// FNV-1a — a stable config fingerprint; cryptographic strength is not needed.
std::string fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    std::ostringstream oss;
    oss << std::hex << h;
    return oss.str();
}

std::string config_fingerprint(const ComposeService& s) {
    json j;
    j["image"] = s.image;
    j["ports"] = s.ports;
    j["env"] = s.env;
    j["volumes"] = s.volumes;
    j["command"] = s.command;
    return fnv1a(j.dump());
}

// Topological order over depends_on; cycle or unknown dep -> error text.
bool topo_sort(const std::vector<ComposeService>& in,
               std::vector<ComposeService>& out, std::string& err) {
    std::map<std::string, const ComposeService*> by_name;
    for (const auto& s : in) by_name[s.name] = &s;
    std::map<std::string, int> state;   // 0 new, 1 visiting, 2 done
    out.clear();
    std::function<bool(const ComposeService&)> visit =
        [&](const ComposeService& s) -> bool {
        int& st = state[s.name];
        if (st == 2) return true;
        if (st == 1) { err = "dependency cycle involving '" + s.name + "'"; return false; }
        st = 1;
        for (const auto& dep : s.depends_on) {
            auto it = by_name.find(dep);
            if (it == by_name.end()) {
                err = "service '" + s.name + "' depends on unknown service '" + dep + "'";
                return false;
            }
            if (!visit(*it->second)) return false;
        }
        st = 2;
        out.push_back(s);
        return true;
    };
    for (const auto& s : in)
        if (!visit(s)) return false;
    return true;
}

std::string basename_of(const std::string& path) {
    auto slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

bool looks_absolute(const std::string& p) {
    return !p.empty() && p[0] == '/';
}

// Direct CLI helpers for verbs Manager doesn't expose yet. Manager::list()
// only shows running containers, so existence checks use `container ls -a`.
services::Outcome<bool> cli_container(const std::vector<std::string>& args) {
    services::Outcome<bool> out;
    std::vector<std::string> argv{"/usr/local/bin/container"};
    for (const auto& a : args) argv.push_back(a);
    auto r = services::cli::run(argv);
    if (!r.started()) {
        out.error = services::ManagerError{services::ManagerError::Kind::CliNotFound,
                                           "could not launch container CLI", {}};
        return out;
    }
    out.value = r.exit_code == 0;
    return out;
}

bool container_exists_any_state(const std::string& name) {
    auto r = services::cli::run({"/usr/local/bin/container", "ls", "-a"});
    if (!r.started() || r.exit_code != 0) return false;
    std::istringstream iss(r.stdout_text);
    std::string line;
    bool first = true;
    while (std::getline(iss, line)) {
        if (first) { first = false; continue; }
        std::istringstream ls(line);
        std::string tok;
        if (ls >> tok && tok == name) return true;
    }
    return false;
}

services::Outcome<bool> rm_container(services::Manager& /*mgr*/, const std::string& name) {
    // `container rm -f` — Manager has no dedicated API yet, go through cli::run.
    services::Outcome<bool> out;
    auto r = services::cli::run({"/usr/local/bin/container", "rm", "-f", name});
    if (!r.started()) {
        out.error = services::ManagerError{services::ManagerError::Kind::CliNotFound,
                                           "could not launch container CLI", {}};
        return out;
    }
    out.value = r.exit_code == 0;
    return out;
}

}  // namespace

std::string app_name_from_dir(const std::string& dir) {
    std::string d = trim_slash(dir);
    auto slash = d.find_last_of('/');
    std::string name = slash == std::string::npos ? d : d.substr(slash + 1);
    // Keep it container-name safe.
    std::string clean;
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')
            clean += c;
        else clean += '-';
    }
    return clean.empty() ? "app" : clean;
}

std::optional<bool> app_up(const std::string& dir,
                           const std::optional<std::string>& name_override,
                           const std::vector<std::pair<std::string, std::string>>& extra_env,
                           services::Manager& mgr,
                           std::vector<std::string>& messages,
                           AppError& err) {
    // 1. Find + parse.
    auto compose_path = find_compose_file(dir);
    if (!compose_path) {
        err = {"no compose.yaml / docker-compose.yml found in " + dir +
               " (searched the directory and subdirectories, depth 2)"};
        return std::nullopt;
    }
    std::map<std::string, std::string> env_extra;
    for (const auto& [k, v] : extra_env) env_extra[k] = v;
    ComposeError cerr;
    auto compose = load_compose(*compose_path, env_extra, cerr);
    if (!compose) {
        err = {cerr.message};
        return std::nullopt;
    }

    std::string app = name_override && !name_override->empty()
                          ? *name_override
                          : app_name_from_dir(dir);
    messages.push_back("app        : " + app);
    messages.push_back("compose    : " + *compose_path);

    // 2. Translate.
    std::vector<ComposeService> ordered;
    std::string terr;
    if (!topo_sort(compose->services, ordered, terr)) {
        err = {terr};
        return std::nullopt;
    }

    std::string host = lan_ip();
    std::set<std::string> svc_names;
    for (const auto& s : ordered) svc_names.insert(s.name);

    AppState state;
    state.name = app;
    state.compose_path = *compose_path;

    for (auto& s : ordered) {
        services::RunOptions ro;
        ro.name = app + "-" + s.name;
        ro.image = s.image;
        ro.args = s.command;

        // Env: rewrite service-name hostnames to the host LAN IP.
        for (auto& [k, v] : s.env) {
            for (const auto& other : svc_names) {
                if (other == s.name || other.empty()) continue;
                if (v == other) v = host;
            }
            ro.env.emplace_back(k, v);
        }

        // Volumes: named volumes pass through; absolute paths bind-mount
        // (with a warning); anything else becomes a named volume.
        for (auto& [src, dst] : s.volumes) {
            if (looks_absolute(src)) {
                messages.push_back("[warn] bind mount " + src + " -> " + dst +
                                   ": virtiofs on macOS 15 forbids chown; "
                                   "stateful services (postgres, mysql) need a named volume");
                ro.volumes.emplace_back(src, dst);
            } else if (src.find('/') == std::string::npos) {
                ro.volumes.emplace_back(src, dst);      // already a named volume
            } else {
                std::string dstbase = basename_of(trim_slash(dst));
                ro.volumes.emplace_back(app + "-" + s.name + "-" + dstbase, dst);
                messages.push_back("[info] volume " + src + " -> " + dst +
                                   " became named volume '" +
                                   app + "-" + s.name + "-" + dstbase + "'");
            }
        }

        std::string fingerprint = config_fingerprint(s);

        // 3. Reconcile — decide BEFORE picking ports: a container that is
        // about to be recreated must not hold its old port hostage (the new
        // one would drift to a different host port).
        bool needs_run = true;
        AppState old;
        if (load_state(app, old)) {
            for (const auto& old_svc : old.services) {
                if (old_svc.service != s.name) continue;
                if (old_svc.config_hash == fingerprint) {
                    // Same config: keep the container if it exists in any state.
                    bool running = false;
                    if (auto lst = mgr.list(); lst.ok()) {
                        for (const auto& info : *lst.value)
                            if (info.name == ro.name) { running = true; break; }
                    }
                    bool exists = container_exists_any_state(ro.name);
                    if (running) {
                        needs_run = false;
                        messages.push_back("[ok] " + s.name + " up to date (" + ro.name + ")");
                    } else if (exists) {
                        // Stopped but unchanged: start it again, don't recreate.
                        needs_run = false;
                        auto started = cli_container({"start", ro.name});
                        if (started.ok() && *started.value) {
                            messages.push_back("[started] " + s.name + " (" + ro.name + ")");
                        } else {
                            messages.push_back("[warn] could not start " + ro.name +
                                               "; remove it and re-run `hostely app up`");
                        }
                    }
                }
            }
        }

        // Ports: auto-pick when the wanted host port is taken. For a service
        // being recreated, the stale container was already stopped above.
        for (auto& [h, c] : s.ports) {
            int wanted = atoi(h.c_str());
            if (wanted <= 0) { ro.ports.emplace_back(h, c); continue; }
            int actual = wanted;
            if (port_in_use(wanted)) {
                actual = pick_free_host_port(wanted + 1);
                messages.push_back("[info] host port " + std::to_string(wanted) +
                                   " in use; " + s.name + " uses " +
                                   std::to_string(actual) + " instead");
            }
            ro.ports.emplace_back(std::to_string(actual), c);
        }

        // Unchanged services keep the ports recorded in state (stable across
        // reconciles even if something else transiently held the port).
        if (!needs_run && old.services.size() > 0) {
            for (const auto& old_svc : old.services) {
                if (old_svc.service == s.name) ro.ports = old_svc.ports;
            }
        }

        AppServiceState st;
        st.service = s.name;
        st.container = ro.name;
        st.image = s.image;
        st.config_hash = fingerprint;
        st.ports = ro.ports;
        if (needs_run) {
            // Stop + remove any stale container with this name (config
            // changed, or a previous failed run left one behind).
            (void)mgr.stop(ro.name);
            (void)rm_container(mgr, ro.name);
            auto r = mgr.run(ro);
            if (!r.ok()) {
                err = {"failed to start " + s.name + ": " + r.error->message};
                if (!r.error->stderr_tail.empty()) err.message += "\n" + r.error->stderr_tail;
                // Deliberately do NOT save_state here: the previous state file
                // still describes the containers that are actually running.
                return std::nullopt;
            }
            messages.push_back("[up] " + s.name + "  (" + ro.name + ", id " +
                               r.value->substr(0, 12) + ")");
        }
        state.services.push_back(st);
    }

    if (!save_state(state, err)) return std::nullopt;
    return true;
}

std::optional<bool> app_stop(const std::string& app, services::Manager& mgr,
                             std::vector<std::string>& messages, AppError& err) {
    AppState st;
    if (!load_state(app, st)) {
        err = {"no known app named '" + app + "' (nothing was ever `hostely app up`'d from it)"};
        return std::nullopt;
    }
    for (auto it = st.services.rbegin(); it != st.services.rend(); ++it) {
        auto r = mgr.stop(it->container);
        if (r.ok()) messages.push_back("[stopped] " + it->container);
        else messages.push_back("[warn] " + it->container + ": " + r.error->message);
    }
    return true;
}

std::optional<std::vector<std::pair<std::string, std::vector<std::string>>>>
app_ps(services::Manager& mgr, AppError& err) {
    auto dir = apps_state_dir();
    std::vector<std::pair<std::string, std::vector<std::string>>> out;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.path().extension().empty() && entry.path().extension() != ".json") continue;
        std::string app = entry.path().stem().string();
        AppState st;
        if (!load_state(app, st)) continue;
        std::vector<std::string> lines;
        for (const auto& svc : st.services) {
            std::string status = "?";
            if (auto lst = mgr.list(); lst.ok()) {
                for (const auto& info : *lst.value)
                    if (info.name == svc.container) { status = info.status; break; }
            }
            std::string ports;
            for (const auto& [h, c] : svc.ports)
                ports += (ports.empty() ? "" : ", ") + h + "->" + c;
            lines.push_back(svc.container + "  " + status + "  " + ports);
        }
        out.emplace_back(app, std::move(lines));
    }
    return out;
}

std::optional<std::string> app_logs(const std::string& app,
                                    const std::optional<std::string>& service,
                                    bool follow, services::Manager& mgr,
                                    AppError& err) {
    AppState st;
    if (!load_state(app, st)) {
        err = {"no known app named '" + app + "'"};
        return std::nullopt;
    }
    std::string text;
    for (const auto& svc : st.services) {
        if (service && *service != svc.service && *service != svc.container) continue;
        auto r = mgr.logs(svc.container, follow);
        if (!r.ok()) {
            err = {svc.container + ": " + r.error->message};
            return std::nullopt;
        }
        text += "── " + svc.container + " ──\n" + *r.value + "\n";
        if (follow) break;   // `container logs -f` blocks; only one can run
    }
    return text;
}

std::optional<bool> app_rm(const std::string& app, services::Manager& mgr,
                           std::vector<std::string>& messages, AppError& err) {
    auto stop = app_stop(app, mgr, messages, err);
    if (!stop) return std::nullopt;
    AppState st;
    load_state(app, st);
    for (const auto& svc : st.services) {
        auto r = rm_container(mgr, svc.container);
        if (r.ok() && *r.value)
            messages.push_back("[removed] " + svc.container);
        else
            messages.push_back("[warn] could not remove " + svc.container);
    }
    std::error_code ec;
    std::filesystem::remove(app_state_file(app), ec);
    return true;
}

}  // namespace hostely::apps