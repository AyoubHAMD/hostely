#include "services/manager.hpp"

#include "services/cli.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace hostely::services {

namespace {

constexpr std::string_view kInstallUrl =
    "https://github.com/apple/container/releases";

std::string tail(std::string s, std::size_t n) {
    if (s.size() <= n) return s;
    return s.substr(s.size() - n);
}

bool contains_ci(std::string_view hay, std::string_view needle) {
    if (needle.size() > hay.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= hay.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            char a = hay[i + j];
            char b = needle[j];
            if (std::tolower(static_cast<unsigned char>(a)) !=
                std::tolower(static_cast<unsigned char>(b))) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

// Parse `container ls` table output. Columns we know on macOS 15:
//   ID   IMAGE   OS   ARCH   STATE    ADDR          PORTS   NAMES
// (NAMES is the last column; STATE is "running" / "stopped" / ...)
//
// We do a best-effort parse: skip the header row, then take the first token
// as ID and the last token as the name. We re-join the middle columns into
// the "image" and "ports" fields loosely. This is OK for v1; we'll tighten
// if/when Apple adds `--format json` to macOS 15.
std::vector<ServiceInfo> parse_ls(const std::string& text) {
    std::vector<ServiceInfo> out;
    std::istringstream iss(text);
    std::string line;
    bool first = true;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        // Skip the header row and any "----..." separator.
        if (first) { first = false; continue; }
        if (line.find_first_not_of(" -") == std::string::npos) continue;

        std::istringstream ls(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (ls >> tok) tokens.push_back(std::move(tok));
        if (tokens.size() < 3) continue;

        ServiceInfo s;
        s.id    = tokens.front();
        s.image = tokens[1];
        // STATE is somewhere in the middle; pick it by best-effort heuristics.
        for (std::size_t i = 2; i + 1 < tokens.size(); ++i) {
            const auto& t = tokens[i];
            if (t == "running" || t == "stopped" || t == "starting" ||
                t == "stopping" || t == "created") {
                s.status = t;
                break;
            }
        }
        if (s.status.empty() && tokens.size() >= 4) s.status = tokens[3];

        // NAMES is conventionally the last token.
        s.name = tokens.back();
        // If more than 2 trailing tokens exist, we can guess "ports" as the
        // second-to-last. Be conservative: only set if it looks like a port.
        if (tokens.size() >= 3) {
            const auto& maybe_ports = tokens[tokens.size() - 2];
            if (maybe_ports.find(':') != std::string::npos ||
                maybe_ports == "-" ||
                maybe_ports.find_first_of("0123456789") != std::string::npos) {
                s.ports = maybe_ports;
            }
        }
        out.push_back(std::move(s));
    }
    return out;
}

// Shared body for `container <args...> --format json`: same guards and
// error mapping as list(), returns the raw stdout JSON. Callers own parsing.
Outcome<std::string> run_json(const std::string& cli_path_,
                              const std::vector<std::string>& args,
                              const char* what) {
    Outcome<std::string> out;
    if (cli_path_.empty()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 install_instructions(), {}};
        return out;
    }
    std::vector<std::string> argv{cli_path_};
    argv.insert(argv.end(), args.begin(), args.end());
    auto r = cli::run(argv);
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        std::string err = tail(r.stderr_text, 2048);
        if (contains_ci(err, "system start")) {
            out.error = ManagerError{ManagerError::Kind::SystemNotStarted,
                "the container system has not been started.\n"
                "  Run `container system start` once, then retry.",
                err};
        } else {
            out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                     what, err};
        }
        return out;
    }
    out.value = r.stdout_text;
    return out;
}

}  // namespace

std::string install_instructions() {
    std::ostringstream oss;
    oss <<
        "Apple's `container` CLI is not installed.\n"
        "Install:\n"
        "  1. Download `container-installer-signed.pkg` from:\n"
        "       " << kInstallUrl << "\n"
        "  2. Run the installer (admin password required).\n"
        "  3. Start the system once:\n"
        "       container system start\n"
        "\n"
        "hostely will then be able to manage containers on this Mac.";
    return oss.str();
}

Manager::Manager(std::string cli_path) : cli_path_(std::move(cli_path)) {}

Outcome<bool> Manager::system_is_ready() {
    Outcome<bool> out;
    if (!available()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "container CLI not on PATH", {}};
        return out;
    }
    // `container system start` is a no-op if already started; safe to call.
    auto r = cli::run({cli_path_, "system", "start"});
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                 "`container system start` failed",
                                 tail(r.stderr_text, 2048)};
        return out;
    }
    out.value = true;
    return out;
}

Outcome<std::string> Manager::run(const RunOptions& opts) {
    Outcome<std::string> out;
    if (!available()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 install_instructions(), {}};
        return out;
    }
    if (opts.name.empty() || opts.image.empty()) {
        out.error = ManagerError{ManagerError::Kind::BadInvocation,
                                 "both --name and --image are required", {}};
        return out;
    }

    std::vector<std::string> argv{cli_path_, "run", "-d", "--name", opts.name};
    for (const auto& [host, container] : opts.ports) {
        argv.push_back("-p");
        argv.push_back(host + ":" + container);
    }
    for (const auto& [k, v] : opts.env) {
        argv.push_back("-e");
        argv.push_back(k + "=" + v);
    }
    for (const auto& [src, dst] : opts.volumes) {
        argv.push_back("-v");
        argv.push_back(src + ":" + dst);
    }
    for (const auto& ns : opts.dns) {
        argv.push_back("--dns");
        argv.push_back(ns);
    }
    if (!opts.memory.empty()) {
        argv.push_back("--memory");
        argv.push_back(opts.memory);
    }
    argv.push_back(opts.image);
    for (const auto& a : opts.args) argv.push_back(a);

    auto r = cli::run(argv);
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        std::string err = tail(r.stderr_text, 2048);
        if (contains_ci(err, "system start")) {
            out.error = ManagerError{ManagerError::Kind::SystemNotStarted,
                "the container system has not been started.\n"
                "  Run `container system start` once, then retry.",
                err};
        } else {
            out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                     "container run failed", err};
        }
        return out;
    }

    // `container run -d` prints the container ID to stdout, optionally with
    // a trailing newline. Trim.
    std::string id = r.stdout_text;
    while (!id.empty() && (id.back() == '\n' || id.back() == '\r' || id.back() == ' '))
        id.pop_back();
    out.value = std::move(id);
    return out;
}

Outcome<std::vector<ServiceInfo>> Manager::list() {
    Outcome<std::vector<ServiceInfo>> out;
    if (!available()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 install_instructions(), {}};
        return out;
    }
    auto r = cli::run({cli_path_, "ls"});
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        std::string err = tail(r.stderr_text, 2048);
        if (contains_ci(err, "system start")) {
            out.error = ManagerError{ManagerError::Kind::SystemNotStarted,
                "the container system has not been started.\n"
                "  Run `container system start` once, then retry.",
                err};
        } else {
            out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                     "container ls failed", err};
        }
        return out;
    }
    out.value = parse_ls(r.stdout_text);
    return out;
}

Outcome<std::string> Manager::list_json() {
    return run_json(cli_path_, {"ls", "--format", "json"}, "container ls failed");
}

Outcome<std::string> Manager::stats_json() {
    return run_json(cli_path_, {"stats", "--format", "json", "--no-stream"},
                    "container stats failed");
}

Outcome<bool> Manager::stop(const std::string& name) {
    Outcome<bool> out;
    if (!available()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 install_instructions(), {}};
        return out;
    }
    if (name.empty()) {
        out.error = ManagerError{ManagerError::Kind::BadInvocation,
                                 "service name is required", {}};
        return out;
    }
    auto r = cli::run({cli_path_, "stop", name});
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        // "not found" is acceptable for idempotency.
        std::string err = tail(r.stderr_text, 2048);
        if (contains_ci(err, "not found") || contains_ci(err, "no such")) {
            out.value = true;   // already stopped
            return out;
        }
        out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                 "container stop failed", err};
        return out;
    }
    out.value = true;
    return out;
}

Outcome<std::string> Manager::logs(const std::string& name, bool follow) {
    Outcome<std::string> out;
    if (!available()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 install_instructions(), {}};
        return out;
    }
    if (name.empty()) {
        out.error = ManagerError{ManagerError::Kind::BadInvocation,
                                 "service name is required", {}};
        return out;
    }
    std::vector<std::string> argv{cli_path_, "logs"};
    if (follow) argv.push_back("-f");
    argv.push_back(name);
    auto r = cli::run(argv);
    if (!r.started()) {
        out.error = ManagerError{ManagerError::Kind::CliNotFound,
                                 "could not launch container CLI", r.stderr_text};
        return out;
    }
    if (r.exit_code != 0) {
        out.error = ManagerError{ManagerError::Kind::NonZeroExit,
                                 "container logs failed",
                                 tail(r.stderr_text, 2048)};
        return out;
    }
    out.value = r.stdout_text;
    return out;
}

}  // namespace hostely::services
