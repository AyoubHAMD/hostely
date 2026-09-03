#pragma once

#include <optional>
#include <string>
#include <vector>

namespace hostely::services {

/// What `hostely run` accepts.
struct RunOptions {
    std::string name;                 // required
    std::string image;                // required, e.g. "nginx:alpine"
    std::vector<std::pair<std::string, std::string>> ports;   // host:container
    std::vector<std::pair<std::string, std::string>> env;     // KEY=VALUE
    std::vector<std::pair<std::string, std::string>> volumes; // src:dst
    std::vector<std::string>          args;     // appended after image
    std::vector<std::string>          dns;      // --dns nameserver IPs
    std::string                       memory;   // --memory, e.g. "8g"; empty = container default
};

/// What `hostely ps` returns, one row per service.
struct ServiceInfo {
    std::string id;          // container ID (short form)
    std::string name;
    std::string image;
    std::string status;      // "running", "stopped", etc.
    std::string ports;       // human-readable
};

/// Why a manager call didn't succeed.
struct ManagerError {
    enum class Kind {
        CliNotFound,
        SystemNotStarted,
        BadInvocation,
        NonZeroExit,
    };
    Kind        kind;
    std::string message;     // short, user-facing
    std::string stderr_tail; // last ~2 KiB of the child's stderr
};

/// Outcome of a manager call. Either it succeeded (and `value` is set) or
/// it failed (and `error` is set).
template <typename T>
struct Outcome {
    std::optional<T>         value;
    std::optional<ManagerError> error;
    bool ok() const { return value.has_value(); }
};

/// Manager talks to the Apple `container` CLI. One instance per CLI path
/// (typically just one — set by config).
class Manager {
public:
    explicit Manager(std::string cli_path);

    /// Is the CLI binary resolvable? (Cheap; just checks PATH.)
    bool available() const { return !cli_path_.empty(); }

    /// Detect whether `container system start` has been run yet. The CLI
    /// emits a specific error if not. Returns true if the system is ready,
    /// false otherwise. If false, the error message includes the hint.
    Outcome<bool> system_is_ready();

    /// Run a service per `opts`. Returns the new container ID.
    Outcome<std::string> run(const RunOptions& opts);

    /// List services owned by hostely. We filter by name prefix when possible
    /// (container CLI supports `-q` quiet + `--format`).
    Outcome<std::vector<ServiceInfo>> list();

    /// Stop a service by name. Idempotent: stopping a stopped service is OK.
    Outcome<bool> stop(const std::string& name);

    /// Stream logs for a service. We *don't* stream here — we capture and
    /// return, so the CLI can render. `follow=true` means block until the
    /// service exits (similar to `docker logs --follow`).
    Outcome<std::string> logs(const std::string& name, bool follow = false);

private:
    std::string cli_path_;
};

/// Print the install instructions for the Apple `container` CLI.
std::string install_instructions();

}  // namespace hostely::services
