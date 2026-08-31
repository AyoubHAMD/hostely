#pragma once

#include <optional>
#include <string>
#include <vector>

namespace hostely::services::cli {

/// Captured result of running a subprocess.
struct Result {
    int          exit_code;   // -1 if the process could not be started
    std::string  stdout_text; // exact bytes the child wrote to stdout
    std::string  stderr_text; // exact bytes the child wrote to stderr
    bool started() const { return exit_code >= 0; }
    bool ok()      const { return exit_code == 0; }
};

/// Run `argv[0]` with the given argv vector (must be NUL-terminated
/// internally; this function takes a vector and writes its own C strings).
/// `argv[0]` is the program name. The first element must be a path or a name
/// resolvable via PATH.
///
/// The child inherits the parent's environment. stdout and stderr are
/// captured into Result.{stdout_text, stderr_text}.
///
/// On failure to fork/exec the child, returns Result{ -1, "", "" }.
Result run(const std::vector<std::string>& argv);

/// Convenience: is `program_name` (e.g. "container") resolvable on PATH?
/// Returns the absolute path if found, std::nullopt otherwise.
std::optional<std::string> which(const std::string& program_name);

}  // namespace hostely::services::cli
