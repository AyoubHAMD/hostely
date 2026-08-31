#pragma once

#include <cstdint>
#include <string>

namespace hostely::resources {

/// What `hostely run --mem-gb N` actually applied.
struct LimitReport {
    bool cpu_seconds_set       = false;
    int  cpu_seconds_value     = 0;
    bool nofile_set            = false;
    int  nofile_value          = 0;
    bool jetsam_attempted      = false;
    bool jetsam_applied        = false;
    std::string jetsam_error;  // populated when jetsam_attempted && !jetsam_applied

    /// Whether we are running as root (uid 0). Jetsam limits only work
    /// as root; we expose this so callers can tell the user "re-run with sudo".
    bool running_as_root = false;
};

/// Apply process resource limits to the current process. Should be called
/// from the child immediately after fork() but before exec(); or in the
/// container-run shim before shelling to `container run`.
///
/// mem_gb <= 0 means "don't apply a Jetsam limit".
/// cpu_seconds <= 0 means "don't apply RLIMIT_CPU".
LimitReport apply_process_limits(int mem_gb = -1, int cpu_seconds = -1,
                                 int nofile  = 65536);

}  // namespace hostely::resources
