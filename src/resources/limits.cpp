#include "resources/limits.hpp"

// syscall(2) is deprecated on macOS 10.12+, but it is the only public way to
// reach memorystatus_control from a non-entitled binary. We use it once per
// `hostely run --mem-gb` invocation and ignore the deprecation warning.
// Remove this pragma when Apple ships a public replacement.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#pragma clang diagnostic pop

#include <cerrno>
#include <cstring>
#include <string>

namespace hostely::resources {

namespace {

// Jetsam (memorystatus_control) constants. These are *not* exposed by the
// macOS public SDK — they live in the private header kern_memorystatus.h.
// Values verified against xnu-3247.x source (newosxbook.com mirror).
//
// IMPORTANT: these are kernel-private and may change between macOS versions.
// When the call fails with ENOTSUP or EINVAL, callers should treat the
// memory limit as advisory only.
constexpr int kMemorystatusCmdSetMemlimitProperties = 6;

struct MemlimitProperties {
    int32_t memlimit_active;
    int32_t memlimit_inactive;
};

}  // namespace

LimitReport apply_process_limits(int mem_gb, int cpu_seconds, int nofile) {
    LimitReport r;
    r.running_as_root = (::geteuid() == 0);

    // ---- RLIMIT_CPU (wall-time seconds) ------------------------------------
    if (cpu_seconds > 0) {
        struct rlimit rl{};
        rl.rlim_cur = static_cast<rlim_t>(cpu_seconds);
        // Hard limit slightly above soft so a SIGXCPU handler could log
        // before SIGKILL. Use 2x as a cushion.
        rl.rlim_max = static_cast<rlim_t>(cpu_seconds * 2);
        if (::setrlimit(RLIMIT_CPU, &rl) == 0) {
            r.cpu_seconds_set   = true;
            r.cpu_seconds_value = cpu_seconds;
        }
    }

    // ---- RLIMIT_NOFILE -----------------------------------------------------
    if (nofile > 0) {
        struct rlimit rl{};
        rl.rlim_cur = static_cast<rlim_t>(nofile);
        rl.rlim_max = static_cast<rlim_t>(nofile);
        // Hard limit raises require root; ignore failure quietly.
        if (::setrlimit(RLIMIT_NOFILE, &rl) == 0) {
            r.nofile_set   = true;
            r.nofile_value = nofile;
        }
    }

    // ---- Jetsam memlimit (root + entitlement required) --------------------
    if (mem_gb > 0) {
        r.jetsam_attempted = true;
        const int32_t bytes = mem_gb * 1024 * 1024 * 1024;
        MemlimitProperties props{};
        props.memlimit_active   = bytes;
        props.memlimit_inactive = bytes;

        // syscall(2) is deprecated on macOS 10.12+, but it is the only public
        // way to reach memorystatus_control from a non-entitled binary. We
        // suppress the warning here; remove when Apple ships a replacement.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        long rc = ::syscall(SYS_memorystatus_control,
                            kMemorystatusCmdSetMemlimitProperties,
                            getpid(), 0, &props, sizeof(props));
#pragma clang diagnostic pop

        if (rc == 0) {
            r.jetsam_applied = true;
        } else {
            r.jetsam_error = std::string(::strerror(errno)) +
                             " (errno " + std::to_string(errno) + ")";
            // EPERM = not root / not entitled.
            // ENOTSUP = kernel doesn't expose the command (older macOS or sandbox).
            // EINVAL = argument rejected.
        }
    }

    return r;
}

}  // namespace hostely::resources
