#pragma once

#include <cstdint>
#include <string>

namespace hostely::resources {

struct MemoryStats {
    std::uint64_t total_bytes = 0;       // hw.memsize
    std::uint64_t free_bytes  = 0;       // host_statistics64.free_count * page_size
    std::uint64_t active_bytes = 0;     // actively used by processes
    std::uint64_t wired_bytes  = 0;      // kernel / locked
    std::uint64_t inactive_bytes = 0;    // reclaimable
    std::uint64_t compressed_bytes = 0; // swapped/compressed

    /// True if the read succeeded (some macOS configs can deny host_statistics64).
    bool valid = false;
};

struct CpuStats {
    int           logical_cores = 0;   // hw.logicalcpu
    int           physical_cores = 0;  // hw.physicalcpu
    double        load_1min      = 0;  // getloadavg
    double        load_5min      = 0;
    double        load_15min     = 0;

    /// Per-process CPU usage since last call (or process start).
    struct ProcSelf {
        std::uint64_t user_time_us   = 0;
        std::uint64_t system_time_us = 0;
        std::uint64_t rss_bytes      = 0;   // resident memory
        std::uint64_t virtual_bytes  = 0;
    } self;

    bool valid = false;
};

/// Read system-wide memory stats. Cheap (two syscalls + one mach call).
MemoryStats read_memory();

/// Read CPU + per-process self stats. Cheap (sysctls + mach call).
CpuStats     read_cpu();

/// Format a byte count as a human-readable string ("22.3 GB", "851 MB").
std::string  human_bytes(std::uint64_t bytes);

}  // namespace hostely::resources
