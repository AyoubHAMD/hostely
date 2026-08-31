#include "resources/system.hpp"

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

namespace hostely::resources {

namespace {

// sysctlbyname helper that returns false on failure.
template <typename T>
bool sysctl_u64(const char* name, T& out) {
    T value = 0;
    size_t size = sizeof(T);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return false;
    out = value;
    return true;
}

bool sysctl_int(const char* name, int& out) {
    int value = 0;
    size_t size = sizeof(int);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return false;
    out = value;
    return true;
}

}  // namespace

MemoryStats read_memory() {
    MemoryStats m;
    std::uint64_t total = 0;
    if (sysctl_u64("hw.memsize", total)) m.total_bytes = total;

    // host_statistics64 returns page counts; convert with hw.memsize / page_size.
    mach_port_t host = mach_host_self();
    vm_size_t   page_size = 0;
    if (host_page_size(host, &page_size) == KERN_SUCCESS && page_size > 0) {
        vm_statistics64_data_t vm{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vm), &count)
                == KERN_SUCCESS) {
            m.free_bytes       = static_cast<std::uint64_t>(vm.free_count)       * page_size;
            m.active_bytes     = static_cast<std::uint64_t>(vm.active_count)     * page_size;
            m.wired_bytes      = static_cast<std::uint64_t>(vm.wire_count)       * page_size;
            m.inactive_bytes   = static_cast<std::uint64_t>(vm.inactive_count)   * page_size;
            m.compressed_bytes = static_cast<std::uint64_t>(vm.compressor_page_count) * page_size;
            m.valid = true;
        }
    }
    return m;
}

CpuStats read_cpu() {
    CpuStats c;
    sysctl_int("hw.logicalcpu",  c.logical_cores);
    sysctl_int("hw.physicalcpu", c.physical_cores);

    double loads[3] = {0, 0, 0};
    if (getloadavg(loads, 3) == 3) {
        c.load_1min  = loads[0];
        c.load_5min  = loads[1];
        c.load_15min = loads[2];
    }

    // Per-self stats. mach_task_self() returns the task port for this process.
    mach_port_t self = mach_task_self();

    // Use MACH_TASK_BASIC_INFO (flavor 20): returns an in-line struct,
    // unlike TASK_BASIC_INFO_64 whose typedef is `task_basic_info_64 *` —
    // that's a legacy "kernel allocates" API and on Apple Silicon it
    // refuses to allocate, crashing the caller.
    mach_task_basic_info_data_t basic{};
    mach_msg_type_number_t basic_count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(self, MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&basic), &basic_count) == KERN_SUCCESS) {
        c.self.rss_bytes     = basic.resident_size;
        c.self.virtual_bytes = basic.virtual_size;
    }

    task_thread_times_info_data_t ttimes{};
    mach_msg_type_number_t ttimes_count = TASK_THREAD_TIMES_INFO_COUNT;
    if (task_info(self, TASK_THREAD_TIMES_INFO,
                  reinterpret_cast<task_info_t>(&ttimes), &ttimes_count) == KERN_SUCCESS) {
        c.self.user_time_us   = ttimes.user_time.seconds   * 1000000ULL
                              + ttimes.user_time.microseconds;
        c.self.system_time_us = ttimes.system_time.seconds * 1000000ULL
                              + ttimes.system_time.microseconds;
    }

    c.valid = true;
    return c;
}

std::string human_bytes(std::uint64_t bytes) {
    char buf[32];
    static const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    double v = static_cast<double>(bytes);
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    if (u == 0) std::snprintf(buf, sizeof(buf), "%llu %s",
                              static_cast<unsigned long long>(bytes), units[u]);
    else        std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return buf;
}

}  // namespace hostely::resources
