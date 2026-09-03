#pragma once

// Sampling layer for `hostely top`: turns raw `container ls/stats` JSON into
// per-container rows with instantaneous rates, and holds the mutex-protected
// snapshot shared between the sampler thread and the renderer.

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace hostely::services { class Manager; }

namespace hostely::top {

struct ContainerRow {
    std::string name;      // container id (stats/ls both key on it)
    std::string image;     // image reference, host prefix stripped
    std::string state;     // "running", ...
    std::string ip;        // container ipv4, "" if none
    std::string ports;     // published ports, human readable
    std::uint64_t cpus = 0;    // cpu limit (0 = unset)
    std::uint64_t mem_bytes = 0;
    std::uint64_t mem_limit = 0;   // 0 = unknown
    std::uint32_t procs = 0;
    // Instantaneous values; -1 means unknown this sample (first sample, or
    // the container vanished). cpu_pct is % of ONE core and can exceed 100.
    double cpu_pct = -1;
    double mem_pct = -1;
    double net_rx_bps = -1, net_tx_bps = -1;
    double blk_rd_bps = -1, blk_wr_bps = -1;
};

struct HostSummary {
    std::uint64_t mem_total = 0, mem_used = 0;
    int cores = 0;
    double load1 = 0, load5 = 0, load15 = 0;
};

struct Snapshot {
    std::vector<ContainerRow> rows;
    HostSummary host;
    std::string error;          // non-empty = last CLI failure; shown as banner
    std::uint32_t missed = 0;   // consecutive stats passes with no output
    int sample_count = 0;
    double interval_sec = 2.0;
};

enum class SortKey { Cpu, Mem, Name, Net };

// Thread-safe shared state between the sampler thread and the renderer.
class SnapshotStore {
public:
    void store(Snapshot snap);           // lock, copy-swap
    Snapshot load() const;

    void set_interval(double sec);
    double interval() const;

    std::atomic<bool> stop{false};
    std::atomic<int>  sort{0};           // SortKey ordinal
    std::atomic<int>  selected{0};       // row index into *rendered* order
    std::atomic<int>  view_offset{0};    // first visible row (scroll)

private:
    mutable std::mutex mu_;
    Snapshot snap_;
    double interval_ = 2.0;              // guarded by mu_
};

// Raw cumulative counters for one container from a stats pass.
struct RawSample {
    std::uint64_t cpu_usec = 0;
    std::uint64_t net_rx = 0, net_tx = 0;
    std::uint64_t blk_rd = 0, blk_wr = 0;
    std::uint64_t mem_bytes = 0, mem_limit = 0;
    std::uint32_t procs = 0;
};

// One blocking sample pass: stats JSON always, ls JSON every `list_every`
// passes (or when stats reports an id missing from the cached list).
// `prev` carries the last raw counters per container for delta math and is
// updated in place. Returns the fresh snapshot (also stored into `store`).
Snapshot sample_once(services::Manager& mgr,
                     const std::map<std::string, RawSample>& prev,
                     std::map<std::string, RawSample>& next,
                     const std::string& cached_list_json,
                     std::string& list_json_out,   // updated cache, "" until first ls
                     int sample_count,
                     double interval_sec,
                     SnapshotStore& store,
                     int list_every = 5);

}  // namespace hostely::top