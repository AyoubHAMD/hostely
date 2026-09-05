#include "top/sample.hpp"

#include "inference/lockfile.hpp"
#include "paths.hpp"
#include "resources/metal_probe.hpp"
#include "resources/system.hpp"
#include "services/manager.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <sys/sysctl.h>

namespace hostely::top {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// SnapshotStore
// ---------------------------------------------------------------------------

void SnapshotStore::store(Snapshot snap) {
    std::lock_guard<std::mutex> lk(mu_);
    snap_ = std::move(snap);
}

Snapshot SnapshotStore::load() const {
    std::lock_guard<std::mutex> lk(mu_);
    return snap_;
}

void SnapshotStore::set_interval(double sec) {
    if (sec < 0.2) sec = 0.2;
    if (sec > 60.0) sec = 60.0;
    std::lock_guard<std::mutex> lk(mu_);
    interval_ = sec;
}

double SnapshotStore::interval() const {
    std::lock_guard<std::mutex> lk(mu_);
    return interval_;
}

// ---------------------------------------------------------------------------
// JSON -> rows
// ---------------------------------------------------------------------------

namespace {

// "docker.io/library/redis:latest" -> "redis:latest"; keep everything after
// the last '/' of the reference, but preserve a leading org like "ghcr.io/x/y".
std::string short_image(const std::string& reference) {
    auto slash = reference.rfind('/');
    if (slash == std::string::npos) return reference;
    return reference.substr(slash + 1);
}

std::string strip_cidr(const std::string& ip) {
    auto slash = ip.find('/');
    return slash == std::string::npos ? ip : ip.substr(0, slash);
}

// Best-effort fill of static fields (image/state/ports/ip/limits) from one
// `container ls --format json` element.
void apply_list_json(ContainerRow& row, const json& entry) {
    if (!entry.is_object()) return;
    const auto& cfg = entry.value("configuration", json::object());
    if (!cfg.is_object()) return;

    const auto& image = cfg.value("image", json::object());
    if (image.is_object()) {
        row.image = short_image(image.value("reference", row.image));
    }
    row.name = entry.value("id", row.name);
    if (row.name.empty()) row.name = cfg.value("id", row.name);

    const auto& status = entry.value("status", json::object());
    if (status.is_object()) {
        row.state = status.value("state", row.state);
        const auto& nets = status.value("networks", json::array());
        if (nets.is_array() && !nets.empty() && nets[0].is_object()) {
            row.ip = strip_cidr(nets[0].value("ipv4Address", ""));
        }
    }

    const auto& ports = cfg.value("publishedPorts", json::array());
    if (ports.is_array()) {
        std::string out;
        for (const auto& p : ports) {
            if (!p.is_object()) continue;
            auto hp = p.value("hostPort", 0);
            auto cp = p.value("containerPort", 0);
            auto proto = p.value("proto", "tcp");
            if (!out.empty()) out += ", ";
            out += std::to_string(hp) + "->" + std::to_string(cp) + "/" + proto;
        }
        row.ports = out;
    }

    const auto& res = cfg.value("resources", json::object());
    if (res.is_object()) {
        row.cpus = res.value("cpuOverhead", 0u) + res.value("cpus", 0u);
        row.mem_limit = res.value("memoryInBytes", 0ull);
    }
}

// Stats pass JSON (flat array) -> raw counters keyed by container id.
std::map<std::string, RawSample> parse_stats(const std::string& text) {
    std::map<std::string, RawSample> out;
    json arr = json::parse(text, nullptr, false);
    if (arr.is_discarded() || !arr.is_array()) return out;
    for (const auto& e : arr) {
        if (!e.is_object()) continue;
        auto id = e.value("id", "");
        if (id.empty()) continue;
        RawSample r;
        r.cpu_usec = e.value("cpuUsageUsec", 0ull);
        r.net_rx   = e.value("networkRxBytes", 0ull);
        r.net_tx   = e.value("networkTxBytes", 0ull);
        r.blk_rd   = e.value("blockReadBytes", 0ull);
        r.blk_wr   = e.value("blockWriteBytes", 0ull);
        r.mem_bytes = e.value("memoryUsageBytes", 0ull);
        r.mem_limit = e.value("memoryLimitBytes", 0ull);
        r.procs     = e.value("numProcesses", 0u);
        out[id] = r;
    }
    return out;
}

// "total = 1024.00M used = 64.00M free = 960.00M" (sysctl vm.swapusage)
void parse_swap_usage(std::uint64_t& total, std::uint64_t& used) {
    char buf[256] = {0};
    std::size_t len = sizeof buf - 1;
    if (sysctlbyname("vm.swapusage", buf, &len, nullptr, 0) != 0) return;
    auto parse_one = [&](const std::string& key) -> std::uint64_t {
        const char* hit = std::strstr(buf, key.c_str());
        if (!hit) return 0;
        double v = std::strtod(hit + key.size(), nullptr);
        // multiplier letter follows the number ("1024.00M")
        const char* p = std::strpbrk(hit + key.size(), "BKMGTP");
        double mult = 1024.0 * 1024.0;   // vm.swapusage reports M
        if (p) {
            switch (*p) {
                case 'B': mult = 1; break;
                case 'K': mult = 1024; break;
                case 'M': mult = 1024.0 * 1024; break;
                case 'G': mult = 1024.0 * 1024 * 1024; break;
                case 'T': mult = 1024.0 * 1024 * 1024 * 1024ull; break;
                default: break;
            }
        }
        return static_cast<std::uint64_t>(v * mult);
    };
    total = parse_one("total = ");
    used = parse_one("used = ");
}

}  // namespace

// ---------------------------------------------------------------------------
// sample_once
// ---------------------------------------------------------------------------

Snapshot sample_once(services::Manager& mgr,
                     const std::map<std::string, RawSample>& prev,
                     std::map<std::string, RawSample>& next,
                     const std::string& cached_list_json,
                     std::string& list_json_out,
                     int sample_count,
                     double interval_sec,
                     SnapshotStore& store,
                     int list_every) {
    Snapshot snap;
    snap.interval_sec = interval_sec;
    snap.sample_count = sample_count;
    snap.host = [] {
        HostSummary h;
        auto m = resources::read_memory();
        auto c = resources::read_cpu();
        h.mem_total = m.total_bytes;
        h.mem_used = m.active_bytes + m.wired_bytes + m.compressed_bytes;
        // Same serve-headroom formula `hostely status` reports: free + inactive
        // minus a 2 GiB safety margin.
        std::uint64_t freeable = m.free_bytes + m.inactive_bytes;
        const std::uint64_t safety = 2ull * 1024 * 1024 * 1024;
        h.avail_bytes = freeable > safety ? freeable - safety : 0;
        h.cores = c.logical_cores;
        h.load1 = c.load_1min; h.load5 = c.load_5min; h.load15 = c.load_15min;

        // Metal ceiling (host-level; containers have no GPU access at all).
        h.gpu_available = hostely_metal_available() != 0;
        if (h.gpu_available) {
            h.gpu_recommended = hostely_metal_recommended_max_bytes();
        }

        // Serving model from the serve lockfile (stale lock = not serving).
        std::string lock_err;
        bool stale = false;
        if (auto lock = inference::read_serve_lock(lock_err, stale);
            lock && !stale && lock_err.empty()) {
            h.serving = true;
            h.model_name = lock->display_name.empty() ? lock->model_path
                                                      : lock->display_name;
            h.model_rss = lock->peak_rss_bytes;
        }

        parse_swap_usage(h.swap_total, h.swap_used);
        return h;
    }();

    // ---- stats (always) --------------------------------------------------
    auto stats = mgr.stats_json();
    if (!stats.ok()) {
        // Keep the rows we already know so the table doesn't flash empty.
        snap.error = stats.error->message;
        auto cached = store.load();
        snap.rows = std::move(cached.rows);
        snap.missed = cached.missed + 1;
        store.store(std::move(snap));
        return snap;
    }
    auto now_map = parse_stats(*stats.value);
    if (now_map.empty()) {
        auto cached = store.load();
        snap.rows = std::move(cached.rows);
        snap.missed = cached.missed + 1;
        store.store(std::move(snap));
        return snap;
    }

    // ---- ls (periodically, or when an id is unknown) ---------------------
    bool need_list = list_json_out.empty() || sample_count % list_every == 0;
    if (!cached_list_json.empty() || need_list) {
        // Cheap containment check: any stats id we haven't seen forces refresh.
        for (const auto& [id, _] : now_map) {
            (void)_;
            if (cached_list_json.find("\"id\"") == std::string::npos ||
                cached_list_json.find(id) == std::string::npos) {
                need_list = true;
                break;
            }
        }
    }
    if (need_list) {
        auto ls = mgr.list_json();
        if (ls.ok()) {
            list_json_out = *ls.value;
        } else {
            // Non-fatal: stats is the primary feed.
            if (list_json_out.empty()) snap.error = ls.error->message + " (ls)";
        }
    }

    // Static rows from the ls cache.
    std::map<std::string, ContainerRow> statics;
    if (!list_json_out.empty()) {
        json arr = json::parse(list_json_out, nullptr, false);
        if (arr.is_array()) {
            for (const auto& e : arr) {
                ContainerRow row;
                apply_list_json(row, e);
                if (!row.name.empty()) statics[row.name] = std::move(row);
            }
        }
    }

    // ---- rows with deltas -------------------------------------------------
    // NOTE: the honest rate window is the wall time between consecutive stats
    // passes. The sampler schedules on `interval_sec`; when the CLI runs slow,
    // rates skew slightly. Acceptable for a dashboard, and cpu_pct sanity
    // checks guard against counter resets.
    const double window = interval_sec > 0.05 ? interval_sec : 2.0;
    for (const auto& [id, raw] : now_map) {
        ContainerRow row;
        auto sit = statics.find(id);
        if (sit != statics.end()) {
            row = sit->second;
            if (row.mem_limit == 0) row.mem_limit = raw.mem_limit;
        } else {
            row.name = id;
            row.state = "running";
            row.mem_limit = raw.mem_limit;
        }
        row.mem_bytes = raw.mem_bytes;
        row.procs = raw.procs;
        if (row.mem_limit > 0) row.mem_pct = 100.0 * raw.mem_bytes / row.mem_limit;

        auto pit = prev.find(id);
        if (pit != prev.end() && window > 0.05) {
            auto dcpu = static_cast<double>(raw.cpu_usec) -
                        static_cast<double>(pit->second.cpu_usec);
            if (dcpu >= 0) {
                row.cpu_pct = dcpu / (window * 1'000'000.0) * 100.0;
                if (row.cpu_pct > 1'000'000.0) row.cpu_pct = -1;  // counter reset
            }
            auto drx = static_cast<double>(raw.net_rx) - static_cast<double>(pit->second.net_rx);
            auto dtx = static_cast<double>(raw.net_tx) - static_cast<double>(pit->second.net_tx);
            if (drx >= 0) row.net_rx_bps = drx / window;
            if (dtx >= 0) row.net_tx_bps = dtx / window;
            auto drd = static_cast<double>(raw.blk_rd) - static_cast<double>(pit->second.blk_rd);
            auto dwr = static_cast<double>(raw.blk_wr) - static_cast<double>(pit->second.blk_wr);
            if (drd >= 0) row.blk_rd_bps = drd / window;
            if (dwr >= 0) row.blk_wr_bps = dwr / window;
        }
        snap.rows.push_back(std::move(row));
        next[id] = raw;
    }

    // Containers that left stats: keep them for `3` passes with unknown rates,
    // then drop.
    if (list_json_out.empty()) {
        // No ls yet — nothing to preserve.
    } else {
        for (const auto& [name, srow] : statics) {
            if (now_map.count(name)) continue;
            auto it = std::find_if(snap.rows.begin(), snap.rows.end(),
                                   [&](const ContainerRow& r) { return r.name == name; });
            if (it != snap.rows.end()) continue;
            ContainerRow row = srow;
            row.state = srow.state == "running" ? "stopped?" : srow.state;
            snap.rows.push_back(std::move(row));
        }
    }

    snap.missed = 0;
    store.store(snap);
    return snap;
}

}  // namespace hostely::top