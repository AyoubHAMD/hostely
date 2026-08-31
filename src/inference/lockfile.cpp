#include "inference/lockfile.hpp"

#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "log/logger.hpp"
#include "paths.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace hostely::inference {

namespace {

bool pid_alive(std::int64_t pid) {
    if (pid <= 0) return false;
    // kill(pid, 0) probes existence: 0 = alive, ESRCH = gone, EPERM = alive
    // but owned by someone else (still "running" from our viewpoint).
    if (::kill(static_cast<pid_t>(pid), 0) == 0) return true;
    return errno == EPERM;
}

std::string now_iso8601() {
    auto now = std::time(nullptr);
    struct tm tm_buf{};
    gmtime_r(&now, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

}  // namespace

bool write_serve_lock(const ServeLock& lock) {
    if (!paths::ensure_state_dir()) return false;
    const fs::path p = paths::serve_lock_file();

    json j = {
        {"pid",            lock.pid},
        {"model_path",     lock.model_path},
        {"display_name",   lock.display_name},
        {"quant_name",     lock.quant_name},
        {"params",         lock.params},
        {"tensor_bytes",   lock.tensor_bytes},
        {"kv_state_bytes", lock.kv_state_bytes},
        {"peak_rss_bytes", lock.peak_rss_bytes},
        {"ctx_size",       lock.ctx_size},
        {"ctx_train",      lock.ctx_train},
        {"port",           lock.port},
        {"started_at",     lock.started_at.empty() ? now_iso8601()
                                                    : lock.started_at},
    };
    std::ofstream out(p);
    if (!out) return false;
    out << j.dump(2) << '\n';
    return out.good();
}

void remove_serve_lock() {
    std::error_code ec;
    fs::remove(paths::serve_lock_file(), ec);
}

std::optional<ServeLock> read_serve_lock(std::string& error_out, bool& stale_out) {
    stale_out = false;
    error_out.clear();
    const fs::path p = paths::serve_lock_file();
    if (!fs::exists(p)) return std::nullopt;   // nothing serving — not an error

    std::ifstream in(p);
    if (!in) {
        error_out = "serve lock exists but could not be read: " + p.string();
        return std::nullopt;
    }
    std::ostringstream ss;
    ss << in.rdbuf();

    ServeLock lock;
    try {
        auto j = json::parse(ss.str());
        lock.pid           = j.value("pid", static_cast<std::int64_t>(0));
        lock.model_path    = j.value("model_path", "");
        lock.display_name  = j.value("display_name", "");
        lock.quant_name    = j.value("quant_name", "");
        lock.params        = j.value("params",   static_cast<std::uint64_t>(0));
        lock.tensor_bytes  = j.value("tensor_bytes",  static_cast<std::uint64_t>(0));
        lock.kv_state_bytes= j.value("kv_state_bytes",static_cast<std::uint64_t>(0));
        lock.peak_rss_bytes= j.value("peak_rss_bytes",static_cast<std::uint64_t>(0));
        lock.ctx_size      = j.value("ctx_size", 0);
        lock.ctx_train     = j.value("ctx_train",0);
        lock.port          = j.value("port", 0);
        lock.started_at    = j.value("started_at", "");
    } catch (const json::exception& e) {
        error_out = std::string("serve lock is corrupt: ") + e.what();
        return std::nullopt;
    }

    if (!pid_alive(lock.pid)) {
        stale_out = true;
        error_out = "lock is stale (pid " + std::to_string(lock.pid) +
                    " is gone; safe to start a new serve)";
    }
    return lock;
}

}  // namespace hostely::inference