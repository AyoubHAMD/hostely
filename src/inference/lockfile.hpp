#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace hostely::inference {

/// Snapshot of a running `hostely serve`, published at
/// `paths::serve_lock_file()` so `hostely status` can show what's loaded
/// without coupling to the server process. Written by `Server::start()`,
/// removed on graceful shutdown (Phase 7c).
struct ServeLock {
    std::int64_t   pid          = 0;   // serving process
    std::string    model_path;         // absolute .gguf path
    std::string    display_name;       // stem, for humans
    std::string    quant_name;         // "Q4_K_M"
    std::uint64_t  params        = 0;  // from llama_model_n_params
    std::uint64_t  tensor_bytes  = 0;
    std::uint64_t  kv_state_bytes = 0;
    std::uint64_t  peak_rss_bytes = 0; // fit-advisor estimate
    std::int32_t   ctx_size      = 0;
    std::int32_t   ctx_train     = 0;
    std::int32_t   port          = 0;
    std::string    started_at;         // ISO8601 UTC
};

/// Write the lockfile (overwrites any previous one). Returns false on I/O
/// failure; the caller should warn but continue serving.
bool write_serve_lock(const ServeLock& lock);

/// Remove the lockfile if it exists. Called on shutdown.
void remove_serve_lock();

/// Read the lockfile. Returns:
///   - nullopt-with-empty-error : no lockfile (nothing is serving)
///   - nullopt-with-error       : lockfile exists but unreadable/stale
///   - a ServeLock              : lockfile present
/// `stale_out` is set true when the pid is not alive anymore.
std::optional<ServeLock> read_serve_lock(std::string& error_out, bool& stale_out);

}  // namespace hostely::inference