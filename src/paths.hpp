#pragma once

#include <filesystem>
#include <string>

namespace hostely::paths {

/// Returns the hostely state directory.
///   ~/Library/Application Support/hostely/
/// Created lazily by `ensure_*` helpers below.
std::filesystem::path state_dir();

/// Returns the hostely log directory.
///   ~/Library/Logs/hostely/
std::filesystem::path log_dir();

/// Returns the canonical config file path.
std::filesystem::path config_file();

/// Returns the canonical models directory.
///   ~/Library/Application Support/hostely/models/
/// Sidecar manifests and GGUF downloads live here (Phase 7a+).
std::filesystem::path models_dir();

/// Returns the canonical serve-state lockfile path (Phase 7c). Used by
/// `hostely serve` to publish "what is currently loaded" to `hostely status`.
std::filesystem::path serve_lock_file();

/// Create `state_dir()` if it does not exist. Returns true on success.
bool ensure_state_dir();

/// Create `log_dir()` if it does not exist. Returns true on success.
bool ensure_log_dir();

/// Create `models_dir()` if it does not exist. Returns true on success.
bool ensure_models_dir();

/// Resolve `~/Library/...` from `$HOME`. Returns empty path if HOME is unset.
std::filesystem::path home_dir();

}  // namespace hostely::paths
