#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace hostely::config {

/// In-memory representation of the hostely config file.
///
/// We do not use a generic key/value TOML layer because we want the schema
/// to be explicit and reviewable. Adding a field is a deliberate change to
/// this struct + the writer below.
struct Config {
    /// Path the config was loaded from (empty if defaults only).
    std::filesystem::path source_path;

    // ---- [general] ---------------------------------------------------------
    // hostely only ships the "llama" engine on Apple Silicon. The field is
    // retained in config for forward compatibility: if llama.cpp ever adds
    // an MLX backend we will surface it here without a config-schema break.
    std::string default_engine = "llama";
    int         default_port  = 8080;

    // ---- [inference] -------------------------------------------------------
    int         ctx_size      = 4096;
    int         gpu_layers    = -1;        // -1 = all
    std::string models_dir    = "";        // set to ~/Library/.../hostely/models on init

    // ---- [services] --------------------------------------------------------
    std::string container_cli = "container";  // overridable for testing
};

/// Returns the default config (does not touch disk).
Config defaults();

/// Load from `path`. If the file does not exist, returns `defaults()` and
/// `source_path` is left empty. Returns std::nullopt on parse failure.
std::optional<Config> load(const std::filesystem::path& path);

/// Write `cfg` to `path` as TOML. Creates parent dirs. Returns true on
/// success. Overwrites an existing file.
bool save(const Config& cfg, const std::filesystem::path& path);

}  // namespace hostely::config
