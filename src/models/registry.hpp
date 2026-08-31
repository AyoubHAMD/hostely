#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hostely::models {

/// What we know about a GGUF that's been pulled into the local registry.
struct Manifest {
    std::filesystem::path   path;        // absolute path to the .gguf file
    std::filesystem::path   manifest_path; // absolute path to the sidecar .toml
    std::string             source;      // canonical URL it was pulled from
    std::string             repo;        // "user/repo" if Hugging Face, else ""
    std::string             quant;       // "Q4_K_M" — the file-suffix quant
    std::string             filename;    // basename of the gguf
    std::uint64_t           size_bytes = 0; // file size on disk
    std::string             license;     // human-readable license string, may be empty
    std::string             pulled_at;   // ISO 8601 UTC; populated by pull
};

/// Scan `dir` for `.toml` sidecars and return a sorted list of manifests.
/// If a sidecar's referenced .gguf file is missing, the manifest is still
/// returned but `path` may be empty (caller should handle gracefully).
std::vector<Manifest> list(const std::filesystem::path& dir);

/// Resolve a user-supplied name to an existing manifest. The `name` can be:
///   1. an absolute path to a .gguf file,
///   2. a logical name (the manifest filename stem, with `--` separators),
///   3. a partial match: a unique prefix of any stem returns that manifest,
///      otherwise an error.
/// Returns std::nullopt with `error_out` populated on miss.
std::optional<Manifest> resolve(const std::filesystem::path& dir,
                                const std::string&            name,
                                std::string&                  error_out);

/// Read a single sidecar manifest. The `.gguf` referenced is not required
/// to exist for this call to succeed (we only parse the sidecar).
std::optional<Manifest> read_manifest(const std::filesystem::path& manifest_path);

/// Write a sidecar manifest TOML. The destination is the file the manifest
/// lives at; we don't dedupe existing files — caller decides.
bool write_manifest(const Manifest& m);

/// Produce the sanitised sidecar stem for a logical name. The convention is
/// `repo--quant` with `--` between segments and filesystem-safe chars. If
/// two files in one repo share a quant (rare), the later pull overwrites the
/// earlier sidecar — the sidecar's `filename` field is authoritative.
std::string make_stem(const std::string& repo,
                      const std::string& quant);

/// Build the path of a sidecar given a registry dir and a logical stem.
std::filesystem::path sidecar_for(const std::filesystem::path& dir,
                                  const std::string&            stem);

}  // namespace hostely::models
