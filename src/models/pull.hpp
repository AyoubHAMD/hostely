#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "models/registry.hpp"

namespace hostely::models {

/// Errors that can come back from a pull operation.
struct PullError {
    std::string message;
    /// Optional: a list of (filename, size_bytes) the user can choose from
    /// when the requested quant wasn't found. Empty if not applicable.
    std::vector<std::pair<std::string, std::uint64_t>> available_files;
};

struct PullRequest {
    /// The operand as the user typed it (e.g. "TheBloke/Repo:Q4_K_M").
    std::string operand;
    /// Where to put the downloaded GGUF and sidecar. Typically `paths::models_dir()`.
    std::filesystem::path dest_dir;
};

/// Download a model. On success, fills `out_manifest` with the manifest as it
/// was written to disk, including the absolute resolved gguf path.
struct PullResult {
    bool                          ok = false;
    PullError                     error;
    Manifest                      manifest;
};

PullResult pull(const PullRequest& req);

/// Canonical list of file-suffix quantization tags we recognise on
/// Hugging Face GGUF filenames. The first match wins during auto-resolution.
extern const std::vector<std::string> kRecognisedQuants;

}  // namespace hostely::models
