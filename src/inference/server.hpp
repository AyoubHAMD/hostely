#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace hostely::inference {

struct ServeOptions {
    std::filesystem::path model_path;   // path to a .gguf file
    int                   port     = 8080;
    int                   ctx_size = 4096;
    int                   gpu_layers = -1;   // -1 = all
    int                   threads  = 0;     //  0 = auto
    bool                  no_fit_check = false; // bypass Phase 7b advisor
};

/// Snapshot of a successfully-loaded model — populated right after
/// `llama_model_load_from_file` succeeds. Used by:
///   - Phase 7b (fit advisor)
///   - Phase 7c (status output) via the serve.lock.json we publish
///   - Phase 7d (multi-turn) for param-aware session sizing.
struct LoadedModel {
    std::filesystem::path path;            // absolute .gguf path
    std::string           display_name;    // filename without extension
    std::string           quant_name;      // "Q4_K_M" — from llama_ftype_name
    std::uint64_t         params           = 0;  // from llama_model_n_params
    std::uint64_t         tensor_bytes     = 0;  // from llama_model_size
    std::int32_t          ctx_train        = 0;  // from llama_model_n_ctx_train
    std::int32_t          n_layer          = 0;
    std::int32_t          n_embd           = 0;
    std::int32_t          n_head           = 0;
    std::int32_t          n_head_kv        = 0;
    // KV cache capacity for the configured n_ctx, computed analytically
    // (n_layer × n_ctx × 2 × n_head_kv × head_dim × 2 bytes for f16).
    // llama_state_get_size() is NOT used: it reports *used* state (~0 right
    // after init), which is useless for a fit check.
    std::uint64_t         kv_state_bytes   = 0;
};

/// OpenAI-compatible inference server backed by llama.cpp.
///
/// Lifecycle:
///   Server s(opts);
///   s.start();             // loads model, binds HTTP, blocks until stop()
///   // from another thread:
///   s.stop();
class Server {
public:
    explicit Server(ServeOptions opts);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    /// Bind the listening socket and serve forever. Returns the process exit
    /// code (0 on clean shutdown via stop(), 1 on load failure, etc.).
    int  start();

    /// Request shutdown. Safe to call from another thread or signal handler.
    void stop();

    const ServeOptions& options() const { return options_; }

    /// After a successful load, this is populated. Used by the Phase 7b fit
    /// advisor and the Phase 7c `hostely status` served-model readout.
    const LoadedModel& loaded() const { return loaded_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ServeOptions          options_;
    LoadedModel           loaded_;     // empty until start() loads
};

}  // namespace hostely::inference
