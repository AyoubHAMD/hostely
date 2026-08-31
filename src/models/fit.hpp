#pragma once

#include <cstdint>
#include <string>

#include "inference/server.hpp"   // for LoadedModel

namespace hostely::models {

enum class Verdict {
    Fits,         // peak_rss ≤ available — proceed silently
    Tight,        // 0 ≤ Δ < 1 GiB — warn but proceed (default)
    WontFit,      // Δ ≥ 1 GiB — error unless --no-fit-check
};

/// One-shot memory estimate for "what would happen if we loaded this model
/// with this context size on this machine right now".
struct FitEstimate {
    std::uint64_t tensor_bytes     = 0;   // weights resident in process
    std::uint64_t kv_state_bytes   = 0;   // llama_state_get_size(ctx)
    std::uint64_t scratch_bytes    = 0;   // llama.cpp + Metal scratch (heuristic)
    std::uint64_t peak_rss_bytes   = 0;   // tensor + kv + scratch
    std::uint64_t available_bytes  = 0;   // free + inactive - safety margin
    std::int64_t  delta_bytes      = 0;   // peak_rss - available (signed)
    std::uint64_t headroom_bytes   = 0;   // safety margin subtracted (default 2 GiB)
    std::uint64_t total_memory     = 0;   // for context in messages
    Verdict       verdict          = Verdict::Fits;
    std::string   message;                // human-readable summary
};

class FitAdvisor {
public:
    /// Compute a fit estimate. Reads system memory fresh each call so the
    /// numbers reflect the current machine state.
    static FitEstimate check(const inference::LoadedModel& model,
                             std::int32_t ctx_size,
                             std::uint64_t headroom_bytes = 2ull << 30);

    /// Decide whether to proceed given the user's override flag. Returns
    /// false only when the verdict is WontFit AND no_fit_check is false.
    static bool should_proceed(const FitEstimate& est, bool no_fit_check);
};

}  // namespace hostely::models