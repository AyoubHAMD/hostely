#include "models/fit.hpp"

#include <algorithm>
#include <sstream>

#include "resources/system.hpp"

namespace hostely::models {

using hostely::resources::human_bytes;
using hostely::resources::MemoryStats;
using hostely::resources::read_memory;

FitEstimate FitAdvisor::check(const inference::LoadedModel& model,
                              std::int32_t ctx_size,
                              std::uint64_t headroom_bytes) {
    FitEstimate est;
    est.tensor_bytes   = model.tensor_bytes;
    est.kv_state_bytes = model.kv_state_bytes;
    est.headroom_bytes = headroom_bytes;

    // Scratch / Metal-buffer overhead: a heuristic. llama.cpp allocates
    // per-thread compute buffers plus Metal command buffers. We use
    // max(256 MiB, 10% of weights), which roughly matches what we observed
    // when loading TinyLlama-1.1B (~763 MB RSS for ~637 MB weights).
    std::uint64_t ten_pct = static_cast<std::uint64_t>(model.tensor_bytes / 10);
    est.scratch_bytes = std::max<std::uint64_t>(256ull << 20, ten_pct);

    est.peak_rss_bytes = est.tensor_bytes + est.kv_state_bytes + est.scratch_bytes;

    auto mem = read_memory();
    est.total_memory = mem.total_bytes;
    // "available" = free + inactive (both reclaimable on demand) minus the
    // safety margin we want to keep for the OS + active containers.
    if (mem.valid) {
        std::uint64_t free_inactive = mem.free_bytes + mem.inactive_bytes;
        est.available_bytes = (free_inactive > headroom_bytes)
                                ? free_inactive - headroom_bytes
                                : 0;
    } else {
        // No memory read — be conservative and pretend nothing is available.
        est.available_bytes = 0;
    }

    est.delta_bytes = static_cast<std::int64_t>(est.peak_rss_bytes) -
                      static_cast<std::int64_t>(est.available_bytes);

    const std::uint64_t kTightGiB = 1ull << 30;
    if (est.delta_bytes < 0) {
        est.verdict = Verdict::Fits;
    } else if (static_cast<std::uint64_t>(est.delta_bytes) < kTightGiB) {
        est.verdict = Verdict::Tight;
    } else {
        est.verdict = Verdict::WontFit;
    }

    std::ostringstream oss;
    switch (est.verdict) {
    case Verdict::Fits:
        oss << "fits with " << human_bytes(static_cast<std::uint64_t>(-est.delta_bytes))
            << " headroom";
        break;
    case Verdict::Tight:
        oss << "tight: peak RSS exceeds available by "
            << human_bytes(static_cast<std::uint64_t>(est.delta_bytes))
            << " (within the 1 GiB tolerance)";
        break;
    case Verdict::WontFit:
        oss << "won't fit: peak RSS exceeds available by "
            << human_bytes(static_cast<std::uint64_t>(est.delta_bytes))
            << "; reduce --ctx-size or pick a smaller quant";
        break;
    }
    est.message = oss.str();
    return est;
}

bool FitAdvisor::should_proceed(const FitEstimate& est, bool no_fit_check) {
    if (est.verdict == Verdict::WontFit && !no_fit_check) return false;
    return true;
}

}  // namespace hostely::models