#pragma once

#include "cli/args.hpp"
#include "config/config.hpp"

namespace hostely::top {

/// `hostely top` — live htop-style dashboard. Returns the process exit code.
int run_top(const hostely::config::Config& cfg, const hostely::cli::ParsedArgs& args);

}  // namespace hostely::top