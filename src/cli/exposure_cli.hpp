#pragma once
// CLI handlers for the exposure stack (proxy / certs / expose / tunnel).
// Built only when HOSTELY_HAVE_OPENSSL is defined.

namespace hostely::cli {
class ParsedArgs;
}

namespace hostely {

int run_proxy(const cli::ParsedArgs& args);
int run_certs(const cli::ParsedArgs& args);
int run_expose(const cli::ParsedArgs& args);
int run_tunnel(const cli::ParsedArgs& args);

}  // namespace hostely