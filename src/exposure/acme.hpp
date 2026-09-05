#pragma once
// Minimal RFC 8555 ACME client — DNS-01 only, ES256 account key.
// Hand-rolled (no maintained C++ ACME library exists). Staging-first:
// HOSTELY_ACME_DIR_URL overrides, default is Let's Encrypt production;
// `hostely certs issue --staging` uses LE staging.
#include <string>
#include <vector>

namespace hostely::exposure {

struct AcmeResult {
    bool ok = false;
    std::string cert_pem;    // full chain, leaf first
    std::string key_pem;     // private key for the cert (PEM)
    std::string error;
};

// Issue a certificate for `domains` (first = primary) via DNS-01.
// `staging` selects LE's staging directory. The account key lives in
// state_dir()/exposure/acme-account.pem (created on first use).
AcmeResult acme_issue_cert(const std::vector<std::string>& domains,
                           bool staging, std::string& error_out);

}  // namespace hostely::exposure