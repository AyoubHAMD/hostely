#pragma once
// Cert store — one directory per domain under certs_dir():
//   certs/<domain>/cert.pem   full chain (leaf first)
//   certs/<domain>/key.pem    private key, 0600
//   certs/<domain>/meta.json  { domain, issuer, issued_at, not_after, source }
// First-class artifact (prior-art lesson: never a single acme.json blob).

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace hostely::exposure {

struct CertInfo {
    std::string domain;
    std::string issuer;
    std::int64_t not_after = 0;    // unix seconds
    std::int64_t not_before = 0;
    std::string source;            // "acme" | "imported"
};

// Store a cert chain + key for `domain`; sets 0600 on key. Returns false on error.
bool cert_store_save(const std::string& domain, const std::string& cert_pem,
                     const std::string& key_pem, const std::string& source);

// Load PEMs; empty strings on absence.
bool cert_store_load(const std::string& domain, std::string& cert_pem,
                     std::string& key_pem);

// Parse a PEM cert chain: subject CN / SANs, issuer, validity. Best-effort.
std::optional<CertInfo> cert_inspect_pem(const std::string& cert_pem);

std::vector<CertInfo> cert_store_list();
bool cert_store_remove(const std::string& domain);

// Days until expiry (negative if expired); -1 if unknown.
int cert_days_remaining(const CertInfo& c);

}  // namespace hostely::exposure