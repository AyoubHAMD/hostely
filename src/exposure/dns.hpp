#pragma once
// DNS provider abstraction — used by ACME DNS-01 challenges and by
// `hostely expose` to create/verify CNAME records.
//
// v1 ships Cloudflare (single Bearer-token JSON API). Route 53 is deferred
// (hand-rolled SigV4 + XML). Providers are configured via environment
// variables so tokens never live in state files:
//   HOSTELY_CF_API_TOKEN   Cloudflare API token (Zone:DNS:Edit)
#include <memory>
#include <string>
#include <vector>

namespace hostely::exposure {

struct DnsRecord {
    std::string id;
    std::string type;
    std::string name;
    std::string content;
    int ttl = 300;
};

class DnsProvider {
public:
    virtual ~DnsProvider() = default;
    // Create (or update an existing TXT record with the same name).
    virtual bool set_txt(const std::string& zone, const std::string& name,
                         const std::string& value) = 0;
    virtual bool remove_txt(const std::string& zone, const std::string& name,
                            const std::string& value) = 0;
    // CNAME helper for `hostely expose` + tunnel custom domains.
    virtual bool set_cname(const std::string& zone, const std::string& name,
                           const std::string& target) = 0;
    // A record (public-IP watcher).
    virtual bool set_a(const std::string& zone, const std::string& name,
                       const std::string& ip) = 0;
    // Best-effort provider label ("cloudflare"), for messages.
    virtual const char* name() const = 0;
};

// Returns nullptr (and explains why on stderr) if no provider is configured.
std::unique_ptr<DnsProvider> make_dns_provider();

// HTTP helpers shared by DNS + ACME (plain HTTPS GET/POST/PATCH/DELETE with
// a JSON body; returns status + body).
struct HttpResult {
    long status = 0;
    std::string body;
    bool ok() const { return status >= 200 && status < 300; }
};
HttpResult https_request(const std::string& method, const std::string& url,
                         const std::vector<std::string>& headers,
                         const std::string& body);

// Minimal DNS-over-HTTPS TXT lookup (Cloudflare resolver) to verify
// propagation. Returns first TXT value found, or "".
std::string dns_query_txt(const std::string& name);

}  // namespace hostely::exposure