#include "exposure/dns.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>

using json = nlohmann::json;

namespace hostely::exposure {

// ---------------------------------------------------------------------------
// HTTPS helper — cpp-httplib SSL client.
// ---------------------------------------------------------------------------

HttpResult https_request(const std::string& method, const std::string& url,
                         const std::vector<std::string>& headers,
                         const std::string& body) {
    HttpResult res;
    // url: https://host[:port]/path
    std::string rest = url;
    const std::string scheme = "https://";
    if (rest.rfind(scheme, 0) != 0) {
        res.status = -1;
        res.body = "only https URLs supported";
        return res;
    }
    rest = rest.substr(scheme.size());
    auto slash = rest.find('/');
    std::string hostport = slash == std::string::npos ? rest : rest.substr(0, slash);
    std::string path = slash == std::string::npos ? "/" : rest.substr(slash);
    std::string host = hostport;
    int port = 443;
    if (auto colon = hostport.rfind(':'); colon != std::string::npos) {
        host = hostport.substr(0, colon);
        port = std::atoi(hostport.c_str() + colon + 1);
    }

    httplib::SSLClient cli(host.c_str(), port);
    cli.set_connection_timeout(15);
    cli.set_read_timeout(30);
    cli.enable_server_certificate_verification(true);

    httplib::Headers hdrs;
    for (const auto& h : headers) {
        if (auto colon = h.find(':'); colon != std::string::npos) {
            hdrs.emplace(h.substr(0, colon), h.substr(colon + 1));
        }
    }
    httplib::Result r;
    if (method == "GET") {
        r = cli.Get(path.c_str(), hdrs);
    } else if (method == "POST") {
        r = cli.Post(path.c_str(), hdrs, body, "application/json");
    } else if (method == "PUT") {
        r = cli.Put(path.c_str(), hdrs, body, "application/json");
    } else if (method == "PATCH") {
        r = cli.Patch(path.c_str(), hdrs, body, "application/json");
    } else if (method == "DELETE") {
        r = cli.Delete(path.c_str(), hdrs, body, "application/json");
    } else {
        res.status = -1;
        res.body = "unsupported method " + method;
        return res;
    }
    if (!r) {
        res.status = -1;
        res.body = httplib::to_string(r.error());
        return res;
    }
    res.status = r->status;
    res.body = r->body;
    return res;
}

std::string dns_query_txt(const std::string& name) {
    // DNS-over-HTTPS (RFC 8484) against Cloudflare's resolver using the
    // name=...&type=TXT JSON shortcut endpoint.
    auto res = https_request(
        "GET",
        "https://cloudflare-dns.com/dns-query?name=" + name + "&type=TXT",
        {"accept: application/dns-json"}, "");
    if (!res.ok()) return "";
    try {
        auto j = json::parse(res.body);
        for (const auto& a : j.at("Answer")) {
            if (a.value("type", 0) == 16) {
                // Strip enclosing quotes DoH JSON keeps.
                std::string v = a.value("data", "");
                if (v.size() >= 2 && v.front() == '"' && v.back() == '"') {
                    v = v.substr(1, v.size() - 2);
                }
                return v;
            }
        }
    } catch (...) {}
    return "";
}

// ---------------------------------------------------------------------------
// Cloudflare
// ---------------------------------------------------------------------------

namespace {

class CloudflareDns : public DnsProvider {
public:
    explicit CloudflareDns(std::string token) : token_(std::move(token)) {}

    const char* name() const override { return "cloudflare"; }

    bool set_txt(const std::string& zone, const std::string& name,
                 const std::string& value) override {
        return upsert("TXT", zone, name, value);
    }
    bool remove_txt(const std::string& zone, const std::string& name,
                    const std::string& value) override {
        return remove_record("TXT", zone, name, value);
    }
    bool set_cname(const std::string& zone, const std::string& name,
                   const std::string& target) override {
        return upsert("CNAME", zone, name, target);
    }

private:
    std::vector<std::string> hdrs() const {
        return {"Authorization: Bearer " + token_, "Content-Type: application/json"};
    }

    // Resolve zone name -> zone id.
    std::string zone_id(const std::string& zone) {
        auto res = https_request(
            "GET",
            "https://api.cloudflare.com/client/v4/zones?name=" + zone,
            hdrs(), "");
        if (!res.ok()) return "";
        try {
            auto j = json::parse(res.body);
            if (j.value("success", false) && !j.at("result").empty()) {
                return j.at("result")[0].value("id", "");
            }
        } catch (...) {}
        return "";
    }

    bool upsert(const std::string& type, const std::string& zone,
                const std::string& name, const std::string& content) {
        std::string zid = zone_id(zone);
        if (zid.empty()) {
            std::cerr << "dns: cannot resolve zone id for '" << zone << "'\n";
            return false;
        }
        // Delete existing records of this type/name first — ACME TXT names
        // are single-use and CNAME upserts should stay authoritative.
        list_and_delete(type, zid, name);

        json body = {{"type", type},
                     {"name", name},
                     {"content", content},
                     {"ttl", 120},
                     {"proxied", false}};
        auto res = https_request(
            "POST",
            "https://api.cloudflare.com/client/v4/zones/" + zid + "/dns_records",
            hdrs(), body.dump());
        if (!res.ok()) {
            std::cerr << "dns: create " << type << " " << name
                      << " failed: HTTP " << res.status << " " << res.body
                      << "\n";
            return false;
        }
        return true;
    }

    void list_and_delete(const std::string& type, const std::string& zid,
                         const std::string& name) {
        auto res = https_request(
            "GET",
            "https://api.cloudflare.com/client/v4/zones/" + zid +
                "/dns_records?type=" + type + "&name=" + name,
            hdrs(), "");
        if (!res.ok()) return;
        try {
            auto j = json::parse(res.body);
            for (const auto& rec : j.at("result")) {
                auto id = rec.value("id", "");
                if (id.empty()) continue;
                https_request(
                    "DELETE",
                    "https://api.cloudflare.com/client/v4/zones/" + zid +
                        "/dns_records/" + id,
                    hdrs(), "");
            }
        } catch (...) {}
    }

    bool remove_record(const std::string& type, const std::string& zone,
                       const std::string& name, const std::string& value) {
        std::string zid = zone_id(zone);
        if (zid.empty()) return false;
        auto res = https_request(
            "GET",
            "https://api.cloudflare.com/client/v4/zones/" + zid +
                "/dns_records?type=" + type + "&name=" + name,
            hdrs(), "");
        if (!res.ok()) return false;
        bool any = false;
        try {
            auto j = json::parse(res.body);
            for (const auto& rec : j.at("result")) {
                if (rec.value("content", "") != value) continue;
                auto id = rec.value("id", "");
                if (id.empty()) continue;
                auto del = https_request(
                    "DELETE",
                    "https://api.cloudflare.com/client/v4/zones/" + zid +
                        "/dns_records/" + id,
                    hdrs(), "");
                any = del.ok() || any;
            }
        } catch (...) {}
        return any;
    }

    std::string token_;
};

}  // namespace

std::unique_ptr<DnsProvider> make_dns_provider() {
    if (const char* tok = std::getenv("HOSTELY_CF_API_TOKEN"); tok && *tok) {
        return std::make_unique<CloudflareDns>(tok);
    }
    return nullptr;
}

}  // namespace hostely::exposure