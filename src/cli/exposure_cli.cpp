#include "cli/exposure_cli.hpp"

#include "cli/args.hpp"
#include "exposure/acme.hpp"
#include "exposure/certs.hpp"
#include "exposure/dns.hpp"
#include "exposure/expose.hpp"
#include "log/logger.hpp"
#include "proxy/proxy.hpp"
#include "tunnel/client.hpp"

#include <cstdlib>
#include <iostream>

using namespace hostely;
using namespace hostely::exposure;
namespace lg = hostely::log;

namespace {

std::string sub(const cli::ParsedArgs& args) {
    auto pos = args.positionals();
    return pos.size() > 1 ? pos[1] : "";
}

std::string operand(const cli::ParsedArgs& args, size_t i) {
    auto pos = args.positionals();
    return pos.size() > i ? pos[i] : "";
}

}  // namespace

// ---------------------------------------------------------------------------
// hostely certs list|issue|rm
// ---------------------------------------------------------------------------

namespace hostely {

int run_certs(const cli::ParsedArgs& args) {
    std::string sc = sub(args);

    if (sc == "list" || sc.empty()) {
        auto certs = cert_store_list();
        if (certs.empty()) {
            std::cout << "no certificates stored\n";
            return 0;
        }
        for (const auto& c : certs) {
            int days = cert_days_remaining(c);
            std::cout << c.domain << "  issuer=" << c.issuer.substr(0, 40)
                      << "  expires_in=" << (days >= 0 ? std::to_string(days) + "d"
                                                        : "expired")
                      << "  source=" << c.source << "\n";
        }
        return 0;
    }

    if (sc == "issue") {
        std::string domain = operand(args, 2);
        if (domain.empty()) {
            std::cerr << "usage: hostely certs issue <domain> [san...] [--staging]\n";
            return 2;
        }
        bool staging = args.has("staging");
        std::vector<std::string> domains{domain};
        auto pos = args.positionals();
        for (size_t i = 3; i < pos.size(); ++i) domains.push_back(pos[i]);

        lg::info("issuing " + std::string(staging ? "staging " : "") +
                 "certificate for " + domain + " via ACME DNS-01...");
        std::string err;
        auto res = acme_issue_cert(domains, staging, err);
        if (!res.ok) {
            lg::error("certificate issuance failed: " + err);
            return 1;
        }
        if (!cert_store_save(domain, res.cert_pem, res.key_pem, "acme")) {
            lg::error("failed to persist certificate for " + domain);
            return 1;
        }
        if (auto info = cert_inspect_pem(res.cert_pem)) {
            lg::info("issued certificate for " + domain +
                     " (valid " + std::to_string(cert_days_remaining(*info)) +
                     " days)");
        }
        return 0;
    }

    if (sc == "rm") {
        std::string domain = operand(args, 2);
        if (domain.empty()) {
            std::cerr << "usage: hostely certs rm <domain>\n";
            return 2;
        }
        if (!cert_store_remove(domain)) {
            lg::error("no certificate stored for " + domain);
            return 1;
        }
        lg::info("removed certificate for " + domain);
        return 0;
    }

    std::cerr << "hostely certs: unknown subcommand '" << sc << "'\n";
    return 2;
}

// ---------------------------------------------------------------------------
// hostely proxy serve
// ---------------------------------------------------------------------------

int run_proxy(const cli::ParsedArgs& args) {
    std::string sc = sub(args);
    if (sc != "serve") {
        std::cerr << "usage: hostely proxy serve [--http N] [--https N]\n";
        return sc.empty() ? 2 : 1;
    }
    ProxyOptions opts;
    if (auto v = args.get("http")) {
        try { opts.http_port = std::stoi(*v); } catch (...) {}
    }
    if (auto v = args.get("https")) {
        try { opts.https_port = std::stoi(*v); } catch (...) {}
    }
    return proxy_serve(opts);
}

// ---------------------------------------------------------------------------
// hostely expose <host> <container> [--port N] [--no-tls] [--off] [--cname target]
//        expose route list|rm <host>
// ---------------------------------------------------------------------------

int run_expose(const cli::ParsedArgs& args) {
    auto pos = args.positionals();
    // `hostely expose route ...`
    if (pos.size() > 1 && pos[1] == "route") {
        std::string sc = pos.size() > 2 ? pos[2] : "";
        if (sc == "list" || sc.empty()) {
            auto routes = routes_load();
            if (routes.empty()) std::cout << "no routes\n";
            for (const auto& r : routes) {
                std::cout << r.host << " -> " << r.target << ":" << r.port
                          << (r.tls ? " (tls)" : " (plain)") << "\n";
            }
            return 0;
        }
        if (sc == "rm") {
            if (pos.size() < 4) {
                std::cerr << "usage: hostely expose route rm <host>\n";
                return 2;
            }
            if (!route_remove(std::string(pos[3]))) {
                lg::error("no route for " + std::string(pos[3]));
                return 1;
            }
            lg::info("removed route " + std::string(pos[3]));
            return 0;
        }
        std::cerr << "hostely expose route: unknown subcommand\n";
        return 2;
    }

    std::string host = operand(args, 1);
    std::string container = operand(args, 2);
    if (host.empty() || container.empty()) {
        std::cerr << "usage: hostely expose <host> <container> [--port N] "
                     "[--no-tls] [--off] [--cname <target>]\n"
                     "       hostely expose route list|rm <host>\n";
        return 2;
    }

    if (args.has("off")) {
        if (!route_remove(host)) {
            lg::warn("no route for " + host);
        }
        lg::info("removed route " + host);
        return 0;
    }

    Route r;
    r.host = host;
    r.target = container;
    if (auto v = args.get("port")) {
        try { r.port = std::stoi(*v); } catch (...) {}
    }
    r.tls = !args.has("no-tls");
    if (!route_add(r)) {
        lg::error("failed to persist route");
        return 1;
    }

    // Optional DNS CNAME (e.g. to a relay) when --cname is given.
    if (auto cname = args.get("cname"); cname && !cname->empty()) {
        auto provider = make_dns_provider();
        if (!provider) {
            lg::warn("no DNS provider configured (set HOSTELY_CF_API_TOKEN); "
                     "create the CNAME manually");
        } else {
            std::string zone = host;
            size_t last = host.rfind('.');
            if (last != std::string::npos) {
                size_t prev = host.rfind('.', last - 1);
                if (prev != std::string::npos) zone = host.substr(prev + 1);
            }
            if (provider->set_cname(zone, host, *cname)) {
                lg::info("CNAME " + host + " -> " + *cname + " created");
            } else {
                lg::error("CNAME creation failed for " + host);
            }
        }
    }

    lg::info("route added: " + host + " -> " + container + ":" +
             std::to_string(r.port) + (r.tls ? " (tls)" : ""));
    return 0;
}

// ---------------------------------------------------------------------------
// hostely tunnel <hostname> --relay host[:port] [--port N]
// Token via HOSTELY_TUNNEL_TOKEN env (never passed on the command line).
// ---------------------------------------------------------------------------

int run_tunnel(const cli::ParsedArgs& args) {
    std::string hostname = operand(args, 1);
    if (hostname.empty()) {
        std::cerr << "usage: hostely tunnel <hostname> --relay <host> "
                     "[--port N] [--local-host IP] [--local-port N]\n"
                     "token is read from $HOSTELY_TUNNEL_TOKEN\n";
        return 2;
    }
    tunnel::TunnelOptions opts;
    opts.hostname = hostname;
    if (auto v = args.get("relay")) {
        opts.relay_host = *v;
        if (auto colon = v->rfind(':'); colon != std::string::npos) {
            opts.relay_host = v->substr(0, colon);
            try { opts.relay_port = std::stoi(v->substr(colon + 1)); } catch (...) {}
        }
    }
    if (auto v = args.get("port")) {
        try { opts.relay_port = std::stoi(*v); } catch (...) {}
    }
    if (auto v = args.get("local-host")) opts.local_host = *v;
    if (auto v = args.get("local-port")) {
        try { opts.local_port = std::stoi(*v); opts.local_port_set = true; } catch (...) {}
    }
    if (opts.relay_host.empty()) {
        std::cerr << "hostely tunnel: --relay is required\n";
        return 2;
    }
    if (const char* tok = std::getenv("HOSTELY_TUNNEL_TOKEN"); tok && *tok) {
        opts.token = tok;
    } else {
        lg::error("HOSTELY_TUNNEL_TOKEN is not set");
        return 2;
    }
    if (opts.local_host.empty()) opts.local_host = "127.0.0.1";
    return tunnel::tunnel_run(opts);
}

}  // namespace hostely