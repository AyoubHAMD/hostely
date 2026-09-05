#include "exposure/acme.hpp"

#include "exposure/dns.hpp"
#include "paths.hpp"

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

using json = nlohmann::json;

namespace hostely::exposure {

namespace {

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::string b64url(const unsigned char* data, size_t len) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    for (size_t i = 0; i < len; i += 3) {
        unsigned n = data[i] << 16;
        if (i + 1 < len) n |= data[i + 1] << 8;
        if (i + 2 < len) n |= data[i + 2];
        out += tbl[(n >> 18) & 63];
        out += tbl[(n >> 12) & 63];
        out += (i + 1 < len) ? tbl[(n >> 6) & 63] : '-';
        out += (i + 2 < len) ? tbl[n & 63] : '-';
    }
    return out;
}

std::string b64url_str(const std::string& s) {
    return b64url(reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

std::string sha256_b64url(const std::string& s) {
    unsigned char md[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(s.data()), s.size(), md);
    return b64url(md, sizeof md);
}

// JWK thumbprint (RFC 7638) for an EC P-256 key.
std::string thumbprint_b64url(EVP_PKEY* key) {
    const EC_KEY* ec = EVP_PKEY_get0_EC_KEY(key);
    if (!ec) return "";
    const EC_GROUP* grp = EC_KEY_get0_group(ec);
    const EC_POINT* pub = EC_KEY_get0_public_key(ec);
    std::array<unsigned char, 65> buf{};
    size_t len = EC_POINT_point2oct(grp, pub, POINT_CONVERSION_UNCOMPRESSED,
                                    buf.data(), buf.size(), nullptr);
    if (len != 65) return "";
    std::string xb(reinterpret_cast<char*>(buf.data() + 1), 32);
    std::string yb(reinterpret_cast<char*>(buf.data() + 33), 32);
    json jwk = {{"crv", "P-256"},
                {"kty", "EC"},
                {"x", b64url(reinterpret_cast<const unsigned char*>(xb.data()), 32)},
                {"y", b64url(reinterpret_cast<const unsigned char*>(yb.data()), 32)}};
    return sha256_b64url(jwk.dump());
}

// Account key: create or load PEM from state_dir()/exposure/acme-account.pem
EVP_PKEY* load_or_create_account_key() {
    auto path = paths::state_dir() / "exposure" / "acme-account.pem";
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (std::filesystem::exists(path)) {
        BIO* bio = BIO_new_file(path.c_str(), "r");
        if (bio) {
            EVP_PKEY* k = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
            BIO_free(bio);
            if (k) return k;
        }
        std::cerr << "acme: cannot read account key, regenerating\n";
    }
    EVP_PKEY* k = EVP_EC_gen("P-256");
    if (!k) return nullptr;
    BIO* bio = BIO_new_file(path.c_str(), "w");
    if (bio) {
        PEM_write_bio_PrivateKey(bio, k, nullptr, nullptr, 0, nullptr, nullptr);
        BIO_free(bio);
        std::filesystem::permissions(
            path, std::filesystem::perms::owner_read |
                      std::filesystem::perms::owner_write);
    }
    return k;
}

// ---------------------------------------------------------------------------
// The ACME session
// ---------------------------------------------------------------------------

struct AcmeCtx {
    std::string directory_url;
    std::string new_nonce;
    std::string new_account;
    std::string new_order;
    EVP_PKEY* key = nullptr;
    std::string kid;  // account URL once registered
};

bool get_directory(AcmeCtx& ctx) {
    auto res = https_request("GET", ctx.directory_url, {}, "");
    if (!res.ok()) {
        std::cerr << "acme: directory fetch failed: HTTP " << res.status << "\n";
        return false;
    }
    try {
        auto j = json::parse(res.body);
        ctx.new_nonce = j.at("newNonce");
        ctx.new_account = j.at("newAccount");
        ctx.new_order = j.at("newOrder");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "acme: bad directory payload: " << e.what() << "\n";
        return false;
    }
}

// Signed POST (JWS, ES256). Retries once on badNonce. `payload` may be null
// (POST-as-GET). populates location/link; body is the parsed JSON.
struct AcmeResponse {
    long status = 0;
    json body;
    std::string raw;
    std::string location;
};

// Sign and send one JWS POST. Returns status < 0 on transport error.
AcmeResponse jws_post(AcmeCtx& ctx, const std::string& url, const json& payload) {
    AcmeResponse out;
    // Fetch a fresh nonce (header of a GET to newNonce).
    std::string nonce;
    {
        std::string rest = ctx.new_nonce.substr(8);  // strip https://
        auto slash = rest.find('/');
        std::string path = rest.substr(slash);
        std::string host = rest.substr(0, slash);
        httplib::SSLClient cli(host.c_str(), 443);
        auto r = cli.Get(path.c_str());
        if (r && r->status == 200) {
            if (auto it = r->headers.find("Replay-Nonce"); it != r->headers.end())
                nonce = it->second;
        }
    }
    if (nonce.empty()) {
        out.status = -1;
        out.raw = "cannot obtain ACME nonce";
        return out;
    }

    json header = {{"alg", "ES256"}, {"nonce", nonce}, {"url", url}};
    if (!ctx.kid.empty()) {
        header["kid"] = ctx.kid;
    } else {
        const EC_KEY* ec = EVP_PKEY_get0_EC_KEY(ctx.key);
        const EC_GROUP* grp = EC_KEY_get0_group(ec);
        const EC_POINT* pub = EC_KEY_get0_public_key(ec);
        unsigned char buf[65];
        EC_POINT_point2oct(grp, pub, POINT_CONVERSION_UNCOMPRESSED, buf,
                           sizeof buf, nullptr);
        header["jwk"] = {
            {"crv", "P-256"},
            {"kty", "EC"},
            {"x", b64url(buf + 1, 32)},
            {"y", b64url(buf + 33, 32)},
        };
    }

    std::string payload_str = payload.is_null() ? "" : payload.dump();
    json flat = {{"protected", b64url_str(header.dump())},
                 {"payload", b64url_str(payload_str)},
                 {"signature", ""}};

    std::string signing_input =
        flat["protected"].get<std::string>() + "." +
        flat["payload"].get<std::string>();
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(signing_input.data()),
           signing_input.size(), digest);

    EVP_MD_CTX* mctx = EVP_MD_CTX_new();
    size_t siglen = 96;
    unsigned char sigbuf[96];
    EVP_DigestSignInit(mctx, nullptr, EVP_sha256(), nullptr, ctx.key);
    EVP_DigestSign(mctx, sigbuf, &siglen, digest, sizeof digest);
    EVP_MD_CTX_free(mctx);

    // DER (r,s) -> raw r||s, each zero-padded to 32 bytes (the ES256 trap —
    // the DER form is NOT what JOSE wants).
    const unsigned char* p = sigbuf;
    ECDSA_SIG* sig = d2i_ECDSA_SIG(nullptr, &p, static_cast<long>(siglen));
    if (!sig) {
        out.status = -1;
        out.raw = "signing failed";
        return out;
    }
    std::string raw(64, '\0');
    BN_bn2binpad(ECDSA_SIG_get0_r(sig),
                 reinterpret_cast<unsigned char*>(raw.data()), 32);
    BN_bn2binpad(ECDSA_SIG_get0_s(sig),
                 reinterpret_cast<unsigned char*>(raw.data() + 32), 32);
    ECDSA_SIG_free(sig);
    flat["signature"] =
        b64url(reinterpret_cast<const unsigned char*>(raw.data()), 64);

    std::string rest = url.substr(8);
    auto slash = rest.find('/');
    std::string path = rest.substr(slash);
    std::string host = rest.substr(0, slash);
    httplib::SSLClient cli(host.c_str(), 443);
    cli.set_read_timeout(60);
    auto r = cli.Post(path.c_str(), flat.dump(), "application/jose+json");
    if (!r) {
        out.status = -1;
        out.raw = httplib::to_string(r.error());
        return out;
    }
    out.status = r->status;
    out.raw = r->body;
    if (auto it = r->headers.find("Location"); it != r->headers.end())
        out.location = it->second;
    try {
        out.body = json::parse(r->body);
    } catch (...) {}
    return out;
}

// POST-as-GET wrapper with one badNonce retry.
AcmeResponse acme_post(AcmeCtx& ctx, const std::string& url,
                       const json& payload) {
    auto res = jws_post(ctx, url, payload);
    if (res.status == 400 || res.status == 403) {
        bool bad = false;
        try {
            bad = res.body.value("type", "").find("badNonce") != std::string::npos;
        } catch (...) {}
        if (bad) res = jws_post(ctx, url, payload);
    }
    return res;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry
// ---------------------------------------------------------------------------

AcmeResult acme_issue_cert(const std::vector<std::string>& domains,
                           bool staging, std::string& error_out) {
    AcmeResult out;
    AcmeCtx ctx;
    ctx.directory_url =
        staging
            ? "https://acme-staging-v02.api.letsencrypt.org/directory"
            : "https://acme-v02.api.letsencrypt.org/directory";
    if (const char* env = std::getenv("HOSTELY_ACME_DIR_URL"); env && *env)
        ctx.directory_url = env;

    ctx.key = load_or_create_account_key();
    if (!ctx.key) {
        error_out = "cannot load/create ACME account key";
        return out;
    }

    if (!get_directory(ctx)) {
        error_out = "cannot fetch ACME directory";
        return out;
    }

    // Register (or re-find) the account.
    json acc_payload = {{"termsOfServiceAgreed", true}};
    auto acc = jws_post(ctx, ctx.new_account, acc_payload);
    if (acc.status != 200 && acc.status != 201) {
        error_out = "account registration failed: HTTP " +
                    std::to_string(acc.status) + " " + acc.raw.substr(0, 400);
        return out;
    }
    if (!acc.location.empty()) {
        ctx.kid = acc.location;
    } else {
        error_out = "account created but no Location/KID returned";
        return out;
    }

    // New order.
    json order_payload = {{"identifiers", json::array()}};
    for (const auto& d : domains) {
        order_payload["identifiers"].push_back({{"type", "dns"}, {"value", d}});
    }
    auto order = acme_post(ctx, ctx.new_order, order_payload);
    if (order.status != 201 || order.location.empty()) {
        error_out = "newOrder failed: HTTP " + std::to_string(order.status) +
                    " " + order.raw.substr(0, 400);
        return out;
    }
    std::string order_url = order.location;
    std::string finalize_url;
    try {
        finalize_url = order.body.at("finalize");
    } catch (...) {
        error_out = "newOrder response missing finalize URL";
        return out;
    }

    // DNS-01 challenges for every authorization.
    auto provider = make_dns_provider();
    if (!provider) {
        error_out = "no DNS provider configured (set HOSTELY_CF_API_TOKEN)";
        return out;
    }
    for (const auto& authz_url :
         order.body.value("authorizations", std::vector<std::string>{})) {
        auto authz = acme_post(ctx, authz_url, json());
        if (authz.status != 200) {
            error_out = "authorization fetch failed: HTTP " +
                        std::to_string(authz.status);
            return out;
        }
        std::string domain;
        try {
            domain = authz.body.at("identifier").at("value");
        } catch (...) {}
        if (authz.body.value("status", "") == "valid") continue;

        std::string token, challenge_url;
        for (const auto& ch : authz.body.value("challenges", json::array())) {
            if (ch.value("type", "") == "dns-01") {
                token = ch.value("token", "");
                challenge_url = ch.value("url", "");
            }
        }
        if (token.empty()) {
            error_out = "no dns-01 challenge offered for " + domain;
            return out;
        }
        std::string key_auth = token + "." + thumbprint_b64url(ctx.key);
        std::string txt = sha256_b64url(key_auth);

        // Zone = last two labels (v1 heuristic; multi-label TLDs out of
        // scope — Route 53 provider later gets public-suffix aware logic).
        std::string zone = domain;
        {
            size_t last = domain.rfind('.');
            if (last != std::string::npos) {
                size_t prev = domain.rfind('.', last - 1);
                if (prev != std::string::npos) zone = domain.substr(prev + 1);
            }
        }
        std::string name = "_acme-challenge." + domain;
        if (name == "_acme-challenge." + zone) name = "_acme-challenge";

        if (!provider->set_txt(zone, name, txt)) {
            error_out = "DNS set failed for " + name;
            return out;
        }

        bool propagated = false;
        for (int i = 0; i < 30; ++i) {
            if (dns_query_txt(name) == txt) {
                propagated = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        if (!propagated)
            std::cerr << "acme: TXT not visible yet, submitting anyway\n";

        auto resp = acme_post(ctx, challenge_url, json());
        if (resp.status < 200 || resp.status >= 300) {
            error_out = "challenge trigger failed: HTTP " +
                        std::to_string(resp.status) + " " + resp.raw.substr(0, 400);
            return out;
        }

        bool valid = false;
        for (int i = 0; i < 30; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            auto poll = acme_post(ctx, authz_url, json());
            auto st = poll.body.value("status", "");
            if (st == "valid") {
                valid = true;
                break;
            }
            if (st == "invalid") {
                error_out = "authorization invalid for " + domain + ": " +
                            poll.raw.substr(0, 500);
                return out;
            }
        }
        if (!valid) {
            error_out = "authorization timeout for " + domain;
            return out;
        }
        provider->remove_txt(zone, name, txt);
    }

    // Finalize: CSR with SANs (P-256 key — fast, universally accepted).
    EVP_PKEY* cert_key = EVP_EC_gen("P-256");
    if (!cert_key) {
        error_out = "cannot generate cert key";
        return out;
    }
    X509_REQ* req = X509_REQ_new();
    X509_REQ_set_version(req, 2);
    X509_NAME* subj = X509_REQ_get_subject_name(req);
    X509_NAME_add_entry_by_txt(
        subj, "CN", MBSTRING_ASC,
        reinterpret_cast<const unsigned char*>(domains[0].c_str()), -1, -1, 0);
    {
        auto* exts = sk_X509_EXTENSION_new_null();
        std::string san;
        for (size_t i = 0; i < domains.size(); ++i) {
            if (i) san += ",";
            san += "DNS:" + domains[i];
        }
        if (auto* ext = X509V3_EXT_conf_nid(nullptr, nullptr,
                                            NID_subject_alt_name, san.c_str()))
            sk_X509_EXTENSION_push(exts, ext);
        X509_REQ_add_extensions(req, exts);
        sk_X509_EXTENSION_pop_free(exts, X509_EXTENSION_free);
    }
    X509_REQ_set_pubkey(req, cert_key);
    X509_REQ_sign(req, cert_key, EVP_sha256());

    int der_len = i2d_X509_REQ(req, nullptr);
    std::string der(static_cast<size_t>(der_len), '\0');
    unsigned char* dp = reinterpret_cast<unsigned char*>(der.data());
    i2d_X509_REQ(req, &dp);
    X509_REQ_free(req);

    auto fin = acme_post(
        ctx, finalize_url,
        {{"csr", b64url(reinterpret_cast<const unsigned char*>(der.data()),
                        der.size())}});
    if (fin.status < 200 || fin.status >= 300) {
        error_out = "finalize failed: HTTP " + std::to_string(fin.status) +
                    " " + fin.raw.substr(0, 400);
        return out;
    }

    // Poll order until valid, then download the chain (leaf first).
    std::string cert_url;
    for (int i = 0; i < 60; ++i) {
        auto poll = acme_post(ctx, order_url, json());
        auto st = poll.body.value("status", "");
        if (st == "valid") {
            cert_url = poll.body.value("certificate", "");
            break;
        }
        if (st == "invalid") {
            error_out = "order invalid: " + poll.raw.substr(0, 500);
            return out;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    if (cert_url.empty()) {
        error_out = "order never became valid";
        return out;
    }
    auto cert_res = acme_post(ctx, cert_url, json());
    if (cert_res.status != 200) {
        error_out =
            "certificate download failed: HTTP " + std::to_string(cert_res.status);
        return out;
    }
    out.cert_pem = cert_res.raw;

    // Export the cert private key as PEM for the caller (proxy needs it).
    BIO* kb = BIO_new(BIO_s_mem());
    PEM_write_bio_PrivateKey(kb, cert_key, nullptr, nullptr, 0, nullptr, nullptr);
    char* kd = nullptr;
    long klen = BIO_get_mem_data(kb, &kd);
    out.key_pem.assign(kd, static_cast<size_t>(klen));
    BIO_free(kb);

    out.ok = !out.cert_pem.empty() && !out.key_pem.empty();
    return out;
}

}  // namespace hostely::exposure