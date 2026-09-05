#include "exposure/certs.hpp"

#include "paths.hpp"

#include <nlohmann/json.hpp>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/x509.h>

#include <chrono>
#include <fstream>
#include <sys/stat.h>
#include <sstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace hostely::exposure {

namespace {

fs::path domain_dir(const std::string& domain) {
    // Domains are hostnames; safe as directory names.
    std::string safe;
    for (char c : domain) {
        safe += (c == '/' || c == '\\') ? '_' : c;
    }
    return paths::certs_dir() / safe;
}

bool write_file_private(const fs::path& p, const std::string& data, bool key) {
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    if (ec) return false;
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    out.close();
    if (!out) return false;
    if (key) {
        ::chmod(p.c_str(), 0600);
    }
    return true;
}

bool read_file(const fs::path& p, std::string& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

}  // namespace

bool cert_store_save(const std::string& domain, const std::string& cert_pem,
                     const std::string& key_pem, const std::string& source) {
    auto dir = domain_dir(domain);
    if (!write_file_private(dir / "cert.pem", cert_pem, false)) return false;
    if (!key_pem.empty() && !write_file_private(dir / "key.pem", key_pem, true))
        return false;

    json meta;
    meta["domain"] = domain;
    meta["source"] = source;
    if (auto info = cert_inspect_pem(cert_pem)) {
        meta["issuer"] = info->issuer;
        meta["not_after"] = info->not_after;
        meta["not_before"] = info->not_before;
    }
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
    meta["issued_at"] = now;
    std::ofstream out(dir / "meta.json", std::ios::trunc);
    if (!out) return false;
    out << meta.dump(2) << '\n';
    return true;
}

bool cert_store_load(const std::string& domain, std::string& cert_pem,
                     std::string& key_pem) {
    auto dir = domain_dir(domain);
    cert_pem.clear();
    key_pem.clear();
    if (!read_file(dir / "cert.pem", cert_pem)) return false;
    read_file(dir / "key.pem", key_pem);
    return true;
}

std::optional<CertInfo> cert_inspect_pem(const std::string& cert_pem) {
    BIO* bio = BIO_new_mem_buf(cert_pem.data(),
                               static_cast<int>(cert_pem.size()));
    if (!bio) return std::nullopt;
    X509* x = PEM_read_bio_X509(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    if (!x) return std::nullopt;

    CertInfo info;
    // Subject CN (fallback; SANs are what matters for matching but CN is a
    // good display name).
    X509_NAME* subj = X509_get_subject_name(x);
    char cn[256] = {0};
    if (X509_NAME_get_text_by_NID(subj, NID_commonName, cn, sizeof cn) > 0) {
        info.domain = cn;
    }

    // Issuer
    char issuer[256] = {0};
    X509_NAME* iss = X509_get_issuer_name(x);
    X509_NAME_oneline(iss, issuer, sizeof issuer);
    // Compact "O=Let's Encrypt, CN=R10" style
    std::string iss_s = issuer;
    info.issuer = iss_s;

    // Simpler: use ASN1_TIME_diff against epoch anchor.
    {
        ASN1_TIME* epoch = ASN1_TIME_new();
        ASN1_TIME_set(epoch, 0);
        int days = 0, secs = 0;
        if (ASN1_TIME_diff(&days, &secs, epoch, X509_get0_notBefore(x)))
            info.not_before = static_cast<std::int64_t>(days) * 86400 + secs;
        if (ASN1_TIME_diff(&days, &secs, epoch, X509_get0_notAfter(x)))
            info.not_after = static_cast<std::int64_t>(days) * 86400 + secs;
        ASN1_TIME_free(epoch);
    }
    X509_free(x);
    return info;
}

std::vector<CertInfo> cert_store_list() {
    std::vector<CertInfo> out;
    std::error_code ec;
    auto root = paths::certs_dir();
    if (!fs::exists(root)) return out;
    for (const auto& entry : fs::directory_iterator(root, ec)) {
        if (!entry.is_directory()) continue;
        std::string cert_pem;
        if (!read_file(entry.path() / "cert.pem", cert_pem)) continue;
        if (auto info = cert_inspect_pem(cert_pem)) {
            // Prefer meta.json source + domain as stored.
            std::ifstream m(entry.path() / "meta.json");
            if (m) {
                try {
                    json j = json::parse(m);
                    if (j.contains("domain")) info->domain = j["domain"];
                    if (j.contains("source")) info->source = j["source"];
                    if (j.contains("issuer")) info->issuer = j["issuer"];
                } catch (...) {}
            }
            out.push_back(*info);
        }
    }
    return out;
}

bool cert_store_remove(const std::string& domain) {
    std::error_code ec;
    fs::remove_all(domain_dir(domain), ec);
    return !ec;
}

int cert_days_remaining(const CertInfo& c) {
    if (c.not_after <= 0) return -1;
    auto now = std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count();
    return static_cast<int>((c.not_after - now) / 86400);
}

}  // namespace hostely::exposure