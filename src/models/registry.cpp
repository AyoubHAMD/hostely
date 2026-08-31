#include "models/registry.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <toml++/toml.hpp>

namespace fs = std::filesystem;
using hostely::models::Manifest;

namespace {

bool valid_stem_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-';
}

}  // namespace

namespace hostely::models {

std::string make_stem(const std::string& repo,
                      const std::string& quant) {
    // Sanitise each segment independently; join with "--" to keep `hostely
    // models rm user--repo--quant` unambiguous (filenames often contain "-").
    auto clean = [](const std::string& in) {
        std::string out;
        out.reserve(in.size());
        for (char c : in) {
            out.push_back(valid_stem_char(c) ? c : '_');
        }
        return out;
    };
    return clean(repo) + "--" + clean(quant);
}

fs::path sidecar_for(const fs::path& dir, const std::string& stem) {
    return dir / (stem + ".toml");
}

std::optional<Manifest> read_manifest(const fs::path& manifest_path) {
    if (!fs::exists(manifest_path)) return std::nullopt;
    try {
        auto tbl = toml::parse_file(manifest_path.string());
        Manifest m;
        m.manifest_path = manifest_path;
        m.path          = fs::path(tbl["path"].value_or(""));
        m.source        = tbl["source"].value_or("");
        m.repo          = tbl["repo"].value_or("");
        m.quant         = tbl["quant"].value_or("");
        m.filename      = tbl["filename"].value_or("");
        m.size_bytes    = tbl["size_bytes"].value_or<uint64_t>(0);
        m.license       = tbl["license"].value_or("");
        m.pulled_at     = tbl["pulled_at"].value_or("");
        // Sanity-correct: if path is relative, anchor to the sidecar's directory.
        if (!m.path.empty() && m.path.is_relative()) {
            m.path = manifest_path.parent_path() / m.path;
        }
        // Verify the gguf actually exists; if not, blank the path but keep
        // the manifest so the user can still see what they pulled.
        if (!m.path.empty() && !fs::exists(m.path)) {
            m.path.clear();
        }
        // Fill in filename if the sidecar didn't have it.
        if (m.filename.empty() && !m.path.empty()) {
            m.filename = m.path.filename().string();
        }
        return m;
    } catch (const toml::parse_error& e) {
        return std::nullopt;
    }
}

bool write_manifest(const Manifest& m) {
    toml::table tbl;
    // `path` is stored relative to the sidecar when possible so the
    // registry is portable across machines.
    fs::path stored_path = m.path;
    if (!stored_path.empty() && !m.manifest_path.empty()) {
        std::error_code ec;
        fs::path rel = fs::relative(stored_path, m.manifest_path.parent_path(), ec);
        if (!ec && !rel.empty() && rel.native()[0] != '.') {
            stored_path = rel;
        }
    }
    tbl.insert_or_assign("path",
        stored_path.generic_string());
    if (!m.source.empty())     tbl.insert_or_assign("source",   m.source);
    if (!m.repo.empty())       tbl.insert_or_assign("repo",     m.repo);
    if (!m.quant.empty())      tbl.insert_or_assign("quant",    m.quant);
    if (!m.filename.empty())   tbl.insert_or_assign("filename", m.filename);
    if (m.size_bytes != 0)     tbl.insert_or_assign("size_bytes", static_cast<int64_t>(m.size_bytes));
    if (!m.license.empty())    tbl.insert_or_assign("license",  m.license);
    if (!m.pulled_at.empty())  tbl.insert_or_assign("pulled_at", m.pulled_at);

    std::ofstream out(m.manifest_path);
    if (!out) return false;
    out << tbl;
    return out.good();
}

std::vector<Manifest> list(const fs::path& dir) {
    std::vector<Manifest> out;
    if (!fs::exists(dir)) return out;

    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        const auto& p = entry.path();
        if (p.extension() != ".toml") continue;
        auto m = read_manifest(p);
        if (!m) continue;
        // Logical name is the stem (filename without `.toml`).
        out.push_back(std::move(*m));
    }

    // Sort by logical stem so `hostely models list` is deterministic.
    std::sort(out.begin(), out.end(),
              [](const Manifest& a, const Manifest& b) {
                  return a.manifest_path.filename().string() <
                         b.manifest_path.filename().string();
              });
    return out;
}

std::optional<Manifest> resolve(const fs::path& dir,
                                const std::string& name,
                                std::string& error_out) {
    // 1. Absolute (or existing) path: just stat it; we still need a manifest
    //    to attach metadata. If there's a sidecar in `dir` whose `path`
    //    matches, prefer it. Otherwise synthesise a minimal manifest without
    //    writing one (so users can serve an arbitrary .gguf).
    fs::path candidate = name;
    if (candidate.string().find('/') != std::string::npos) {
        if (!fs::exists(candidate)) {
            error_out = "model file not found: " + candidate.string();
            return std::nullopt;
        }
        // Try to attach metadata from any matching sidecar. Comparing
        // canonical paths is portable and good enough — the gguf lives next
        // to its sidecar so filepath collisions are user error.
        std::error_code ec;
        fs::path cand_canon = fs::weakly_canonical(candidate, ec);
        if (ec) cand_canon = fs::absolute(candidate);
        for (const auto& m : list(dir)) {
            if (m.path.empty()) continue;
            fs::path m_canon = fs::weakly_canonical(m.path, ec);
            if (ec) m_canon = fs::absolute(m.path);
            if (m_canon == cand_canon) return m;
        }
        Manifest m;
        m.path = fs::absolute(candidate);
        m.manifest_path.clear();   // not in registry yet
        m.filename = candidate.filename().string();
        if (fs::exists(m.path)) m.size_bytes = fs::file_size(m.path);
        return m;
    }

    // 2. Logical name (full stem, with `--` separators).
    auto all = list(dir);
    for (const auto& m : all) {
        if (m.manifest_path.stem().string() == name) return m;
    }

    // 3. Unique-prefix match (case-sensitive on filesystem, so we don't have
    //    to pretend macOS HFS+ is case-insensitive here — that bites users
    //    on APFS sometimes). Case-insensitive unique-prefix is more friendly.
    auto lc = [](std::string s) {
        for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return s;
    };
    std::string needle_lc = lc(name);
    std::vector<Manifest> hits;
    for (const auto& m : all) {
        std::string stem_lc = lc(m.manifest_path.stem().string());
        if (stem_lc.rfind(needle_lc, 0) == 0) hits.push_back(m);
    }
    if (hits.size() == 1) return hits.front();
    if (hits.empty()) {
        std::ostringstream oss;
        oss << "no model in registry matches '" << name << "'.\n"
            << "  Try `hostely models list` or pass an absolute .gguf path.";
        error_out = oss.str();
        return std::nullopt;
    }
    std::ostringstream oss;
    oss << "'" << name << "' is ambiguous; matches " << hits.size()
        << " models. Be more specific.";
    error_out = oss.str();
    return std::nullopt;
}

}  // namespace hostely::models
