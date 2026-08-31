#include "models/pull.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include "services/cli.hpp"

namespace fs = std::filesystem;
using nlohmann::json;
using hostely::models::Manifest;
using hostely::models::PullError;
using hostely::models::PullRequest;
using hostely::models::PullResult;

namespace hostely::models {

const std::vector<std::string> kRecognisedQuants = {
    "F32", "F16", "Q8_0",
    "Q6_K", "Q5_K_L", "Q5_K_M", "Q5_K_S", "Q5_1", "Q5_0",
    "Q4_K_L", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
    "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q3_K_XS",
    "Q2_K", "Q2_K_S",
    "IQ4_XS", "IQ4_NL", "IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_XS", "IQ2_XXS", "IQ1_M",
    "BF16",
};

namespace {

bool looks_like_url(const std::string& s) {
    return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0;
}

// Convert operands like "user/repo:Q4_K_M" or "user/repo/path/file.gguf" or
// a URL into a (kind, payload) tuple. `kind` is one of:
//   "url", "repo-quant", "repo-file".
struct Resolved {
    std::string kind;
    std::string repo;          // "user/repo" or "" for direct url
    std::string quant;         // "Q4_K_M" or "" if not a quant query
    std::string file_in_repo;  // filename inside the HF repo, or "" for direct url
    std::string url;           // full URL to download from
};

std::optional<Resolved> resolve_operand(const std::string& operand) {
    Resolved r;

    if (looks_like_url(operand)) {
        r.kind = "url";
        r.url  = operand;
        // Heuristically extract a filename.
        auto slash = operand.find_last_of('/');
        r.file_in_repo = (slash == std::string::npos) ? "model.gguf" : operand.substr(slash + 1);

        // Try to detect a quant suffix in the filename — this lets the
        // sidecar carry a meaningful `--Q4_K_M` tag rather than `--url`.
        for (const auto& q : kRecognisedQuants) {
            std::string suffix = "." + q + ".gguf";
            const auto& f = r.file_in_repo;
            if (f.size() > suffix.size() &&
                f.compare(f.size() - suffix.size(), suffix.size(), suffix) == 0) {
                r.quant = q;
                break;
            }
        }
        return r;
    }

    // Identify `user/repo[:quant]` or `user/repo/<path/file.gguf>` forms.
    auto colon = operand.find(':');
    auto first_slash = operand.find('/');
    if (first_slash == std::string::npos) return std::nullopt; // not valid

    // The colon, if present, must be *after* the slash (otherwise it's not
    // user/repo:quant — we don't allow colons in repo paths).
    if (colon != std::string::npos && colon > first_slash) {
        r.repo = operand.substr(0, colon);
        r.quant = operand.substr(colon + 1);
        r.kind  = "repo-quant";
        return r;
    }

    // Either `user/repo` (no quant — we will pick Q4_K_M by default) or
    // `user/repo/path/file.gguf`.
    auto second_slash = operand.find('/', first_slash + 1);
    if (second_slash == std::string::npos) {
        r.repo = operand;
        r.quant = "Q4_K_M";   // default
        r.kind  = "repo-quant";
        return r;
    }
    // Treat as a path inside a repo.
    auto last_slash = operand.find_last_of('/');
    r.repo          = operand.substr(0, last_slash);
    r.file_in_repo  = operand.substr(last_slash + 1);
    r.kind          = "repo-file";
    r.url           = "https://huggingface.co/" + r.repo + "/resolve/main/" + r.file_in_repo;
    return r;
}

// Parse HF model API JSON (`{"siblings":[{"rfilename":"..."}], ...}`) and
// pick the most appropriate sibling for the requested quant.
struct HfSibling {
    std::string rfilename;
    std::uint64_t size_bytes = 0;
};

bool parse_api_json(const std::string& body, std::vector<HfSibling>& out) {
    try {
        auto j = json::parse(body);
        if (!j.contains("siblings") || !j["siblings"].is_array()) return false;
        for (const auto& s : j["siblings"]) {
            if (!s.contains("rfilename")) continue;
            std::string fname = s.value("rfilename", "");
            if (fname.empty()) continue;
            HfSibling sib;
            sib.rfilename = std::move(fname);
            sib.size_bytes = s.value("size", static_cast<std::uint64_t>(0));
            out.push_back(std::move(sib));
        }
        return !out.empty();
    } catch (const json::exception&) {
        return false;
    }
}

// Pick the largest file ending in `.<quant>.gguf`. If none ends in the
// requested quant, fall back to the file with the alphabetically nearest quant.
bool pick_sibling(const std::vector<HfSibling>& sibs,
                  const std::string& quant,
                  std::size_t& idx_out,
                  std::vector<std::pair<std::string, std::uint64_t>>& available_out) {
    std::vector<std::size_t> gguf_idx;
    for (std::size_t i = 0; i < sibs.size(); ++i) {
        const auto& f = sibs[i].rfilename;
        if (f.size() >= 5 && f.compare(f.size() - 5, 5, ".gguf") == 0) {
            gguf_idx.push_back(i);
            available_out.emplace_back(f, sibs[i].size_bytes);
        }
    }
    if (gguf_idx.empty()) return false;

    // `file.Q4_K_M.gguf` — from the end: ".gguf" (5), the quant, then a '.'.
    auto ends_with_quant = [&](const std::string& fname, const std::string& q) -> bool {
        // 1 (dot) + q.size() + 5 (.gguf) + at least 1 char before the dot.
        if (fname.size() < q.size() + 7) return false;
        auto quant_at = fname.size() - 5 - q.size();
        if (fname.compare(quant_at, q.size(), q) != 0) return false;
        if (fname[quant_at - 1] != '.') return false;
        if (fname.compare(fname.size() - 5, 5, ".gguf") != 0) return false;
        return true;
    };

    // Pass 1: exact quant match. Pick the largest (first wins on ties, so
    // a repo reporting all sizes as 0 still resolves deterministically).
    std::size_t best_exact = std::string::npos;
    std::uint64_t best_exact_size = 0;
    for (auto i : gguf_idx) {
        if (ends_with_quant(sibs[i].rfilename, quant)) {
            if (best_exact == std::string::npos ||
                sibs[i].size_bytes > best_exact_size) {
                best_exact = i;
                best_exact_size = sibs[i].size_bytes;
            }
        }
    }
    if (best_exact != std::string::npos) { idx_out = best_exact; return true; }

    // Pass 2: skip the quant and pick the smallest quant we recognise.
    std::string best_quant;
    std::size_t best_quant_idx = std::string::npos;
    std::uint64_t best_quant_size = std::numeric_limits<std::uint64_t>::max();
    for (auto i : gguf_idx) {
        const auto& f = sibs[i].rfilename;
        for (const auto& q : kRecognisedQuants) {
            if (ends_with_quant(f, q)) {
                if (best_quant_idx == std::string::npos ||
                    sibs[i].size_bytes < best_quant_size) {
                    best_quant_size = sibs[i].size_bytes;
                    best_quant = q;
                    best_quant_idx = i;
                }
                break;
            }
        }
    }
    if (best_quant_idx == std::string::npos) {
        // Fallback: just pick the largest gguf in the repo (first on ties).
        std::uint64_t biggest = 0;
        std::size_t biggest_idx = std::string::npos;
        for (auto i : gguf_idx) {
            if (biggest_idx == std::string::npos ||
                sibs[i].size_bytes > biggest) {
                biggest = sibs[i].size_bytes;
                biggest_idx = i;
            }
        }
        if (biggest_idx != std::string::npos) { idx_out = biggest_idx; return true; }
        return false;
    }
    idx_out = best_quant_idx;
    return true;
}

// License pulled from the model cardData, if present (best-effort).
std::string extract_license(const std::string& body) {
    try {
        auto j = json::parse(body);
        if (j.contains("cardData") && j["cardData"].is_object()) {
            auto& c = j["cardData"];
            if (c.contains("license") && c["license"].is_string()) {
                return c["license"].get<std::string>();
            }
        }
    } catch (...) {}
    return "";
}

std::string now_iso8601() {
    auto now = std::time(nullptr);
    struct tm tm_buf{};
#if defined(_WIN32)
    gmtime_s(&tm_buf, &now);
#else
    gmtime_r(&now, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

}  // namespace

PullResult pull(const PullRequest& req) {
    PullResult result;

    auto resolved = resolve_operand(req.operand);
    if (!resolved) {
        result.error.message =
            "could not interpret '" + req.operand + "' as a Hugging Face repo, "
            "a `repo:quant` selector, or a direct URL.";
        return result;
    }

    // Direct URL: skip HF API resolution entirely.
    Resolved& r = *resolved;
    std::vector<HfSibling> sibs;
    std::string license;
    if (r.kind == "repo-quant") {
        // Hit HF model API. `blobs=true` makes the API include per-file sizes
        // (the plain form returns siblings without sizes, which breaks the
        // "pick the largest" selection below).
        std::string api_url = "https://huggingface.co/api/models/" + r.repo + "?blobs=true";
        auto api = services::cli::run({"/usr/bin/curl", "-fsSL", "--connect-timeout", "15", "--max-time", "30", api_url});
        if (!api.started() || api.exit_code != 0) {
            result.error.message =
                "huggingface.co model API failed: " + api.stderr_text +
                " (HTTP " + std::to_string(api.exit_code) + ")";
            return result;
        }
        if (!parse_api_json(api.stdout_text, sibs)) {
            result.error.message =
                "could not parse Hugging Face model metadata for '" + r.repo + "'.";
            return result;
        }
        license = extract_license(api.stdout_text);

        std::size_t idx = 0;
        if (!pick_sibling(sibs, r.quant, idx, result.error.available_files)) {
            result.error.message =
                "no GGUF files found in '" + r.repo + "'.";
            return result;
        }

        // Did we land on the requested quant? If not, warn via the manifest's
        // quant field and update the URL accordingly.
        const HfSibling& chosen = sibs[idx];
        r.file_in_repo = chosen.rfilename;
        r.url = "https://huggingface.co/" + r.repo + "/resolve/main/" + r.file_in_repo;
        // Pull out the actual quant we ended up with: look for a known quant
        // in the filename.
        r.quant.clear();
        for (const auto& q : kRecognisedQuants) {
            std::string suffix = "." + q + ".gguf";
            if (chosen.rfilename.size() >= suffix.size() &&
                chosen.rfilename.compare(chosen.rfilename.size() - suffix.size(),
                                         suffix.size(), suffix) == 0) {
                r.quant = q;
                break;
            }
        }
        if (r.quant.empty()) r.quant = "unknown";
        result.manifest.size_bytes = chosen.size_bytes;
    } else if (r.kind == "repo-file") {
        // Already have a full URL. We don't know the size up front; let curl
        // tell us via the trailing marker line. The model API would also
        // work but is overkill if the user named an exact file.
        result.manifest.size_bytes = 0;
        license.clear();   // unknown without a separate API call
    } else {
        // Direct URL
        result.manifest.size_bytes = 0;
    }

    // Resolve a destination path.
    fs::create_directories(req.dest_dir);

    // When there's no repo context (direct URL or repo/file form), synthesize
    // a stable, descriptive stem from the source URL — `hostely models rm
    // <stem>` should always work even for one-off pulls.
    std::string stem;
    if (r.repo.empty()) {
        // Use the URL's last two meaningful path segments for uniqueness.
        // Example: https://huggingface.co/USER/REPO/resolve/main/f.Q4_K_M.gguf
        //   → USER--REPO--f_Q4_K_M
        auto host = r.url.find("://");
        std::string path = (host == std::string::npos) ? r.url : r.url.substr(host + 3);
        auto scheme_end = path.find('/');
        std::string rest = (scheme_end == std::string::npos) ? "" : path.substr(scheme_end + 1);
        while (!rest.empty() && rest.front() == '/') rest.erase(rest.begin());
        // Take the last two segments.
        std::string segs[2] = {"", ""};
        for (int i = 0; i < 2; ++i) {
            auto slash = rest.find('/');
            if (slash == std::string::npos) { segs[i] = rest; rest.clear(); break; }
            segs[i] = rest.substr(0, slash);
            rest = rest.substr(slash + 1);
            if (rest.empty()) break;
        }
        std::string a = segs[0], b = segs[1];
        stem = (a.empty() ? "url" : a) + "--" +
               (b.empty() ? std::string("file") : b) + "--" +
               (r.quant.empty() ? std::string("url") : r.quant);
    } else {
        stem = make_stem(r.repo, r.quant);
    }

    fs::path dest_gguf = req.dest_dir / r.file_in_repo;
    fs::path sidecar   = sidecar_for(req.dest_dir, stem);
    fs::path tmp_gguf  = dest_gguf.string() + ".tmp";

    // If the final file already exists with the right size, skip download.
    if (fs::exists(dest_gguf) && fs::exists(sidecar)) {
        std::uint64_t on_disk = fs::file_size(dest_gguf);
        if (result.manifest.size_bytes != 0 && on_disk == result.manifest.size_bytes) {
            // Already there.
            Manifest m;
            m.path          = fs::absolute(dest_gguf);
            m.manifest_path = fs::absolute(sidecar);
            m.source        = r.url;
            m.repo          = r.repo;
            m.quant         = r.quant;
            m.filename      = r.file_in_repo;
            m.size_bytes    = on_disk;
            m.license       = license;
            m.pulled_at     = now_iso8601();
            if (write_manifest(m)) {
                result.manifest = std::move(m);
                result.ok = true;
                return result;
            }
        }
    }

    // Spawn curl. Use -fL (fail on 4xx/5xx, follow redirects), -C - (resume),
    // -w to print our marker line at the end. We want stderr on the user's
    // terminal so the progress bar shows.
    std::string marker = "hostely_marker:";   // something unlikely
    std::string size_marker = "hostely_done:"; // intentionally distinct

    std::vector<std::string> argv = {
        "/usr/bin/curl",
        "-fL",
        "--connect-timeout", "15",
        "--retry", "3",
        "-C", "-",
        "-o", tmp_gguf.string(),
        "-w", "\n" + marker + "%{http_code}:%{size_download}:%{time_total}\n",
        r.url,
    };
    auto dl = services::cli::run(argv);
    if (!dl.started()) {
        result.error.message = "could not launch /usr/bin/curl";
        return result;
    }

    // Parse curl's final marker line from stdout.
    bool saw_marker = false;
    long http_code = -1;
    std::uint64_t dl_bytes = 0;
    double dl_secs = 0.0;
    {
        std::istringstream iss(dl.stdout_text);
        std::string line;
        while (std::getline(iss, line)) {
            auto pos = line.find(marker);
            if (pos == std::string::npos) continue;
            std::string rest = line.substr(pos + marker.size());
            std::snprintf(nullptr, 0, "%s", rest.c_str());   // noop
            int scanned = std::sscanf(rest.c_str(), "%ld:%llu:%lf",
                                      &http_code,
                                      (unsigned long long*)&dl_bytes,
                                      &dl_secs);
            saw_marker = scanned == 3;
        }
    }

    if (dl.exit_code != 0 || !saw_marker || http_code >= 400) {
        fs::remove(tmp_gguf);
        std::ostringstream oss;
        oss << "curl exited " << dl.exit_code << " (HTTP " << http_code << ")\n"
            << dl.stderr_text;
        result.error.message = oss.str();
        return result;
    }

    // Move tmp into place.
    std::error_code ec;
    fs::rename(tmp_gguf, dest_gguf, ec);
    if (ec) {
        result.error.message = "could not move downloaded file into place: " + ec.message();
        return result;
    }

    if (result.manifest.size_bytes == 0) result.manifest.size_bytes = dl_bytes;
    Manifest m;
    m.path          = fs::absolute(dest_gguf);
    m.manifest_path = fs::absolute(sidecar);
    m.source        = r.url;
    m.repo          = r.repo;
    m.quant         = r.quant;
    m.filename      = r.file_in_repo;
    m.size_bytes    = result.manifest.size_bytes != 0
                         ? result.manifest.size_bytes
                         : dl_bytes;
    m.license       = license;
    m.pulled_at     = now_iso8601();
    if (!write_manifest(m)) {
        // Don't fail — the gguf is downloaded. We just couldn't persist metadata.
        m.manifest_path.clear();
    }
    result.manifest = std::move(m);
    result.ok = true;
    return result;
}

}  // namespace hostely::models
