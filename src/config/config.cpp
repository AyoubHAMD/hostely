#include "config/config.hpp"

#include <toml++/toml.hpp>

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace hostely::config {

Config defaults() {
    Config c;
    // Resolve models_dir only if HOME is set; otherwise leave empty so the
    // save() writer can skip the field.
    if (const char* home = std::getenv("HOME"); home && *home) {
        fs::path p = fs::path(home) / "Library" / "Application Support"
                                       / "hostely" / "models";
        c.models_dir = p.string();
    }
    return c;
}

std::optional<Config> load(const fs::path& path) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        Config c = defaults();
        return c;
    }

    // toml++ (default build) throws on parse error and returns a toml::table.
    toml::table res;
    try {
        res = toml::parse_file(path.string());
    } catch (const toml::parse_error&) {
        return std::nullopt;
    }

    Config c = defaults();
    c.source_path = path;

    if (auto v = res["general"]["default_engine"].value<std::string>())
        c.default_engine = *v;
    if (auto v = res["general"]["default_port"].value<int>())
        c.default_port = *v;

    if (auto v = res["inference"]["ctx_size"].value<int>())
        c.ctx_size = *v;
    if (auto v = res["inference"]["gpu_layers"].value<int>())
        c.gpu_layers = *v;
    if (auto v = res["inference"]["models_dir"].value<std::string>())
        c.models_dir = *v;

    if (auto v = res["services"]["container_cli"].value<std::string>())
        c.container_cli = *v;

    return c;
}

bool save(const Config& cfg, const fs::path& path) {
    std::error_code ec;
    if (auto parent = path.parent_path(); !parent.empty()) {
        fs::create_directories(parent, ec);
        if (ec) return false;
    }

    toml::table t;

    {
        auto g = toml::table{
            {"default_engine", cfg.default_engine},
            {"default_port",  cfg.default_port},
        };
        t.insert("general", std::move(g));
    }
    {
        auto i = toml::table{
            {"ctx_size",    cfg.ctx_size},
            {"gpu_layers",  cfg.gpu_layers},
            {"models_dir",  cfg.models_dir},
        };
        t.insert("inference", std::move(i));
    }
    {
        auto s = toml::table{
            {"container_cli", cfg.container_cli},
        };
        t.insert("services", std::move(s));
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;
    out << t;
    return out.good();
}

}  // namespace hostely::config
