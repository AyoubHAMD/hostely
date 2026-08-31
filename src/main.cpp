// hostely — entry point.
//
// Phase 2 scope (this file):
//   - All of Phase 1 (CLI parsing, logger, init, doctor) is preserved.
//   - `hostely run <name> --image <img> [--port h:c]...` shells out to the
//     Apple `container` CLI and prints the new container ID.
//   - `hostely ps` lists running services in a tabular format.
//   - `hostely stop <name>` is idempotent.
//   - `hostely logs <name> [--follow]` returns captured logs.
//   - `hostely doctor` now actually probes the `container` CLI.
//
// Phase 3+ adds: serve / models / status / resources.

#include "cli/args.hpp"
#include "config/config.hpp"
#include "inference/lockfile.hpp"
#include "inference/server.hpp"
#include "log/logger.hpp"
#include "models/pull.hpp"
#include "models/registry.hpp"
#include "paths.hpp"
#include "resources/limits.hpp"
#include "resources/metal_probe.hpp"
#include "resources/system.hpp"
#include "services/cli.hpp"
#include "services/manager.hpp"

#include <atomic>
#include <csignal>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kBanner =
    "hostely 0.1.0 (phase 2 — services)";

namespace fs = std::filesystem;

// ----------------------------------------------------------------------------
// help / version
// ----------------------------------------------------------------------------

void print_help() {
    std::cout <<
        "Usage: hostely <command> [options]\n"
        "\n"
        "Commands:\n"
        "  init        create config dir + default config.toml\n"
        "  run         run a service from an OCI image\n"
        "  ps          list running services\n"
        "  stop        stop a service\n"
        "  logs        show logs for a service\n"
        "  serve       serve an LLM (llama.cpp / ggml-metal backend)\n"
        "  models      pull/list/path/rm local GGUF models from Hugging Face\n"
        "  status      system-wide CPU/RAM/GPU memory pressure\n"
        "  resources   detailed per-service resource accounting\n"
        "  doctor      verify hostely can run on this machine\n"
        "\n"
        "Global options:\n"
        "  --log-level LEVEL   debug | info | warn | error (default: info)\n"
        "  --no-log-file       disable the rotating logfile sink\n"
        "  --help              print this message and exit\n"
        "  --version           print version and exit\n"
        "\n"
        "Run 'hostely <command> --help' for command-specific options.\n";
}

// ----------------------------------------------------------------------------
// shared helpers
// ----------------------------------------------------------------------------

// Build a Manager. If the CLI is not on PATH, prints install instructions to
// stderr and returns std::nullopt.
std::optional<hostely::services::Manager> make_manager(
        const hostely::config::Config& cfg,
        bool warn_if_missing = true) {
    using namespace hostely;
    auto path = services::cli::which(cfg.container_cli);
    if (!path) {
        if (warn_if_missing) {
            std::cerr << services::install_instructions() << '\n';
        }
        return std::nullopt;
    }
    return services::Manager(*path);
}

void print_manager_error(const hostely::services::ManagerError& e) {
    using namespace hostely;
    using namespace hostely::log;
    switch (e.kind) {
        case services::ManagerError::Kind::CliNotFound:
            error(e.message);
            if (!e.stderr_tail.empty()) {
                error("--- container stderr ---");
                error(e.stderr_tail);
            }
            break;
        case services::ManagerError::Kind::SystemNotStarted:
            warn(e.message);
            break;
        case services::ManagerError::Kind::BadInvocation:
            error(e.message);
            break;
        case services::ManagerError::Kind::NonZeroExit:
            error(e.message);
            if (!e.stderr_tail.empty()) {
                error("--- container stderr ---");
                error(e.stderr_tail);
            }
            break;
    }
}

// ----------------------------------------------------------------------------
// init
// ----------------------------------------------------------------------------

int run_init() {
    using namespace hostely;
    using namespace hostely::log;

    info("hostely init");

    if (!paths::ensure_state_dir()) {
        error("could not create state directory");
        return 1;
    }
    if (!paths::ensure_log_dir()) {
        error("could not create log directory");
        return 1;
    }

    auto cfg_path = paths::config_file();
    if (std::filesystem::exists(cfg_path)) {
        info("config already exists, leaving it untouched:");
        info("  " + cfg_path.string());
        return 0;
    }

    auto cfg = config::defaults();
    if (!config::save(cfg, cfg_path)) {
        error("failed to write config file at " + cfg_path.string());
        return 1;
    }
    info("wrote default config:");
    info("  " + cfg_path.string());
    info("state directory:");
    info("  " + paths::state_dir().string());
    info("log directory:");
    info("  " + paths::log_dir().string());
    return 0;
}

// ----------------------------------------------------------------------------
// run
// ----------------------------------------------------------------------------

// Parse "--port 8080:80" repeated. The flag is repeatable.
std::vector<std::pair<std::string, std::string>>
parse_ports(const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    std::vector<std::pair<std::string, std::string>> out;
    // args.get returns the value of the LAST occurrence; we want all of them,
    // so iterate options_ directly.
    for (std::size_t i = 0; i < args.options().size(); ++i) {
        if (args.options()[i].name != "port") continue;
        const auto& v = args.options()[i].value;
        auto colon = v.find(':');
        if (colon == std::string::npos) {
            out.emplace_back(v, v);  // same port on host and container
        } else {
            out.emplace_back(v.substr(0, colon), v.substr(colon + 1));
        }
    }
    return out;
}

int run_run(const hostely::config::Config& cfg,
            const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    using namespace hostely::log;

    auto operands = args.operands();
    if (operands.empty()) {
        std::cerr << "Usage: hostely run <name> --image <image> [--port h:c]...\n";
        return 2;
    }
    services::RunOptions opts;
    opts.name = std::string(operands[0]);
    if (auto v = args.get("image")) opts.image = *v;
    opts.ports = parse_ports(args);
    // TODO(phase 2+): --env K=V, --volume src:dst, -- arg...

    if (opts.image.empty()) {
        error("--image is required");
        return 2;
    }

    // Apply resource limits to our own process before spawning the
    // container child. Note: on macOS these limits apply to hostely itself,
    // not to the container (which runs in its own VM). The intent for v1
    // is to protect the hostely manager from runaway containers spawned
    // repeatedly. Jetsam memlimit will only apply when running as root.
    int mem_gb = -1;
    if (auto v = args.get("mem-gb")) mem_gb = std::stoi(*v);
    if (mem_gb > 0) {
        auto limits = resources::apply_process_limits(mem_gb);
        if (limits.jetsam_attempted) {
            if (limits.jetsam_applied) {
                info("applied Jetsam memlimit: " + std::to_string(mem_gb) + " GB");
            } else {
                warn("Jetsam memlimit not applied: " + limits.jetsam_error);
                if (!limits.running_as_root) {
                    warn("(this usually requires running hostely as root)");
                }
            }
        }
    }

    auto mgr = make_manager(cfg);
    if (!mgr) return 1;

    auto outcome = mgr->run(opts);
    if (!outcome.ok()) {
        print_manager_error(*outcome.error);
        return 1;
    }
    info("started '" + opts.name + "' as " + outcome.value.value());
    std::cout << outcome.value.value() << '\n';
    return 0;
}

// ----------------------------------------------------------------------------
// ps
// ----------------------------------------------------------------------------

std::string pad_right(const std::string& s, std::size_t width) {
    if (s.size() >= width) return s;
    return s + std::string(width - s.size(), ' ');
}

int run_ps(const hostely::config::Config& cfg) {
    using namespace hostely;
    using namespace hostely::log;

    auto mgr = make_manager(cfg);
    if (!mgr) return 1;

    auto outcome = mgr->list();
    if (!outcome.ok()) {
        print_manager_error(*outcome.error);
        return 1;
    }

    const auto& rows = outcome.value.value();
    if (rows.empty()) {
        std::cout << "(no services running)\n";
        return 0;
    }

    // Compute column widths.
    std::size_t w_name = 4, w_image = 5, w_status = 6, w_ports = 5, w_id = 2;
    for (const auto& r : rows) {
        w_name   = std::max(w_name,   r.name.size());
        w_image  = std::max(w_image,  r.image.size());
        w_status = std::max(w_status, r.status.size());
        w_ports  = std::max(w_ports,  r.ports.size());
        w_id     = std::max(w_id,     r.id.size());
    }

    auto row = [&](const std::string& a, const std::string& b,
                   const std::string& c, const std::string& d,
                   const std::string& e) {
        std::cout << pad_right(a, w_name) << "  "
                  << pad_right(b, w_image) << "  "
                  << pad_right(c, w_status) << "  "
                  << pad_right(d, w_ports) << "  "
                  << pad_right(e, w_id) << '\n';
    };

    row("NAME", "IMAGE", "STATUS", "PORTS", "ID");
    for (const auto& r : rows) row(r.name, r.image, r.status, r.ports, r.id);
    return 0;
}

// ----------------------------------------------------------------------------
// stop
// ----------------------------------------------------------------------------

int run_stop(const hostely::config::Config& cfg,
             const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    using namespace hostely::log;

    auto operands = args.operands();
    if (operands.empty()) {
        std::cerr << "Usage: hostely stop <name>\n";
        return 2;
    }

    auto mgr = make_manager(cfg);
    if (!mgr) return 1;

    for (auto name : operands) {
        auto outcome = mgr->stop(std::string(name));
        if (!outcome.ok()) {
            warn(std::string("stop(") + std::string(name) + ") failed:");
            print_manager_error(*outcome.error);
            return 1;
        }
        info("stopped " + std::string(name));
    }
    return 0;
}

// ----------------------------------------------------------------------------
// logs
// ----------------------------------------------------------------------------

int run_logs(const hostely::config::Config& cfg,
             const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    using namespace hostely::log;

    auto operands = args.operands();
    if (operands.empty()) {
        std::cerr << "Usage: hostely logs <name> [--follow]\n";
        return 2;
    }
    bool follow = args.has("follow") || args.has("f");

    auto mgr = make_manager(cfg);
    if (!mgr) return 1;

    auto outcome = mgr->logs(std::string(operands[0]), follow);
    if (!outcome.ok()) {
        print_manager_error(*outcome.error);
        return 1;
    }
    std::cout << outcome.value.value();
    if (!outcome.value.value().empty() &&
        outcome.value.value().back() != '\n') std::cout << '\n';
    return 0;
}

// ----------------------------------------------------------------------------
// serve (Phase 3)
// ----------------------------------------------------------------------------

namespace {
// Global pointer the SIGINT handler uses to stop the server cleanly.
std::atomic<hostely::inference::Server*> g_active_server{nullptr};

void handle_sigint(int) {
    if (auto* s = g_active_server.load()) s->stop();
}
}  // namespace

int run_serve(const hostely::config::Config& cfg,
              const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    using namespace hostely::log;

    auto operands = args.operands();
    if (operands.empty()) {
        std::cerr <<
            "Usage: hostely serve <name|model.gguf> [options]\n"
            "\n"
            "Options:\n"
            "  --port N           listening port (default: " << cfg.default_port << ")\n"
            "  --ctx-size N       context size in tokens (default: " << cfg.ctx_size << ")\n"
            "  --gpu-layers N     layers to offload to Metal (-1 = all)\n"
            "  --threads N        CPU threads (0 = auto)\n"
            "  --engine NAME      llama (only one supported on this build)\n"
            "  --no-fit-check     skip the Phase 7b pre-load memory check\n";
        return 2;
    }

    // Resolve the operand: an existing path (or one that contains '/')
    // is used as-is; otherwise it's a logical name looked up against the
    // models registry. Models pulled by `hostely models pull user/repo:quant`
    // arrive as `user/repo:quant` strings, so we fall through to name
    // resolution when the path doesn't exist.
    std::string operand(operands[0]);
    fs::path resolved_path;
    fs::path candidate_path = operand;
    bool resolved = false;
    if (fs::exists(candidate_path)) {
        resolved_path = candidate_path;
        resolved = true;
    } else {
        std::string err;
        auto m = models::resolve(paths::models_dir(), operand, err);
        if (m && !m->path.empty()) {
            resolved_path = m->path;
            resolved = true;
            info("resolved '" + operand + "' -> " + resolved_path.string());
        } else if (fs::path(operand).extension() == ".gguf") {
            // Looks like an explicit .gguf path; honour the user's intent
            // and produce the same "not found" error the old code did.
        }
    }
    if (!resolved) {
        error("model file not found: " + operand);
        error("(use `hostely models list` to see pulled models, or pass an absolute path)");
        return 1;
    }

    inference::ServeOptions opts;
    opts.model_path = resolved_path;
    if (args.has("no-fit-check")) opts.no_fit_check = true;

    // --engine: we only support "llama" (ggml-metal on Apple Silicon).
    // If the user passes --engine mlx we surface a clear message rather
    // than silently ignore it; the option exists for forward compatibility
    // when/if llama.cpp adds an MLX backend.
    std::string engine = "llama";
    if (auto v = args.get("engine")) engine = *v;
    if (engine != "llama") {
        error("engine '" + engine + "' is not supported on this build.");
        error("on Apple Silicon, hostely uses llama.cpp + ggml-metal,");
        error("which is the recommended Apple-native inference path.");
        error("if you need a separate MLX server, run mlx-lm externally");
        error("and point your client at its port.");
        return 2;
    }

    if (auto v = args.get("port"))      opts.port      = std::stoi(*v);
    else                                opts.port      = cfg.default_port;
    if (auto v = args.get("ctx-size"))  opts.ctx_size  = std::stoi(*v);
    else                                opts.ctx_size  = cfg.ctx_size;
    if (auto v = args.get("gpu-layers")) opts.gpu_layers = std::stoi(*v);
    else                                 opts.gpu_layers = cfg.gpu_layers;
    if (auto v = args.get("threads"))    opts.threads    = std::stoi(*v);

    info("starting inference server (engine=llama / ggml-metal)");
    info("  model     : " + opts.model_path.string());
    info("  port      : " + std::to_string(opts.port));
    info("  ctx_size  : " + std::to_string(opts.ctx_size));
    info("  gpu_layers: " + std::to_string(opts.gpu_layers));
    info("  threads   : " + std::to_string(opts.threads));

    inference::Server srv(opts);
    g_active_server.store(&srv);

    // Forward SIGINT to the server so Ctrl-C exits cleanly.
    std::signal(SIGINT, handle_sigint);

    int rc = srv.start();
    g_active_server.store(nullptr);
    return rc;
}

// ----------------------------------------------------------------------------
// status (Phase 5)
// ----------------------------------------------------------------------------

int run_status() {
    using namespace hostely;
    using namespace hostely::log;

    info("hostely status");

    // ---- memory --------------------------------------------------------
    auto mem = resources::read_memory();
    if (mem.valid) {
        std::cout << "memory\n"
                  << "  total     : " << resources::human_bytes(mem.total_bytes) << "\n"
                  << "  free      : " << resources::human_bytes(mem.free_bytes) << "\n"
                  << "  active    : " << resources::human_bytes(mem.active_bytes) << "\n"
                  << "  inactive  : " << resources::human_bytes(mem.inactive_bytes) << "\n"
                  << "  wired     : " << resources::human_bytes(mem.wired_bytes) << "\n"
                  << "  compressed: " << resources::human_bytes(mem.compressed_bytes) << "\n";
        if (mem.total_bytes > 0) {
            std::uint64_t used = mem.active_bytes + mem.wired_bytes + mem.compressed_bytes;
            int pct = static_cast<int>(used * 100 / mem.total_bytes);
            std::cout << "  used%     : " << pct << "%\n";
        }
    } else {
        warn("could not read memory stats");
    }

    // ---- cpu ------------------------------------------------------------
    auto cpu = resources::read_cpu();
    if (cpu.valid) {
        std::cout << "cpu\n"
                  << "  logical cores  : " << cpu.logical_cores << "\n"
                  << "  physical cores : " << cpu.physical_cores << "\n"
                  << "  load (1/5/15m) : "
                  << cpu.load_1min << " / "
                  << cpu.load_5min << " / "
                  << cpu.load_15min << "\n"
                  << "  self rss       : "
                  << resources::human_bytes(cpu.self.rss_bytes) << "\n"
                  << "  self virtual   : "
                  << resources::human_bytes(cpu.self.virtual_bytes) << "\n";
    } else {
        warn("could not read cpu stats");
    }

    // ---- model headroom (Phase 7c) ---------------------------------------
    // Same accounting the Phase 7b fit advisor uses: free+inactive minus a
    // 2 GiB safety margin is what's available to a new `hostely serve`.
    if (mem.valid) {
        constexpr std::uint64_t kSafety = 2ull << 30;   // 2 GiB
        std::uint64_t free_inactive = mem.free_bytes + mem.inactive_bytes;
        std::uint64_t avail = (free_inactive > kSafety) ? free_inactive - kSafety
                                                        : 0;
        std::cout << "memory model headroom\n"
                  << "  available for serve : " << resources::human_bytes(avail)
                  << "   (= free + inactive - 2 GiB safety)\n"
                  << "  hostely process rss : "
                  << resources::human_bytes(cpu.valid ? cpu.self.rss_bytes : 0)
                  << "\n";
    }

    // ---- served model (Phase 7c) -----------------------------------------
    // Reads the serve.lock.json published by a running `hostely serve`.
    // Absent lockfile = nothing serving. Stale pid = crash leftover.
    {
        std::string lock_err;
        bool stale = false;
        auto lock = inference::read_serve_lock(lock_err, stale);
        if (lock) {
            if (stale) {
                warn("served model : " + lock_err);
            } else {
                // Parameter count as "1.10 B" / "1100 M" — a count, not bytes.
                auto human_params = [](std::uint64_t n) {
                    std::ostringstream oss;
                    if (n >= 1'000'000'000)
                        oss << std::fixed << std::setprecision(2)
                            << (n / 1e9) << " B";
                    else if (n >= 1'000'000)
                        oss << std::fixed << std::setprecision(1)
                            << (n / 1e6) << " M";
                    else
                        oss << n;
                    return oss.str();
                };
                std::cout << "served model\n"
                          << "  name   : " << lock->display_name << "\n"
                          << "  quant  : " << (lock->quant_name.empty() ? "-"
                                                                        : lock->quant_name)
                          << "\n"
                          << "  params : " << human_params(lock->params) << "\n"
                          << "  ctx    : " << lock->ctx_size
                          << " / " << lock->ctx_train << " trained\n"
                          << "  ram    : " << resources::human_bytes(lock->peak_rss_bytes)
                          << " peak (estimated)\n"
                          << "  port   : " << lock->port << "\n"
                          << "  pid    : " << lock->pid << "\n"
                          << "  since  : " << lock->started_at << "\n";
            }
        } else if (!lock_err.empty()) {
            warn("served model : " + lock_err);
        }
        // Absent lock + empty error: nothing serving — print nothing.
    }

    // ---- metal ----------------------------------------------------------
    // On Apple Silicon, unified memory means CPU RSS is also (roughly) the
    // GPU's working set. We still probe Metal because `recommendedMaxWorkingSetSize`
    // is the only public signal for when the OS will start evicting.
    //
    // Caveat: `currentAllocatedSize` only counts allocations made through
    // Metal's public allocator by *this process*. Frameworks like llama.cpp
    // keep weights in private pools invisible to it, so the "current alloc"
    // number can be near zero even when this process holds gigabytes on the
    // GPU. Use `self rss` above for an honest per-process figure.
    if (hostely_metal_available()) {
        char name[64];
        hostely_metal_device_name(name, sizeof(name));
        auto cur  = hostely_metal_current_allocated_bytes();
        auto max  = hostely_metal_recommended_max_bytes();
        std::cout << "metal\n"
                  << "  device              : " << name << "\n"
                  << "  recommended max     : " << resources::human_bytes(max) << "\n"
                  << "  current (pub. API)  : " << resources::human_bytes(cur) << "\n";
        if (max > 0) {
            int pct = static_cast<int>(cur * 100 / max);
            std::cout << "  utilisation         : " << pct << "% (public-API only)\n";
        }
        if (max > 0 && cur > max) {
            warn("Metal public-API allocation exceeds recommended maximum; "
                 "OS may evict and degrade inference speed.");
        }
    } else {
        std::cout << "metal\n"
                  << "  device : (none / Metal unavailable)\n";
    }

    return 0;
}

// ----------------------------------------------------------------------------
// models (Phase 7a)
// ----------------------------------------------------------------------------

int run_models(const hostely::cli::ParsedArgs& args) {
    using namespace hostely;
    using namespace hostely::log;

    // `hostely models <subcommand> ...` — args.command() returns "models";
    // the subcommand is the first operand.
    auto operands = args.operands();

    if (operands.empty()) {
        std::cerr <<
            "Usage: hostely models <subcommand> [args]\n"
            "\n"
            "Subcommands:\n"
            "  pull <repo>[:quant] | <url>     download a GGUF (Phase 7a)\n"
            "  list                              show registry contents\n"
            "  path <name>                       print the resolved path\n"
            "  rm <name>                         remove a model + its sidecar\n";
        return 2;
    }

    // Copy operands into std::string so the rest of the function isn't
    // juggling string_views.
    std::vector<std::string> ops;
    ops.reserve(operands.size());
    for (auto v : operands) ops.emplace_back(v);

    auto subcmd = ops.front();
    std::vector<std::string> rest(ops.begin() + 1, ops.end());

    auto dir = paths::models_dir();
    if (!paths::ensure_models_dir()) {
        error("could not create models directory: " + dir.string());
        return 1;
    }

    if (subcmd == "list") {
        auto items = hostely::models::list(dir);
        if (items.empty()) {
            std::cout << "(no models in registry — `hostely models pull` to add one)\n";
            return 0;
        }
        std::printf("%-50s  %-9s  %-12s  %s\n",
                    "NAME", "QUANT", "SIZE", "LICENSE");
        for (const auto& m : items) {
            std::string name = m.manifest_path.stem().string();
            if (name.size() > 50) name = name.substr(0, 47) + "...";
            std::printf("%-50s  %-9s  %-12s  %s\n",
                        name.c_str(),
                        m.quant.empty() ? "-" : m.quant.c_str(),
                        m.size_bytes ? resources::human_bytes(m.size_bytes).c_str() : "-",
                        m.license.empty() ? "-" : m.license.c_str());
        }
        return 0;
    }

    if (subcmd == "path") {
        if (rest.empty()) {
            std::cerr << "Usage: hostely models path <name>\n"; return 2;
        }
        std::string err;
        auto m = hostely::models::resolve(dir, rest[0], err);
        if (!m) { error(err); return 1; }
        if (m->path.empty()) {
            error("matched manifest, but the .gguf file is missing: " + m->manifest_path.string());
            return 1;
        }
        std::cout << m->path.string() << "\n";
        return 0;
    }

    if (subcmd == "rm") {
        if (rest.empty()) {
            std::cerr << "Usage: hostely models rm <name>\n"; return 2;
        }
        std::string err;
        auto m = hostely::models::resolve(dir, rest[0], err);
        if (!m) { error(err); return 1; }
        if (!m->manifest_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(m->manifest_path, ec);
        }
        if (!m->path.empty()) {
            std::error_code ec;
            std::filesystem::remove(m->path, ec);
        }
        info("removed " + rest[0]);
        return 0;
    }

    if (subcmd == "pull") {
        if (rest.empty()) {
            std::cerr << "Usage: hostely models pull <repo>[:quant] | <url>\n"; return 2;
        }
        hostely::models::PullRequest req{ rest[0], dir };
        info("pulling " + rest[0] + " into " + dir.string());
        auto r = hostely::models::pull(req);
        if (!r.ok) {
            if (!r.error.available_files.empty()) {
                std::cout << "available files in this repo:\n";
                for (const auto& f : r.error.available_files) {
                    std::cout << "  " << f.first << "  (" << f.second << " bytes)\n";
                }
            }
            error("pull failed: " + r.error.message);
            return 1;
        }
        info("saved -> " + r.manifest.path.string());
        return 0;
    }

    error("unknown models subcommand: '" + subcmd + "'");
    return 2;
}

// ----------------------------------------------------------------------------
// doctor
// ----------------------------------------------------------------------------

int run_doctor() {
    using namespace hostely;
    using namespace hostely::log;

    info("hostely doctor");

    // --- basic paths -------------------------------------------------------
    auto home = paths::home_dir();
    if (home.empty()) {
        warn("$HOME is not set; hostely cannot resolve state/log directories.");
    } else {
        info("HOME                : " + home.string());
    }
    info("state dir           : " + paths::state_dir().string() +
         (paths::ensure_state_dir() ? "" : " (created)"));
    info("log dir             : " + paths::log_dir().string() +
         (paths::ensure_log_dir() ? "" : " (created)"));
    info("config file         : " + paths::config_file().string() +
         (std::filesystem::exists(paths::config_file())
              ? " (exists)" : " (missing; run 'hostely init')"));

    // --- container CLI -----------------------------------------------------
    auto cfg = config::load(paths::config_file()).value_or(config::defaults());
    auto cli_path = services::cli::which(cfg.container_cli);
    if (cli_path) {
        info("container CLI       : " + *cli_path);
        services::Manager mgr(*cli_path);
        auto sys_ready = mgr.system_is_ready();
        if (sys_ready.ok() && sys_ready.value.value()) {
            info("container system    : ready");
        } else if (sys_ready.error) {
            warn("container system    : not ready — " + sys_ready.error->message);
        }
    } else {
        warn("container CLI       : not on PATH (looking for `" +
             cfg.container_cli + "`)");
        info(services::install_instructions());
    }

    info("llama.cpp build     : linked (ggml-metal enabled)");
    if (hostely_metal_available()) {
        char name[64];
        hostely_metal_device_name(name, sizeof(name));
        auto max = hostely_metal_recommended_max_bytes();
        info("Metal device        : " + std::string(name) +
             " (recommended max " + resources::human_bytes(max) + ")");
    } else {
        warn("Metal device        : not available");
    }
    info("Jetsam memlimit     : requires root + entitlement");

    info("doctor complete.");
    return 0;
}

// ----------------------------------------------------------------------------
// dispatch
// ----------------------------------------------------------------------------

hostely::log::Level parse_level(std::string_view s) {
    using hostely::log::Level;
    if (s == "debug") return Level::Debug;
    if (s == "info")  return Level::Info;
    if (s == "warn")  return Level::Warn;
    if (s == "error") return Level::Error;
    return Level::Info;  // default
}

}  // namespace

int main(int argc, const char* const argv[]) {
    using namespace hostely;
    using namespace hostely::log;

    auto args = cli::ParsedArgs::from_argv(argc, argv);

    // --version is silent on stderr.
    if (args.has("version") || args.has("v")) {
        std::cout << "hostely " << HOSTELY_VERSION << '\n';
        return 0;
    }
    if (args.has("help") || args.has("h") || args.command().empty()) {
        print_help();
        return 0;
    }

    // From here on, we want logs.
    Level level = Level::Info;
    if (auto v = args.get("log-level")) level = parse_level(*v);
    bool file_log = !args.has("no-log-file");

    if (file_log) {
        if (paths::ensure_log_dir()) {
            auto log_path = paths::log_dir() / "hostely.log";
            init(level, log_path.string());
        } else {
            init(level);  // stderr only
            warn("could not create log directory; file logging disabled.");
        }
    } else {
        init(level);
    }

    info(kBanner);

    // Load config (defaults if missing — same as config::load does).
    auto cfg = config::load(paths::config_file()).value_or(config::defaults());

    auto cmd = args.command();
    if (cmd == "init")    return run_init();
    if (cmd == "doctor")  return run_doctor();
    if (cmd == "run")     return run_run(cfg, args);
    if (cmd == "ps")      return run_ps(cfg);
    if (cmd == "stop")    return run_stop(cfg, args);
    if (cmd == "logs")    return run_logs(cfg, args);
    if (cmd == "serve")   return run_serve(cfg, args);
    if (cmd == "status")  return run_status();
    if (cmd == "models")  return run_models(args);

    // Remaining commands get a stub for now so --help is honest about scope.
    if (cmd == "resources") {
        warn("'resources' is not implemented yet (planned for a later phase).");
        return 1;
    }

    std::cerr << "hostely: unknown command '" << cmd << "'\n"
              << "Run 'hostely --help' for usage.\n";
    return 2;
}
