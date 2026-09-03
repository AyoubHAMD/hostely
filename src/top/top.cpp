#include "top/top.hpp"

#include "services/cli.hpp"
#include "services/manager.hpp"
#include "top/render.hpp"
#include "top/sample.hpp"
#include "top/tui.hpp"

#include <unistd.h>

#include <chrono>
#include <iostream>
#include <map>
#include <thread>
#include <utility>

namespace hostely::top {

namespace {

using clock = std::chrono::steady_clock;

// Flags: --interval N (seconds), --sort cpu|mem|name|net, --once.
double parse_interval(const cli::ParsedArgs& args) {
    if (auto v = args.get("interval")) {
        try {
            return std::stod(*v);
        } catch (...) { /* fall through to default */ }
    }
    return 2.0;
}

SortKey parse_sort(const cli::ParsedArgs& args) {
    if (auto v = args.get("sort")) {
        if (*v == "mem")  return SortKey::Mem;
        if (*v == "name") return SortKey::Name;
        if (*v == "net")  return SortKey::Net;
    }
    return SortKey::Cpu;
}

void print_manager_error(const services::ManagerError& e) {
    std::cerr << e.message << '\n';
    if (!e.stderr_tail.empty()) {
        std::cerr << "--- container stderr ---\n" << e.stderr_tail << '\n';
    }
}

int print_usage() {
    std::cout << "Usage: hostely top [--interval N] [--sort cpu|mem|name|net] [--once]\n"
                 "\n"
                 "Live dashboard of containers: CPU, memory, network, disk, processes.\n"
                 "\n"
                 "Keys: q quit   ↑↓ select   s sort   +/- interval   k stop (confirm)\n";
    return 0;
}

}  // namespace

int run_top(const config::Config& cfg, const cli::ParsedArgs& args) {
    const double interval = parse_interval(args);
    const SortKey sort_key = parse_sort(args);
    const bool once = args.has("once");

    if (args.has("help") || args.has("h")) return print_usage();

    auto cli_path = services::cli::which(cfg.container_cli);
    if (!cli_path) {
        std::cerr << services::install_instructions() << '\n';
        return 1;
    }
    services::Manager mgr(*cli_path);

    // Fast pre-flight so a stopped container system gets a clean error
    // instead of a broken TUI.
    auto ready = mgr.system_is_ready();
    if (!ready.ok()) {
        print_manager_error(*ready.error);
        return 1;
    }

    if (!once && !isatty(STDOUT_FILENO)) {
        std::cerr << "hostely top: stdout is not a terminal; use --once for a "
                     "one-shot snapshot.\n";
        return 1;
    }

    if (once) {
        // Two passes so deltas exist; plain text to stdout (scriptable).
        std::map<std::string, RawSample> a, b;
        std::string list_json;
        SnapshotStore store;
        store.set_interval(interval);
        sample_once(mgr, a, b, "", list_json, 1, interval, store);
        auto snap = sample_once(mgr, b, a, "", list_json, 2, interval, store);
        std::cout << render_snapshot_text(snap, sort_key);
        return 0;
    }

    Terminal term;
    if (!term.ready()) {
        std::cerr << "hostely top: terminal is not capable of live rendering.\n";
        return 1;
    }

    // ---- sampler thread ----------------------------------------------------
    SnapshotStore store;
    store.set_interval(interval);
    store.sort = static_cast<int>(sort_key);

    std::map<std::string, RawSample> prev, next;
    std::string list_json;
    int sample_count = 0;

    std::thread sampler([&] {
        while (!store.stop.load()) {
            double itv = store.interval();
            auto t0 = clock::now();
            ++sample_count;
            sample_once(mgr, prev, next, "", list_json, sample_count, itv, store);
            // The pass just filled `next` from `prev`; swap so the next
            // iteration computes deltas against what we have now.
            std::swap(prev, next);
            // Sleep the remaining interval; cli::run latency eats into it.
            // Interval changes take effect after the current pass completes.
            auto elapsed = std::chrono::duration<double>(clock::now() - t0);
            double rest = itv - elapsed.count();
            for (double slept = 0; slept < rest && !store.stop.load(); slept += 0.1) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });

    // ---- render loop -------------------------------------------------------
    term.enter();
    std::string confirm_kill;
    int selected = 0;
    int view_offset = 0;

    auto draw = [&] {
        auto snap = store.load();
        auto key = static_cast<SortKey>(store.sort.load());
        selected = store.selected.load();
        auto lines = render_screen(snap, key, selected, confirm_kill,
                                   term.size(), term.color(), view_offset);
        std::cout << "\x1b[H";
        for (std::size_t i = 0; i < lines.size(); ++i) {
            std::cout << lines[i] << "\x1b[K";
            if (i + 1 < lines.size()) std::cout << "\r\n";
        }
        std::cout.flush();
    };

    try {
        while (true) {
            if (term.resized()) {
                std::cout << "\x1b[2J";
                view_offset = 0;
            }
            draw();

            // Frame pacing: poll keys for ~250ms, repaint at ~4 fps.
            for (int i = 0; i < 12; ++i) {
                if (Terminal::interrupted()) throw std::runtime_error("interrupted");
                auto pressed = term.read_key(20);
                if (!pressed) continue;
                Key key = *pressed;
                if (key == Key::Other || key == Key::Enter) continue;

                if (key == Key::Quit) {
                    if (!confirm_kill.empty()) {
                        confirm_kill.clear();   // Esc cancels the stop prompt
                        break;
                    }
                    throw std::runtime_error("quit");
                }

                if (confirm_kill.empty()) {
                    switch (key) {
                        case Key::Up: {
                            int sel = store.selected.load();
                            store.selected.store(sel > 0 ? sel - 1 : 0);
                            break;
                        }
                        case Key::Down: {
                            int sel = store.selected.load() + 1;
                            auto snap = store.load();
                            int max = static_cast<int>(snap.rows.size()) - 1;
                            if (sel > max) sel = max;
                            store.selected.store(sel);
                            break;
                        }
                        case Key::Sort: {
                            int s = store.sort.load();
                            store.sort.store((s + 1) % 4);
                            store.selected.store(0);
                            view_offset = 0;
                            break;
                        }
                        case Key::IntervalUp:
                            store.set_interval(store.interval() * 2.0);
                            break;
                        case Key::IntervalDown:
                            store.set_interval(store.interval() / 2.0);
                            break;
                        case Key::Kill: {
                            auto snap = store.load();
                            auto rows = snap.rows;
                            sort_rows(rows, static_cast<SortKey>(store.sort.load()));
                            int sel = store.selected.load();
                            if (sel >= 0 && sel < static_cast<int>(rows.size())) {
                                confirm_kill = rows[sel].name;
                            }
                            break;
                        }
                        default: break;
                    }
                } else if (key == Key::Kill) {
                    // Second press: do the stop. Blocks a few seconds while
                    // the CLI runs; the sampler keeps sampling.
                    auto name = confirm_kill;
                    confirm_kill.clear();
                    auto r = mgr.stop(name);
                    if (!r.ok()) {
                        auto snap = store.load();
                        snap.error = r.error->message;
                        store.store(std::move(snap));
                    }
                }
            }
        }
    } catch (const std::exception&) {
        // quit or interrupted — the normal exit path
    }

    store.stop.store(true);
    if (sampler.joinable()) sampler.join();
    term.leave();
    return 0;
}

}  // namespace hostely::top