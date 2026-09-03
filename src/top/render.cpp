#include "top/render.hpp"

#include "resources/system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>

namespace hostely::top {

namespace {

// ---------------------------------------------------------------------------
// formatting helpers
// ---------------------------------------------------------------------------

// UTF-8 glyphs as byte strings (char literals can't hold them).
constexpr const char* kBarFill = "\xe2\x96\x88";   // █
constexpr const char* kBarRest = "\xe2\x96\x91";   // ░
constexpr const char* kMarker  = "\xe2\x96\xb8";   // ▸
constexpr const char* kDots    = "\xe2\x80\xa6";   // …

std::string pad(const std::string& s, std::size_t w) {
    std::string out = s;
    if (out.size() < w) out.append(w - out.size(), ' ');
    return out;
}

// Clip for display. Widths are in bytes (good enough for the ASCII-heavy
// container names/images we render); a "…" suffix when cut.
std::string clip(const std::string& s, std::size_t w) {
    if (s.size() <= w) return s;
    if (w <= 3) return s.substr(0, w);
    return s.substr(0, w - 3) + kDots;
}

// Compact byte string: 918M, 1.0G, 42K, 3B (one decimal below 100).
std::string compact_bytes(double bytes) {
    if (bytes < 0) return "-";
    static const char* units[] = {"B", "K", "M", "G", "T"};
    double v = bytes;
    int u = 0;
    while (v >= 1024.0 && u < 4) { v /= 1024.0; ++u; }
    char buf[32];
    if (v >= 100.0 || u == 0)
        std::snprintf(buf, sizeof buf, "%.0f%s", v, units[u]);
    else
        std::snprintf(buf, sizeof buf, "%.1f%s", v, units[u]);
    return buf;
}

std::string fmt_rate(double bps) {
    if (bps < 0) return "-";
    return compact_bytes(bps) + "/s";
}

std::string fmt_pct(double pct) {
    if (pct < 0) return "-";
    char buf[16];
    if (pct >= 100.0) std::snprintf(buf, sizeof buf, "%.0f%%", pct);
    else std::snprintf(buf, sizeof buf, "%.1f%%", pct);
    return buf;
}

// 10-cell gauge. ASCII bars ('#'/'-') when color is off — block glyphs stay
// tied to the color path, which implies a capable terminal.
std::string bar(double pct, bool color) {
    const int cells = 10;
    int filled = 0;
    if (pct > 0) filled = static_cast<int>(std::lround(pct / 100.0 * cells));
    filled = std::clamp(filled, 0, cells);
    std::string s;
    if (color) {
        for (int i = 0; i < cells; ++i) s += (i < filled ? kBarFill : kBarRest);
    } else {
        for (int i = 0; i < cells; ++i) s += (i < filled ? '#' : '-');
    }
    return s;
}

// Threshold color for a gauge; green/yellow/red by load bands.
const char* heat(double pct, double yellow, double red) {
    if (pct >= red)   return "\x1b[31m";
    if (pct >= yellow) return "\x1b[33m";
    return "\x1b[32m";
}

std::string colorize(const std::string& text, const char* code, bool color) {
    if (!color) return text;
    return std::string(code) + text + "\x1b[0m";
}

std::string dim(const std::string& text, bool color) {
    return colorize(text, "\x1b[2m", color);
}

}  // namespace

// ---------------------------------------------------------------------------
// sorting
// ---------------------------------------------------------------------------

void sort_rows(std::vector<ContainerRow>& rows, SortKey key) {
    auto unknown_last = [](double v) { return v < 0; };
    switch (key) {
        case SortKey::Cpu:
            std::sort(rows.begin(), rows.end(), [&](const ContainerRow& a, const ContainerRow& b) {
                if (unknown_last(a.cpu_pct) != unknown_last(b.cpu_pct))
                    return unknown_last(b.cpu_pct);
                if (a.cpu_pct != b.cpu_pct) return a.cpu_pct > b.cpu_pct;
                return a.name < b.name;
            });
            break;
        case SortKey::Mem:
            std::sort(rows.begin(), rows.end(), [](const ContainerRow& a, const ContainerRow& b) {
                if (a.mem_bytes != b.mem_bytes) return a.mem_bytes > b.mem_bytes;
                return a.name < b.name;
            });
            break;
        case SortKey::Net:
            std::sort(rows.begin(), rows.end(), [](const ContainerRow& a, const ContainerRow& b) {
                auto na = (a.net_rx_bps > 0 ? a.net_rx_bps : 0) + (a.net_tx_bps > 0 ? a.net_tx_bps : 0);
                auto nb = (b.net_rx_bps > 0 ? b.net_rx_bps : 0) + (b.net_tx_bps > 0 ? b.net_tx_bps : 0);
                if (na != nb) return na > nb;
                return a.name < b.name;
            });
            break;
        case SortKey::Name:
            std::sort(rows.begin(), rows.end(), [](const ContainerRow& a, const ContainerRow& b) {
                return a.name < b.name;
            });
            break;
    }
}

// ---------------------------------------------------------------------------
// table rendering
// ---------------------------------------------------------------------------

namespace {

struct ColumnSpec {
    int name_w = 22;
};

constexpr int kMemW = 12;    // "918M/1.0G "
constexpr int kBarW = 10;

// One table row; widths match the header in render_screen below.
std::string render_row_line(const ContainerRow& r,
                            const ColumnSpec& col,
                            std::size_t total_width,
                            bool color,
                            bool selected) {
    std::string name_col = pad(clip(r.name, static_cast<std::size_t>(col.name_w)),
                               static_cast<std::size_t>(col.name_w));

    std::string mem = compact_bytes(static_cast<double>(r.mem_bytes));
    if (r.mem_limit > 0) mem += "/" + compact_bytes(static_cast<double>(r.mem_limit));
    mem = pad(mem, kMemW);

    std::string membar = bar(r.mem_pct, color);
    std::string mempct = pad(fmt_pct(r.mem_pct), 7);

    std::string net = pad(fmt_rate(r.net_rx_bps), 9) + " " + pad(fmt_rate(r.net_tx_bps), 9);
    std::string disk = pad(fmt_rate(r.blk_rd_bps), 9) + " " + pad(fmt_rate(r.blk_wr_bps), 9);

    std::string cpu = pad(fmt_pct(r.cpu_pct), 7);
    std::string procs = pad(r.procs > 0 ? std::to_string(r.procs) : "-", 3);

    std::string body = name_col + cpu + mem + membar + " " + mempct + net +
                       "  " + disk + "  " + procs;
    std::string image = "";
    if (total_width > body.size() + 1) {
        image = " " + clip(r.image, total_width - body.size() - 1);
    }

    if (!color) {
        return (selected ? "> " : "  ") + body + image;
    }

    std::ostringstream c;
    c << (selected ? "\x1b[1m" : "");
    c << (selected ? kMarker : " ") << " \x1b[0m";
    c << name_col;
    // CPU
    if (r.cpu_pct >= 0) c << heat(r.cpu_pct, 50, 80) << cpu << "\x1b[0m";
    else c << dim(cpu, true);
    // MEM + gauge
    if (r.mem_pct >= 0) {
        const char* hs = heat(r.mem_pct, 60, 85);
        c << hs << mem << "\x1b[0m" << hs << membar << "\x1b[0m";
    } else {
        c << dim(mem, true) << dim(membar, true);
    }
    c << " " << dim(mempct + net, true) << "  " << dim(disk + "  " + procs, true);
    c << image;
    return c.str();
}

}  // namespace

std::vector<std::string> render_screen(const Snapshot& snap,
                                       SortKey sort_key,
                                       int selected,
                                       const std::string& confirm_kill,
                                       TermSize size,
                                       bool color,
                                       int& view_offset) {
    std::vector<std::string> lines;
    const int width = size.cols > 0 ? size.cols : 80;
    const int height = size.rows > 0 ? size.rows : 24;

    auto rows = snap.rows;
    sort_rows(rows, sort_key);
    if (selected >= static_cast<int>(rows.size())) selected = static_cast<int>(rows.size()) - 1;
    if (selected < 0) selected = 0;

    const char* sort_names[] = {"CPU", "MEM", "NAME", "NET"};
    const char* sn = sort_names[static_cast<int>(sort_key)];

    // ---- header ------------------------------------------------------------
    {
        std::ostringstream h;
        h << " hostely top   " << snap.rows.size() << " container"
          << (snap.rows.size() == 1 ? "" : "s") << "   "
          << (snap.sample_count > 0 ? "live" : "sampling…") << "   "
          << colorize(std::string("[sort: ") + sn + "]", "\x1b[1m", color)
          << "   q quit   ↑↓ select   s sort   +/- interval   k stop";
        lines.push_back(h.str());
    }

    // host cpu line (load-1 normalized by cores, as a rough "how busy")
    {
        double load = snap.host.cores > 0 ? snap.host.load1 / snap.host.cores : 0;
        double cpup = std::clamp(load, 0.0, 1.0) * 100.0;
        std::ostringstream h;
        std::string b = bar(cpup, color);
        h << " cpu " << (color ? heat(cpup, 50, 80) + b + "\x1b[0m" : b) << " "
          << fmt_pct(cpup) << "   load " << snap.host.load1 << " "
          << snap.host.load5 << " " << snap.host.load15 << "   "
          << snap.host.cores << " cores";
        lines.push_back(h.str());
    }

    // host mem line
    {
        double memp = snap.host.mem_total > 0
                          ? 100.0 * static_cast<double>(snap.host.mem_used) /
                                static_cast<double>(snap.host.mem_total)
                          : 0;
        std::ostringstream h;
        std::string b = bar(memp, color);
        h << " mem " << (color ? heat(memp, 60, 85) + b + "\x1b[0m" : b) << " "
          << fmt_pct(memp) << " of " << resources::human_bytes(snap.host.mem_total)
          << "   used " << resources::human_bytes(snap.host.mem_used);
        lines.push_back(h.str());
    }

    lines.push_back(std::string(static_cast<std::size_t>(std::min(width, 100)), '-'));

    // ---- rows --------------------------------------------------------------
    const int header_h = 5;   // top line + cpu + mem + divider + col header
    const int footer_h = 2;
    int visible = height - header_h - footer_h;
    if (visible < 1) visible = 1;

    int offset = view_offset;
    if (offset > selected) offset = selected;
    if (offset < selected - visible + 1) offset = selected - visible + 1;
    if (offset < 0) offset = 0;
    if (offset > static_cast<int>(rows.size()) - 1) offset = static_cast<int>(rows.size()) - 1;
    if (offset < 0) offset = 0;
    view_offset = offset;

    ColumnSpec col;
    if (width < 100) col.name_w = 16;

    if (rows.empty()) {
        lines.push_back("   no containers running — start one with `hostely run …` "
                        "or `hostely app up`");
    } else {
        std::ostringstream hdr;
        hdr << "  " << pad("NAME", static_cast<std::size_t>(col.name_w))
            << pad("CPU%", 7) << pad("MEM", kMemW) << pad("MEM", kBarW) << " "
            << pad("MEM%", 7) << pad("NET RX", 9) << " " << pad("NET TX", 9)
            << "  " << pad("DISK RD", 9) << " " << pad("DISK WR", 9)
            << "  " << pad("P", 3) << "IMAGE";
        lines.push_back(dim(hdr.str(), color));

        int shown = 0;
        for (int i = offset; i < static_cast<int>(rows.size()) && shown < visible; ++i, ++shown) {
            lines.push_back(render_row_line(rows[i], col,
                                            static_cast<std::size_t>(width),
                                            color, i == selected));
        }
        if (offset + shown < static_cast<int>(rows.size())) {
            lines.push_back(dim("   … " + std::to_string(rows.size() - offset - shown) +
                                " more (↑↓ to scroll)", color));
        }
    }

    lines.push_back(std::string(static_cast<std::size_t>(std::min(width, 100)), '-'));

    // ---- footer ------------------------------------------------------------
    if (!confirm_kill.empty()) {
        lines.push_back(colorize(" press k again to stop '" + confirm_kill +
                                 "' — Esc to cancel", "\x1b[31m", color));
    } else if (!snap.error.empty()) {
        lines.push_back(colorize(" " + clip(snap.error,
                                 static_cast<std::size_t>(std::max(width - 2, 0))),
                                 "\x1b[31m", color));
    } else {
        std::uint64_t total_mem = 0;
        for (const auto& r : snap.rows) total_mem += r.mem_bytes;
        std::ostringstream f;
        f << " " << rows.size() << " containers   total mem "
          << resources::human_bytes(total_mem) << "   sample #" << snap.sample_count;
        lines.push_back(f.str());
    }

    return lines;
}

std::string render_snapshot_text(const Snapshot& snap, SortKey sort_key) {
    std::ostringstream out;
    auto rows = snap.rows;
    sort_rows(rows, sort_key);

    // Name column fits the longest container id (capped so wide names don't
    // push everything else off screen).
    std::size_t name_w = 4;
    for (const auto& r : rows) name_w = std::max(name_w, r.name.size());
    name_w = std::min(name_w, std::size_t{32}) + 2;

    out << pad("NAME", name_w)
        << "CPU%    MEM          MEM%     NET RX     NET TX"
           "    DISK RD    DISK WR     P  IMAGE\n";
    for (const auto& r : rows) {
        std::string mem = compact_bytes(static_cast<double>(r.mem_bytes));
        if (r.mem_limit > 0) mem += "/" + compact_bytes(static_cast<double>(r.mem_limit));
        out << pad(clip(r.name, name_w), name_w)
            << pad(fmt_pct(r.cpu_pct), 8)
            << pad(mem, 13)
            << pad(fmt_pct(r.mem_pct), 9)
            << pad(fmt_rate(r.net_rx_bps), 11) << pad(fmt_rate(r.net_tx_bps), 11)
            << pad(fmt_rate(r.blk_rd_bps), 11) << pad(fmt_rate(r.blk_wr_bps), 12)
            << pad(r.procs > 0 ? std::to_string(r.procs) : "-", 3)
            << r.image << "\n";
    }
    return out.str();
}

}  // namespace hostely::top