#include "top/render.hpp"

#include "resources/system.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace hostely::top {

namespace {

// ---------------------------------------------------------------------------
// glyphs + formatting helpers
// ---------------------------------------------------------------------------

constexpr const char* kBarFill = "\xe2\x96\x88";   // █
constexpr const char* kBarRest = "\xe2\x96\x91";   // ░
constexpr const char* kMarker  = "\xe2\x96\xb8";   // ▸
constexpr const char* kDots    = "\xe2\x80\xa6";   // …
constexpr const char* kUp      = "\xe2\x86\x91";   // ↑
constexpr const char* kDown    = "\xe2\x86\x93";   // ↓

// box drawing (rounded)
constexpr const char* kBTopL = "\xe2\x95\xad";     // ╭
constexpr const char* kBTopR = "\xe2\x95\xae";     // ╮
constexpr const char* kBBotL = "\xe2\x95\xb0";     // ╰
constexpr const char* kBBotR = "\xe2\x95\xaf";     // ╯
constexpr const char* kBHorz = "\xe2\x94\x80";     // ─
constexpr const char* kBVert = "\xe2\x94\x82";     // │

// theme: dim cyan frames, colored bodies (btop-ish)
constexpr const char* kFrame  = "\x1b[2;36m";
constexpr const char* kTitle  = "\x1b[1;36m";
constexpr const char* kCpuCol = "\x1b[92m";        // bright green
constexpr const char* kMemCol = "\x1b[95m";        // bright magenta
constexpr const char* kNetCol = "\x1b[96m";        // bright cyan
constexpr const char* kReset  = "\x1b[0m";

// Column-aware padding: pads to *display columns* (the … ellipsis and ▸
// marker are multi-byte but single-column; byte padding shifts every column
// after a clipped name).
int disp_width(const std::string& s);
std::string pad(const std::string& s, std::size_t w) {
    int dw = disp_width(s);
    std::string out = s;
    if (static_cast<std::size_t>(dw) < w) out.append(w - static_cast<std::size_t>(dw), ' ');
    return out;
}

// Clip for display to `w` columns (bytes != columns for multi-byte glyphs);
// a "…" suffix when cut.
std::string clip(const std::string& s, std::size_t w) {
    if (static_cast<std::size_t>(disp_width(s)) <= w) return s;
    if (w <= 1) return s.substr(0, w);
    // walk to w-1 display columns, then append the 1-col ellipsis
    std::string out;
    std::size_t cols = 0;
    for (std::size_t i = 0; i < s.size();) {
        if (cols + 1 > w - 1) break;
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        out += s.substr(i, len);
        i += len;
        ++cols;
    }
    return out + kDots;
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
    return std::string(code) + text + kReset;
}

std::string dim(const std::string& text, bool color) {
    return colorize(text, "\x1b[2m", color);
}

// ---------------------------------------------------------------------------
// panels (rounded boxes with an embedded title)
// ---------------------------------------------------------------------------

// ╭─ title ────…────╮   (title may carry ANSI; width counts display cols)
std::string panel_top(const std::string& title, int width, bool color) {
    // display width of title (ANSI escapes are 0 cols; UTF-8 glyphs 1 col).
    // Names we pass are ASCII, so byte length is fine after stripping escapes.
    int tw = 0;
    for (std::size_t i = 0; i < title.size();) {
        if (title[i] == '\x1b') {  // skip escape sequence
            while (i < title.size() && title[i] != 'm') ++i;
            ++i;
            continue;
        }
        // count one column per UTF-8 lead byte (all our glyphs are 1-col)
        unsigned char c = static_cast<unsigned char>(title[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        i += len;
        ++tw;
    }
    int inner = width - 2 - tw - 2 - 1;   // corners + " title " + right side
    if (inner < 1) inner = 1;
    std::string s;
    if (color) s += kFrame;
    s += kBTopL + std::string(kBHorz) + (color ? kReset : "") + (color ? kTitle : "")
         + " " + title + " " + (color ? kReset : "") + (color ? kFrame : "");
    for (int i = 0; i < inner; ++i) s += kBHorz;
    s += kBTopR;
    if (color) s += kReset;
    return s;
}

std::string panel_bottom(int width, bool color) {
    std::string s;
    if (color) s += kFrame;
    s += kBBotL;
    for (int i = 0; i < width - 2; ++i) s += kBHorz;
    s += kBBotR;
    if (color) s += kReset;
    return s;
}

// Display width of a string in terminal columns: ANSI escapes count 0,
// multi-byte UTF-8 glyphs count 1 (all glyphs we emit are single-column).
int disp_width(const std::string& s) {
    int w = 0;
    for (std::size_t i = 0; i < s.size();) {
        if (s[i] == '\x1b') {  // ANSI escape: zero display width
            while (i < s.size() && s[i] != 'm') ++i;
            ++i;
            continue;
        }
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        i += len;
        ++w;
    }
    return w;
}

std::string clip_cols(const std::string& s, int cols) {
    if (disp_width(s) <= cols) return s;
    std::string out;
    int w = 0;
    for (std::size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        std::size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        if (w + 1 > cols - 1 && cols >= 1) break;
        out += s.substr(i, len);
        i += len;
        ++w;
    }
    return out + kDots;
}

std::string panel_line(const std::string& text, int width, bool color) {
    int inner = std::max(width - 4, 1);
    std::string body = clip_cols(text, inner);
    int pad_n = inner - disp_width(body);
    std::string s;
    if (color) s += kFrame;
    s += kBVert;
    if (color) s += kReset;
    s += " " + body;
    if (pad_n > 0) s += std::string(static_cast<std::size_t>(pad_n), ' ');
    if (color) s += kFrame;
    s += kBVert;
    if (color) s += kReset;
    return s;
}

// ---------------------------------------------------------------------------
// braille history graph (2 pixel-columns per cell, 4 pixel-rows per cell row)
// ---------------------------------------------------------------------------

// Braille bit for pixel row r (0 = bottom) in the left/right sub-column.
constexpr unsigned char kDotsL[4] = {0x40, 0x04, 0x02, 0x01};  // dot7,3,2,1
constexpr unsigned char kDotsR[4] = {0x80, 0x20, 0x10, 0x08};  // dot8,6,5,4

// values in [0,1]; produces `rows` lines of `cells` braille characters
// (one display column each, two history samples per column).
// The newest value renders at the right edge.
std::vector<std::string> braille_graph(const std::vector<double>& values,
                                       int cells, int rows) {
    std::vector<std::string> out(static_cast<std::size_t>(rows));
    if (cells < 1 || rows < 1) return out;
    const int px_cols = cells * 2;
    const int px_rows = rows * 4;
    for (int row = 0; row < rows; ++row) {
        // output `row` covers pixel rows [ (rows-1-row)*4, +4 ) from bottom
        int base = (rows - 1 - row) * 4;
        std::string line;
        for (int c = 0; c < cells; ++c) {
            unsigned char bits = 0;
            for (int sub = 0; sub < 2; ++sub) {
                int idx = static_cast<int>(values.size()) - px_cols + c * 2 + sub;
                if (idx < 0) continue;
                double v = values[static_cast<std::size_t>(idx)];
                if (v < 0) v = 0;
                v = std::clamp(v, 0.0, 1.0);
                int h = static_cast<int>(std::lround(v * px_rows));
                if (v > 0 && h == 0) h = 1;
                for (int r = 0; r < 4; ++r) {
                    int prow = base + r;             // pixel row from bottom
                    if (prow < h) bits |= (sub == 0 ? kDotsL : kDotsR)[r];
                }
            }
            unsigned int cp = 0x2800u + bits;
            std::string ch;
            ch += static_cast<char>(0xe0 | (cp >> 12));
            ch += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
            ch += static_cast<char>(0x80 | (cp & 0x3f));
            line += ch;
        }
        out[static_cast<std::size_t>(row)] = line;
    }
    return out;
}

// ---------------------------------------------------------------------------
// history (per-sample ring buffers, advanced once per new sample)
// ---------------------------------------------------------------------------

struct GraphHistory {
    std::vector<double> cpu, mem, rx, tx;
    int last_sample = -1;

    void push(const Snapshot& snap) {
        if (snap.sample_count == last_sample) return;
        last_sample = snap.sample_count;
        double cores = snap.host.cores > 0 ? static_cast<double>(snap.host.cores) : 1.0;
        double cpu = std::clamp(snap.host.load1 / cores, 0.0, 1.0);
        double mem = snap.host.mem_total > 0
            ? static_cast<double>(snap.host.mem_used) / static_cast<double>(snap.host.mem_total)
            : 0.0;
        double r = 0, t = 0;
        for (const auto& row : snap.rows) {
            if (row.net_rx_bps > 0) r += row.net_rx_bps;
            if (row.net_tx_bps > 0) t += row.net_tx_bps;
        }
        auto add = [](std::vector<double>& v, double x) {
            v.push_back(x);
            if (v.size() > kHistCap) v.erase(v.begin());
        };
        add(this->cpu, cpu);
        add(this->mem, mem);
        add(rx, r);
        add(tx, t);
    }

    static constexpr std::size_t kHistCap = 300;
};

GraphHistory& hist() {
    static GraphHistory h;
    return h;
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
    c << (selected ? kMarker : " ") << " " << kReset;
    c << name_col;
    // CPU
    if (r.cpu_pct >= 0) c << heat(r.cpu_pct, 50, 80) << cpu << kReset;
    else c << dim(cpu, true);
    // MEM + gauge
    if (r.mem_pct >= 0) {
        const char* hs = heat(r.mem_pct, 60, 85);
        c << hs << mem << kReset << hs << membar << kReset;
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

    hist().push(snap);
    const auto& H = hist().cpu;
    const auto& Hm = hist().mem;
    const auto& Hrx = hist().rx;

    // ---- panel layout ------------------------------------------------------
    // row 1: [ cpu | mem ]   row 2: [ net ]   row 3+: containers   last: status
    const int graph_h = 4;        // border + 2 braille rows + border
    int half_w = width / 2;
    if (width < 80) half_w = width;   // stack when narrow

    if (color) lines.push_back("\x1b[?25l");  // cursor off (harmless if repeated)

    if (half_w < width) {
        // cpu + mem panels side-by-side: build each, then merge line-by-line.
        int rx_w = width - half_w;
        double cores = snap.host.cores > 0 ? static_cast<double>(snap.host.cores) : 1.0;
        double cpup = std::clamp(snap.host.load1 / cores, 0.0, 1.0) * 100.0;
        double memp = snap.host.mem_total > 0
            ? 100.0 * static_cast<double>(snap.host.mem_used) / static_cast<double>(snap.host.mem_total)
            : 0.0;

        std::vector<std::string> lcpu, lmem;
        lcpu.push_back(panel_top("cpu", half_w, color));
        for (const auto& gl : braille_graph(H, half_w - 4, 2))
            lcpu.push_back(panel_line(colorize(gl, kCpuCol, color), half_w, color));
        std::ostringstream t;
        t << std::fixed << std::setprecision(2);
        t << fmt_pct(cpup) << "  load " << snap.host.load1 << " " << snap.host.load5
          << " " << snap.host.load15 << "  " << snap.host.cores << " cores";
        lcpu.push_back(panel_line(t.str(), half_w, color));
        lcpu.push_back(panel_bottom(half_w, color));

        lmem.push_back(panel_top("mem", rx_w, color));
        for (const auto& gl : braille_graph(Hm, rx_w - 4, 2))
            lmem.push_back(panel_line(colorize(gl, kMemCol, color), rx_w, color));
        std::ostringstream mt;
        mt << fmt_pct(memp) << " of " << resources::human_bytes(snap.host.mem_total)
           << "  used " << resources::human_bytes(snap.host.mem_used);
        lmem.push_back(panel_line(mt.str(), rx_w, color));
        lmem.push_back(panel_bottom(rx_w, color));

        for (std::size_t i = 0; i < lcpu.size(); ++i) {
            lines.push_back(lcpu[i] + (i < lmem.size() ? lmem[i] : ""));
        }
    } else {
        // narrow terminal: one combined line per metric, no graphs
        double cores = snap.host.cores > 0 ? static_cast<double>(snap.host.cores) : 1.0;
        double cpup = std::clamp(snap.host.load1 / cores, 0.0, 1.0) * 100.0;
        double memp = snap.host.mem_total > 0
            ? 100.0 * static_cast<double>(snap.host.mem_used) / static_cast<double>(snap.host.mem_total)
            : 0.0;
        std::ostringstream t;
        t << std::fixed << std::setprecision(2);
        t << "cpu " << fmt_pct(cpup) << "  load " << snap.host.load1 << " "
          << snap.host.load5 << " " << snap.host.load15 << "  "
          << snap.host.cores << " cores";
        lines.push_back(panel_top("host", width, color));
        lines.push_back(panel_line(t.str(), width, color));
        lines.push_back(panel_line(std::string("mem ") + fmt_pct(memp) + " of " +
                                   resources::human_bytes(snap.host.mem_total),
                                   width, color));
        lines.push_back(panel_bottom(width, color));
    }

    // ---- net panel ---------------------------------------------------------
    double trx = 0, ttx = 0;
    for (const auto& r : rows) {
        if (r.net_rx_bps > 0) trx += r.net_rx_bps;
        if (r.net_tx_bps > 0) ttx += r.net_tx_bps;
    }
    int net_w = width;
    lines.push_back(panel_top("net", net_w, color));
    {
        auto ng = braille_graph(Hrx, net_w - 4, 2);
        for (const auto& gl : ng)
            lines.push_back(panel_line(colorize(gl, kNetCol, color), net_w, color));
    }
    std::ostringstream nt;
    nt << "rx " << fmt_rate(trx) << "   tx " << fmt_rate(ttx);
    lines.push_back(panel_line(nt.str(), net_w, color));
    lines.push_back(panel_bottom(net_w, color));

    // ---- containers panel --------------------------------------------------
    const int header_h = graph_h + graph_h + graph_h + 1;  // panels + status
    const int footer_h = 2;                                 // bottom border + col hdr
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

    lines.push_back(panel_top("containers", width, color));

    if (rows.empty()) {
        lines.push_back(panel_line("  no containers running — start one with "
                                   "`hostely run …` or `hostely app up`",
                                   width, color));
    } else {
        std::ostringstream hdr;
        hdr << "  " << pad("NAME", static_cast<std::size_t>(col.name_w))
            << pad("CPU%", 7) << pad("MEM", kMemW) << pad("MEM", kBarW) << " "
            << pad("MEM%", 7) << pad("NET RX", 9) << " " << pad("NET TX", 9)
            << "  " << pad("DISK RD", 9) << " " << pad("DISK WR", 9)
            << "  " << pad("P", 3) << "IMAGE";
        lines.push_back(panel_line(dim(hdr.str(), color), width, color));

        int shown = 0;
        for (int i = offset; i < static_cast<int>(rows.size()) && shown < visible; ++i, ++shown) {
            lines.push_back(panel_line(
                render_row_line(rows[i], col, static_cast<std::size_t>(width),
                                color, i == selected),
                width, color));
        }
        if (offset + shown < static_cast<int>(rows.size())) {
            lines.push_back(panel_line(dim("   … " + std::to_string(rows.size() - offset - shown) +
                                           " more (" + kUp + kDown + " to scroll)", color),
                                       width, color));
        }
    }
    lines.push_back(panel_bottom(width, color));

    // ---- status bar --------------------------------------------------------
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
        const char* sort_names[] = {"CPU", "MEM", "NAME", "NET"};
        std::ostringstream f;
        f << " " << rows.size() << " containers   total mem "
          << resources::human_bytes(total_mem) << "   sample #" << snap.sample_count
          << "   ";
        if (color) f << "\x1b[2m";
        f << "q quit  " << kUp << kDown << " select  s sort [" << sort_names[static_cast<int>(sort_key)]
          << "]  -/+ interval  k stop";
        if (color) f << kReset;
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