#pragma once

// Rendering for `hostely top`: one snapshot -> terminal lines. Both the live
// TUI and the non-TTY one-shot printer go through these helpers.

#include <string>
#include <vector>

#include "top/sample.hpp"
#include "top/tui.hpp"

namespace hostely::top {

/// One formatted row of the container table, already padded.
struct RenderedRow {
    std::string name;
    std::string line;    // full row text (columns joined)
    bool running = true;
};

/// Sort rows per `key`. Mutates `rows` order.
void sort_rows(std::vector<ContainerRow>& rows, SortKey key);

/// Render the full screen for `size` into `out` (plain text lines, ANSI only
/// when `color`). `selected` is an index into the *sorted* rows; `confirm_kill`
/// names the container pending a stop confirmation ("" = none).
std::vector<std::string> render_screen(const Snapshot& snap,
                                       SortKey sort_key,
                                       int selected,
                                       const std::string& confirm_kill,
                                       TermSize size,
                                       bool color,
                                       int& view_offset);

/// One-shot plain-text table (no ANSI) for pipes / `--once`.
std::string render_snapshot_text(const Snapshot& snap, SortKey sort_key);

}  // namespace hostely::top