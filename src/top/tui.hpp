#pragma once

// Minimal terminal control for `hostely top` — no ncurses, no new deps.
// RAII wrapper over termios raw mode + ANSI alternate screen, with poll-based
// key decoding and SIGWINCH tracking.

#include <csignal>
#include <optional>

namespace hostely::top {

struct TermSize {
    int rows = 24;
    int cols = 80;
};

// Key codes returned by Terminal::read_key.
enum class Key : int {
    None = 0,        // timed out, no input
    Quit,            // q or Esc
    Up, Down,        // arrows
    Sort,            // s
    IntervalUp,      // + / =
    IntervalDown,    // -
    Kill,            // k
    Cancel,          // Esc while confirming (also mapped to Quit upstream)
    Enter,           // \r / \n
    Other,           // anything we don't handle
};

class Terminal {
public:
    Terminal();
    ~Terminal();

    Terminal(const Terminal&) = delete;
    Terminal& operator=(const Terminal&) = delete;

    /// False when stdout isn't a TTY (piped) or TERM is dumb — callers should
    /// fall back to the one-shot snapshot printer.
    bool ready() const { return ready_; }

    void enter();      // alt screen, hide cursor, raw mode
    void leave();      // restore termios + screen; idempotent

    TermSize size() const;
    std::optional<Key> read_key(int timeout_ms);
    bool resized() const;      // SIGWINCH since last call; clears the flag
    bool color() const { return color_; }

    /// SIGINT/SIGTERM while the TUI is up: set by handlers installed in
    /// tui.cpp; the render loop checks and leaves cleanly.
    static bool interrupted();
    static void clear_interrupted();

private:
    bool ready_ = false;
    bool color_ = false;
    bool raw_active_ = false;
    void*  saved_termios_ = nullptr;   // struct termios, heap-held to keep
                                       // this header free of <termios.h>
};

}  // namespace hostely::top