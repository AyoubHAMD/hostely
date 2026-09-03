#include "top/tui.hpp"

#include <poll.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/signal.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace hostely::top {

namespace {

std::atomic<bool> g_interrupted{false};

void handle_signal(int sig) {
    (void)sig;
    g_interrupted.store(true);
}

// SIGWINCH handler — only sets the flag; Terminal::resized() reads it.
// (One global flag is fine: only one top session per process.)
volatile sig_atomic_t g_winch = 0;
void handle_winch(int) { g_winch = 1; }

constexpr const char* kAltScreenOn  = "\x1b[?1049h\x1b[H";
constexpr const char* kAltScreenOff = "\x1b[?1049l";
constexpr const char* kHideCursor   = "\x1b[?25l";
constexpr const char* kShowCursor   = "\x1b[?25h";

}  // namespace

bool Terminal::interrupted()     { return g_interrupted.load(); }
void Terminal::clear_interrupted() { g_interrupted.store(false); }

Terminal::Terminal() {
    const char* term = std::getenv("TERM");
    ready_ = isatty(STDOUT_FILENO) != 0 &&
             !(term && std::strcmp(term, "dumb") == 0);
    // NO_COLOR disables color; TERM=dumb already excluded above.
    color_ = ready_ && std::getenv("NO_COLOR") == nullptr;

    if (ready_) {
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::signal(SIGWINCH, handle_winch);
    }
}

Terminal::~Terminal() { leave(); }

void Terminal::enter() {
    if (!ready_ || raw_active_) return;

    auto* saved = new termios{};
    if (tcgetattr(STDIN_FILENO, saved) != 0) {
        delete saved;
        return;
    }
    saved_termios_ = saved;

    termios raw = *saved;
    cfmakeraw(&raw);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    raw_active_ = true;

    std::fputs(kAltScreenOn, stdout);
    std::fputs(kHideCursor, stdout);
    std::fflush(stdout);
}

void Terminal::leave() {
    if (raw_active_) {
        tcsetattr(STDIN_FILENO, TCSANOW,
                  static_cast<termios*>(saved_termios_));
        raw_active_ = false;
    }
    if (ready_) {
        std::fputs(kShowCursor, stdout);
        std::fputs(kAltScreenOff, stdout);
        std::fflush(stdout);
    }
    delete static_cast<termios*>(saved_termios_);
    saved_termios_ = nullptr;
}

TermSize Terminal::size() const {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0 &&
        ws.ws_col > 0) {
        return {static_cast<int>(ws.ws_row), static_cast<int>(ws.ws_col)};
    }
    return {};
}

std::optional<Key> Terminal::read_key(int timeout_ms) {
    if (!raw_active_) return std::nullopt;

    pollfd pfd{STDIN_FILENO, POLLIN, 0};
    int rc = ::poll(&pfd, 1, timeout_ms);
    if (rc <= 0) return std::nullopt;

    char buf[8];
    ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
    if (n <= 0) return std::nullopt;

    if (n == 1) {
        switch (buf[0]) {
            case 'q': return Key::Quit;
            case 's': return Key::Sort;
            case 'k': return Key::Kill;
            case '+': case '=': return Key::IntervalUp;
            case '-': case '_': return Key::IntervalDown;
            case '\r': case '\n': return Key::Enter;
            case '\x1b': return Key::Quit;
            case '\x03': g_interrupted.store(true); return Key::Quit;
            default: return Key::Other;
        }
    }
    // Escape sequences: "\x1b[A" / "\x1b[B" (may arrive in one read).
    if (buf[0] == '\x1b' && n >= 3 && buf[1] == '[') {
        switch (buf[2]) {
            case 'A': return Key::Up;
            case 'B': return Key::Down;
            default: return Key::Other;
        }
    }
    return Key::Other;
}

bool Terminal::resized() const {
    if (g_winch == 0) return false;
    g_winch = 0;
    return true;
}

}  // namespace hostely::top