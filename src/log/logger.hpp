#pragma once

#include <string>
#include <string_view>

namespace hostely::log {

/// Severity levels for log messages.
enum class Level {
    Debug,
    Info,
    Warn,
    Error,
};

/// Configure the logger. Must be called once at startup (typically from
/// `main`). After init, every `emit()` writes to stderr AND to the rotating
/// logfile (best-effort: if the file cannot be opened, stderr still works).
///
/// `level` filters out messages below the chosen severity (e.g. setting
/// Level::Warn hides Debug and Info). Defaults to Level::Info.
/// `max_bytes` is the size at which the logfile is rotated (default 1 MiB).
/// `keep_files` is how many rotated copies to keep (default 3).
void init(Level level = Level::Info,
          std::string_view logfile_path = {},
          std::size_t max_bytes = 1u << 20,
          std::size_t keep_files = 3);

/// Like init(), but the stderr sink stays off — for full-screen commands
/// (`hostely top`) where stderr output would corrupt the display. The log
/// file still receives everything.
void init_file_only(Level level = Level::Info,
                    std::string_view logfile_path = {},
                    std::size_t max_bytes = 1u << 20,
                    std::size_t keep_files = 3);

/// Emit one log line to all configured sinks.
void emit(Level level, std::string_view message);

/// Convenience wrappers so callers don't have to spell out the enum.
inline void debug(std::string_view m) { emit(Level::Debug, m); }
inline void info (std::string_view m) { emit(Level::Info,  m); }
inline void warn (std::string_view m) { emit(Level::Warn,  m); }
inline void error(std::string_view m) { emit(Level::Error, m); }

/// Convert a level to its short tag ("INFO", "WARN", ...).
std::string_view level_tag(Level level);

}  // namespace hostely::log
