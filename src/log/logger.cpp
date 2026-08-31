#include "log/logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

namespace hostely::log {

namespace {

// ---- globals guarded by sink_mutex() --------------------------------------

std::mutex& sink_mutex() {
    static std::mutex m;
    return m;
}

Level& level_ref() {
    static Level lvl = Level::Info;
    return lvl;
}

std::ofstream& file_stream() {
    static std::ofstream s;
    return s;
}

fs::path& file_path_ref() {
    static fs::path p;
    return p;
}

std::size_t& max_bytes_ref() {
    static std::size_t n = 1u << 20;
    return n;
}

std::size_t& keep_files_ref() {
    static std::size_t n = 3;
    return n;
}

std::atomic<std::size_t>& bytes_written_ref() {
    static std::atomic<std::size_t> n{0};
    return n;
}

bool& file_enabled_ref() {
    static bool b = false;
    return b;
}

// ---- helpers --------------------------------------------------------------

std::string timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t   = system_clock::to_time_t(now);
    auto ms  = duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000;

    std::tm tm_buf{};
    localtime_r(&t, &tm_buf);

    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S")
        << '.' << std::setw(3) << std::setfill('0') << ms;
    return oss.str();
}

void rotate_if_needed(std::size_t incoming_size) {
    auto& bytes = bytes_written_ref();
    auto total = bytes.load() + incoming_size;
    if (total < max_bytes_ref() || max_bytes_ref() == 0) {
        bytes.store(total);
        return;
    }

    // Close current, rotate hostely.log -> hostely.log.1, .1 -> .2, etc.
    file_stream().close();

    const auto& base = file_path_ref();
    if (!base.empty()) {
        // Drop the oldest if we'd exceed keep_files.
        fs::path oldest = base;
        oldest += "." + std::to_string(keep_files_ref());
        std::error_code ec;
        fs::remove(oldest, ec);

        // Shift .N -> .(N+1) for N from keep-1 down to 1.
        for (std::size_t i = keep_files_ref(); i >= 2; --i) {
            fs::path from = base;
            from += "." + std::to_string(i - 1);
            fs::path to   = base;
            to   += "." + std::to_string(i);
            fs::rename(from, to, ec);
        }
        // hostely.log -> hostely.log.1
        fs::rename(base, base.string() + ".1", ec);

        // Reopen.
        file_stream().open(base, std::ios::out | std::ios::trunc);
        bytes.store(0);
    }
}

}  // namespace

std::string_view level_tag(Level level) {
    switch (level) {
        case Level::Debug: return "DEBUG";
        case Level::Info:  return "INFO";
        case Level::Warn:  return "WARN";
        case Level::Error: return "ERROR";
    }
    return "?";
}

void init(Level level, std::string_view logfile_path,
          std::size_t max_bytes, std::size_t keep_files) {
    std::lock_guard<std::mutex> lock(sink_mutex());
    level_ref()       = level;
    max_bytes_ref()   = max_bytes;
    keep_files_ref()  = keep_files;
    bytes_written_ref().store(0);
    file_enabled_ref() = false;

    if (!logfile_path.empty()) {
        fs::path p(logfile_path);
        if (auto parent = p.parent_path(); !parent.empty()) {
            std::error_code ec;
            fs::create_directories(parent, ec);
        }
        file_stream().open(p, std::ios::out | std::ios::app);
        if (file_stream().is_open()) {
            file_path_ref() = p;
            file_enabled_ref() = true;
            // Best-effort: current file size counts toward rotation budget.
            std::error_code ec;
            auto sz = fs::file_size(p, ec);
            if (!ec) bytes_written_ref().store(static_cast<std::size_t>(sz));
        }
    }
}

void emit(Level level, std::string_view message) {
    if (static_cast<int>(level) < static_cast<int>(level_ref())) return;

    std::string line;
    line.reserve(message.size() + 32);
    line += timestamp();
    line += " [";
    line += level_tag(level);
    line += "] ";
    line.append(message.data(), message.size());
    line += '\n';

    std::lock_guard<std::mutex> lock(sink_mutex());
    std::cerr << line;
    if (file_enabled_ref() && file_stream().is_open()) {
        rotate_if_needed(line.size());
        file_stream() << line;
        file_stream().flush();
    }
}

}  // namespace hostely::log
