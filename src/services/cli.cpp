#include "services/cli.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <optional>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace hostely::services::cli {

namespace {

// Read all of `fd` into a string until EOF. Returns "" on error.
std::string drain(int fd) {
    std::string out;
    char buf[4096];
    while (true) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n == 0) break;
        if (n < 0) {
            if (errno == EINTR) continue;
            return out;
        }
        out.append(buf, static_cast<std::size_t>(n));
    }
    return out;
}

// posix_spawn a child process with stdout/stderr redirected to pipes.
// Returns the child's pid, or -1 on failure. *out_fd / *err_fd are the
// parent ends of the pipes.
pid_t spawn_with_capture(const std::vector<std::string>& argv,
                         int* out_fd, int* err_fd) {
    if (argv.empty() || argv[0].empty()) return -1;

    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (::pipe(out_pipe) != 0) return -1;
    if (::pipe(err_pipe) != 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        return -1;
    }

    // Build NUL-terminated argv for exec.
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& s : argv) cargv.push_back(const_cast<char*>(s.c_str()));
    cargv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[1]);

    pid_t pid = -1;
    int rc = ::posix_spawnp(&pid, cargv[0], &actions, nullptr,
                            cargv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);

    // Parent: close the child ends, return the read ends.
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);
    if (rc != 0) {
        ::close(out_pipe[0]);
        ::close(err_pipe[0]);
        return -1;
    }
    *out_fd = out_pipe[0];
    *err_fd = err_pipe[0];
    return pid;
}

}  // namespace

Result run(const std::vector<std::string>& argv) {
    Result r;
    int out_fd = -1, err_fd = -1;
    pid_t pid = spawn_with_capture(argv, &out_fd, &err_fd);
    if (pid < 0) {
        r.exit_code = -1;
        return r;
    }

    // Read both pipes (block until child closes them).
    r.stdout_text = drain(out_fd);
    r.stderr_text = drain(err_fd);
    ::close(out_fd);
    ::close(err_fd);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    if (WIFEXITED(status))       r.exit_code = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) r.exit_code = 128 + WTERMSIG(status);
    else                          r.exit_code = -1;
    return r;
}

std::optional<std::string> which(const std::string& program_name) {
    if (program_name.empty()) return std::nullopt;

    // If it contains a slash, treat as a path and just stat it.
    if (program_name.find('/') != std::string::npos) {
        if (::access(program_name.c_str(), X_OK) == 0) return program_name;
        return std::nullopt;
    }

    const char* path_env = std::getenv("PATH");
    if (!path_env || !*path_env) return std::nullopt;

    std::string path_str(path_env);
    std::size_t start = 0;
    while (start <= path_str.size()) {
        std::size_t end = path_str.find(':', start);
        if (end == std::string::npos) end = path_str.size();

        std::string dir = path_str.substr(start, end - start);
        if (!dir.empty()) {
            std::string candidate = dir + "/" + program_name;
            if (::access(candidate.c_str(), X_OK) == 0) return candidate;
        }
        if (end == path_str.size()) break;
        start = end + 1;
    }
    return std::nullopt;
}

}  // namespace hostely::services::cli
