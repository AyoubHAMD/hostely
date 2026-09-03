#pragma once

// Minimal docker-compose reader for `hostely app up`.
//
// We deliberately do NOT ship a YAML library. Compose files use a small,
// well-behaved subset of YAML — block mappings, block sequences, plain and
// quoted scalars, comments — and everything else (anchors, flow style,
// multi-line scalars) is rare in the wild and rejected loudly rather than
// mis-parsed. This file owns that subset plus compose's ${VAR:-default}
// interpolation.

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace hostely::apps {

// One service from a compose file, already normalized.
struct ComposeService {
    std::string name;
    std::string image;                       // required by us (build: unsupported in v1)
    std::vector<std::pair<std::string, std::string>> ports;   // host:container
    std::vector<std::pair<std::string, std::string>> env;     // KEY=VALUE
    std::vector<std::pair<std::string, std::string>> volumes; // src:dst
    std::vector<std::string> command;        // split words; empty = image default
    std::vector<std::string> depends_on;     // service names, conditions ignored
    std::vector<std::string> dns;            // custom nameserver IPs for the service
    std::string mem_limit;                   // compose mem_limit, passed as --memory
};

struct ComposeFile {
    std::string path;                        // where it was found
    std::vector<ComposeService> services;    // in file order
};

struct ComposeError {
    std::string message;
};

// Parse a compose file. `extra_env` values are consulted before the process
// environment during ${...} interpolation (this is how `hostely app up
// --env K=V` feeds compose defaults). Throws ComposeError.
std::optional<ComposeFile> load_compose(const std::string& path,
                                        std::map<std::string, std::string> extra_env,
                                        ComposeError& err);

// Search a directory (and its subdirectories, depth <= 2) for a compose
// file: compose.yaml, compose.yml, docker-compose.yaml, docker-compose.yml.
// Prefers the shallowest match, then the canonical names in that order.
std::optional<std::string> find_compose_file(const std::string& dir);

}  // namespace hostely::apps