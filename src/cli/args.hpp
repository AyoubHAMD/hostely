#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace hostely::cli {

/// One parsed command-line option.
struct Option {
    std::string name;          // e.g. "port"  (without leading "--")
    std::string value;         // empty string for boolean flags
    bool present = false;      // true if --name appeared on the command line
};

/// Result of parsing argv. Holds every flag/value pair and the trailing
/// positional arguments (the subcommand and its operands).
///
/// Design notes:
///   - We keep the full record of every flag so `hostely --help` can later
///     inspect what the user typed (handy for `--verbose` etc.).
///   - We do NOT validate option names here; the caller decides which
///     options it knows about. This keeps the parser dumb and reusable.
class ParsedArgs {
public:
    /// Build from raw argv (argv[0] is the program name and is skipped).
    static ParsedArgs from_argv(int argc, const char* const argv[]);

    /// True if --name (or -name) was given at all.
    bool has(std::string_view name) const;

    /// Value of --name, or std::nullopt if it was not given, or the empty
    /// string if it was given as a boolean flag.
    std::optional<std::string> get(std::string_view name) const;

    /// First positional argument (typically the subcommand name).
    /// Empty if there are no positionals.
    std::string_view command() const;

    /// All positional arguments after the command.
    std::vector<std::string_view> operands() const;

    /// Direct access for debug printing.
    const std::vector<Option>& options() const { return options_; }
    const std::vector<std::string>& positionals() const { return positionals_; }

private:
    std::vector<Option> options_;
    std::vector<std::string> positionals_;  // index 0 is the subcommand if any
};

}  // namespace hostely::cli
