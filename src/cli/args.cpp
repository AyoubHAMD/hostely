#include "cli/args.hpp"

#include <algorithm>
#include <string_view>

namespace hostely::cli {

namespace {

// Strip leading dashes from a token: "--port" -> "port", "-v" -> "v",
// "--" -> "". Returns the stripped name; `was_long` is set to true for "--".
std::string_view strip_dashes(std::string_view token, bool& was_long) {
    if (token.size() >= 2 && token[0] == '-' && token[1] == '-') {
        was_long = true;
        return token.substr(2);
    }
    if (token.size() >= 2 && token[0] == '-') {
        was_long = false;
        return token.substr(1);
    }
    was_long = false;
    return token;
}

}  // namespace

ParsedArgs ParsedArgs::from_argv(int argc, const char* const argv[]) {
    ParsedArgs out;

    // Index 0 is the program name; skip it.
    for (int i = 1; i < argc; ++i) {
        std::string_view token = argv[i];

        // Bare "--" ends option processing; everything after is positional.
        if (token == "--") {
            for (int j = i + 1; j < argc; ++j) {
                out.positionals_.emplace_back(argv[j]);
            }
            break;
        }

        bool was_long = false;
        std::string_view name_part = strip_dashes(token, was_long);

        // If the token didn't start with '-', it's a positional.
        if (name_part.data() == token.data()) {
            out.positionals_.emplace_back(token);
            continue;
        }

        Option opt;
        opt.name.assign(name_part);

        // Look for "--name=value" form.
        if (auto eq = name_part.find('='); eq != std::string_view::npos) {
            opt.name.assign(name_part.substr(0, eq));
            opt.value.assign(name_part.substr(eq + 1));
            opt.present = true;
            out.options_.push_back(std::move(opt));
            continue;
        }

        // "--name value" form: only if the next argv is NOT itself an option.
        // A flag with no value gets an empty string.
        if (i + 1 < argc) {
            std::string_view next = argv[i + 1];
            bool next_is_option = next.size() >= 2 && next[0] == '-';
            // Special case: "-" alone is often used as stdin and is not a flag.
            if (next_is_option && next != "-") {
                opt.value.clear();
                opt.present = true;
                out.options_.push_back(std::move(opt));
                continue;
            }
            opt.value.assign(next);
            opt.present = true;
            out.options_.push_back(std::move(opt));
            ++i;  // consume the value
            continue;
        }

        // Trailing flag with no following token.
        opt.value.clear();
        opt.present = true;
        out.options_.push_back(std::move(opt));
    }

    return out;
}

bool ParsedArgs::has(std::string_view name) const {
    return std::any_of(options_.begin(), options_.end(),
        [&](const Option& o) { return o.name == name; });
}

std::optional<std::string> ParsedArgs::get(std::string_view name) const {
    for (const auto& o : options_) {
        if (o.name == name) return o.value;
    }
    return std::nullopt;
}

std::string_view ParsedArgs::command() const {
    if (positionals_.empty()) return {};
    return positionals_.front();
}

std::vector<std::string_view> ParsedArgs::operands() const {
    std::vector<std::string_view> out;
    out.reserve(positionals_.size() > 0 ? positionals_.size() - 1 : 0);
    for (size_t i = 1; i < positionals_.size(); ++i) {
        out.push_back(positionals_[i]);
    }
    return out;
}

}  // namespace hostely::cli
