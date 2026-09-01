#include "apps/compose.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <stdlib.h>

#include <dirent.h>
#include <sys/stat.h>

namespace hostely::apps {

namespace {

// ---------------------------------------------------------------------------
// YAML subset node
// ---------------------------------------------------------------------------

struct Node {
    enum class Kind { Scalar, Map, Seq } kind = Kind::Scalar;
    std::string scalar;
    std::map<std::string, Node> map;
    std::vector<Node> seq;
};

std::string strip_comment(const std::string& s) {
    // '#' outside quotes starts a comment.
    bool in_s = false, in_d = false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '\'' && !in_d) in_s = !in_s;
        else if (c == '"' && !in_s) in_d = !in_d;
        else if (c == '#' && !in_s && !in_d &&
                 (i == 0 || std::isspace(static_cast<unsigned char>(s[i - 1])))) {
            return s.substr(0, i);
        }
    }
    return s;
}

std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string unquote(const std::string& s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        std::string out = s.substr(1, s.size() - 2);
        // Collapse doubled quote escapes ('' inside '...', "" inside "...")
        std::string cleaned;
        for (std::size_t i = 0; i < out.size(); ++i) {
            if (i + 1 < out.size() && out[i] == out[i + 1] &&
                (out[i] == '\'' || out[i] == '"')) {
                cleaned += out[i];
                ++i;
            } else {
                cleaned += out[i];
            }
        }
        return cleaned;
    }
    return s;
}

struct Line {
    int indent;
    std::string text;   // comment-stripped, right-trimmed
};

std::vector<Line> read_lines(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<Line> lines;
    std::string raw;
    while (std::getline(in, raw)) {
        std::string s = trim(strip_comment(raw));
        if (s.empty()) continue;
        // Only reject obvious YAML features we don't support, loudly.
        if (s == "---" || s == "...") continue;
        if (s.rfind("&", 0) == 0 || s.rfind("*", 0) == 0 ||
            s.rfind("<<:", 0) == 0) {
            throw std::runtime_error(
                "unsupported YAML feature (anchors/aliases) in " + path);
        }
        int indent = 0;
        while (indent < static_cast<int>(raw.size()) &&
               raw[indent] == ' ') ++indent;
        if (raw.find('\t') != std::string::npos && raw.find('\t') < static_cast<std::size_t>(indent))
            throw std::runtime_error("tabs are not valid YAML indentation in " + path);
        lines.push_back({indent, s});
    }
    return lines;
}

struct Parser {
    const std::vector<Line>& lines;
    std::size_t pos = 0;

    bool eof() const { return pos >= lines.size(); }
    const Line& cur() const { return lines[pos]; }

    Node parse_block(int indent) {
        if (eof()) return {};
        if (cur().text.rfind("- ", 0) == 0 || cur().text == "-") {
            return parse_seq(indent);
        }
        return parse_map(indent);
    }

    Node parse_seq(int indent) {
        Node n; n.kind = Node::Kind::Seq;
        while (!eof() && cur().indent == indent &&
               (cur().text.rfind("- ", 0) == 0 || cur().text == "-")) {
            std::string rest = cur().text == "-" ? "" : cur().text.substr(2);
            ++pos;
            if (rest.empty()) {
                // Nested block under the dash.
                if (!eof() && cur().indent > indent) {
                    n.seq.push_back(parse_block(cur().indent));
                } else {
                    n.seq.push_back(Node{});
                }
            } else if (rest.find(": ") != std::string::npos ||
                       (rest.size() > 1 && rest.back() == ':')) {
                // "- key: value" — an inline map item; it may continue with
                // sibling keys at indent + 2.
                Node item; item.kind = Node::Kind::Map;
                parse_kv_into(item, rest, indent + 2);
                while (!eof() && cur().indent == indent + 2 &&
                       cur().text.rfind("- ", 0) != 0) {
                    parse_kv_into(item, cur().text, indent + 2);
                    ++pos;
                }
                n.seq.push_back(std::move(item));
            } else {
                Node v; v.kind = Node::Kind::Scalar; v.scalar = unquote(trim(rest));
                n.seq.push_back(std::move(v));
            }
        }
        return n;
    }

    void parse_kv_into(Node& n, const std::string& text, int /*indent*/) {
        // Split "key: value" (first ': ' outside quotes, or trailing ':').
        std::string key, val;
        bool in_s = false, in_d = false;
        std::size_t split = std::string::npos;
        for (std::size_t i = 0; i < text.size(); ++i) {
            char c = text[i];
            if (c == '\'' && !in_d) in_s = !in_s;
            else if (c == '"' && !in_s) in_d = !in_d;
            else if (c == ':' && !in_s && !in_d) {
                if (i + 1 == text.size()) { split = i; break; }
                if (text[i + 1] == ' ') { split = i; break; }
            }
        }
        if (split == std::string::npos)
            throw std::runtime_error("cannot parse YAML line: " + text);
        key = unquote(trim(text.substr(0, split)));
        val = trim(text.substr(split + 1));
        if (val.empty()) {
            // Nested block (map or seq) or null.
            if (!eof() && cur().indent > 0 &&
                (cur().text.rfind("- ", 0) == 0 || cur().text == "-" ||
                 cur().text.find(": ") != std::string::npos || cur().text.back() == ':')) {
                // Distinguish by the child's shape: dash -> seq, else map.
                Node child = cur().text.rfind("- ", 0) == 0 || cur().text == "-"
                                 ? parse_seq(cur().indent)
                                 : parse_map(cur().indent);
                n.map[key] = std::move(child);
            } else if (!eof() && cur().indent > 0 &&
                       cur().text.rfind("- ", 0) == 0) {
                n.map[key] = parse_seq(cur().indent);
            } else {
                Node v; v.scalar = "";  // null
                n.map[key] = std::move(v);
            }
        } else {
            Node v; v.kind = Node::Kind::Scalar; v.scalar = unquote(val);
            n.map[key] = std::move(v);
        }
    }

    Node parse_map(int indent) {
        Node n; n.kind = Node::Kind::Map;
        while (!eof() && cur().indent == indent &&
               cur().text.rfind("- ", 0) != 0 && cur().text != "-") {
            std::string text = cur().text;
            ++pos;
            parse_kv_into(n, text, indent);
        }
        return n;
    }
};

// ---------------------------------------------------------------------------
// ${VAR:-default} interpolation
// ---------------------------------------------------------------------------

std::string interpolate(const std::string& in,
                        const std::map<std::string, std::string>& extra) {
    std::string out;
    for (std::size_t i = 0; i < in.size();) {
        if (in.compare(i, 2, "${") == 0) {
            std::size_t end = in.find('}', i + 2);
            if (end == std::string::npos) { out += in[i++]; continue; }
            std::string expr = in.substr(i + 2, end - i - 2);
            i = end + 1;
            // ${VAR:-default}, ${VAR-default}, ${VAR}, ${VAR:?err}
            std::string name = expr, def;
            bool have_def = false;
            auto sep = expr.find(":-");
            if (sep != std::string::npos) {
                name = expr.substr(0, sep); def = expr.substr(sep + 2);
                have_def = true;
            } else if ((sep = expr.find('-')) != std::string::npos &&
                       expr.find(":?") == std::string::npos) {
                name = expr.substr(0, sep); def = expr.substr(sep + 1);
                have_def = true;
            }
            name = trim(name);
            auto it = extra.find(name);
            const char* env = getenv(name.c_str());
            std::string val = it != extra.end() ? it->second
                            : env               ? env
                                                : "";
            if (!val.empty()) out += val;
            else if (have_def) out += def;
            // unset without default -> empty (compose warns; we proceed)
            continue;
        }
        out += in[i++];
    }
    return out;
}

std::string node_scalar(const Node& n) {
    return n.kind == Node::Kind::Scalar ? n.scalar : "";
}

// environment: map form (KEY: val) or list form (KEY=val / KEY)
std::vector<std::pair<std::string, std::string>>
parse_env(const Node& n, const std::map<std::string, std::string>& extra) {
    std::vector<std::pair<std::string, std::string>> out;
    if (n.kind == Node::Kind::Map) {
        for (const auto& [k, v] : n.map)
            out.emplace_back(k, interpolate(node_scalar(v), extra));
    } else if (n.kind == Node::Kind::Seq) {
        for (const auto& item : n.seq) {
            std::string s = interpolate(node_scalar(item), extra);
            auto eq = s.find('=');
            if (eq == std::string::npos) {
                const char* env = getenv(s.c_str());
                if (env) out.emplace_back(s, env);
            } else {
                out.emplace_back(s.substr(0, eq), s.substr(eq + 1));
            }
        }
    }
    return out;
}

std::vector<std::pair<std::string, std::string>>
parse_ports(const Node& n) {
    std::vector<std::pair<std::string, std::string>> out;
    auto add = [&](const std::string& spec) {
        auto colon = spec.rfind(':');  // "127.0.0.1:3000:3000" — last colon
        if (colon == std::string::npos) { out.emplace_back(spec, spec); return; }
        out.emplace_back(spec.substr(0, colon), spec.substr(colon + 1));
    };
    if (n.kind == Node::Kind::Seq)
        for (const auto& item : n.seq) add(trim(node_scalar(item)));
    else if (n.kind == Node::Kind::Scalar && !n.scalar.empty())
        add(trim(n.scalar));
    return out;
}

std::vector<std::pair<std::string, std::string>>
parse_volumes(const Node& n) {
    std::vector<std::pair<std::string, std::string>> out;
    if (n.kind != Node::Kind::Seq) return out;
    for (const auto& item : n.seq) {
        std::string s = trim(node_scalar(item));
        auto colon = s.find(':');
        if (colon == std::string::npos) continue;   // anonymous volume — skip
        std::string src = s.substr(0, colon);
        std::string dst = s.substr(colon + 1);
        // Drop mode suffixes like ":ro" / ":rw".
        if (auto m = dst.rfind(':'); m != std::string::npos) dst = dst.substr(0, m);
        out.emplace_back(src, dst);
    }
    return out;
}

std::vector<std::string> parse_command(const Node& n) {
    std::vector<std::string> out;
    if (n.kind == Node::Kind::Seq) {
        for (const auto& item : n.seq) out.push_back(node_scalar(item));
    } else if (n.kind == Node::Kind::Scalar && !n.scalar.empty()) {
        // Flow-style sequence written inline: ["yarn", "worker:prod"]
        std::string s = trim(n.scalar);
        if (s.front() == '[' && s.back() == ']') {
            std::string body = s.substr(1, s.size() - 2);
            std::size_t i = 0;
            while (i < body.size()) {
                // skip whitespace/comma separators
                while (i < body.size() &&
                       (std::isspace(static_cast<unsigned char>(body[i])) ||
                        body[i] == ',')) ++i;
                std::string item;
                if (i < body.size() && (body[i] == '"' || body[i] == '\'')) {
                    char q = body[i++];
                    while (i < body.size() && body[i] != q) item += body[i++];
                    if (i < body.size()) ++i;   // closing quote
                } else {
                    while (i < body.size() && body[i] != ',') item += body[i++];
                }
                item = trim(item);
                if (!item.empty()) out.push_back(item);
            }
        } else {
            // shell-string form: naive whitespace split is fine for compose files
            std::istringstream iss(s);
            std::string w;
            while (iss >> w) out.push_back(w);
        }
    }
    return out;
}

std::vector<std::string> parse_depends_on(const Node& n) {
    std::vector<std::string> out;
    if (n.kind == Node::Kind::Seq) {
        for (const auto& item : n.seq) out.push_back(trim(node_scalar(item)));
    } else if (n.kind == Node::Kind::Map) {
        // depends_on: {db: {condition: ...}}
        for (const auto& [k, v] : n.map) out.push_back(k);
    }
    return out;
}

}  // namespace

std::optional<ComposeFile> load_compose(const std::string& path,
                                        std::map<std::string, std::string> extra_env,
                                        ComposeError& err) {
    try {
        // .env alongside the compose file feeds interpolation (compose's own
        // behavior). Explicit extra_env wins over .env; process env last.
        std::map<std::string, std::string> env = std::move(extra_env);
        {
            auto slash = path.find_last_of('/');
            std::string dir = slash == std::string::npos ? "." : path.substr(0, slash);
            std::ifstream dot_env(dir + "/.env");
            std::string line;
            while (std::getline(dot_env, line)) {
                line = trim(strip_comment(line));
                auto eq = line.find('=');
                if (eq == std::string::npos || line.empty()) continue;
                std::string k = trim(line.substr(0, eq));
                std::string v = unquote(trim(line.substr(eq + 1)));
                if (!env.count(k)) env[k] = v;
            }
        }

        std::vector<Line> lines = read_lines(path);
        if (lines.empty()) {
            err = {"compose file is empty: " + path};
            return std::nullopt;
        }
        // Find top-level `services:` and parse only its subtree.
        Parser p{lines};
        Node root = p.parse_map(lines.front().indent);
        auto svcs = root.map.find("services");
        if (svcs == root.map.end() || svcs->second.kind != Node::Kind::Map) {
            err = {"no top-level `services:` block in " + path};
            return std::nullopt;
        }
        ComposeFile out;
        out.path = path;
        for (const auto& [name, body] : svcs->second.map) {
            ComposeService s;
            s.name = name;
            s.image = interpolate(node_scalar(body.map.count("image")
                                                  ? body.map.at("image")
                                                  : Node{}),
                                  env);
            if (s.image.empty())
                throw std::runtime_error(
                    "service '" + name + "' has no image (hostely app v1 requires "
                    "prebuilt images; `build:` is not supported yet)");
            if (auto it = body.map.find("ports"); it != body.map.end()) {
                s.ports = parse_ports(it->second);
                for (auto& [h, c] : s.ports) {
                    h = interpolate(h, env);
                    c = interpolate(c, env);
                }
            }
            if (auto it = body.map.find("environment"); it != body.map.end())
                s.env = parse_env(it->second, env);
            if (auto it = body.map.find("volumes"); it != body.map.end()) {
                s.volumes = parse_volumes(it->second);
                for (auto& [src, dst] : s.volumes) {
                    src = interpolate(src, env);
                    dst = interpolate(dst, env);
                }
            }
            if (auto it = body.map.find("command"); it != body.map.end())
                s.command = parse_command(it->second);
            if (auto it = body.map.find("depends_on"); it != body.map.end())
                s.depends_on = parse_depends_on(it->second);
            out.services.push_back(std::move(s));
        }
        if (out.services.empty()) {
            err = {"`services:` block defines no services in " + path};
            return std::nullopt;
        }
        return out;
    } catch (const std::exception& e) {
        err = {e.what()};
        return std::nullopt;
    }
}

std::optional<std::string> find_compose_file(const std::string& dir) {
    static const char* names[] = {"compose.yaml", "compose.yml",
                                  "docker-compose.yaml", "docker-compose.yml"};
    // Depth <= 2, shallowest first.
    std::vector<std::string> dirs = {dir};
    auto list_dir = [](const std::string& d, std::vector<std::string>& out) {
        if (DIR* dp = opendir(d.c_str())) {
            while (auto* ent = readdir(dp)) {
                std::string n = ent->d_name;
                if (n == "." || n == ".." || n[0] == '.') continue;
                std::string sub = d + "/" + n;
                struct stat st{};
                if (stat(sub.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
                    out.push_back(sub);
            }
            closedir(dp);
        }
    };
    std::vector<std::string> depth1, depth2;
    list_dir(dir, depth1);
    for (const auto& d : depth1) list_dir(d, depth2);
    for (const auto& d : dirs)
        for (const char* n : names) {
            std::string p = d + "/" + n;
            struct stat st{};
            if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return p;
        }
    for (const auto& d : depth1)
        for (const char* n : names) {
            std::string p = d + "/" + n;
            struct stat st{};
            if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return p;
        }
    for (const auto& d : depth2)
        for (const char* n : names) {
            std::string p = d + "/" + n;
            struct stat st{};
            if (stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode)) return p;
        }
    return std::nullopt;
}

}  // namespace hostely::apps