#include "inference/session.hpp"

#include <algorithm>

namespace hostely::inference {

// ----------------------------------------------------------------------------
// SeqAllocator
// ----------------------------------------------------------------------------

SeqAllocator::SeqAllocator(std::size_t max_seqs)
    : max_(max_seqs < 2 ? 2 : max_seqs), in_use_(max_ < 2 ? 2 : max_seqs, false) {}

std::optional<llama_seq_id> SeqAllocator::take() {
    // Round-robin scan from the cursor; every id before the cursor has been
    // handed out at least once, so a full sweep finds any freed slot.
    for (std::size_t i = 1; i < max_; ++i) {
        std::size_t idx = (next_ + i - 1) % max_;
        if (idx == 0) continue;               // 0 is reserved for stateless
        if (!in_use_[idx]) {
            in_use_[idx] = true;
            next_ = (idx + 1) % max_;
            return static_cast<llama_seq_id>(idx);
        }
    }
    return std::nullopt;
}

void SeqAllocator::give_back(llama_seq_id id) {
    if (id > 0 && static_cast<std::size_t>(id) < max_) {
        in_use_[static_cast<std::size_t>(id)] = false;
    }
}

// ----------------------------------------------------------------------------
// SessionTable
// ----------------------------------------------------------------------------

SessionTable::SessionTable(std::size_t max_sessions) : max_(max_sessions) {}

Session* SessionTable::find(const std::string& id) {
    auto it = map_.find(id);
    if (it == map_.end()) return nullptr;
    touch(it->second);
    return &it->second;
}

const Session* SessionTable::peek(const std::string& id) const {
    auto it = map_.find(id);
    return it == map_.end() ? nullptr : &it->second;
}

std::optional<llama_seq_id> SessionTable::insert(const std::string& id,
                                                 llama_seq_id seq) {
    std::optional<llama_seq_id> evicted;
    if (map_.size() >= max_ && map_.find(id) == map_.end()) {
        // Evict least-recently-used.
        auto victim = std::min_element(
            map_.begin(), map_.end(),
            [](const auto& a, const auto& b) {
                return a.second.last_used < b.second.last_used;
            });
        evicted = victim->second.seq_id;
        map_.erase(victim);
    }
    Session s;
    s.id = id;
    s.seq_id = seq;
    s.last_used = std::chrono::steady_clock::now();
    map_[id] = std::move(s);
    return evicted;
}

Session* SessionTable::find_by_prefix(const std::vector<llama_token>& prompt_tokens) {
    Session* best = nullptr;
    std::size_t best_len = 0;
    for (auto& [id, s] : map_) {
        if (s.tokens.empty()) continue;
        if (s.tokens.size() > prompt_tokens.size()) continue;
        if (!std::equal(s.tokens.begin(), s.tokens.end(), prompt_tokens.begin())) continue;
        if (s.tokens.size() > best_len) {
            best_len = s.tokens.size();
            best = &s;
        }
    }
    if (best) touch(*best);
    return best;
}

std::optional<llama_seq_id> SessionTable::evict_lru() {
    if (map_.empty()) return std::nullopt;
    auto victim = std::min_element(
        map_.begin(), map_.end(),
        [](const auto& a, const auto& b) {
            return a.second.last_used < b.second.last_used;
        });
    llama_seq_id seq = victim->second.seq_id;
    map_.erase(victim);
    return seq;
}

void SessionTable::touch(Session& s) {
    s.last_used = std::chrono::steady_clock::now();
}

}  // namespace hostely::inference