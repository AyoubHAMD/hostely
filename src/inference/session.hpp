#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <llama.h>

namespace hostely::inference {

/// Allocates llama_seq_id values for multi-turn sessions. Seq 0 is reserved
/// for the stateless single-shot path (OpenAI-compatible requests without a
/// session), so sessions draw from 1..max-1.
class SeqAllocator {
public:
    explicit SeqAllocator(std::size_t max_seqs);

    /// Number of ids available for sessions (= llama_n_seq_max(ctx) - 1).
    std::size_t capacity() const { return max_ - 1; }

    /// Take a free id, or nullopt when all are in use.
    std::optional<llama_seq_id> take();

    /// Return an id to the pool (called when a session is evicted/removed).
    void give_back(llama_seq_id id);

private:
    std::size_t max_ = 0;
    std::size_t next_ = 1;              // round-robin cursor
    std::vector<bool>  in_use_;
};

/// One conversation. Owns a sequence id and the exact token history that has
/// been prefilled into the KV cache on that sequence — the token vector is
/// what lets us verify on the next turn that the new prompt really does share
/// a prefix with what's cached (chat templates append turns, so a prefix
/// match means the cached KV is still valid).
struct Session {
    std::string        id;
    llama_seq_id       seq_id = 0;
    std::vector<llama_token> tokens;   // everything in KV on this seq
    std::chrono::steady_clock::time_point last_used{};
};

/// Session table with LRU eviction. Capacity = SeqAllocator capacity. The
/// server calls `evict_lru()` when the table is full before inserting a new
/// session; the returned seq id must then be dropped from the KV cache via
/// llama_memory_seq_rm(mem, seq, -1, -1).
class SessionTable {
public:
    explicit SessionTable(std::size_t max_sessions);

    Session* find(const std::string& id);      // updates last_used
    const Session* peek(const std::string& id) const;

    /// Insert a new session; if the table is at capacity, evicts the LRU
    /// entry and returns its seq id so the caller can free its KV.
    std::optional<llama_seq_id> insert(const std::string& id,
                                       llama_seq_id seq);

    /// Longest session whose `tokens` form a prefix of `prompt_tokens`.
    /// Used as the fallback session resolver when the client sends no
    /// X-Session-Id — KV reuse is detected from the prompt itself.
    Session* find_by_prefix(const std::vector<llama_token>& prompt_tokens);

    /// Drop the least-recently-used session and return its seq id (for the
    /// caller to free the KV and recycle the id). Nullopt when empty.
    std::optional<llama_seq_id> evict_lru();

    std::size_t size() const { return map_.size(); }
    std::size_t capacity() const { return max_; }

private:
    void touch(Session& s);

    std::size_t max_ = 0;
    std::unordered_map<std::string, Session> map_;
};

}  // namespace hostely::inference