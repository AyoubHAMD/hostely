#include "inference/server.hpp"

#include "inference/lockfile.hpp"
#include "inference/session.hpp"
#include "log/logger.hpp"
#include "models/fit.hpp"
#include "paths.hpp"

// llama.cpp C API. Public headers live under third_party/llama.cpp/include
// and third_party/llama.cpp/ggml/include; both are exposed via the `llama`
// CMake target's INTERFACE_INCLUDE_DIRECTORIES, so this single include is
// enough for the C API.
#include <llama.h>
#include <chat.h>   // llama.cpp common: jinja chat templates + tool-call PEG parser

// cpp-httplib + nlohmann/json (single-header libs).
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace hostely::inference {

using nlohmann::json;

// ----------------------------------------------------------------------------
// Server::Impl
// ----------------------------------------------------------------------------

struct Server::Impl {
    // llama.cpp handles.
    llama_model*   model  = nullptr;
    llama_context* ctx    = nullptr;
    const llama_vocab* vocab = nullptr;
    llama_sampler* smpl   = nullptr;

    // Jinja chat templates + tool-call PEG parser (llama.cpp common). Built
    // from the GGUF metadata; falls back to chatml for template-less models.
    common_chat_templates_ptr tmpls_;

    // HTTP.
    httplib::Server svr;
    std::thread     server_thread;
    std::atomic<bool> stop_requested{false};

    // Serialise inference — single-request at a time for v1.
    std::mutex infer_mu;

    // Phase 7d: session-keyed KV cache. seq 0 is the stateless path;
    // sessions draw from the rest. Re-initialised in load_model() once we
    // know llama_n_seq_max(ctx).
    SeqAllocator seq_alloc_{2};
    SessionTable sessions_{0};

    ServeOptions opts;

    ~Impl() {
        if (smpl) llama_sampler_free(smpl);
        if (ctx)  llama_free(ctx);
        if (model) llama_model_free(model);
        // llama.cpp has no llama_backend_free(); backend init is global and
        // intentionally leaks across server lifetimes.
    }

    bool load_model(LoadedModel& out) {
        using namespace hostely::log;
        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = opts.gpu_layers;
        info("loading model: " + opts.model_path.string());
        model = llama_model_load_from_file(opts.model_path.string().c_str(), mparams);
        if (!model) {
            error("llama_model_load_from_file failed");
            return false;
        }
        vocab = llama_model_get_vocab(model);

        auto cparams = llama_context_default_params();
        cparams.n_ctx        = static_cast<uint32_t>(opts.ctx_size);
        // Multi-turn sessions each own a KV sequence. With kv_unified the
        // sequences share ONE cache of n_ctx tokens — every session sees the
        // full context window and the KV memory cost doesn't scale with
        // n_seq_max. The default (split) mode would divide n_ctx across the
        // 32 sequences (2048/32 = 64 tokens each), which is for llama.cpp's
        // parallel-slot server, not for us.
        cparams.n_seq_max    = 32;
        cparams.kv_unified   = true;
        cparams.n_threads    = opts.threads > 0 ? opts.threads
                              : std::max(1u, std::thread::hardware_concurrency() / 2);
        cparams.n_threads_batch = cparams.n_threads;
        info("context: n_ctx=" + std::to_string(cparams.n_ctx) +
             " n_threads=" + std::to_string(cparams.n_threads));

        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            error("llama_new_context_with_model failed");
            return false;
        }

        // Greedy sampler for v1 — simple, deterministic, fast.
        smpl = llama_sampler_init_greedy();

        // Jinja chat templates from GGUF metadata (chatml fallback). This is
        // what lets /v1/messages and /v1/chat/completions render tool
        // definitions and tool calls the way the model was trained to see
        // them — the manual llama_chat_apply_template path has no notion of
        // tools.
        tmpls_ = common_chat_templates_init(model, opts.chat_template);

        // Phase 7d: size the session machinery from the real context.
        const std::size_t n_seq = llama_n_seq_max(ctx);
        seq_alloc_ = SeqAllocator(n_seq);
        sessions_  = SessionTable(seq_alloc_.capacity());

        // Populate LoadedModel so the Phase 7b fit advisor + Phase 7c
        // status output can read real numbers from llama.cpp.
        out = LoadedModel{};
        out.path           = opts.model_path;
        out.display_name   = opts.model_path.stem().string();
        out.quant_name     = llama_ftype_name(llama_model_ftype(model));
        out.params         = llama_model_n_params(model);
        out.tensor_bytes   = llama_model_size(model);
        out.ctx_train      = llama_model_n_ctx_train(model);
        out.n_layer        = llama_model_n_layer(model);
        out.n_embd         = llama_model_n_embd(model);
        out.n_head         = llama_model_n_head(model);
        out.n_head_kv      = llama_model_n_head_kv(model);
        // KV capacity for the configured n_ctx, analytic. The default KV
        // element type is f16 (2 bytes). head_dim = n_embd / n_head; with
        // GQA only n_head_kv heads are cached. Some models (MoE/SSM hybrids)
        // don't fit this formula — they yield 0 and the estimate is simply
        // lowballed rather than wrong-by-17-bytes.
        if (out.n_head > 0 && out.n_head_kv > 0 && out.n_layer > 0) {
            std::uint64_t head_dim = static_cast<std::uint64_t>(out.n_embd) /
                                     static_cast<std::uint64_t>(out.n_head);
            out.kv_state_bytes =
                2 /* K+V */ * 2 /* f16 bytes */ *
                static_cast<std::uint64_t>(out.n_layer) *
                static_cast<std::uint64_t>(opts.ctx_size) *
                static_cast<std::uint64_t>(out.n_head_kv) * head_dim;
        }
        return true;
    }

    // Build a prompt string from a chat-style messages array. We support the
    // most common manual format; if the model has a chat template (GGUF
    // metadata), we use llama_chat_apply_template instead.
    std::string build_prompt_from_messages(const json& messages,
                                           std::string& out_model_name) {
        // Try the model's native chat template first.
        const char* tmpl = llama_model_chat_template(model, nullptr);
        if (tmpl && *tmpl) {
            // llama_chat_message holds raw char*s, so the strings must
            // outlive the apply_template call — collect OWNED copies first,
            // then build the view array pointing into them.
            std::vector<std::string> roles, contents;
            roles.reserve(messages.size());
            contents.reserve(messages.size());
            for (const auto& m : messages) {
                roles.push_back(m.value("role", "user"));
                contents.push_back(m.value("content", ""));
            }
            std::vector<llama_chat_message> cmsgs;
            cmsgs.reserve(messages.size());
            for (std::size_t i = 0; i < roles.size(); ++i) {
                cmsgs.push_back({roles[i].c_str(), contents[i].c_str()});
            }
            int needed = llama_chat_apply_template(
                tmpl, cmsgs.data(), cmsgs.size(),
                /*add_ass*/ true,
                /*buf*/ nullptr, /*length*/ 0);
            if (needed > 0) {
                std::string buf(static_cast<size_t>(needed) + 1, '\0');
                int written = llama_chat_apply_template(
                    tmpl, cmsgs.data(), cmsgs.size(),
                    true, buf.data(), static_cast<int32_t>(buf.size()));
                if (written > 0) return buf.substr(0, written);
            }
            // fall through to manual if template returned nothing useful
        }

        // Minimal manual template: "<role>: <content>\n" per turn.
        std::ostringstream oss;
        for (const auto& m : messages) {
            const auto role = m.value("role", "user");
            const auto content = m.value("content", "");
            oss << role << ": " << content << "\n";
        }
        oss << "assistant:";
        return oss.str();
    }

    // ---- tool-calling (llama.cpp common) -----------------------------------
    // Both wire protocols are converted into llama.cpp's chat model, rendered
    // with the model's own jinja template (which knows how to present tool
    // definitions + tool calls), and the generated text is parsed back with
    // the PEG parser selected by that template. This is the same machinery
    // llama-server uses — reimplementing per-model formats here would be a
    // correctness trap.

    // OpenAI:  tools: [{type:"function", function:{name, description,
    //                   parameters}}]
    // Anthropic: tools: [{name, description, input_schema}]
    std::vector<common_chat_tool> to_common_tools(const json& tools, bool anthropic) {
        std::vector<common_chat_tool> out;
        out.reserve(tools.size());
        for (const auto& t : tools) {
            if (!t.is_object()) continue;
            common_chat_tool ct;
            if (anthropic) {
                ct.name        = t.value("name", "");
                ct.description = t.value("description", "");
                if (t.contains("input_schema"))
                    ct.parameters = t["input_schema"].dump();
            } else {
                const json& fn = t.contains("function") ? t["function"] : t;
                ct.name        = fn.value("name", "");
                ct.description = fn.value("description", "");
                if (fn.contains("parameters"))
                    ct.parameters = fn["parameters"].dump();
            }
            if (!ct.name.empty()) out.push_back(std::move(ct));
        }
        return out;
    }

    // Read a message content field that may be a string, null (assistant
    // tool_call turns), or absent — nlohmann's value() throws on null.
    static std::string content_str(const json& m) {
        return m.contains("content") && m["content"].is_string()
            ? m["content"].get<std::string>() : std::string{};
    }

    // OpenAI messages -> common model. "tool" results and assistant
    // tool_calls map onto the corresponding common fields so the template can
    // render a full agent transcript.
    std::vector<common_chat_msg> openai_to_common(const json& messages) {
        std::vector<common_chat_msg> out;
        out.reserve(messages.size());
        for (const auto& m : messages) {
            common_chat_msg c;
            c.role = m.value("role", "user");
            if (c.role == "tool") {
                c.tool_call_id = m.value("tool_call_id", "");
                c.tool_name    = m.value("name", "");
            }
            c.content = content_str(m);
            if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
                for (const auto& tc : m["tool_calls"]) {
                    const json& fn = tc.value("function", json::object());
                    common_chat_tool_call call;
                    call.id        = tc.value("id", "");
                    call.name      = fn.value("name", "");
                    call.arguments = fn.value("arguments", "");
                    c.tool_calls.push_back(std::move(call));
                }
            }
            out.push_back(std::move(c));
        }
        return out;
    }

    // Anthropic messages (+ optional top-level system) -> common model.
    // tool_use blocks become assistant tool_calls; tool_result blocks become
    // tool messages keyed by tool_use_id.
    std::vector<common_chat_msg> anthropic_to_common(const json& messages,
                                                     const std::string& system) {
        std::vector<common_chat_msg> out;
        out.reserve(messages.size() + 1);
        if (!system.empty()) {
            common_chat_msg sys;
            sys.role    = "system";
            sys.content = system;
            out.push_back(std::move(sys));
        }
        for (const auto& m : messages) {
            const bool assistant = m.value("role", "user") == "assistant";
            common_chat_msg c;
            c.role    = assistant ? "assistant" : "user";
            c.content = flatten_anthropic_content(m.value("content", json{}));
            if (m.value("content", json{}).is_array()) {
                for (const auto& block : m["content"]) {
                    if (!block.is_object()) continue;
                    const std::string type = block.value("type", "");
                    if (assistant && type == "tool_use") {
                        common_chat_tool_call call;
                        call.id        = block.value("id", "");
                        call.name      = block.value("name", "");
                        if (block.contains("input"))
                            call.arguments = block["input"].dump();
                        c.tool_calls.push_back(std::move(call));
                    } else if (!assistant && type == "tool_result") {
                        common_chat_msg tr;
                        tr.role         = "tool";
                        tr.tool_call_id = block.value("tool_use_id", "");
                        // tool_result content is a string or content blocks.
                        if (block.contains("content")) {
                            if (block["content"].is_string())
                                tr.content = block["content"].get<std::string>();
                            else
                                tr.content = flatten_anthropic_content(block["content"]);
                        }
                        out.push_back(std::move(tr));
                    }
                }
            }
            // Skip pure tool_result user messages — already emitted above.
            if (!c.content.empty() || !c.tool_calls.empty())
                out.push_back(std::move(c));
        }
        return out;
    }

    // Render the conversation (optionally with tools) through the model's
    // own jinja template. Returns the prompt + the PEG parser spec that
    // understands how this template expresses tool calls.
    common_chat_params apply_chat_template(const std::vector<common_chat_msg>& msgs,
                                           const std::vector<common_chat_tool>* tools) {
        common_chat_templates_inputs in;
        in.messages = msgs;
        in.use_jinja = true;
        if (tools && !tools->empty()) {
            in.tools = *tools;
            in.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;
        }
        return common_chat_templates_apply(tmpls_.get(), in);
    }

    // Parse generated text into content + tool calls.
    common_chat_msg parse_generation(const std::string& text,
                                     const common_chat_params& chat_params,
                                     bool partial) {
        common_chat_parser_params pp(chat_params);
        // The ctor from chat_params only copies format + generation_prompt;
        // without the arena the parser degrades to "pure content", which
        // makes the rendered generation prompt leak into the content.
        pp.parser.load(chat_params.parser);
        return common_chat_parse(text, partial, pp);
    }

    // Stop sequences suggested by the template (e.g. the tool-call end tag).
    // Applied after generation so a chatty model that emits them literally
    // doesn't leak template artifacts into the content.
    std::string apply_stops(std::string text, const common_chat_params& chat_params) {
        for (const auto& stop : chat_params.additional_stops) {
            if (stop.empty()) continue;
            const auto pos = text.find(stop);
            if (pos != std::string::npos) text.resize(pos);
        }
        return text;
    }

    // Stable ids for wire responses: OpenAI wants "call_...", Anthropic
    // "toolu_...". The parser leaves ids empty for most formats.
    static std::string gen_id(const char* prefix) {
        static std::atomic<uint64_t> counter{0};
        return std::string(prefix) +
               std::to_string(std::chrono::system_clock::now()
                                  .time_since_epoch().count()) +
               "-" + std::to_string(counter.fetch_add(1));
    }

    // Run inference on a prompt. Streaming variant: `hooks.on_token` fires
    // with incremental text as it is generated (SSE for OpenAI/Anthropic
    // clients); `hooks.on_prefill` fires once the prompt token count is
    // known, before generation starts. Both may be null.
    //
    // `session_id` (Phase 7d): when non-empty, the request joins that
    // session's KV sequence — turn N only prefills the tokens *after* what's
    // already cached, so multi-turn chat doesn't re-prefill the whole
    // history. When empty, we look for an existing session whose cached
    // tokens form a prefix of this prompt (KV reuse detected from the prompt
    // itself); if none matches, the stateless single-shot path runs on
    // sequence 0.
    struct CompletionStats {
        int prompt_tokens     = 0;
        int completion_tokens = 0;
    };

    struct StreamHooks {
        std::function<void(int prompt_tokens)> on_prefill;
        std::function<void(const std::string& piece)> on_token;
    };

    // Detokenize a token sequence in one pass (exact, avoids the per-token
    // leading-space loss of naive piece-by-piece output).
    std::string detokenize(const std::vector<llama_token>& tokens) const {
        if (tokens.empty()) return {};
        std::vector<char> buf(tokens.size() * 16 + 64);
        int n = llama_detokenize(vocab, tokens.data(),
                                 static_cast<int32_t>(tokens.size()),
                                 buf.data(), static_cast<int32_t>(buf.size()),
                                 /*remove_special*/ true,
                                 /*unparse_special*/ false);
        if (n < 0) {
            buf.resize(static_cast<std::size_t>(-n));
            n = llama_detokenize(vocab, tokens.data(),
                                 static_cast<int32_t>(tokens.size()),
                                 buf.data(), static_cast<int32_t>(buf.size()),
                                 true, false);
        }
        return n > 0 ? std::string(buf.data(), static_cast<std::size_t>(n))
                     : std::string{};
    }

    std::string complete_stream(const std::string& prompt, int max_tokens,
                                const std::string& session_id,
                                bool allow_session_fallback,
                                const StreamHooks& hooks,
                                CompletionStats* stats = nullptr,
                                const std::vector<std::string>* stops = nullptr) {
        std::lock_guard<std::mutex> lock(infer_mu);

        llama_memory_t mem = llama_get_memory(ctx);

        // Tokenize the full prompt (session path relies on the tokenizer
        // being deterministic so the prefix property holds turn-to-turn).
        // parse_special=true: jinja chat templates emit real special tokens
        // (<|im_start|>, <|tool_call|>, ...); tokenizing them as plain text
        // makes tool-capable models echo the tags instead of acting on them.
        const int prompt_cap = static_cast<int>(prompt.size()) + 16;
        std::vector<llama_token> prompt_tokens(prompt_cap);
        int n_prompt = llama_tokenize(
            vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
            prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()),
            /*add_special*/ true, /*parse_special*/ true);
        if (n_prompt < 0) {
            prompt_tokens.resize(-n_prompt);
            n_prompt = llama_tokenize(
                vocab, prompt.data(), static_cast<int32_t>(prompt.size()),
                prompt_tokens.data(), static_cast<int32_t>(prompt_tokens.size()),
                true, true);
        }
        if (n_prompt <= 0) return {};
        prompt_tokens.resize(n_prompt);
        if (stats) stats->prompt_tokens = n_prompt;

        // ---- resolve the session ------------------------------------------
        Session* sess = nullptr;
        llama_seq_id seq = 0;
        std::size_t n_past = 0;              // tokens already in KV on `seq`

        if (!session_id.empty()) {
            sess = sessions_.find(session_id);
            if (!sess) {
                auto allocated = seq_alloc_.take();
                if (!allocated) {
                    // All seq ids are in use. The LRU session must go —
                    // free its KV and recycle its id before continuing.
                    auto evicted = sessions_.evict_lru();
                    if (evicted) {
                        if (mem) llama_memory_seq_rm(mem, *evicted, -1, -1);
                        seq_alloc_.give_back(*evicted);
                        allocated = seq_alloc_.take();
                    }
                }
                if (!allocated) {
                    log::warn("session table full; falling back to stateless");
                } else {
                    // Inserting may evict the LRU session; free its KV and
                    // return its seq id to the pool for future sessions.
                    auto evicted = sessions_.insert(session_id, *allocated);
                    if (evicted) {
                        if (mem) llama_memory_seq_rm(mem, *evicted, -1, -1);
                        seq_alloc_.give_back(*evicted);
                    }
                    sess = sessions_.find(session_id);
                }
            }
        } else if (allow_session_fallback) {
            // No explicit session id — detect KV reuse from the prompt.
            sess = sessions_.find_by_prefix(prompt_tokens);
        }

        if (sess) {
            seq = sess->seq_id;
            n_past = sess->tokens.size();
            // Reuse the longest common prefix of the cached tokens and this
            // prompt's tokens. An exact-prefix requirement is too strict:
            // BPE merges at message boundaries (e.g. the template's header
            // "\n" fusing with a reply starting "\n\n") make re-tokenized
            // history diverge by a few tokens even when the conversation is
            // unchanged. Anything past the divergence point is dropped from
            // the KV and re-prefilled.
            std::size_t k = 0;
            while (k < n_past && k < prompt_tokens.size() &&
                   sess->tokens[k] == prompt_tokens[k]) {
                ++k;
            }
            if (k < n_past) {
                if (mem) llama_memory_seq_rm(mem, seq,
                                              static_cast<llama_pos>(k), -1);
                sess->tokens.resize(k);
                n_past = k;
            }
        } else {
            // Stateless single-shot on seq 0: clear only seq 0's entries —
            // NOT llama_memory_clear, which would wipe every session's KV.
            if (mem) llama_memory_seq_rm(mem, 0, -1, -1);
        }

        const uint32_t n_ctx = llama_n_ctx(ctx);
        if (static_cast<uint32_t>(prompt_tokens.size()) +
                static_cast<uint32_t>(max_tokens) >= n_ctx) {
            log::warn("prompt + max_tokens exceeds context; truncating");
            max_tokens = static_cast<int>(n_ctx) -
                         static_cast<int>(prompt_tokens.size()) - 1;
            if (max_tokens <= 0) {
                // Context is exhausted for this session — evict it so the
                // next turn starts fresh rather than spinning.
                if (sess && mem) {
                    llama_memory_seq_rm(mem, seq, -1, -1);
                    sess->tokens.clear();
                }
                return {};
            }
        }

        // ---- prefill the tokens beyond n_past ------------------------------
        log::debug("complete: sess=" + (sess ? sess->id : std::string("-")) +
                   " seq=" + std::to_string(seq) +
                   " n_past=" + std::to_string(n_past) +
                   " n_prompt=" + std::to_string(prompt_tokens.size()));
        // llama_batch_get_one hardcodes seq 0, so sessions need a manual
        // batch with an explicit seq id and absolute positions.
        const std::size_t n_feed = prompt_tokens.size() - n_past;
        // Prefill must be chunked to n_batch — a single batch larger than
        // n_batch trips GGML_ASSERT(n_tokens_all <= cparams.n_batch).
        const std::size_t n_chunk = std::min<std::size_t>(
            std::max<std::size_t>(n_feed, 1),
            static_cast<std::size_t>(llama_n_batch(ctx)));
        llama_batch batch = llama_batch_init(n_chunk, /*embd*/ 0, /*n_seq_max*/ 1);

        auto fill_batch = [&](llama_token tok, llama_pos pos, bool want_logits) {
            batch.token[0]     = tok;
            batch.pos[0]       = pos;
            batch.n_seq_id[0]  = 1;              // MUST be set —
            batch.seq_id[0][0] = seq;            // llama_batch_init mallocs
            batch.logits[0]    = want_logits;    // without initialising it.
            batch.n_tokens     = 1;
        };

        bool prefill_ok = true;
        for (std::size_t start = 0; prefill_ok && start < n_feed; start += n_chunk) {
            // One batch per n_batch-sized chunk — decoding token-by-token was
            // ~10x slower and bought nothing.
            const std::size_t len = std::min(n_chunk, n_feed - start);
            for (std::size_t i = 0; i < len; ++i) {
                batch.token[i]        = prompt_tokens[n_past + start + i];
                batch.pos[i]          = static_cast<llama_pos>(n_past + start + i);
                batch.n_seq_id[i]     = 1;
                batch.seq_id[i][0]    = seq;
                batch.logits[i]       = (start + i + 1 == n_feed);
            }
            batch.n_tokens = static_cast<int32_t>(len);
            if (llama_decode(ctx, batch) != 0) {
                log::error("llama_decode (prefill) failed");
                prefill_ok = false;
            }
        }
        if (!prefill_ok) {
            llama_batch_free(batch);
            return {};
        }

        // The session now holds exactly the prompt tokens.
        std::size_t n_total = prompt_tokens.size();

        if (hooks.on_prefill) hooks.on_prefill(n_prompt);

        // ---- generate -------------------------------------------------------
        std::vector<llama_token> generated;   // tokens decoded into the KV
        generated.reserve(static_cast<std::size_t>(std::max(max_tokens, 0)));
        std::string emitted;                  // detokenized text sent so far
        emitted.reserve(1024);
        const llama_token eos = llama_vocab_eos(vocab);
        for (int i = 0; i < max_tokens; ++i) {
            llama_token id = llama_sampler_sample(smpl, ctx, -1);
            llama_sampler_accept(smpl, id);

            if (id == eos) break;

            // Feed the new token back for the next step — on the session's
            // sequence, at the position just past everything decoded so far.
            fill_batch(id, static_cast<llama_pos>(n_total), /*want_logits*/ true);
            if (llama_decode(ctx, batch) != 0) {
                log::warn("llama_decode (step) failed; stopping generation");
                break;
            }
            generated.push_back(id);
            ++n_total;

            // Streaming detokenization: piece-by-piece is lossy on
            // SentencePiece vocabularies (leading-space markers vanish),
            // and it broke the KV session prefix match. So detokenize the
            // whole sequence and emit only the confirmed tail. The
            // quadratic re-detok is bounded by n_ctx and stays in the
            // noise next to llama_decode (one matmul per byte vs a
            // full layer pass per token). Also computed when stop
            // sequences are set — they are text-level, not token-level.
            const bool want_full = hooks.on_token || stops;
            std::string full;
            if (want_full) {
                full = detokenize(generated);
                // Template stop sequences (tool-call end tags): stop as soon
                // as one appears; the tail is trimmed by apply_stops().
                if (stops) {
                    bool hit = false;
                    for (const auto& s : *stops) {
                        if (!s.empty() && full.find(s) != std::string::npos) {
                            hit = true;
                            break;
                        }
                    }
                    if (hit) {
                        llama_memory_t m2 = llama_get_memory(ctx);
                        if (m2) llama_memory_seq_rm(m2, seq, n_total, -1);
                        break;
                    }
                }
            }
            if (hooks.on_token) {
                if (full.size() > emitted.size() &&
                    full.compare(0, emitted.size(), emitted) == 0) {
                    hooks.on_token(full.substr(emitted.size()));
                    emitted = std::move(full);
                } else if (full != emitted) {
                    // Shouldn't happen (detok is a pure prefix function
                    // here), but never corrupt a stream: resend the tail
                    // conservatively.
                    hooks.on_token(full);
                    emitted = std::move(full);
                }
            }
        }
        llama_batch_free(batch);

        // Detokenize the WHOLE generated stream at once for the non-stream
        // return path (exact, no per-token loss).
        std::string out = detokenize(generated);

        // Persist the session: its KV now holds prompt + generated tokens,
        // which is exactly the prefix the next turn's prompt must extend.
        if (sess) {
            sess->tokens = prompt_tokens;
            sess->tokens.insert(sess->tokens.end(),
                                generated.begin(), generated.end());
        }
        if (stats) stats->completion_tokens = static_cast<int>(generated.size());
        return out;
    }

    std::string complete(const std::string& prompt, int max_tokens,
                         const std::string& session_id = {},
                         bool allow_session_fallback = true,
                         CompletionStats* stats = nullptr) {
        StreamHooks none;
        return complete_stream(prompt, max_tokens, session_id,
                               allow_session_fallback, none, stats);
    }

    // ---- HTTP handlers ----------------------------------------------------

    void handle_models(const httplib::Request&, httplib::Response& res) {
        json body = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id",       opts.model_path.stem().string()},
                    {"object",   "model"},
                    {"owned_by", "hostely"},
                }
            })},
        };
        res.set_content(body.dump(), "application/json");
    }

    void handle_completions(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        std::string prompt = body.value("prompt", std::string{});
        int max_tokens = body.value("max_tokens", 256);
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"prompt is required"})", "application/json");
            return;
        }

        // /v1/completions is always stateless per the OpenAI contract.
        CompletionStats stats;
        std::string text = complete(prompt, max_tokens,
                                    /*session_id*/ {},
                                    /*allow_session_fallback*/ false, &stats);
        if (text.empty() && stats.prompt_tokens > 0) {
            // complete() returns "" only on decode failure (e.g. KV pool
            // exhausted by live sessions) — that's a server error, not an
            // empty completion, and must not be masked by a 200.
            res.status = 503;
            res.set_content(json{{"error",
                "inference failed; the KV pool may be exhausted — restart "
                "the server or retry with a shorter prompt"}}.dump(),
                "application/json");
            return;
        }

        json resp = {
            {"id",       "cmpl-" + std::to_string(
                              std::chrono::system_clock::now().time_since_epoch().count())},
            {"object",   "text_completion"},
            {"created",  std::time(nullptr)},
            {"model",    opts.model_path.stem().string()},
            {"choices",  json::array({{
                {"text",          text},
                {"index",         0},
                {"finish_reason", "stop"},
            }})},
            {"usage", {{"prompt_tokens", stats.prompt_tokens},
                       {"completion_tokens", stats.completion_tokens},
                       {"total_tokens", stats.prompt_tokens + stats.completion_tokens}}},
        };
        res.set_content(resp.dump(), "application/json");
    }

    // ---- shared HTTP helpers ----------------------------------------------

    std::string api_error(const std::string& type, const std::string& message) const {
        return json{{"type", "error"},
                    {"error", {{"type", type}, {"message", message}}}}.dump();
    }

    // Flatten one Anthropic message: content is either a string or an array
    // of blocks ({type:"text",text:...}). Tool blocks are ignored for now
    // (Phase C); their text is skipped rather than sent to the model.
    std::string flatten_anthropic_content(const json& content) {
        if (content.is_string()) return content.get<std::string>();
        if (!content.is_array()) return {};
        std::string out;
        for (const auto& block : content) {
            if (!block.is_object()) continue;
            if (block.value("type", "") == "text") {
                if (!out.empty()) out += "\n";
                out += block.value("text", "");
            }
        }
        return out;
    }

    // Anthropic /v1/messages — the protocol Claude Code and the Claude SDKs
    // speak. System prompt arrives top-level; any model string is accepted
    // and resolved to whatever is loaded (Ollama-style aliasing). Tool
    // definitions are rendered through the model's jinja template and
    // tool_use / tool_result blocks round-trip through llama.cpp's chat
    // model, so agentic clients drive the local model end to end.
    void handle_messages(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(api_error("invalid_request_error", e.what()),
                            "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(api_error("invalid_request_error",
                                      "messages array is required"),
                            "application/json");
            return;
        }

        const std::string system = body.contains("system")
            ? flatten_anthropic_content(body["system"]) : std::string{};
        std::vector<common_chat_tool> tools;
        if (body.contains("tools") && body["tools"].is_array())
            tools = to_common_tools(body["tools"], /*anthropic*/ true);

        common_chat_params chat_params;
        try {
            chat_params = apply_chat_template(
                anthropic_to_common(body["messages"], system),
                tools.empty() ? nullptr : &tools);
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(api_error("api_error",
                    std::string("chat template failed: ") + e.what()),
                    "application/json");
            return;
        }

        std::string model_name = opts.model_path.stem().string();
        const int max_tokens = body.value("max_tokens", 256);
        const bool want_stream = body.value("stream", false);
        const std::string session_id = req.get_header_value("X-Session-Id");
        const auto id = std::string("msg_") +
                  std::to_string(std::chrono::system_clock::now()
                                     .time_since_epoch().count());

        // Build the wire content-block array from a parsed generation.
        auto make_blocks = [&](const common_chat_msg& parsed,
                               const char** stop_reason) -> json {
            json blocks = json::array();
            if (!parsed.content.empty())
                blocks.push_back({{"type", "text"}, {"text", parsed.content}});
            for (const auto& tc : parsed.tool_calls) {
                json input = json::object();
                try { input = json::parse(tc.arguments); }
                catch (const std::exception&) {
                    input = json::object();   // malformed args -> empty input
                }
                blocks.push_back({{"type", "tool_use"},
                                  {"id", tc.id.empty() ? gen_id("toolu_") : tc.id},
                                  {"name", tc.name},
                                  {"input", input}});
            }
            *stop_reason = parsed.tool_calls.empty() ? "end_turn" : "tool_use";
            if (blocks.empty())
                blocks.push_back({{"type", "text"}, {"text", ""}});
            return blocks;
        };

        if (!want_stream) {
            CompletionStats stats;
            std::string text = complete_stream(chat_params.prompt, max_tokens,
                                               session_id,
                                               /*allow_session_fallback*/ true,
                                               {}, &stats,
                                               &chat_params.additional_stops);
            if (text.empty() && stats.prompt_tokens > 0 &&
                stats.completion_tokens == 0) {
                res.status = 503;
                res.set_content(api_error("overloaded_error",
                        "inference failed; the KV pool may be exhausted"),
                        "application/json");
                return;
            }
            text = apply_stops(std::move(text), chat_params);
            common_chat_msg parsed;
            if (chat_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY ||
                !chat_params.additional_stops.empty()) {
                try { parsed = parse_generation(text, chat_params, false); }
                catch (const std::exception&) { parsed.content = text; }
            } else {
                parsed.content = text;
            }
            const char* stop_reason = "end_turn";
            json resp = {
                {"id",            id},
                {"type",          "message"},
                {"role",          "assistant"},
                {"model",         model_name},
                {"content",       make_blocks(parsed, &stop_reason)},
                {"stop_reason",   stop_reason},
                {"stop_sequence", nullptr},
                {"usage",         {{"input_tokens", stats.prompt_tokens},
                                   {"output_tokens", stats.completion_tokens}}},
            };
            res.set_content(resp.dump(), "application/json");
            return;
        }

        // ---- SSE stream (Anthropic event protocol) ------------------------
        // message_start is emitted from the prefill hook, so it carries the
        // real input-token count. The inference mutex is held for the whole
        // stream: one request at a time by design for v1 — a slow client
        // throttles the queue, not memory (each pending request holds only
        // its JSON body). With tools, blocks are opened/closed dynamically:
        // text deltas extend a text block; parsed tool calls appear as
        // tool_use blocks fed by input_json_delta fragments.
        res.set_chunked_content_provider(
            "text/event-stream",
            [this, chat_params = std::move(chat_params), max_tokens, session_id,
             model_name, id](std::size_t, httplib::DataSink& sink) -> bool {
                auto send = [&](const std::string& event, const json& data) {
                    std::string chunk = "event: " + event + "\ndata: " +
                                        data.dump() + "\n\n";
                    return sink.write(chunk.data(), chunk.size());
                };

                const bool with_tools = chat_params.format !=
                                            COMMON_CHAT_FORMAT_CONTENT_ONLY ||
                                        !chat_params.additional_stops.empty();

                CompletionStats stats;
                StreamHooks hooks;

                int next_block = -1;
                bool block_open = false;

                auto open_block = [&](const char* type) {
                    ++next_block;
                    block_open = true;
                    json block = {{"type", type}};
                    if (std::string(type) == "tool_use")
                        block["input"] = json::object();
                    send("content_block_start",
                         {{"type", "content_block_start"},
                          {"index", next_block},
                          {"content_block", block}});
                };
                auto close_block = [&]() {
                    if (!block_open) return;
                    send("content_block_stop",
                         {{"type", "content_block_stop"}, {"index", next_block}});
                    block_open = false;
                };
                auto message_start = [&](int input_tokens) {
                    send("message_start", {
                        {"type", "message_start"},
                        {"message", {
                            {"id", id}, {"type", "message"},
                            {"role", "assistant"}, {"model", model_name},
                            {"content", json::array()},
                            {"stop_reason", nullptr},
                            {"usage", {{"input_tokens", input_tokens},
                                       {"output_tokens", 0}}},
                        }}});
                };
                auto ensure_started = [&](int input_tokens) {
                    if (next_block >= 0 || block_open) return;
                    message_start(input_tokens);
                    if (!with_tools) open_block("text");
                };

                // Text streams live. The moment the partial parser sees a
                // tool call forming, streaming stops and everything after is
                // buffered — tool_use blocks are emitted atomically from the
                // final parse below. Splitting a tool call's JSON across
                // incremental fragments gains nothing (clients parse the
                // block on content_block_stop) and risks torn output when
                // the partial parser revises the call mid-stream.
                bool buffering = false;
                std::string acc;
                hooks.on_prefill = [&](int n_prompt) { ensure_started(n_prompt); };
                hooks.on_token = [&](const std::string& piece) {
                    if (piece.empty()) return;
                    if (!with_tools) {
                        ensure_started(0);
                        send("content_block_delta", {
                            {"type", "content_block_delta"},
                            {"index", next_block},
                            {"delta", {{"type", "text_delta"}, {"text", piece}}}});
                        return;
                    }
                    if (buffering) { acc += piece; return; }
                    acc += piece;
                    ensure_started(0);
                    try {
                        auto parsed = parse_generation(acc, chat_params, true);
                        if (!parsed.tool_calls.empty()) {
                            buffering = true;   // stop streaming text
                            return;
                        }
                        if (!parsed.content.empty() && !block_open) {
                            open_block("text");
                        }
                        if (block_open) {
                            send("content_block_delta", {
                                {"type", "content_block_delta"},
                                {"index", next_block},
                                {"delta", {{"type", "text_delta"},
                                           {"text", parsed.content}}}});
                            acc.clear();   // this content is confirmed
                        }
                    } catch (const std::exception&) {
                        // Parser hiccup mid-stream: emit raw text, never hang.
                        if (!block_open) open_block("text");
                        send("content_block_delta", {
                            {"type", "content_block_delta"},
                            {"index", next_block},
                            {"delta", {{"type", "text_delta"}, {"text", piece}}}});
                        acc.clear();
                    }
                };

                std::string text = complete_stream(chat_params.prompt, max_tokens,
                                                   session_id,
                                                   /*allow_session_fallback*/ true,
                                                   hooks, &stats,
                                                   &chat_params.additional_stops);
                if (next_block < 0 && !block_open) message_start(stats.prompt_tokens);
                close_block();

                // Final, authoritative parse: emit text (if it wasn't fully
                // streamed) and any tool_use blocks as complete blocks.
                text = apply_stops(std::move(text), chat_params);
                common_chat_msg parsed;
                if (with_tools) {
                    try { parsed = parse_generation(text, chat_params, false); }
                    catch (const std::exception&) { parsed.content = text; }
                } else {
                    parsed.content = text;
                }
                const char* stop_reason = "end_turn";
                if (!with_tools) {
                    if (!parsed.content.empty()) {
                        open_block("text");
                        send("content_block_delta", {
                            {"type", "content_block_delta"},
                            {"index", next_block},
                            {"delta", {{"type", "text_delta"},
                                       {"text", parsed.content}}}});
                        close_block();
                    }
                } else {
                    // text that was still buffered when tool calls started
                    if (!parsed.content.empty()) {
                        open_block("text");
                        send("content_block_delta", {
                            {"type", "content_block_delta"},
                            {"index", next_block},
                            {"delta", {{"type", "text_delta"},
                                       {"text", parsed.content}}}});
                        close_block();
                    }
                    for (const auto& tc : parsed.tool_calls) {
                        json input = json::object();
                        try { input = json::parse(tc.arguments); }
                        catch (const std::exception&) {}
                        open_block("tool_use");
                        send("content_block_delta", {
                            {"type", "content_block_delta"},
                            {"index", next_block},
                            {"delta", {{"type", "input_json_delta"},
                                       {"partial_json", input.dump()}}}});
                        close_block();
                    }
                    if (!parsed.tool_calls.empty()) stop_reason = "tool_use";
                }

                send("message_delta", {
                    {"type", "message_delta"},
                    {"delta", {{"stop_reason", stop_reason},
                               {"stop_sequence", nullptr}}},
                    {"usage", {{"output_tokens", stats.completion_tokens}}}});
                send("message_stop", {{"type", "message_stop"}});
                sink.done();
                return true;
            });
    }

    // OpenAI /v1/chat/completions — with tool calling. The conversation is
    // rendered through the model's jinja template (tools included) and the
    // generated text is PEG-parsed back into content + tool_calls.
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            return;
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"messages array is required"})", "application/json");
            return;
        }

        int max_tokens = body.value("max_tokens", 256);
        // Phase 7d: a client-managed session id keeps one KV sequence per
        // conversation. Without it, complete() tries prefix-matching against
        // existing sessions and otherwise runs stateless.
        std::string session_id = req.get_header_value("X-Session-Id");

        std::vector<common_chat_tool> tools;
        if (body.contains("tools") && body["tools"].is_array())
            tools = to_common_tools(body["tools"], /*anthropic*/ false);

        common_chat_params chat_params;
        try {
            chat_params = apply_chat_template(openai_to_common(body["messages"]),
                                              tools.empty() ? nullptr : &tools);
        } catch (const std::exception& e) {
            // A template that can't render the conversation must not take
            // the whole server down.
            res.status = 500;
            res.set_content(json{{"error",
                std::string("chat template failed: ") + e.what()}}.dump(),
                "application/json");
            return;
        }
        std::string prompt = std::move(chat_params.prompt);
        if (std::getenv("HOSTELY_DEBUG_PROMPT")) {
            log::info("prompt >>> " + prompt.substr(0, 2000));
            log::info("format=" +
                      std::string(common_chat_format_name(chat_params.format)));
        }

        const bool want_stream = body.value("stream", false);
        const std::string model_name = opts.model_path.stem().string();

        // Streaming: same chunk protocol as OpenAI (chat.completion.chunk
        // deltas, finish_reason chunk, then `data: [DONE]`). With tools, the
        // accumulated text is partially PEG-parsed per token and diffed, so
        // content and tool_calls stream incrementally.
        if (want_stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [this, prompt = std::move(prompt), max_tokens, session_id,
                 model_name, chat_params = std::move(chat_params)](
                    std::size_t, httplib::DataSink& sink) -> bool {
                    auto send = [&](const json& obj) {
                        std::string chunk = "data: " + obj.dump() + "\n\n";
                        return sink.write(chunk.data(), chunk.size());
                    };
                    auto chunk_obj = [&](const json& delta, const char* finish) {
                        // NB: a null const char* would route into nlohmann's
                        // string ctor (strlen crash) — box it explicitly.
                        json finish_reason = finish ? json(finish) : json(nullptr);
                        return json{
                            {"id",      "chatcmpl-" + std::to_string(
                                          std::chrono::system_clock::now()
                                              .time_since_epoch().count())},
                            {"object",  "chat.completion.chunk"},
                            {"created", std::time(nullptr)},
                            {"model",   model_name},
                            {"choices", json::array({{
                                {"index", 0},
                                {"delta", delta},
                                {"finish_reason", finish_reason},
                            }})}};
                    };

                    const bool with_tools = !chat_params.additional_stops.empty() ||
                                            chat_params.format !=
                                                COMMON_CHAT_FORMAT_CONTENT_ONLY;
                    send(chunk_obj(json{{"role", "assistant"}}, nullptr));

                    std::string acc;
                    common_chat_msg prev;
                    CompletionStats stats;
                    StreamHooks hooks;
                    hooks.on_token = [&](const std::string& piece) {
                        if (piece.empty()) return;
                        acc += piece;
                        if (!with_tools) {
                            send(chunk_obj(json{{"content", piece}}, nullptr));
                            return;
                        }
                        // Partial parse + diff -> incremental deltas.
                        try {
                            auto parsed = parse_generation(acc, chat_params, /*partial*/ true);
                            for (const auto& d : common_chat_msg_diff::compute_diffs(prev, parsed)) {
                                if (!d.content_delta.empty())
                                    send(chunk_obj(json{{"content", d.content_delta}}, nullptr));
                                if (d.tool_call_index != std::string::npos) {
                                    json tc = {{"index", d.tool_call_index}};
                                    if (!d.tool_call_delta.name.empty() ||
                                        !d.tool_call_delta.id.empty()) {
                                        tc["id"] = d.tool_call_delta.id.empty()
                                            ? gen_id("call_") : d.tool_call_delta.id;
                                        tc["type"] = "function";
                                        tc["function"] = {
                                            {"name", d.tool_call_delta.name},
                                            {"arguments", d.tool_call_delta.arguments}};
                                    } else {
                                        tc["function"] = {
                                            {"arguments", d.tool_call_delta.arguments}};
                                    }
                                    send(chunk_obj(json{{"tool_calls", json::array({tc})}},
                                                   nullptr));
                                }
                            }
                            prev = std::move(parsed);
                        } catch (const std::exception&) {
                            // Parser hiccup mid-stream: fall back to raw text.
                            send(chunk_obj(json{{"content", piece}}, nullptr));
                        }
                    };
                    std::string text = complete_stream(prompt, max_tokens, session_id,
                                                       /*allow_session_fallback*/ true,
                                                       hooks, &stats,
                                                       &chat_params.additional_stops);
                    if (text.empty() && stats.completion_tokens == 0 && acc.empty())
                        text = acc;

                    const char* finish = "stop";
                    common_chat_msg parsed;
                    if (with_tools) {
                        try {
                            parsed = parse_generation(apply_stops(acc, chat_params),
                                                      chat_params, /*partial*/ false);
                            if (!parsed.tool_calls.empty()) finish = "tool_calls";
                        } catch (const std::exception&) {
                            parsed.content = acc;
                        }
                    } else {
                        parsed.content = acc;
                    }

                    json final_chunk = chunk_obj(json::object(), finish);
                    final_chunk["usage"] = {
                        {"prompt_tokens", stats.prompt_tokens},
                        {"completion_tokens", stats.completion_tokens},
                        {"total_tokens", stats.prompt_tokens + stats.completion_tokens}};
                    send(final_chunk);
                    sink.write("data: [DONE]\n\n", 14);
                    sink.done();
                    return true;
                });
            return;
        }

        CompletionStats stats;
        std::string text = complete_stream(prompt, max_tokens, session_id,
                                           /*allow_session_fallback*/ true,
                                           {}, &stats, &chat_params.additional_stops);
        if (text.empty() && stats.prompt_tokens > 0 && stats.completion_tokens == 0) {
            res.status = 503;
            res.set_content(json{{"error",
                "inference failed; the KV pool may be exhausted — restart "
                "the server or retry with a shorter prompt"}}.dump(),
                "application/json");
            return;
        }

        text = apply_stops(std::move(text), chat_params);
        if (std::getenv("HOSTELY_DEBUG_PROMPT")) {
            log::info("raw gen >>> " + text.substr(0, 500));
        }
        common_chat_msg parsed;
        bool has_tools = false;
        if (chat_params.format != COMMON_CHAT_FORMAT_CONTENT_ONLY ||
            !chat_params.additional_stops.empty()) {
            try {
                parsed = parse_generation(text, chat_params, /*partial*/ false);
                has_tools = !parsed.tool_calls.empty();
            } catch (const std::exception&) {
                parsed.content = text;
            }
        } else {
            parsed.content = text;
        }

        json message = {{"role", "assistant"}};
        if (has_tools) {
            if (!parsed.content.empty()) message["content"] = parsed.content;
            else message["content"] = nullptr;
            json tcs = json::array();
            for (std::size_t i = 0; i < parsed.tool_calls.size(); ++i) {
                const auto& tc = parsed.tool_calls[i];
                tcs.push_back({
                    {"id",        tc.id.empty() ? gen_id("call_") : tc.id},
                    {"type",      "function"},
                    {"index",     i},
                    {"function",  {{"name", tc.name},
                                   {"arguments", tc.arguments}}},
                });
            }
            message["tool_calls"] = tcs;
        } else {
            message["content"] = parsed.content;
        }

        json resp = {
            {"id",       "chatcmpl-" + std::to_string(
                              std::chrono::system_clock::now().time_since_epoch().count())},
            {"object",   "chat.completion"},
            {"created",  std::time(nullptr)},
            {"model",    opts.model_path.stem().string()},
            {"choices",  json::array({{
                {"index",         0},
                {"message",       message},
                {"finish_reason", has_tools ? "tool_calls" : "stop"},
            }})},
            {"usage", {{"prompt_tokens", stats.prompt_tokens},
                       {"completion_tokens", stats.completion_tokens},
                       {"total_tokens", stats.prompt_tokens + stats.completion_tokens}}},
        };
        res.set_content(resp.dump(), "application/json");
    }

    void register_routes() {
        svr.Get ("/v1/models",            [this](auto& r, auto& s){ handle_models(r, s); });
        svr.Post("/v1/completions",       [this](auto& r, auto& s){ handle_completions(r, s); });
        svr.Post("/v1/chat/completions",  [this](auto& r, auto& s){ handle_chat_completions(r, s); });
        svr.Post("/v1/messages",          [this](auto& r, auto& s){ handle_messages(r, s); });
        svr.Get ("/health",               [](auto&, auto& s){
            s.set_content(R"({"status":"ok"})", "application/json");
        });
    }
};

// ----------------------------------------------------------------------------
// Server (public)
// ----------------------------------------------------------------------------

Server::Server(ServeOptions opts) : options_(std::move(opts)) {}

Server::~Server() = default;

int Server::start() {
    using namespace hostely::log;
    impl_ = std::make_unique<Impl>();
    impl_->opts = options_;

    // Single global backend init — safe to call multiple times in a process.
    llama_backend_init();

    // Register routes BEFORE attempting model load so /health remains
    // reachable even when the model file is missing or corrupt — useful for
    // diagnostics.
    impl_->register_routes();
    impl_->svr.set_payload_max_length(16 * 1024 * 1024);
    impl_->svr.set_read_timeout(120, 0);

    if (!impl_->load_model(loaded_)) return 1;

    // Phase 7b: fit check. We have real tensor_bytes + KV state now, so
    // the numbers the user sees are actual — not a heuristic.
    std::uint64_t peak_rss = 0;
    {
        using namespace hostely::models;
        auto est = FitAdvisor::check(loaded_, options_.ctx_size);
        peak_rss = est.peak_rss_bytes;
        switch (est.verdict) {
        case Verdict::Fits:
            info("fit check : " + est.message);
            break;
        case Verdict::Tight:
            warn("fit check : " + est.message);
            break;
        case Verdict::WontFit:
            error("fit check : " + est.message);
            if (!options_.no_fit_check) {
                error("refusing to load; pass --no-fit-check to override.");
                return 1;
            }
            warn("--no-fit-check set; loading anyway at the user's request");
            break;
        }
    }

    // Phase 7c: publish what we just loaded so `hostely status` can report
    // it. Best-effort — a failure here is a warning, not a fatal error.
    {
        ServeLock lock;
        lock.pid            = ::getpid();
        lock.model_path     = loaded_.path.string();
        lock.display_name   = loaded_.display_name;
        lock.quant_name     = loaded_.quant_name;
        lock.params         = loaded_.params;
        lock.tensor_bytes   = loaded_.tensor_bytes;
        lock.kv_state_bytes = loaded_.kv_state_bytes;
        lock.peak_rss_bytes = peak_rss;
        lock.ctx_size       = options_.ctx_size;
        lock.ctx_train      = loaded_.ctx_train;
        lock.port           = options_.port;
        if (!write_serve_lock(lock)) {
            warn("could not write serve lockfile: " +
                 paths::serve_lock_file().string());
        }
    }

    info("listening on http://0.0.0.0:" + std::to_string(options_.port));
    info("endpoints:");
    info("  GET  /v1/models");
    info("  POST /v1/completions");
    info("  POST /v1/chat/completions  (OpenAI-compatible, stream supported)");
    info("  POST /v1/messages           (Anthropic-compatible, stream supported)");
    info("  GET  /health");

    bool ok = impl_->svr.listen("0.0.0.0", options_.port);
    if (!ok && !impl_->stop_requested.load()) {
        error("server failed to bind");
        remove_serve_lock();
        return 1;
    }
    info("server stopped");
    remove_serve_lock();
    return 0;
}

void Server::stop() {
    if (!impl_) return;
    impl_->stop_requested.store(true);
    impl_->svr.stop();
    remove_serve_lock();
}

}  // namespace hostely::inference
