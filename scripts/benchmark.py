#!/usr/bin/env python3
"""Benchmark hostely (and optionally Ollama) on an OpenAI-compatible endpoint.

Measures, on the same machine and same GGUF:
  1. Generation throughput (tok/s) at fixed prompt, 256 generated tokens
  2. Prefill rate (tok/s) at ~256 / 1024 / 2048-token prompts, 8 tokens out
  3. Session KV reuse: turn-2 re-prefill cost vs turn-1 (hostely only)

Usage:
  python3 scripts/benchmark.py --name hostely --url http://localhost:8090
  python3 scripts/benchmark.py --name ollama --url http://localhost:11434 --model tinyllama:1.1b-chat-q4_K_M --no-session
"""
import argparse, json, time, urllib.request, statistics, sys

def post(url, path, payload, headers=None, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t0

def wait_health(url):
    for _ in range(600):
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(url.rstrip("/") + path, timeout=1) as r:
                    if r.status == 200: return
            except Exception:
                pass
        time.sleep(0.5)
    sys.exit(f"server at {url} never became healthy")

def token_count(url, text):
    """hostely has no tokenize endpoint; estimate via /v1/completions echo is wasteful.
    We report estimated tokens (chars/4) and label them as estimates."""
    return max(1, len(text) // 4)

def bench_generation(url, model, prompt, n_out):
    """Returns (latency_s, out_tokens, prompt_tokens)."""
    payload = {"model": model, "prompt": prompt, "max_tokens": n_out, "temperature": 0}
    data, dt = post(url, "/v1/completions", payload)
    u = data.get("usage") or {}
    out = u.get("completion_tokens") or n_out
    return dt, out, u.get("prompt_tokens") or 0

def bench_sessions(url, model):
    """Two-turn chat with X-Session-Id. Turn 1 prefills everything; turn 2
    repeats history + adds a few tokens. Returns (turn1_s, turn2_s, prompt1_est, prompt2_est)."""
    story = ("The lighthouse keeper wrote in his log: " +
             "The sea was restless. Waves rose and fell against the ancient stone, "
             "and the gulls wheeled overhead in widening circles as dusk settled. " * 40)
    m1 = [{"role": "user", "content": story + " What did the keeper write about? Reply in one sentence."}]
    m2 = m1 + [{"role": "assistant", "content": "The keeper wrote about a restless sea, rising waves, gulls at dusk."},
               {"role": "user", "content": "And what happened next? Reply in one short sentence."}]
    sid = "bench-s1-%d" % time.time_ns()   # unique per run — a leftover
    # session from an earlier run would make turn-1 look pre-cached
    s = {"model": model, "messages": m1, "max_tokens": 32, "temperature": 0}
    _, t1 = post(url, "/v1/chat/completions", s, headers={"X-Session-Id": sid})
    s2 = {"model": model, "messages": m2, "max_tokens": 32, "temperature": 0}
    _, t2 = post(url, "/v1/chat/completions", s2, headers={"X-Session-Id": sid})
    return t1, t2, token_count(url, json.dumps(m1)), token_count(url, json.dumps(m2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="any")
    ap.add_argument("--name", default="server")
    ap.add_argument("--no-session", action="store_true", help="skip session reuse test")
    args = ap.parse_args()
    wait_health(args.url)

    base_prompt = "Count from 1 to 300, numbers separated by commas." * 3
    rows = []

    # warmup
    bench_generation(args.url, args.model, base_prompt, 8)

    # 1. generation throughput (tokens counted from server-reported usage)
    best = 0.0
    runs = []
    gen_prompt_tokens = 0
    for _ in range(3):
        dt, n, pt = bench_generation(args.url, args.model, base_prompt, 256)
        gen_prompt_tokens = pt
        runs.append(f"{n}tok/{dt*1000:.0f}ms")
        best = max(best, n / max(dt - pt / 1500.0, 0.02))
    gen_best = best
    print(f"[gen ] best of 3: {gen_best:6.1f} tok/s  ({'; '.join(runs)})")

    # 2. prefill scaling
    for approx in (256, 1024, 2048):
        prompt = ("In the old library there were many books about ships, maps, "
                  "islands, storms, anchors, sails, and distant harbours. " * (approx * 4 // 110))
        dt, n_out, real = bench_generation(args.url, args.model, prompt, 8)
        # subtract ~generation cost using measured gen rate
        prefill = max(dt - n_out / max(gen_best, 1), 1e-6)
        rows.append((real, real / prefill, dt))
        print(f"[pref] {real:5d} tok  total {dt*1000:7.0f} ms  ~{real/prefill:6.0f} tok/s prefill")

    # 3. session reuse
    if not args.no_session:
        t1, t2, p1, p2 = bench_sessions(args.url, args.model)
        print(f"[sess] turn1 {t1*1000:6.0f} ms (prompt ~{p1} tok)  turn2 {t2*1000:6.0f} ms (prompt ~{p2} tok, mostly cached)")
        print(f"[sess] turn-2 speedup: {t1/t2:.1f}x")

    print(f"\nRESULTS {args.name}: gen={gen_best:.1f} tok/s; prefill "
          + " ".join(f"{e}tok={r:.0f}tok/s" for e, r, _ in rows))

if __name__ == "__main__":
    main()