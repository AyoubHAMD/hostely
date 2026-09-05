# hostely Exposure Roadmap — `hostely proxy`, `hostely tunnel`, `hostely expose`

> Goal: a hostely user goes from "I have a container running" to "my team opens
> `https://myservice.mydomain.com` and it works" in one command — with hostely
> owning domains, DNS, certificates, router, and routing end to end.

## The one hard constraint (design around it, don't fight it)

Every exposure path terminates in one of two places:

1. **Inbound path** — the internet reaches the Mac directly. Requires the router
   to forward ports. Some routers (e.g. Freebox) refuse to forward privileged
   ports (< 16384), so a clean `https://…` URL is *impossible* on them. DNS,
   certs, and a perfect proxy on the Mac cannot fix this — the packets never
   arrive.
2. **Outbound path** — the Mac dials *out* to a relay that owns a real 443
   (a VPS). Nothing on the router needs to change; ISP CGNAT and dynamic IPs
   are irrelevant.

Therefore: **the tunnel is the default path for clean URLs; the port-forward is
the optimization when the router allows it.** hostely should detect which path
is available and pick automatically, and never make the user reason about it.

## Guiding decisions

- **One command per outcome, not per mechanism.** `hostely expose web
  example.com` is the surface; proxy/tunnel/port-forward/ACME are internals.
- **Zero new daemons.** Everything runs in the single hostely binary (matches
  the existing posix_spawn, no-daemon philosophy). A long-lived `hostely proxy
  serve` process is a *command*, not a background service; on macOS it can be
  installed as a LaunchAgent by `hostely proxy install`.
- **Zero heavyweight deps stays true.** Standalone asio (header-only) +
  Boost.Beast for HTTP/WS, OpenSSL 3 for TLS (SNI switch, ALPN, and the
  ECDSA/HMAC primitives ACME needs — the same backend cpp-httplib and drogon
  already use). miniupnpc + libnatpmp (BSD) for router automation.
  No embedded nginx/envoy, no aws-sdk-cpp, no websocketpp/IXWebSocket (dormant).
- **Fail loudly, degrade visibly.** `hostely doctor --exposure` reports exactly
  which path is possible and what blocks it (router refuses 443, DNS not
  authoritative, etc.).
- **Security default**: the proxy terminates TLS itself; containers never
  receive plaintext; secrets (DNS API tokens) live in the hostely config dir
  with 0600, never in compose files.

## Phases

### Phase 0 — TLS foundation (the only true prerequisite)
- Link OpenSSL (Homebrew on macOS; static where practical).
- TLS server context, SNI-based cert selection, cert hot-reload (watch the
  cert dir; swap contexts without dropping connections).
- ACME client (RFC 8555): account key, JWS signing, order flow.
  **DNS-01 only** — it needs no inbound port at all, which sidesteps every
  router problem and is required for wildcard certs.
- DNS provider abstraction: **Cloudflare first** (one Bearer-token JSON call),
  Route 53 later (hand-rolled SigV4 + XML, ~500 LOC — only on demand).
- Output: `hostely certs issue app.steerai.autos` and `*.steerai.autos`
  wildcard issuance + auto-renewal loop.
- *Exit criteria*: `curl https://app.steerai.autos` against hostely's own TLS
  listener passes with a public cert, renewal verified by shortening the
  renew window.

### Phase 1 — `hostely proxy` (the reverse proxy)
- HTTP/1.1 + WebSocket reverse proxy on 80/443, Host-header (SNI) routing.
  Streaming bodies end-to-end; WS upgrade pass-through (AppFlowy-style apps
  die without it).
- **Container-first routing model**:
  ```
  hostely run web --image nginx --port 8080:80 --host web.steerai.autos
  hostely expose web web.steerai.autos        # attach/detach routes
  ```
  Routes are derived from the existing container registry (`hostely ps` data +
  an `expose` table in the hostely store), so *there is no nginx.conf to
  maintain*. Route changes hot-reload.
- Static vhosts too (serve a directory): `hostely expose --static ./site
  docs.steerai.autos`.
- Sensible per-vhost defaults: HTTP→HTTPS redirect, HSTS, gzip; opt-in client
  body limits and access-log format.
- Observability: access log to the rotating logfile; live per-vhost request
  counters surfaced in `hostely top`.
- *Exit criteria*: the AppFlowy nginx container is replaced by hostely proxy on
  a test machine; web + WS + auth flows work unchanged.

### Phase 2 — Router & DNS automation (make inbound set itself up)
- NAT-PMP/PCP client (Apple routers speak it; Freebox supports NAT-PMP) and a
  UPnP IGD fallback: `hostely expose` asks the router for the external mapping
  automatically. If the router grants 443 — great, inbound path works with a
  clean URL. If it refuses (Freebox < 16384), hostely knows and records it.
- Public-IP watcher → DNS A/AAAA update via the same provider layer as ACME
  (this replaces every user's home-grown "ip turning script").
- Router capability probe: NAT-PMP supported? UPnP? privileged ports allowed?
  Cached in the store; surfaced in `doctor --exposure`.
- *Exit criteria*: on a NAT-PMP router, `hostely expose web web.example.com`
  results in DNS + router mapping + cert with zero manual steps.

### Phase 3 — `hostely tunnel` (the default clean-URL path)
- Outbound-only client in hostely: persistent multiplexed connection(s) to a
  relay; relay forwards `host` → stream. TCP first (HTTP/1.1 + WS ride on it),
  HTTP/2 later.
- **Reference relay, two distributions**:
  - `hostely-relay` — tiny open-source server (~200 lines, static binary)
    anyone runs on a $3 VPS; config = one token per client.
  - Optionally, a hosted relay for users with no VPS (cheap/free tier) — this
    is the natural sponsor/open-core line for the project. Keep the protocol
    open either way.
- Pairing UX: `hostely tunnel pair <relay> <token>` once; after that
  `hostely expose web web.example.com` transparently goes via tunnel.
- Reconnect with jittered backoff, offline queueing of nothing (stateless
  forwarding), latency telemetry.
- *Exit criteria*: a Mac behind a Freebox serves a container at a port-less
  HTTPS URL through the relay; kill the WAN mid-transfer and the tunnel
  reconnects.

### Phase 4 — `hostely expose` end-to-end UX (the promise)
```
hostely expose web web.example.com
  → checks/derives DNS provider + credentials
  → creates/updates DNS (A record, or tunnel CNAME)
  → issues/reuses wildcard cert (DNS-01)
  → picks path: router-forward if possible, else tunnel
  → wires the route, starts/updates proxy or tunnel
  → prints the working URL and runs a real external probe to confirm
```
- `hostely doctor --exposure`: machine-readable verdict of the whole chain
  (domain authoritative? creds valid? router limits? path chosen? reachable
  from outside via a probe?).
- Idempotent re-runs; `--forget` to tear down cleanly (DNS optional delete,
  route removal, cert GC).

### Phase 5 — polish & product surface
- HTTP/2 and HTTP/3 (QUIC) at the edge where OpenSSL allows; keep containers
  unchanged.
- Rate limiting / basic bot filtering per vhost.
- `hostely top` gains an "exposure" section: per-vhost RPS, active tunnels,
  cert expiry countdown.
- Docs: "expose your first app in 5 minutes" guide; comparison table vs
  nginx-proxy-manager / Caddy / Cloudflare Tunnel.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| TLS/ACME is the long pole | DNS-01 only, two providers max; cut scope before adding HTTP-01 |
| Proxy bugs affect *all* apps at once | Per-vhost circuit breaker; easy `--rollback` to previous route table |
| WebSocket/streaming subtleties | Test suite with a WS echo container in CI from Phase 1, not bolted on later |
| Tunnel relay is a new thing to run | Keep `hostely-relay` a single static binary + one config; publish a 60-second setup guide |
| Users on double-NAT/CGNAT assume inbound works | `doctor --exposure` detects CGNAT early and pushes the tunnel path |

## Sequencing

| Phase | Depends on | Rough size |
|---|---|---|
| 0 TLS + ACME (DNS-01) | — | the big rock: TLS + ACME + 2 DNS providers |
| 1 proxy | 0 | medium (routing, WS, hot reload) |
| 2 router/DNS automation | 1 | small (NAT-PMP is a simple binary protocol) |
| 3 tunnel | 0 | medium (client + relay) |
| 4 `expose` UX | 1 + (2 or 3) | small (orchestration + probes) |
| 5 polish | 4 | open-ended |

Phases 1 and 3 can proceed in parallel once Phase 0 lands; Phase 2 is
independent and can fill any gap.

## Validation against prior art (deep research, Sep 2026)

Findings from due-diligence on Caddy/Traefik/NPM/Pangolin, the PaaS platforms
(Cloudron, Coolify, Dokploy, CapRover, Dokku, Cosmos, CasaOS, YunoHost),
tunnels (frp, rathole, bore, cloudflared, ngrok, Tailscale Funnel,
localtunnel/PageKite), and the C++ library landscape.

### The plan survives — with these confirmations
- **The gap is real and demanded.** CasaOS — the highest-download self-hosting
  dashboard — has *no* domain/TLS layer, and it is its most-requested missing
  feature. No established C++ analogue of Caddy exists.
- **Routes-from-container-registry beats label scraping.** Traefik guesses the
  backend port (classic 502 bug class); hostely owns the spec at `run` time.
  Rebuild the full desired route set per event and atomically swap — never
  patch deltas (Dokploy's stale-config 502s).
- **DNS-01-only is the right call.** HTTP-01-by-default is the #1 "cert not
  issued" complaint driver across Coolify/CapRover/Dokploy; Cloudron's
  wildcard-DNS-01 default produces the fewest. Also enables wildcards, which
  hide hostnames from CT logs.
- **In-process ACME avoids an entire CVE class.** nginx-proxy-manager's
  security history is dominated by shell-command injection in exactly the
  cert/DNS-credential path hostely would implement — in C++ with no shelling
  out, that class disappears.
- **Tunnel-as-default is genuinely differentiated.** Only Cosmos (VPN-based)
  and YunoHost (miniupnpc) ship anything comparable; everyone else delegates.

### Design revisions adopted from the research
1. **Tunnel protocol**: one persistent control connection + yamux-style
   in-tree mux over a single outbound TLS-TCP connection (frp/rathole shape);
   **no WebSocket framing** (Telebit's admitted regret), **no QUIC in v1**
   (a C++ QUIC stack is a project in itself; N parallel TCP connections is the
   cheap fallback if HOL blocking bites).
2. **Backhaul auth**: rathole's Noise_NK pattern — per-relay token + relay
   static X25519 key, MITM-resistant without certificates, using OpenSSL
   primitives already in the tree.
3. **TLS terminates at the relay**; custom domains via **CNAME verification —
   no nameserver lock-in** (the anti-cloudflared move); relay prefers a
   wildcard cert (`*.relay.domain`) over per-host issuance to dodge LE
   rate-limit lockouts (Tailscale's ~34 h lockout warning).
4. **Heartbeats 10 s / kill at 15–30 s** (ngrok's defaults) — NAT mapping
   expiry is the single most common "tunnel died" report. Remember the
   last-working transport across reconnects (cloudflared's lesson).
5. **Relay abuse controls from day one**: token required before any data path,
   per-token stream/rate caps, unclaimed inbound streams dropped in seconds,
   unknown Host/Token closed and logged.
6. **Config lifecycle is the #1 industry bug class**: generated config must be
   ephemeral + validated before apply + reconciled idempotently, with
   user-escape-hatch snippet dirs (Dokku's model) — never a fragile single
   `acme.json`-style blob (Coolify's "delete and restart" fix).
7. **On-demand TLS needs Caddy's "ask" check**: hostely's allowlist is
   "does a running container claim this hostname" — otherwise cert-minting
   abuse.
8. **Cert store as first-class artifact**: cert + key + metadata per domain,
   0600, atomic swap; LE **staging endpoint first**, retain (not revoke) certs
   on uninstall, renew only in-use domains plus a 502-block so HTTP-01-style
   renewals still work for undeployed apps (Dokku).
9. **Router automation ladder**: NAT-PMP/PCP first (Apple gateways speak it),
   UPnP IGD fallback (miniupnpc + libnatpmp, the Transmission pairing).
   YunoHost's ops lesson: refresh mappings on a schedule, never drop a mapping
   because a refresh failed; diagnose missing hairpinning explicitly.
10. **Isolation of planes**: proxy, auth, and tunnel must not share one trust
    plane (Cosmos's 9.8-CVSS history). Exposure is opt-in per container;
    `doctor --exposure` diagnoses DNS, router limits, CGNAT, and hairpinning.

### C++ stack (validated)
| Component | Choice | Why |
|---|---|---|
| Networking | standalone asio + Boost.Beast | one kqueue loop, HTTP/1.1 + WS in-tree, header-only |
| TLS | OpenSSL 3 (Apache-2.0) | SNI switch, ALPN, ACME crypto, brew-native |
| ACME | hand-rolled RFC 8555 DNS-01 (~1k LOC) | no maintained C++ lib exists (acme-lw = reference only); ES256 raw r‖s is the classic trap |
| DNS providers | Cloudflare first, Route 53 deferred | one Bearer call vs hand-rolled SigV4+XML |
| Port mapping | libnatpmp + miniupnpc (BSD) | same maintainer, Transmission precedent |
| Tunnel mux | yamux wire format (in-tree or cpp-yamux, MIT) | open spec, Go reference for interop tests |
| Proxy architecture | Pipy-style per-connection filter chains | best C++ precedent for an embeddable proxy |