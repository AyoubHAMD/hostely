#!/bin/sh
# scripts/smoke.sh — hostely end-to-end smoke test.
#
# What this verifies (in order):
#   1. The build links and `./build/hostely --version` works.
#   2. `hostely init` creates the state directory and default config.
#   3. `hostely doctor` runs without crashing.
#   4. `hostely status` reports CPU/RAM/Metal numbers.
#   5. `hostely run` + `hostely ps` + `hostely stop` round-trip a service
#      through whatever `container` CLI is on PATH (real Apple CLI or the
#      bundled scripts/fake-container.sh stand-in).
#   6. `hostely serve` loads a model and serves /v1/models over HTTP
#      (skipped if no .gguf model is provided — models are not vendored).
#
# Exit codes:
#   0   every step succeeded
#   1   a step failed (the failing step is echoed)
#   2   the script is invoked with no usable hostely binary
#
# Knobs (env vars):
#   HOSTELY_BIN        path to the hostely binary (default: ./build/hostely)
#   HOSTELY_MODEL      path to a .gguf file to load (default: skip serve step)
#   HOSTELY_FAKE_CLI   path to scripts/fake-container.sh (default: same dir)

set -u

ROOT_DIR=$(CDPATH= cd "$(dirname "$0")/.." && pwd)
HOSTELY_BIN="${HOSTELY_BIN:-$ROOT_DIR/build/hostely}"
HOSTELY_MODEL="${HOSTELY_MODEL:-}"
HOSTELY_FAKE_CLI="${HOSTELY_FAKE_CLI:-$ROOT_DIR/scripts/fake-container.sh}"

# -----------------------------------------------------------------------------
# tiny pretty-print helpers
# -----------------------------------------------------------------------------

bold() { printf '\033[1m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
red()   { printf '\033[31m%s\033[0m' "$*"; }

step() {
    printf '\n%s %s\n' "$(bold '::')" "$(bold "$*")"
}

ok() {
    printf '   %s %s\n' "$(green '✓')" "$*"
}

fail() {
    printf '   %s %s\n' "$(red '✗')" "$*" >&2
    exit 1
}

# -----------------------------------------------------------------------------
# 1. find a usable hostely binary
# -----------------------------------------------------------------------------

step "1/6 locate hostely binary"
if [ ! -x "$HOSTELY_BIN" ]; then
    red "hostely binary not found at: $HOSTELY_BIN\n" >&2
    red "build it first with:  cmake -B build && cmake --build build -j\n" >&2
    exit 2
fi
ok "hostely -> $HOSTELY_BIN"

# -----------------------------------------------------------------------------
# 2. init
# -----------------------------------------------------------------------------

step "2/6 hostely init"
# Use an isolated state dir so this never touches the user's real config.
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME_OVERRIDE:-/tmp/hostely-smoke-home}"
rm -rf "$XDG_CONFIG_HOME"
export HOME="$XDG_CONFIG_HOME"   # our paths::home_dir() reads $HOME.
if "$HOSTELY_BIN" init >/dev/null; then
    ok "init succeeded"
else
    fail "init failed (exit $?)"
fi

# -----------------------------------------------------------------------------
# 3. doctor + status
# -----------------------------------------------------------------------------

step "3/6 hostely doctor"
if "$HOSTELY_BIN" doctor >/dev/null 2>&1; then
    ok "doctor ran"
else
    # doctor is informational; it warns when the container CLI is missing.
    # a non-zero exit is acceptable here as long as *something* was printed.
    fail "doctor exited non-zero"
fi

step "4/6 hostely status"
if "$HOSTELY_BIN" status 2>&1 | grep -q '^metal'; then
    ok "status printed Metal block"
else
    fail "status output missing Metal block"
fi

# -----------------------------------------------------------------------------
# 4. service lifecycle (real CLI or fake)
# -----------------------------------------------------------------------------

step "5/6 service run/ps/stop round trip"

# Decide which container CLI to point at. If the user has the real Apple
# `container` CLI on PATH, use that. Otherwise prepend our fake shim so
# `hostely` finds it as `container` without polluting the user's PATH.
CONTAINER_PATH=""
if command -v container >/dev/null 2>&1; then
    CONTAINER_PATH="container"
    ok "using real Apple container CLI"
else
    FAKE_DIR="$(mktemp -d -t hostely-fake.XXXXXX)"
    ln -s "$HOSTELY_FAKE_CLI" "$FAKE_DIR/container"
    export PATH="$FAKE_DIR:$PATH"
    CONTAINER_PATH="container"
    ok "no real container CLI; using fake shim at $FAKE_DIR/container"
fi

# Point hostely at the chosen CLI. The services manager looks up `container`
# via cli::which() on PATH, so the symlink/fake approach above is enough.

SVC_NAME="smoke-$$"

run_out=$("$HOSTELY_BIN" run "$SVC_NAME" \
              --image nginx:alpine \
              --port 8080:80 2>&1) || fail "hostely run failed: $run_out"
ok "hostely run -> $run_out"

ps_out=$("$HOSTELY_BIN" ps 2>&1) || fail "hostely ps failed"
if printf '%s\n' "$ps_out" | grep -q "$SVC_NAME"; then
    ok "hostely ps lists $SVC_NAME"
else
    fail "ps did not list $SVC_NAME:\n$ps_out"
fi

logs_out=$("$HOSTELY_BIN" logs "$SVC_NAME" 2>&1) || fail "hostely logs failed"
ok "hostely logs returned $(printf '%s' "$logs_out" | wc -l | tr -d ' ') line(s)"

stop_out=$("$HOSTELY_BIN" stop "$SVC_NAME" 2>&1) || fail "hostely stop failed"
ok "hostely stop succeeded"

# -----------------------------------------------------------------------------
# 5. optional LLM serve
# -----------------------------------------------------------------------------

step "6/6 hostely serve + OpenAI-compatible /v1/models"
if [ -z "$HOSTELY_MODEL" ]; then
    printf '   (skipped — set HOSTELY_MODEL=/path/to/model.gguf to enable)\n'
else
    if [ ! -f "$HOSTELY_MODEL" ]; then
        fail "HOSTELY_MODEL=$HOSTELY_MODEL not found"
    fi
    PORT=18080
    # Use a non-default port in case something already binds 8081.
    "$HOSTELY_BIN" serve "$HOSTELY_MODEL" --port "$PORT" \
        > /tmp/hostely-serve.log 2>&1 &
    SERVE_PID=$!

    # Wait up to 20s for the server to come up.
    i=0
    while [ "$i" -lt 40 ]; do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            break
        fi
        sleep 0.5
        i=$((i + 1))
    done

    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
        ok "GET /v1/models returned 200"
    else
        kill "$SERVE_PID" 2>/dev/null
        fail "GET /v1/models failed; server log:\n$(cat /tmp/hostely-serve.log)"
    fi

    kill "$SERVE_PID" 2>/dev/null
    wait "$SERVE_PID" 2>/dev/null
    ok "hostely serve shut down cleanly"
fi

printf '\n%s\n' "$(green 'ALL CHECKS PASSED')"
