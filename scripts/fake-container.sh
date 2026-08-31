#!/bin/sh
# fake-container.sh — a stand-in for Apple's `container` CLI used by the
# hostely smoke test. Lets `hostely run/ps/stop/logs` work without the real
# Apple runtime (which requires macOS 26 and admin install).
#
# State lives in $HOSTELY_FAKE_STATE (default /tmp/hostely-fake) as a single
# tab-separated file: <id>\t<name>\t<image>\t<state>\t<ports>
#
# The CLI only implements the subset hostely actually invokes:
#   system start       exit 0
#   run -d --name N [-p host:cont] [-e K=V] [-v src:dst] IMAGE [args...]
#   ls                 print table with header
#   stop NAME          exit 0 (idempotent)
#   logs [-f] NAME     print captured stdout (we don't really capture)
# Anything else exits 2.

set -eu

STATE_DIR="${HOSTELY_FAKE_STATE:-/tmp/hostely-fake}"
STATE_FILE="$STATE_DIR/services.tsv"
LOG_DIR="$STATE_DIR/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"

cmd="${1:-}"
shift 2>/dev/null || true

gen_id() {
    # 12-char hex, simple.
    od -An -tx1 -N6 /dev/urandom | tr -d ' \n'
}

case "$cmd" in
    system)
        sub="${1:-}"
        case "$sub" in
            start|status) exit 0 ;;
            *) echo "fake-container: unknown system subcommand: $sub" >&2; exit 2 ;;
        esac
        ;;

    run)
        # Parse a subset of the real CLI's flags. We only need to consume
        # the ones hostely passes (see src/services/manager.cpp).
        name=""
        ports=""
        image=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -d) shift ;;
                --name) name="$2"; shift 2 ;;
                -p)     ports="$ports $2"; shift 2 ;;
                -e)     shift 2 ;;        # env: drop on the floor
                -v)     shift 2 ;;        # volumes: drop on the floor
                --*)    shift ;;          # any other long flag we don't care about
                -*)     shift ;;          # ditto for short
                *)
                    if [ -z "$image" ]; then image="$1"; fi
                    shift
                    ;;
            esac
        done

        if [ -z "$name" ] || [ -z "$image" ]; then
            echo "fake-container: --name and IMAGE are required" >&2
            exit 2
        fi

        id=$(gen_id)
        printf '%s\t%s\t%s\trunning\t%s\n' "$id" "$name" "$image" \
            "${ports# }" >> "$STATE_FILE"

        # Seed a tiny log file so `hostely logs` has something to show.
        printf 'fake-container: service %s (%s) started\n' "$name" "$image" \
            > "$LOG_DIR/$name.log"

        printf '%s\n' "$id"
        exit 0
        ;;

    ls)
        # Print a table matching the columns parse_ls() expects.
        #   ID   IMAGE   OS   ARCH   STATE    ADDR          PORTS   NAMES
        printf 'ID\tIMAGE\tOS\tARCH\tSTATE\tADDR\tPORTS\tNAMES\n'
        printf -- '------------------------------------\n'
        if [ -s "$STATE_FILE" ]; then
            awk -F'\t' '{
                printf "%s\t%s\tlinux\tarm64\t%s\t-\t%s\t%s\n",
                    $1, $3, $4, $5, $2
            }' "$STATE_FILE"
        fi
        exit 0
        ;;

    stop)
        name="${1:-}"
        if [ -z "$name" ]; then
            echo "fake-container: stop requires NAME" >&2; exit 2
        fi
        if [ -s "$STATE_FILE" ]; then
            # Drop any rows for this name. awk match is exact on $2.
            awk -F'\t' -v n="$name" '$2 != n' "$STATE_FILE" > "$STATE_FILE.tmp"
            mv "$STATE_FILE.tmp" "$STATE_FILE"
        fi
        exit 0
        ;;

    logs)
        follow=0
        if [ "${1:-}" = "-f" ]; then follow=1; shift; fi
        name="${1:-}"
        if [ -z "$name" ]; then
            echo "fake-container: logs requires NAME" >&2; exit 2
        fi
        if [ -s "$LOG_DIR/$name.log" ]; then
            cat "$LOG_DIR/$name.log"
        else
            echo "fake-container: no logs for $name" >&2
            exit 1
        fi
        [ "$follow" = "1" ] && sleep 0.1   # don't actually hang
        exit 0
        ;;

    '')
        echo "fake-container: no subcommand given" >&2; exit 2 ;;
    *)
        echo "fake-container: unknown subcommand: $cmd" >&2; exit 2 ;;
esac
