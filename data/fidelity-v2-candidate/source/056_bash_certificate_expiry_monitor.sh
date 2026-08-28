#!/usr/bin/env bash
set -euo pipefail

# Monitor TLS certificate expiry for a list of HTTPS endpoints.
#
# Input file format:
#   hostname[:port] [label]
#
# Example:
#   api.example.net:443 production-api
#   portal.example.net customer-portal
#
# The script prints one TSV line per endpoint and exits nonzero if any
# certificate is expired, unreachable, or inside the critical threshold.

usage() {
    cat <<'EOF'
Usage: cert-expiry-monitor.sh [options] ENDPOINT_FILE

Options:
  --warning DAYS      Warning threshold, default 30
  --critical DAYS     Critical threshold, default 14
  --timeout SECONDS   openssl connection timeout, default 8
  --json              Emit JSON instead of TSV
EOF
}

warning_days=30
critical_days=14
timeout_seconds=8
json_mode=0

while (($#)); do
    case "$1" in
        --warning)
            warning_days=$2
            shift 2
            ;;
        --critical)
            critical_days=$2
            shift 2
            ;;
        --timeout)
            timeout_seconds=$2
            shift 2
            ;;
        --json)
            json_mode=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "unknown option: $1" >&2
            exit 2
            ;;
        *)
            break
            ;;
    esac
done

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
input=$1

for value in "$warning_days" "$critical_days" "$timeout_seconds"; do
    [[ "$value" =~ ^[0-9]+$ ]] || {
        echo "thresholds and timeout must be nonnegative integers" >&2
        exit 2
    }
done

(( critical_days <= warning_days )) || {
    echo "--critical must be <= --warning" >&2
    exit 2
}

[[ -r "$input" ]] || {
    echo "cannot read endpoint file: $input" >&2
    exit 2
}

command -v openssl >/dev/null || {
    echo "openssl not found" >&2
    exit 2
}
command -v timeout >/dev/null || {
    echo "timeout command not found" >&2
    exit 2
}

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

declare -a json_items=()
overall=0

json_escape() {
    python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

parse_endpoint() {
    local token=$1
    if [[ "$token" == \[*\]:* ]]; then
        host="${token%%]*}"
        host="${host#[}"
        port="${token##*:}"
    elif [[ "$token" == *:* ]]; then
        host="${token%%:*}"
        port="${token##*:}"
    else
        host=$token
        port=443
    fi
}

status_for_days() {
    local days=$1
    if (( days < 0 )); then
        printf 'EXPIRED'
    elif (( days <= critical_days )); then
        printf 'CRITICAL'
    elif (( days <= warning_days )); then
        printf 'WARNING'
    else
        printf 'OK'
    fi
}

check_one() {
    local endpoint=$1
    local label=$2

    parse_endpoint "$endpoint"
    local cert="$tmpdir/cert.$RANDOM.pem"
    local stderr_file="$tmpdir/stderr.$RANDOM"

    if ! timeout "${timeout_seconds}s" \
        openssl s_client \
        -connect "${host}:${port}" \
        -servername "$host" \
        -showcerts \
        </dev/null 2>"$stderr_file" |
        awk '
            /-----BEGIN CERTIFICATE-----/ { capture=1 }
            capture { print }
            /-----END CERTIFICATE-----/ { exit }
        ' > "$cert"
    then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$endpoint" "$label" "UNREACHABLE" "-" "-" "-"
        return 2
    fi

    if ! grep -q 'BEGIN CERTIFICATE' "$cert"; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$endpoint" "$label" "NO_CERT" "-" "-" "-"
        return 2
    fi

    local subject issuer not_before not_after end_epoch now_epoch remaining seconds days
    subject=$(openssl x509 -in "$cert" -noout -subject | sed 's/^subject=//')
    issuer=$(openssl x509 -in "$cert" -noout -issuer | sed 's/^issuer=//')
    not_before=$(openssl x509 -in "$cert" -noout -startdate | cut -d= -f2-)
    not_after=$(openssl x509 -in "$cert" -noout -enddate | cut -d= -f2-)

    if ! end_epoch=$(date -u -d "$not_after" +%s 2>/dev/null); then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$endpoint" "$label" "BAD_DATE" "-" "$not_after" "$subject"
        return 2
    fi

    now_epoch=$(date -u +%s)
    seconds=$((end_epoch - now_epoch))
    if (( seconds >= 0 )); then
        days=$((seconds / 86400))
    else
        days=$((- ((-seconds + 86399) / 86400) ))
    fi

    local status
    status=$(status_for_days "$days")

    printf '%s\t%s\t%s\t%d\t%s\t%s\n' \
        "$endpoint" "$label" "$status" "$days" "$not_after" "$subject"

    case "$status" in
        OK) return 0 ;;
        WARNING) return 1 ;;
        CRITICAL|EXPIRED) return 2 ;;
        *) return 2 ;;
    esac
}

if (( json_mode == 0 )); then
    printf 'endpoint\tlabel\tstatus\tdays_remaining\tnot_after\tsubject\n'
fi

while IFS= read -r raw || [[ -n "$raw" ]]; do
    line=${raw%%#*}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    [[ -n "$line" ]] || continue

    endpoint=${line%%[[:space:]]*}
    if [[ "$line" == "$endpoint" ]]; then
        label="$endpoint"
    else
        label=${line#"$endpoint"}
        label=${label#"${label%%[![:space:]]*}"}
    fi

    result_file="$tmpdir/result.$RANDOM"
    rc=0
    check_one "$endpoint" "$label" > "$result_file" || rc=$?

    IFS=$'\t' read -r out_endpoint out_label status days not_after subject < "$result_file"

    if (( json_mode )); then
        item=$(python3 - "$out_endpoint" "$out_label" "$status" "$days" "$not_after" "$subject" <<'PY'
import json,sys
endpoint,label,status,days,not_after,subject=sys.argv[1:]
payload = {
    "endpoint": endpoint,
    "label": label,
    "status": status,
    "days_remaining": None if days == "-" else int(days),
    "not_after": None if not_after == "-" else not_after,
    "subject": None if subject == "-" else subject,
}
print(json.dumps(payload, separators=(",", ":")))
PY
)
        json_items+=("$item")
    else
        cat "$result_file"
    fi

    if (( rc > overall )); then
        overall=$rc
    fi
done < "$input"

if (( json_mode )); then
    printf '[\n'
    for ((i=0; i<${#json_items[@]}; i++)); do
        printf '  %s' "${json_items[$i]}"
        (( i + 1 < ${#json_items[@]} )) && printf ','
        printf '\n'
    done
    printf ']\n'
fi

exit "$overall"
