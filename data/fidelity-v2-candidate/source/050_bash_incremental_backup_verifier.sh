#!/usr/bin/env bash
set -euo pipefail

# Verify a directory of timestamped incremental backups.
#
# Expected layout:
#   backups/
#     2026-08-01T020000Z/
#       manifest.tsv
#       root/
#     2026-08-02T020000Z/
#       manifest.tsv
#       root/
#
# manifest.tsv columns:
# relative_path<TAB>size_bytes<TAB>sha256
#
# The script checks manifest syntax, path safety, file sizes, checksums,
# duplicate snapshot identifiers, and continuity between adjacent snapshots.

usage() {
    cat <<'EOF'
Usage:
  backup-verify.sh [--sample N] [--strict] BACKUP_ROOT

Options:
  --sample N    Verify at most N content hashes per snapshot; 0 means all.
  --strict      Treat continuity warnings as failures.
EOF
}

sample_limit=0
strict=0

while (($#)); do
    case "$1" in
        --sample)
            [[ $# -ge 2 ]] || { echo "missing argument for --sample" >&2; exit 2; }
            sample_limit=$2
            shift 2
            ;;
        --strict)
            strict=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
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
backup_root=$1

[[ -d "$backup_root" ]] || { echo "not a directory: $backup_root" >&2; exit 2; }
[[ "$sample_limit" =~ ^[0-9]+$ ]] || { echo "--sample must be nonnegative integer" >&2; exit 2; }

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

errors=0
warnings=0

say_error() {
    printf 'ERROR: %s\n' "$*" >&2
    errors=$((errors + 1))
}

say_warn() {
    printf 'WARN : %s\n' "$*" >&2
    warnings=$((warnings + 1))
}

safe_relpath() {
    local p=$1
    [[ -n "$p" ]] || return 1
    [[ "$p" != /* ]] || return 1
    [[ "$p" != *$'\n'* ]] || return 1

    local part
    IFS='/' read -r -a pieces <<< "$p"
    for part in "${pieces[@]}"; do
        [[ -n "$part" ]] || return 1
        [[ "$part" != "." && "$part" != ".." ]] || return 1
    done
}

validate_snapshot_name() {
    [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{6}Z$ ]]
}

load_manifest() {
    local snapshot=$1
    local manifest="$snapshot/manifest.tsv"
    local normalized=$2

    : > "$normalized"

    [[ -f "$manifest" ]] || {
        say_error "$snapshot: missing manifest.tsv"
        return 1
    }

    local line_no=0
    local rel size digest extra
    while IFS=$'\t' read -r rel size digest extra || [[ -n "${rel:-}${size:-}${digest:-}${extra:-}" ]]; do
        line_no=$((line_no + 1))
        [[ -n "${rel:-}" ]] || continue
        [[ "${rel:0:1}" != "#" ]] || continue

        if [[ -n "${extra:-}" ]]; then
            say_error "$manifest:$line_no: expected exactly 3 tab-separated columns"
            continue
        fi

        if ! safe_relpath "$rel"; then
            say_error "$manifest:$line_no: unsafe relative path: $rel"
            continue
        fi

        if [[ ! "$size" =~ ^[0-9]+$ ]]; then
            say_error "$manifest:$line_no: invalid size: $size"
            continue
        fi

        if [[ ! "$digest" =~ ^[0-9a-fA-F]{64}$ ]]; then
            say_error "$manifest:$line_no: invalid sha256: $digest"
            continue
        fi

        printf '%s\t%s\t%s\n' "$rel" "$size" "${digest,,}" >> "$normalized"
    done < "$manifest"

    LC_ALL=C sort -t $'\t' -k1,1 "$normalized" -o "$normalized"

    local dupes
    dupes=$(cut -f1 "$normalized" | uniq -d)
    if [[ -n "$dupes" ]]; then
        while IFS= read -r p; do
            say_error "$manifest: duplicate path: $p"
        done <<< "$dupes"
    fi
}

verify_content() {
    local snapshot=$1
    local normalized=$2
    local root="$snapshot/root"
    local checked=0

    [[ -d "$root" ]] || {
        say_error "$snapshot: missing root/"
        return
    }

    while IFS=$'\t' read -r rel expected_size expected_hash; do
        local path="$root/$rel"
        if [[ ! -f "$path" ]]; then
            say_error "$snapshot: missing file: $rel"
            continue
        fi

        local actual_size
        actual_size=$(stat -c '%s' -- "$path")
        if [[ "$actual_size" != "$expected_size" ]]; then
            say_error "$snapshot: size mismatch for $rel: expected=$expected_size actual=$actual_size"
            continue
        fi

        if (( sample_limit == 0 || checked < sample_limit )); then
            local actual_hash
            actual_hash=$(sha256sum -- "$path" | awk '{print $1}')
            if [[ "$actual_hash" != "$expected_hash" ]]; then
                say_error "$snapshot: checksum mismatch for $rel"
            fi
            checked=$((checked + 1))
        fi
    done < "$normalized"

    # Detect files present on disk but absent from manifest.
    local disk_list="$tmpdir/disk.$RANDOM"
    local manifest_list="$tmpdir/manifest.$RANDOM"
    (
        cd "$root"
        find . -type f -print0 |
            while IFS= read -r -d '' p; do
                printf '%s\n' "${p#./}"
            done |
            LC_ALL=C sort
    ) > "$disk_list"

    cut -f1 "$normalized" > "$manifest_list"
    while IFS= read -r extra; do
        [[ -n "$extra" ]] && say_warn "$snapshot: untracked file: $extra"
    done < <(comm -23 "$disk_list" "$manifest_list")
}

compare_snapshots() {
    local previous_name=$1
    local previous_manifest=$2
    local current_name=$3
    local current_manifest=$4

    local prev_paths="$tmpdir/prev.paths"
    local cur_paths="$tmpdir/cur.paths"
    cut -f1 "$previous_manifest" > "$prev_paths"
    cut -f1 "$current_manifest" > "$cur_paths"

    local added removed
    added=$(comm -13 "$prev_paths" "$cur_paths" | wc -l)
    removed=$(comm -23 "$prev_paths" "$cur_paths" | wc -l)

    local changed=0 unchanged=0
    while IFS=$'\t' read -r path cur_size cur_hash; do
        local prev_line
        prev_line=$(awk -F '\t' -v p="$path" '$1 == p { print; exit }' "$previous_manifest")
        [[ -n "$prev_line" ]] || continue
        local prev_path prev_size prev_hash
        IFS=$'\t' read -r prev_path prev_size prev_hash <<< "$prev_line"
        if [[ "$cur_size" == "$prev_size" && "$cur_hash" == "$prev_hash" ]]; then
            unchanged=$((unchanged + 1))
        else
            changed=$((changed + 1))
        fi
    done < "$current_manifest"

    printf 'CONTINUITY %s -> %s added=%d removed=%d changed=%d unchanged=%d\n' \
        "$previous_name" "$current_name" "$added" "$removed" "$changed" "$unchanged"

    local total
    total=$(wc -l < "$current_manifest")
    if (( total > 0 && removed > total / 2 )); then
        say_warn "$current_name: unusually large deletion relative to current snapshot"
    fi
}

mapfile -t snapshots < <(
    find "$backup_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' |
        LC_ALL=C sort
)

if ((${#snapshots[@]} == 0)); then
    echo "no snapshots found" >&2
    exit 2
fi

declare -A seen
previous_name=
previous_manifest=

for name in "${snapshots[@]}"; do
    if ! validate_snapshot_name "$name"; then
        say_warn "ignoring directory with non-snapshot name: $name"
        continue
    fi
    if [[ -n "${seen[$name]:-}" ]]; then
        say_error "duplicate snapshot identifier: $name"
        continue
    fi
    seen[$name]=1

    snapshot="$backup_root/$name"
    normalized="$tmpdir/$name.tsv"

    echo "SNAPSHOT $name"
    if load_manifest "$snapshot" "$normalized"; then
        verify_content "$snapshot" "$normalized"
        if [[ -n "$previous_name" ]]; then
            compare_snapshots "$previous_name" "$previous_manifest" "$name" "$normalized"
        fi
        previous_name=$name
        previous_manifest="$normalized"
    fi
done

printf 'SUMMARY errors=%d warnings=%d\n' "$errors" "$warnings"

if (( errors > 0 )); then
    exit 1
fi
if (( strict && warnings > 0 )); then
    exit 1
fi
exit 0
