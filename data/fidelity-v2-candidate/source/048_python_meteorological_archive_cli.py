#!/usr/bin/env python3
"""Validate and summarize a directory of meteorological station archives.

Archive layout:
    station_id/
        metadata.json
        observations-YYYY-MM.csv
        checksums.sha256

The CLI verifies required files, timestamp ordering, plausible ranges, checksum
records, and emits a JSON report suitable for unattended batch processing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

REQUIRED_COLUMNS = (
    "timestamp_utc",
    "air_temp_c",
    "relative_humidity_pct",
    "station_pressure_hpa",
    "wind_speed_mps",
    "wind_direction_deg",
    "precip_mm",
)

RANGES = {
    "air_temp_c": (-90.0, 60.0),
    "relative_humidity_pct": (0.0, 100.0),
    "station_pressure_hpa": (450.0, 1100.0),
    "wind_speed_mps": (0.0, 120.0),
    "wind_direction_deg": (0.0, 360.0),
    "precip_mm": (0.0, 1000.0),
}


@dataclass
class Issue:
    severity: str
    code: str
    path: str
    detail: str


@dataclass
class StationReport:
    station_id: str
    files_checked: int
    observations: int
    first_timestamp: str | None
    last_timestamp: str | None
    issues: list[Issue]

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: malformed checksum line")
        digest, name = parts
        name = name.lstrip("*")
        if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise ValueError(f"{path}:{line_no}: invalid SHA-256 digest")
        result[name] = digest.lower()
    return result


def parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return dt.astimezone(timezone.utc)


def parse_float(value: str, field: str) -> float:
    if value == "":
        raise ValueError(f"{field} is blank")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite")
    return number


def csv_rows(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing columns: {', '.join(missing)}")
        for line_no, row in enumerate(reader, 2):
            yield line_no, row


def check_observation_file(path: Path, issues: list[Issue]) -> tuple[int, datetime | None, datetime | None]:
    count = 0
    first: datetime | None = None
    last: datetime | None = None
    previous: datetime | None = None

    try:
        iterator = csv_rows(path)
        for line_no, row in iterator:
            count += 1
            try:
                ts = parse_timestamp(row["timestamp_utc"])
            except Exception as exc:
                issues.append(Issue("error", "bad_timestamp", str(path), f"line {line_no}: {exc}"))
                continue

            if previous is not None and ts <= previous:
                issues.append(
                    Issue(
                        "error",
                        "non_monotonic_time",
                        str(path),
                        f"line {line_no}: {ts.isoformat()} <= {previous.isoformat()}",
                    )
                )
            previous = ts
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)

            for field, (lo, hi) in RANGES.items():
                try:
                    value = parse_float(row[field], field)
                except Exception as exc:
                    issues.append(Issue("error", "bad_value", str(path), f"line {line_no}: {exc}"))
                    continue
                if not lo <= value <= hi:
                    issues.append(
                        Issue(
                            "warning",
                            "range",
                            str(path),
                            f"line {line_no}: {field}={value} outside [{lo}, {hi}]",
                        )
                    )
    except Exception as exc:
        issues.append(Issue("error", "csv_structure", str(path), str(exc)))

    return count, first, last


def validate_metadata(path: Path, station_dir_name: str, issues: list[Issue]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        issues.append(Issue("error", "metadata_json", str(path), str(exc)))
        return

    required = ("station_id", "name", "latitude", "longitude", "elevation_m")
    missing = [k for k in required if k not in data]
    if missing:
        issues.append(Issue("error", "metadata_missing", str(path), ", ".join(missing)))
        return

    if str(data["station_id"]) != station_dir_name:
        issues.append(
            Issue(
                "error",
                "station_id_mismatch",
                str(path),
                f"metadata station_id={data['station_id']!r}, directory={station_dir_name!r}",
            )
        )

    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        elev = float(data["elevation_m"])
        if not (-90 <= lat <= 90):
            raise ValueError("latitude out of range")
        if not (-180 <= lon <= 180):
            raise ValueError("longitude out of range")
        if not (-500 <= elev <= 9000):
            raise ValueError("elevation out of range")
    except Exception as exc:
        issues.append(Issue("error", "metadata_coordinates", str(path), str(exc)))


def validate_station(station_dir: Path, verify_checksums: bool) -> StationReport:
    issues: list[Issue] = []
    station_id = station_dir.name
    metadata = station_dir / "metadata.json"

    if metadata.exists():
        validate_metadata(metadata, station_id, issues)
    else:
        issues.append(Issue("error", "missing_metadata", str(metadata), "required file not found"))

    observation_files = sorted(station_dir.glob("observations-????-??.csv"))
    if not observation_files:
        issues.append(Issue("error", "missing_observations", str(station_dir), "no monthly CSV files found"))

    total = 0
    first: datetime | None = None
    last: datetime | None = None

    for path in observation_files:
        count, file_first, file_last = check_observation_file(path, issues)
        total += count
        if file_first is not None:
            first = file_first if first is None else min(first, file_first)
        if file_last is not None:
            last = file_last if last is None else max(last, file_last)

    if verify_checksums:
        manifest_path = station_dir / "checksums.sha256"
        if not manifest_path.exists():
            issues.append(Issue("error", "missing_checksum_manifest", str(manifest_path), "not found"))
        else:
            try:
                manifest = parse_checksum_manifest(manifest_path)
                candidates = [metadata, *observation_files]
                for path in candidates:
                    if not path.exists():
                        continue
                    expected = manifest.get(path.name)
                    if expected is None:
                        issues.append(Issue("warning", "checksum_missing_entry", str(path), "not in manifest"))
                        continue
                    actual = sha256_file(path)
                    if actual != expected:
                        issues.append(
                            Issue("error", "checksum_mismatch", str(path), f"expected {expected}, got {actual}")
                        )
            except Exception as exc:
                issues.append(Issue("error", "checksum_manifest", str(manifest_path), str(exc)))

    return StationReport(
        station_id=station_id,
        files_checked=len(observation_files) + (1 if metadata.exists() else 0),
        observations=total,
        first_timestamp=first.isoformat() if first else None,
        last_timestamp=last.isoformat() if last else None,
        issues=issues,
    )


def find_station_dirs(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def render_human(reports: list[StationReport]) -> None:
    for report in reports:
        status = "OK" if report.ok else "FAIL"
        print(
            f"{report.station_id:20} {status:4} "
            f"rows={report.observations:8d} issues={len(report.issues):3d}"
        )
        for issue in report.issues:
            print(f"  {issue.severity.upper():7} {issue.code:24} {issue.detail}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate meteorological station archives")
    p.add_argument("root", type=Path, help="archive root containing one directory per station")
    p.add_argument("--no-checksums", action="store_true", help="skip checksums.sha256 verification")
    p.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p.add_argument("--fail-on-warning", action="store_true", help="return nonzero if warnings exist")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"error: {args.root} is not a directory", file=sys.stderr)
        return 2

    reports = [
        validate_station(path, verify_checksums=not args.no_checksums)
        for path in find_station_dirs(args.root)
    ]

    if args.json:
        payload = []
        for report in reports:
            item = asdict(report)
            item["ok"] = report.ok
            payload.append(item)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        render_human(reports)

    errors = any(not r.ok for r in reports)
    warnings = any(i.severity == "warning" for r in reports for i in r.issues)
    if errors or (args.fail_on_warning and warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
