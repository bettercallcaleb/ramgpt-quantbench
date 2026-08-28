#!/usr/bin/env python3
"""Streaming geospatial CSV transformer.

Reads records containing latitude/longitude, validates coordinates, optionally
filters to a bounding box, computes a Web Mercator projection, derives a
geohash-like grid key, and writes normalized CSV without loading the entire
input into memory.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

EARTH_RADIUS_M = 6378137.0
MAX_MERCATOR_LAT = 85.05112878


@dataclass(frozen=True)
class Bounds:
    west: float
    south: float
    east: float
    north: float

    def contains(self, lon: float, lat: float) -> bool:
        if not (self.south <= lat <= self.north):
            return False
        if self.west <= self.east:
            return self.west <= lon <= self.east
        # Dateline-crossing bounding box.
        return lon >= self.west or lon <= self.east


@dataclass
class Counters:
    input_rows: int = 0
    output_rows: int = 0
    invalid_rows: int = 0
    filtered_rows: int = 0


def parse_bounds(text: str) -> Bounds:
    try:
        west, south, east, north = map(float, text.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "bounds must be west,south,east,north"
        ) from exc

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise argparse.ArgumentTypeError("longitude bounds must be within [-180, 180]")
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise argparse.ArgumentTypeError("latitude bounds must be within [-90, 90]")
    if south > north:
        raise argparse.ArgumentTypeError("south must not exceed north")
    return Bounds(west, south, east, north)


def parse_coordinate(value: str, name: str) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"{name} is empty")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite")
    return number


def web_mercator(lon: float, lat: float) -> tuple[float, float]:
    clamped_lat = max(-MAX_MERCATOR_LAT, min(MAX_MERCATOR_LAT, lat))
    x = EARTH_RADIUS_M * math.radians(lon)
    y = EARTH_RADIUS_M * math.log(
        math.tan(math.pi / 4 + math.radians(clamped_lat) / 2)
    )
    return x, y


def tile_key(lon: float, lat: float, precision: int) -> str:
    """Return a deterministic quadtree-style key.

    This is intentionally not a standards-compliant geohash. The key records
    repeated longitude/latitude bisection and is useful for partitioning output.
    """
    if not (1 <= precision <= 24):
        raise ValueError("precision must be between 1 and 24")

    west, east = -180.0, 180.0
    south, north = -90.0, 90.0
    digits: list[str] = []

    for _ in range(precision):
        mid_lon = (west + east) / 2
        mid_lat = (south + north) / 2
        east_half = lon >= mid_lon
        north_half = lat >= mid_lat

        quadrant = (2 if north_half else 0) + (1 if east_half else 0)
        digits.append(str(quadrant))

        if east_half:
            west = mid_lon
        else:
            east = mid_lon

        if north_half:
            south = mid_lat
        else:
            north = mid_lat

    return "".join(digits)


def normalize_row(
    row: dict[str, str],
    lat_col: str,
    lon_col: str,
    precision: int,
) -> dict[str, str]:
    lat = parse_coordinate(row.get(lat_col, ""), lat_col)
    lon = parse_coordinate(row.get(lon_col, ""), lon_col)

    if not -90 <= lat <= 90:
        raise ValueError(f"latitude out of range: {lat}")
    if not -180 <= lon <= 180:
        raise ValueError(f"longitude out of range: {lon}")

    x, y = web_mercator(lon, lat)
    result = dict(row)
    result[lat_col] = f"{lat:.8f}"
    result[lon_col] = f"{lon:.8f}"
    result["mercator_x_m"] = f"{x:.3f}"
    result["mercator_y_m"] = f"{y:.3f}"
    result["spatial_key"] = tile_key(lon, lat, precision)
    return result


def transform(
    source: TextIO,
    target: TextIO,
    *,
    lat_col: str,
    lon_col: str,
    bounds: Bounds | None,
    precision: int,
    reject_writer: csv.DictWriter | None = None,
) -> Counters:
    reader = csv.DictReader(source)
    if reader.fieldnames is None:
        raise ValueError("input has no header")

    missing = [name for name in (lat_col, lon_col) if name not in reader.fieldnames]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    output_fields = list(reader.fieldnames)
    for extra in ("mercator_x_m", "mercator_y_m", "spatial_key"):
        if extra not in output_fields:
            output_fields.append(extra)

    writer = csv.DictWriter(target, fieldnames=output_fields, lineterminator="\n")
    writer.writeheader()

    counters = Counters()

    for line_no, row in enumerate(reader, 2):
        counters.input_rows += 1

        try:
            lat = parse_coordinate(row.get(lat_col, ""), lat_col)
            lon = parse_coordinate(row.get(lon_col, ""), lon_col)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("coordinate outside valid geographic range")
        except Exception as exc:
            counters.invalid_rows += 1
            if reject_writer is not None:
                reject_writer.writerow({
                    "line_number": line_no,
                    "reason": str(exc),
                    "raw_record": repr(row),
                })
            continue

        if bounds is not None and not bounds.contains(lon, lat):
            counters.filtered_rows += 1
            continue

        normalized = normalize_row(row, lat_col, lon_col, precision)
        writer.writerow(normalized)
        counters.output_rows += 1

    return counters


def open_text(path: str, mode: str) -> TextIO:
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    return open(path, mode, encoding="utf-8", newline="")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stream-transform geospatial CSV records")
    p.add_argument("input", help="input CSV path or - for stdin")
    p.add_argument("output", help="output CSV path or - for stdout")
    p.add_argument("--lat-column", default="latitude")
    p.add_argument("--lon-column", default="longitude")
    p.add_argument("--bounds", type=parse_bounds, help="west,south,east,north")
    p.add_argument("--precision", type=int, default=10)
    p.add_argument("--rejects", help="write invalid rows to a separate CSV")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input == "-" and args.output == "-":
        # This is valid, but diagnostics must stay on stderr.
        pass
    if not 1 <= args.precision <= 24:
        print("error: --precision must be between 1 and 24", file=sys.stderr)
        return 2

    source = open_text(args.input, "r")
    target = open_text(args.output, "w")
    reject_fh: TextIO | None = None
    reject_writer: csv.DictWriter | None = None

    try:
        if args.rejects:
            reject_fh = open(args.rejects, "w", encoding="utf-8", newline="")
            reject_writer = csv.DictWriter(
                reject_fh,
                fieldnames=["line_number", "reason", "raw_record"],
                lineterminator="\n",
            )
            reject_writer.writeheader()

        counters = transform(
            source,
            target,
            lat_col=args.lat_column,
            lon_col=args.lon_column,
            bounds=args.bounds,
            precision=args.precision,
            reject_writer=reject_writer,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if source is not sys.stdin:
            source.close()
        if target is not sys.stdout:
            target.close()
        if reject_fh is not None:
            reject_fh.close()

    print(
        f"input={counters.input_rows} "
        f"output={counters.output_rows} "
        f"invalid={counters.invalid_rows} "
        f"filtered={counters.filtered_rows}",
        file=sys.stderr,
    )

    return 1 if counters.invalid_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
