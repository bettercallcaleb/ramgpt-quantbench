#!/usr/bin/env python3
"""Validate Artifact Bench V1 JSON Schemas and comparison-class definitions."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "schemas"
SCHEMA_FILES = (
    "artifact-record-v1.schema.json",
    "artifact-result-v1.schema.json",
    "pairwise-comparison-v1.schema.json",
    "leaderboard-v1.schema.json",
)


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    for name in SCHEMA_FILES:
        schema = load_json(SCHEMA_DIR / name)
        Draft202012Validator.check_schema(schema)
        print(f"SCHEMA_VALID: {name}")

    classes = load_json(ROOT / "protocol" / "comparison-classes-v1.json")
    assert classes["schema"] == "ramgpt-artifact-comparison-classes-v1"
    assert classes["protocol_version"] == "artifact-ranking-v1"
    assert len(classes["classes"]) == 1
    initial = classes["classes"][0]
    assert initial["comparison_class"] == "qwen3-8b-standard-q4-k-m-v1"
    assert initial["candidate_artifacts"] == []
    assert initial["requirements"]["imatrix_status"] == "NONE"
    assert initial["requirements"]["requantization_status"] == "NOT_REQUANTIZED"
    print("COMPARISON_CLASSES_VALID: comparison-classes-v1.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
