#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

PUBLIC = Path("artifact-bench/public")

RELEASE = PUBLIC / "qwen3-4b-q4-k-m-open-artifact-rank-v1"

ENTRY = RELEASE / "leaderboard-entry.json"
MANIFEST = RELEASE / "release-manifest.json"

EXPECTED_ENTRY_SHA = (
    "c5873da79bbe7bd1785703c06181a1df"
    "c093024cbbf7ab4d1e28e3d2ca5048d5"
)

EXPECTED_MANIFEST_SHA = (
    "12a0ab022e8c781ead79f5ef7817fa8a"
    "0a910a48c472183d7c85afce02f5644b"
)


def sha(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)

            if not b:
                break

            h.update(b)

    return h.hexdigest()


print("============================================================")
print("1. RELEASE IDENTITY")
print("============================================================")

entry_sha = sha(ENTRY)
manifest_sha = sha(MANIFEST)

print("entry_sha256:", entry_sha)
print("release_manifest_sha256:", manifest_sha)

if entry_sha != EXPECTED_ENTRY_SHA:
    raise RuntimeError(
        "leaderboard-entry.json no longer matches frozen release"
    )

if manifest_sha != EXPECTED_MANIFEST_SHA:
    raise RuntimeError(
        "release-manifest.json no longer matches frozen release"
    )

print("RELEASE_IDENTITY: PASS")


entry = json.loads(ENTRY.read_text())
manifest = json.loads(MANIFEST.read_text())


print()
print("============================================================")
print("2. RELEASE SEMANTICS")
print("============================================================")

checks = {
    "entry_schema":
        entry["schema"]
        == "ramgpt-quantbench-public-leaderboard-entry",

    "entry_schema_version":
        entry["schema_version"] == 1,

    "entry_status":
        entry["status"] == "VERIFIED",

    "badge":
        entry["badge"] == "RAMGPT_QUANTBENCH_VERIFIED",

    "comparison_class":
        entry["comparison_class"]
        == "OPEN_ARTIFACT_DOWNLOAD_RANK",

    "model":
        entry["model"] == "Qwen/Qwen3-4B",

    "quant":
        entry["quantization_label"] == "Q4_K_M",

    "winner":
        entry["winner"] == "Unsloth",

    "strict_recipe_ineligible":
        entry["strict_recipe_rank"]["eligible"] is False,

    "recommended_not_awarded":
        entry["ramgpt_recommended"]["awarded"] is False,

    "release_manifest_status":
        manifest["status"] == "PASS",

    "release_manifest_binds_entry":
        manifest["files"]["leaderboard-entry.json"]
        == entry_sha,
}

for k, v in checks.items():
    print(f"{k}: {'PASS' if v else 'FAIL'}")

if not all(checks.values()):
    raise RuntimeError("release semantic validation failed")

print("RELEASE_SEMANTICS: PASS")


print()
print("============================================================")
print("3. BUILD GLOBAL INDEX")
print("============================================================")

rankings = []

for artifact in entry["artifacts"]:
    rankings.append({
        "rank":
            artifact["rank"],

        "publisher":
            artifact["publisher"],

        "artifact_sha256":
            artifact["artifact_sha256"],

        "mean_kl_bf16_to_artifact":
            artifact["mean_kl_bf16_to_artifact"],

        "changed_top1_count":
            artifact["changed_top1_count"],
    })


index = {
    "schema":
        "ramgpt-quantbench-public-leaderboard-index",

    "schema_version":
        1,

    "status":
        "PASS",

    "ranking_definition":
        {
            "unit":
                "artifact",

            "primary_metric":
                "mean KL(BF16 || artifact)",

            "primary_metric_direction":
                "lower_is_better",

            "weighted_composite":
                None,

            "better_definition":
                "Lower behavioral divergence from the pinned "
                "native BF16 reference under the applicable "
                "versioned RAMGPT QuantBench contract.",
        },

    "entry_count":
        1,

    "entries": [
        {
            "benchmark_id":
                entry["benchmark_id"],

            "model":
                entry["model"],

            "quantization_label":
                entry["quantization_label"],

            "comparison_class":
                entry["comparison_class"],

            "status":
                entry["status"],

            "badge":
                entry["badge"],

            "winner":
                entry["winner"],

            "rankings":
                rankings,

            "primary_metric":
                entry["primary_metric"],

            "paired_uncertainty":
                entry["paired_uncertainty"],

            "strict_recipe_rank":
                entry["strict_recipe_rank"],

            "ramgpt_recommended":
                entry["ramgpt_recommended"],

            "release": {
                "directory":
                    str(RELEASE),

                "leaderboard_entry_sha256":
                    entry_sha,

                "release_manifest_sha256":
                    manifest_sha,
            },
        }
    ],
}


OUT = PUBLIC / "leaderboard-index-v1.json"

OUT.write_text(
    json.dumps(
        index,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
)


# Read it back from disk. Do not validate only the in-memory object.
frozen = json.loads(OUT.read_text())


print()
print("============================================================")
print("4. INDEX VALIDATION")
print("============================================================")

index_checks = {
    "schema":
        frozen["schema"]
        == "ramgpt-quantbench-public-leaderboard-index",

    "schema_version":
        frozen["schema_version"] == 1,

    "status":
        frozen["status"] == "PASS",

    "entry_count":
        frozen["entry_count"] == 1,

    "actual_entries":
        len(frozen["entries"]) == 1,

    "model":
        frozen["entries"][0]["model"]
        == "Qwen/Qwen3-4B",

    "quant":
        frozen["entries"][0]["quantization_label"]
        == "Q4_K_M",

    "winner":
        frozen["entries"][0]["winner"]
        == "Unsloth",

    "rank_1":
        frozen["entries"][0]["rankings"][0]["publisher"]
        == "Unsloth",

    "rank_2":
        frozen["entries"][0]["rankings"][1]["publisher"]
        == "ggml-org",

    "entry_hash_binding":
        frozen["entries"][0]["release"][
            "leaderboard_entry_sha256"
        ] == EXPECTED_ENTRY_SHA,

    "manifest_hash_binding":
        frozen["entries"][0]["release"][
            "release_manifest_sha256"
        ] == EXPECTED_MANIFEST_SHA,
}

for k, v in index_checks.items():
    print(f"{k}: {'PASS' if v else 'FAIL'}")

if not all(index_checks.values()):
    raise RuntimeError("global index validation failed")


index_sha = sha(OUT)

print()
print("============================================================")
print("GLOBAL LEADERBOARD REGISTRY")
print("============================================================")

print("INDEX_SCHEMA_VERSION: 1")
print("ENTRY_COUNT:", frozen["entry_count"])

print(
    "FIRST_BENCHMARK:",
    frozen["entries"][0]["benchmark_id"],
)

print(
    "FIRST_WINNER:",
    frozen["entries"][0]["winner"],
)

print("LEADERBOARD_INDEX_SHA256:", index_sha)

print()
print("PUBLIC_LEADERBOARD_INDEX_V1: PASS")
