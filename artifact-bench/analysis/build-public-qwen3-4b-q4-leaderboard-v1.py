#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

ROOT = Path(".")

SRC = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "open-artifact-rank-v1"
)

OUT = Path(
    "artifact-bench/public/"
    "qwen3-4b-q4-k-m-open-artifact-rank-v1"
)

OUT.mkdir(parents=True, exist_ok=True)

VERIFIED = SRC / "verified-result.json"
CARD = SRC / "result-card.md"
FREEZE = SRC / "freeze-manifest.json"

EXPECTED = {
    VERIFIED:
        "6a4624c40f412087c5de23ef15c3479cb406cf68f8835cfd2ecf89c1c2bea4cd",

    CARD:
        "55fa093ec7804b50a743a84a6dcd0cdf14b82136bbdcc6a32846a3d5d8b1f03a",

    FREEZE:
        "34500999556dcc3037c93f49c95d74c4411d1f5e31de9045cb363b9e63244d9c",
}


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
print("1. FROZEN SOURCE CHECK")
print("============================================================")

for path, expected in EXPECTED.items():
    got = sha(path)

    print(path)
    print(" expected:", expected)
    print(" actual:  ", got)

    if got != expected:
        raise RuntimeError(
            f"frozen source changed: {path}"
        )

print("FROZEN_SOURCE_CHECK: PASS")


v = json.loads(VERIFIED.read_text())

required = {
    "status": "RAMGPT_QUANTBENCH_VERIFIED",
    "comparison_class": "OPEN_ARTIFACT_DOWNLOAD_RANK",
    "model": "Qwen/Qwen3-4B",
    "nominal_quantization": "Q4_K_M",
    "winner": "Unsloth",
}

for k, expected in required.items():
    got = v.get(k)

    if got != expected:
        raise RuntimeError(
            f"{k}: expected {expected!r}, got {got!r}"
        )

if v["strict_recipe_rank"]["eligible"] is not False:
    raise RuntimeError(
        "strict recipe rank unexpectedly eligible"
    )

if v["ramgpt_recommended"] is not False:
    raise RuntimeError(
        "RAMGPT_RECOMMENDED must not be awarded"
    )

print("VERIFIED_RESULT_SEMANTICS: PASS")


rank = v["rank"]

if len(rank) != 2:
    raise RuntimeError("expected exactly two ranked artifacts")

if rank[0]["rank"] != 1 or rank[0]["publisher"] != "Unsloth":
    raise RuntimeError("rank 1 mismatch")

if rank[1]["rank"] != 2 or rank[1]["publisher"] != "ggml-org":
    raise RuntimeError("rank 2 mismatch")


entry = {
    "schema":
        "ramgpt-quantbench-public-leaderboard-entry",

    "schema_version": 1,

    "benchmark_id":
        v["benchmark_id"],

    "status":
        "VERIFIED",

    "badge":
        "RAMGPT_QUANTBENCH_VERIFIED",

    "model":
        v["model"],

    "quantization_label":
        v["nominal_quantization"],

    "comparison_class":
        v["comparison_class"],

    "winner":
        v["winner"],

    "primary_metric": {
        "name":
            "mean KL(BF16 || artifact)",

        "direction":
            "lower_is_better",

        "ggml_org":
            v["ggml_org_mean_kl"],

        "unsloth":
            v["unsloth_mean_kl"],

        "unsloth_relative_reduction_percent":
            100.0
            * v[
                "relative_mean_kl_reduction_vs_ggml_org"
            ],
    },

    "paired_uncertainty": {
        "delta_definition":
            "mean_KL_ggml-org - mean_KL_Unsloth",

        "delta":
            v["delta_kl_ggml_minus_unsloth"],

        "ci_95":
            v["paired_segment_bootstrap_95_ci"],

        "resampling_unit":
            v["bootstrap"]["resampling_unit"],

        "segments":
            v["bootstrap"]["segment_count"],

        "replicates":
            v["bootstrap"]["replicates"],

        "seed":
            v["bootstrap"]["seed"],
    },

    "decision_stability": {
        "ggml_org_top1_flips":
            v["decision_stability"][
                "ggml_org_changed_top1_count"
            ],

        "unsloth_top1_flips":
            v["decision_stability"][
                "unsloth_changed_top1_count"
            ],

        "ggml_org_hcdfr":
            v["decision_stability"][
                "ggml_org_hcdfr"
            ],

        "unsloth_hcdfr":
            v["decision_stability"][
                "unsloth_hcdfr"
            ],

        "ggml_org_max_flipped_bf16_margin":
            v["decision_stability"][
                "ggml_org_max_flipped_bf16_margin"
            ],

        "unsloth_max_flipped_bf16_margin":
            v["decision_stability"][
                "unsloth_max_flipped_bf16_margin"
            ],
    },

    "artifacts": [
        {
            "rank": x["rank"],
            "publisher": x["publisher"],
            "artifact_sha256":
                x["artifact_sha256"],
            "mean_kl_bf16_to_artifact":
                x["mean_kl_bf16_to_artifact"],
            "changed_top1_count":
                x["changed_top1_count"],
        }
        for x in rank
    ],

    "strict_recipe_rank": {
        "eligible": False,
        "status": "INELIGIBLE",
        "reason":
            v["strict_recipe_rank"]["reason"],
    },

    "ramgpt_recommended": {
        "awarded": False,
        "status": "NOT_AWARDED",
    },

    "verification": {
        "canonical_analysis":
            "PASS",

        "independent_raw_rql_validation":
            v["independent_validation"]["status"],

        "bf16_full_reference_determinism":
            "PASS",

        "full_scored_positions":
            v["benchmark_contract"]["positions"],

        "vocab_size":
            v["benchmark_contract"]["vocab_size"],

        "full_vocab_fp32_logits":
            v["benchmark_contract"][
                "full_vocab_fp32_logits"
            ],
    },

    "scope":
        v["claim_scope"],

    "frozen_evidence": {
        "verified_result_sha256":
            sha(VERIFIED),

        "result_card_sha256":
            sha(CARD),

        "freeze_manifest_sha256":
            sha(FREEZE),
    },
}


entry_path = OUT / "leaderboard-entry.json"

entry_path.write_text(
    json.dumps(
        entry,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
)


readme = f"""# Qwen3-4B Q4_K_M — RAMGPT QuantBench Open Artifact Rank V1

**RAMGPT_QUANTBENCH_VERIFIED**

## Winner

**Unsloth**

| Rank | Artifact | Mean KL ↓ | BF16-relative top-1 flips |
|---:|---|---:|---:|
| 1 | Unsloth Q4_K_M | {rank[0]["mean_kl_bf16_to_artifact"]:.12g} | {rank[0]["changed_top1_count"]:,} |
| 2 | ggml-org Q4_K_M | {rank[1]["mean_kl_bf16_to_artifact"]:.12g} | {rank[1]["changed_top1_count"]:,} |

The tested Unsloth Q4_K_M artifact produced
{entry["primary_metric"]["unsloth_relative_reduction_percent"]:.2f}%
lower mean KL divergence from the pinned native BF16 reference than
the tested ggml-org artifact.

The paired 128-segment bootstrap interval for

`mean_KL_ggml-org - mean_KL_Unsloth`

is

`[{entry["paired_uncertainty"]["ci_95"][0]:.12g},
{entry["paired_uncertainty"]["ci_95"][1]:.12g}]`.

Because the complete interval is above zero, the frozen verdict is:

**UNSLOTH_LOWER_KL**

## Important qualification

This is an **Open Artifact / Download Rank**.

It is not a controlled same-recipe quantization comparison.

The Unsloth artifact explicitly records importance-matrix provenance,
while the tested ggml-org artifact does not expose equivalent recipe
provenance.

Therefore:

**STRICT_RECIPE_RANK: INELIGIBLE**

No **RAMGPT_RECOMMENDED** designation is awarded.

## What "better" means here

The ranking measures behavioral fidelity to the pinned native BF16
reference under the versioned RAMGPT QuantBench execution contract.

It does not claim universal superiority in downstream capability,
accuracy, generation quality, speed, memory use, or every workload.

## Verification

- 65,536 aligned scored positions
- 128 independent segments
- 151,936-token vocabulary
- Full-vocabulary FP32 logits
- Native BF16 canonical reference
- Two complete BF16 captures bitwise identical
- Primary full-RQL analysis: PASS
- Independent raw-RQL numerical validation: PASS
- 10,000 paired segment bootstrap replicates
- Weighted composite: none

## Frozen evidence

`verified-result.json`

`{sha(VERIFIED)}`

`result-card.md`

`{sha(CARD)}`

`freeze-manifest.json`

`{sha(FREEZE)}`
"""

(OUT / "README.md").write_text(readme)


badge = {
    "schema":
        "ramgpt-quantbench-verification-badge",

    "schema_version": 1,

    "benchmark_id":
        v["benchmark_id"],

    "designation":
        "RAMGPT_QUANTBENCH_VERIFIED",

    "status":
        "PASS",

    "winner":
        "Unsloth",

    "recommended":
        False,

    "verified_result_sha256":
        sha(VERIFIED),
}

(OUT / "badge.json").write_text(
    json.dumps(
        badge,
        indent=2,
        sort_keys=True,
    ) + "\n"
)


manifest = {
    "schema":
        "ramgpt-quantbench-public-leaderboard-release",

    "schema_version": 1,

    "status":
        "PASS",

    "files": {
        "leaderboard-entry.json":
            sha(OUT / "leaderboard-entry.json"),

        "README.md":
            sha(OUT / "README.md"),

        "badge.json":
            sha(OUT / "badge.json"),
    },

    "source_verified_result_sha256":
        sha(VERIFIED),
}

(OUT / "release-manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n"
)


print()
print("============================================================")
print("PUBLIC LEADERBOARD ENTRY")
print("============================================================")

print("MODEL: Qwen/Qwen3-4B")
print("QUANT: Q4_K_M")
print("CLASS: OPEN_ARTIFACT_DOWNLOAD_RANK")
print("RANK_1: Unsloth")
print("RANK_2: ggml-org")
print(
    "UNSLOTH_KL_REDUCTION_PERCENT:",
    f"{entry['primary_metric']['unsloth_relative_reduction_percent']:.4f}"
)
print("BADGE: RAMGPT_QUANTBENCH_VERIFIED")
print("STRICT_RECIPE_RANK: INELIGIBLE")
print("RAMGPT_RECOMMENDED: NOT_AWARDED")

print()
print("=== PUBLIC FILE HASHES ===")

for name in (
    "leaderboard-entry.json",
    "README.md",
    "badge.json",
    "release-manifest.json",
):
    p = OUT / name
    print(sha(p), p)

print()
print("PUBLIC_LEADERBOARD_ENTRY_V1: PASS")
