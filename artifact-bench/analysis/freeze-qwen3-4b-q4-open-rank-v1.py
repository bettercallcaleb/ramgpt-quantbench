#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

ROOT = Path(".")
OUT = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "open-artifact-rank-v1"
)

CANON = OUT / "canonical-results.json"
VALID = OUT / "independent-validation.json"

EXPECTED = {
    "canonical-results.json":
        "bb87f88cdcbf27562c3be390001e1347f24482675c40d4427630f7da63119a9e",

    "independent-validation.json":
        "ac98a6b1f53a1466174f1b699eea6bc5c0fdb4e9b86d6f9f7a0cf49f19263828",
}

BF16_RQL_SHA = (
    "f3e7253312f222ac28b44c31ba60c4e9866af0bc9a36566d88a1af69f7417a19"
)

GGML_RQL_SHA = (
    "922ca7f188e8a6232e01eeecb8d11b46b4843314bc89b7ff8f678edbdc0ce9b7"
)

UNSLOTH_RQL_SHA = (
    "9628e1a13c97344b659c6ec677e445abebc831ca8d92949140825b8cd174f402"
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
print("1. EVIDENCE FREEZE PREFLIGHT")
print("============================================================")

for name, expected in EXPECTED.items():
    path = OUT / name
    got = sha(path)

    print(name)
    print(" expected:", expected)
    print(" actual:  ", got)

    if got != expected:
        raise RuntimeError(f"frozen evidence mismatch: {name}")

print("EVIDENCE_HASH_PREFLIGHT: PASS")


canon = json.loads(CANON.read_text())
valid = json.loads(VALID.read_text())

if canon["status"] != "PASS":
    raise RuntimeError("canonical analysis is not PASS")

if valid["status"] != "PASS":
    raise RuntimeError("independent validation is not PASS")

pair = canon["paired_primary_comparison"]

if pair["open_artifact_rank_winner"] != "unsloth":
    raise RuntimeError("unexpected canonical winner")

if pair["rank_verdict"] != "UNSLOTH_LOWER_KL":
    raise RuntimeError("unexpected canonical rank verdict")

if not all(valid["checks"].values()):
    raise RuntimeError("independent validation contains failed checks")


ggml = canon["artifacts"]["ggml-org"]
unsloth = canon["artifacts"]["unsloth"]

ggml_kl = ggml["mean_kl_bf16_to_artifact"]
unsloth_kl = unsloth["mean_kl_bf16_to_artifact"]

relative_kl_reduction = (
    (ggml_kl - unsloth_kl) / ggml_kl
)

flip_reduction = (
    (
        ggml["changed_top1_count"]
        - unsloth["changed_top1_count"]
    )
    / ggml["changed_top1_count"]
)


verified = {
    "schema":
        "ramgpt-quantbench-verified-open-artifact-result",

    "schema_version": 1,

    "benchmark_id":
        "qwen3-4b-q4-k-m-open-artifact-rank-v1",

    "status":
        "RAMGPT_QUANTBENCH_VERIFIED",

    "comparison_class":
        "OPEN_ARTIFACT_DOWNLOAD_RANK",

    "model":
        "Qwen/Qwen3-4B",

    "nominal_quantization":
        "Q4_K_M",

    "rank": [
        {
            "rank": 1,
            "publisher": "Unsloth",
            "artifact_sha256":
                canon["inputs"]["unsloth_model"]["sha256"],
            "mean_kl_bf16_to_artifact":
                unsloth_kl,
            "changed_top1_count":
                unsloth["changed_top1_count"],
        },
        {
            "rank": 2,
            "publisher": "ggml-org",
            "artifact_sha256":
                canon["inputs"]["ggml_org_model"]["sha256"],
            "mean_kl_bf16_to_artifact":
                ggml_kl,
            "changed_top1_count":
                ggml["changed_top1_count"],
        },
    ],

    "winner":
        "Unsloth",

    "primary_metric":
        "mean KL(BF16 || artifact), lower is better",

    "ggml_org_mean_kl":
        ggml_kl,

    "unsloth_mean_kl":
        unsloth_kl,

    "relative_mean_kl_reduction_vs_ggml_org":
        relative_kl_reduction,

    "delta_kl_ggml_minus_unsloth":
        pair["point_estimate_delta_kl"],

    "paired_segment_bootstrap_95_ci": [
        pair["ci_95_low"],
        pair["ci_95_high"],
    ],

    "bootstrap": {
        "seed": pair["bootstrap_seed"],
        "replicates": pair["bootstrap_replicates"],
        "resampling_unit": pair["resampling_unit"],
        "segment_count": pair["segment_count"],
        "positions_per_segment":
            pair["positions_per_segment"],
    },

    "decision_stability": {
        "ggml_org_changed_top1_count":
            ggml["changed_top1_count"],

        "unsloth_changed_top1_count":
            unsloth["changed_top1_count"],

        "relative_flip_reduction_vs_ggml_org":
            flip_reduction,

        "ggml_org_hcdfr":
            ggml["hcdf"],

        "unsloth_hcdfr":
            unsloth["hcdf"],

        "ggml_org_max_flipped_bf16_margin":
            ggml["max_flipped_bf16_margin"],

        "unsloth_max_flipped_bf16_margin":
            unsloth["max_flipped_bf16_margin"],
    },

    "independent_validation": {
        "status": valid["status"],
        "implementation": valid["implementation"],
        "sha256": sha(VALID),
        "checks": valid["checks"],
    },

    "strict_recipe_rank": {
        "eligible": False,

        "reason":
            "Unsloth explicitly records importance-matrix "
            "provenance while ggml-org does not expose "
            "equivalent recipe provenance. This result ranks "
            "downloadable artifacts, not controlled "
            "quantization algorithms.",
    },

    "provenance": {
        "ggml_org":
            "Grade B: model matched; source/recipe uncertain",

        "unsloth":
            "Grade C: recipe variant; explicit imatrix",
    },

    "canonical_reference": {
        "model_sha256":
            canon["inputs"]["bf16_model"]["sha256"],

        "rql_sha256":
            BF16_RQL_SHA,

        "determinism":
            "Two complete independent captures were "
            "bitwise identical across 9,957,277,696 "
            "FP32 logits.",
    },

    "quantized_rql_sha256": {
        "ggml_org": GGML_RQL_SHA,
        "unsloth": UNSLOTH_RQL_SHA,
    },

    "benchmark_contract": canon["benchmark"],

    "weighted_composite":
        None,

    "ramgpt_recommended":
        False,

    "claim_scope":
        "Behavioral fidelity to the pinned native BF16 "
        "reference under this versioned RAMGPT QuantBench "
        "execution contract. This is not a claim of universal "
        "model quality or downstream capability.",
}


(OUT / "verified-result.json").write_text(
    json.dumps(
        verified,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
)


pct_kl = 100.0 * relative_kl_reduction
pct_flip = 100.0 * flip_reduction

card = f"""# RAMGPT QuantBench Verified Result

## Qwen3-4B Q4_K_M Open Artifact Rank V1

**Status:** RAMGPT_QUANTBENCH_VERIFIED

**Winner:** Unsloth

| Rank | Artifact | Mean KL(BF16 || artifact) | Top-1 flips |
|---:|---|---:|---:|
| 1 | Unsloth Q4_K_M | {unsloth_kl:.12g} | {unsloth["changed_top1_count"]:,} |
| 2 | ggml-org Q4_K_M | {ggml_kl:.12g} | {ggml["changed_top1_count"]:,} |

The tested Unsloth artifact has {pct_kl:.2f}% lower mean
BF16-relative KL than the tested ggml-org artifact under the
RAMGPT QuantBench Qwen3-4B Open Artifact Rank V1 contract.

The paired 128-segment bootstrap estimate for

`mean_KL_ggml-org - mean_KL_Unsloth`

is `{pair["point_estimate_delta_kl"]:.12g}`, with a frozen 95%
percentile interval of
`[{pair["ci_95_low"]:.12g}, {pair["ci_95_high"]:.12g}]`.

The entire interval is above zero, therefore the frozen ranking
verdict is:

**UNSLOTH_LOWER_KL**

Unsloth also produced {pct_flip:.2f}% fewer BF16-relative top-1
flips ({unsloth["changed_top1_count"]:,} versus
{ggml["changed_top1_count"]:,}).

## Interpretation

This is an **Open Artifact / Download Rank**. It answers which of
these exact downloadable Q4_K_M artifacts is behaviorally closer to
the pinned native BF16 reference under this benchmark contract.

It is **not** a controlled same-recipe quantization experiment.

The Unsloth GGUF explicitly records importance-matrix provenance.
The ggml-org artifact does not expose equivalent recipe provenance.
Therefore:

**STRICT_RECIPE_RANK: INELIGIBLE**

The result must not be interpreted as proof that one publisher's
quantization algorithm is universally superior.

## Verification

- Scored positions: 65,536
- Independent segments: 128
- Positions per segment: 512
- Vocabulary: 151,936
- Full-vocabulary FP32 logits
- BF16 reference determinism: bitwise identical across two complete
  independent captures
- Primary analysis: PASS
- Independent raw-RQL numerical validation: PASS
- Bootstrap replicates: 10,000
- Bootstrap seed: 20260823
- Resampling unit: segment
- Weighted composite: none

## Artifact identities

### Native BF16 reference

`{canon["inputs"]["bf16_model"]["sha256"]}`

### ggml-org Q4_K_M

`{canon["inputs"]["ggml_org_model"]["sha256"]}`

### Unsloth Q4_K_M

`{canon["inputs"]["unsloth_model"]["sha256"]}`

## Scope

"Better" in this result means lower behavioral divergence from the
pinned native BF16 reference under the versioned RAMGPT QuantBench
contract. It does not mean universally better capability, accuracy,
generation quality, or suitability for every workload.

No `RAMGPT_RECOMMENDED` designation is awarded by this result.
"""

(OUT / "result-card.md").write_text(card)


freeze = {
    "schema":
        "ramgpt-quantbench-open-artifact-result-freeze-manifest",

    "schema_version": 1,

    "status": "PASS",

    "files": {
        "canonical-results.json": sha(CANON),
        "independent-validation.json": sha(VALID),
        "verified-result.json":
            sha(OUT / "verified-result.json"),
        "result-card.md":
            sha(OUT / "result-card.md"),
    },
}

(OUT / "freeze-manifest.json").write_text(
    json.dumps(
        freeze,
        indent=2,
        sort_keys=True,
    ) + "\n"
)


print()
print("============================================================")
print("VERIFIED RESULT")
print("============================================================")

print("WINNER: Unsloth")
print("GGML_ORG_MEAN_KL:", f"{ggml_kl:.12g}")
print("UNSLOTH_MEAN_KL:", f"{unsloth_kl:.12g}")
print(
    "UNSLOTH_RELATIVE_KL_REDUCTION_PERCENT:",
    f"{pct_kl:.4f}",
)

print(
    "PAIRED_BOOTSTRAP_95_CI:",
    [
        pair["ci_95_low"],
        pair["ci_95_high"],
    ],
)

print("STRICT_RECIPE_RANK: INELIGIBLE")
print("RAMGPT_QUANTBENCH_VERIFIED: PASS")
print("RAMGPT_RECOMMENDED: NOT_AWARDED")

print()
print("=== FREEZE HASHES ===")

for p in (
    OUT / "verified-result.json",
    OUT / "result-card.md",
    OUT / "freeze-manifest.json",
):
    print(sha(p), p)

print()
print("QWEN3_4B_Q4_K_M_OPEN_ARTIFACT_RESULT_FREEZE: PASS")
