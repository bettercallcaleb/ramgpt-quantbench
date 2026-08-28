# Artifact Ranking Protocol V1

Status: frozen protocol definition. This document defines no candidate results.

## Evaluation unit and identity

The unit is an immutable GGUF artifact, not a publisher. Identity comprises model family, named model/release, publisher, nominal quant preset, filename, and artifact SHA-256. The SHA-256 is canonical: the same filename with another digest is another artifact. A producer can therefore have multiple separately ranked records.

## Provenance grades

### `GRADE_A_VERIFIED_SAME_SOURCE`

Strong evidence verifies the exact base model, architecture, tokenizer, upstream source/revision, nominal recipe, requantization status, imatrix status, and artifact SHA-256.

### `GRADE_B_MODEL_MATCHED_SOURCE_UNCERTAIN`

The model/release, architecture, tokenizer, and preset are compatible, but exact source revision or the complete recipe cannot be proven.

### `GRADE_C_RECIPE_VARIANT`

The artifact uses a custom, imatrix, dynamic, UD, IQ, altered, or other materially variant recipe. It must not participate in a strict Standard Q4_K_M class unless a later protocol creates an appropriate variant class.

### `GRADE_D_INELIGIBLE`

Available identity or compatibility evidence cannot support a fair comparison against the frozen reference contract.

Grades describe artifact provenance and eligibility, not publisher quality.

## Comparison classes

Artifacts compete only inside a machine-readable class. The initial `qwen3-8b-standard-q4-k-m-v1` class requires the named Qwen3-8B release, compatible architecture/tokenizer, nominal standard Q4_K_M, no imatrix, no requantization, and no dynamic/UD/IQ/material recipe alteration. Class rules are frozen in `comparison-classes-v1.json`. Variant artifacts may be recorded at Grades C/D but are not strict-class competitors.

## Primary fidelity ranking

The primary metric is

`mean KL(native BF16 || artifact)`,

with lower values better. Every artifact must use exactly aligned within-model positions against the same canonical native-BF16 reference. No weighted composite score is permitted.

## Paired comparison rule

Artifact A versus B uses a paired nonparametric segment bootstrap:

- resampling unit: segment;
- replicates: 10,000;
- default seed: 20260823;
- primary difference: `Delta KL = mean_KL_A - mean_KL_B`;
- interval: paired 95% percentile confidence interval.

Verdicts are immutable:

- CI entirely below zero: `A_HAS_CLEARLY_LOWER_KL`;
- CI entirely above zero: `B_HAS_CLEARLY_LOWER_KL`;
- CI includes zero: `STATISTICALLY_INDISTINGUISHABLE_ON_MEAN_KL`.

The leaderboard must not force a unique winner in the final case and may assign `JOINT_FIDELITY_TIER_1` to the tied leading group.

## Decision-stability report

Report separately from Fidelity Rank: top-1 flip rate, HCDFR@1, HCDFR@2, HCDFR@4, and maximum flipped BF16 margin. Do not combine them with KL. The flag `HIGH_MARGIN_STABILITY_EXCEPTION` is reserved, but its triggering policy is intentionally unset in V1; it must not be fitted after observing future artifact outcomes.

## Exclusive failure forensics

For each aligned position relative to the shared BF16 reference, assign exactly one class:

- `BOTH_STABLE`;
- `A_ONLY_FLIP`;
- `B_ONLY_FLIP`;
- `BOTH_FLIP_SAME_ALTERNATE`;
- `BOTH_FLIP_DIFFERENT_ALTERNATE`.

For both exclusive sets, future reports must provide N; BF16-margin median, P90, P99, and maximum; and counts at margin thresholds >=1, >=2, >=4, >=8, and >=16. This is a core RAMGPT Artifact Forensics output.

## Badges

`RAMGPT_QUANTBENCH_VERIFIED` means the exact artifact identity is frozen, provenance is graded, benchmark execution is complete, integrity validation passed, and a result manifest is frozen. It is not an endorsement. `RAMGPT_RECOMMENDED` is undefined and must not be awarded under V1.

## Immutable benchmark identity

Every published record binds `benchmark_version`, `model_reference_sha256`, `reference_rql_sha256`, `corpus_sha256`, `execution_contract_sha256`, `llama_cpp_commit`, `analysis_protocol_version`, and `artifact_sha256`. Published records never silently change; corrections or protocol changes require a new immutable version and explicit supersession metadata.

Timestamps may document observation or creation events, but they are excluded from deterministic scientific identity.

## Core and future validation

QuantBench Core is a public, frozen, reproducible corpus and protocol. A future QuantBench Validation track may use rotating validation sets to reduce benchmark gaming. Protocol V1 creates no validation corpus and makes no validation-set result.
