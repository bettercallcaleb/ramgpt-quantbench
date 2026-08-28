import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
H64 = "a" * 64
H40 = "b" * 40


def schema(name):
    with (SCHEMAS / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate(name, instance):
    Draft202012Validator(schema(name), format_checker=Draft202012Validator.FORMAT_CHECKER).validate(instance)


class SchemaTests(unittest.TestCase):
    def test_artifact_record_accepts_complete_identity(self):
        record = {
            "schema_version": "artifact-record-v1", "artifact_id": "example-artifact", "publisher": "Example",
            "repository": "example/repository", "filename": "model.Q4_K_M.gguf", "sha256": H64, "bytes": 1,
            "model_name": "Qwen3-8B", "model_family": "Qwen3", "model_revision_claimed": None,
            "model_revision_verified": None, "architecture": "qwen3", "tokenizer_identity": "unverified-example",
            "nominal_quant_type": "Q4_K_M", "comparison_class": "qwen3-8b-standard-q4-k-m-v1",
            "imatrix_status": "UNKNOWN", "requantization_status": "UNKNOWN",
            "provenance_grade": "GRADE_B_MODEL_MATCHED_SOURCE_UNCERTAIN",
            "provenance_evidence": [{"kind": "HASH", "description": "Example-only test evidence"}],
            "download_source": "example-only", "status": "PREFLIGHT"
        }
        validate("artifact-record-v1.schema.json", record)
        bad = copy.deepcopy(record); bad["sha256"] = "not-a-hash"
        with self.assertRaises(ValidationError):
            validate("artifact-record-v1.schema.json", bad)

    def test_artifact_result_rejects_composite_score(self):
        result = {
            "schema_version": "artifact-result-v1",
            "artifact": {"artifact_id": "a", "artifact_sha256": H64, "provenance_grade": "GRADE_A_VERIFIED_SAME_SOURCE", "comparison_class": "c"},
            "benchmark": {"benchmark_version": "v", "model_reference_sha256": H64, "reference_rql_sha256": H64,
                          "corpus_sha256": H64, "execution_contract_sha256": H64, "llama_cpp_commit": H40,
                          "analysis_protocol_version": "artifact-ranking-v1", "artifact_sha256": H64},
            "metrics": {"mean_kl": 0, "median_kl": 0, "p90_kl": 0, "p99_kl": 0, "delta_nll": 0,
                        "ppl_ratio": 1, "flip_count": 0, "flip_rate": 0, "hcdfr_at_1": 0,
                        "hcdfr_at_2": 0, "hcdfr_at_4": 0, "max_flipped_bf16_margin": 0},
            "bootstrap": {"resampling_unit": "segment", "replicates": 10000, "seed": 20260823,
                          "confidence_level": 0.95, "intervals": {"mean_kl": [0, 0]}},
            "integrity": {"artifact_identity": "PASS", "position_alignment": "PASS", "execution_contract": "PASS",
                          "metric_validation": "PASS", "manifest_frozen": "PASS"},
            "verification_status": "RAMGPT_QUANTBENCH_VERIFIED", "result_manifest_sha256": H64
        }
        validate("artifact-result-v1.schema.json", result)
        bad = copy.deepcopy(result); bad["weighted_composite_score"] = 1
        with self.assertRaises(ValidationError):
            validate("artifact-result-v1.schema.json", bad)

    def test_pairwise_requires_frozen_rule_and_forensics(self):
        summary = {"n": 0, "median": None, "p90": None, "p99": None, "max": None,
                   "counts_ge": {"1": 0, "2": 0, "4": 0, "8": 0, "16": 0}}
        pair = {
            "schema_version": "pairwise-comparison-v1", "comparison_class": "c", "benchmark_version": "v",
            "artifact_a": {"artifact_id": "a", "sha256": H64}, "artifact_b": {"artifact_id": "b", "sha256": H64},
            "paired_delta_kl": {"definition": "mean_KL_A_minus_mean_KL_B", "estimate": 0, "ci_95": [-1, 1]},
            "primary_verdict": "STATISTICALLY_INDISTINGUISHABLE_ON_MEAN_KL",
            "position_classes": {"both_stable": 1, "a_only_flip": 0, "b_only_flip": 0,
                                 "both_flip_same_alternate": 0, "both_flip_different_alternate": 0},
            "a_only_margin_summary": summary, "b_only_margin_summary": summary,
            "bootstrap": {"resampling_unit": "segment", "replicates": 10000, "seed": 20260823,
                          "confidence_level": 0.95, "interval_method": "paired_percentile"}
        }
        validate("pairwise-comparison-v1.schema.json", pair)
        bad = copy.deepcopy(pair); bad["primary_verdict"] = "FORCED_WINNER"
        with self.assertRaises(ValidationError):
            validate("pairwise-comparison-v1.schema.json", bad)

    def test_leaderboard_supports_empty_skeleton_and_rejects_recommended(self):
        board = {"schema_version": "leaderboard-v1", "comparison_class": "c", "benchmark_version": "v",
                 "analysis_protocol_version": "artifact-ranking-v1", "primary_metric": "mean_KL_BF16_to_artifact",
                 "ranking_policy": "lower_is_better_no_weighted_composite", "eligible_artifacts": [],
                 "fidelity_ranking": [], "tie_groups": []}
        validate("leaderboard-v1.schema.json", board)
        bad = copy.deepcopy(board); bad["badge"] = "RAMGPT_RECOMMENDED"
        with self.assertRaises(ValidationError):
            validate("leaderboard-v1.schema.json", bad)


if __name__ == "__main__":
    unittest.main()
