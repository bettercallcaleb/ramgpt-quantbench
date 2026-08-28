#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import struct
import time

import numpy as np


ROOT = Path(".")
OUT = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/open-artifact-rank-v1"
)
OUT.mkdir(parents=True, exist_ok=True)

HEADER = 321
N = 65536
V = 151936
EXPECTED_RQL_BYTES = HEADER + N * V * 4

BOOTSTRAP_SEED = 20260823
BOOTSTRAP_REPS = 10000
CHUNK = 32

BF16_MODEL = Path("models/Qwen3-4B-BF16.gguf")
GGML_MODEL = Path(
    "artifact-bench/downloads/qwen3-4b-q4-k-m/"
    "ggml-org/Qwen3-4B-Q4_K_M.gguf"
)
UNSLOTH_MODEL = Path(
    "artifact-bench/downloads/qwen3-4b-q4-k-m/"
    "unsloth/Qwen3-4B-Q4_K_M.gguf"
)

BF16_RQL = Path(
    "artifact-bench/results/qwen3-4b-bf16-reference/run-a/bf16.rql"
)
GGML_RQL = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "ggml-org/run-a/ggml-org.rql"
)
UNSLOTH_RQL = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "unsloth/run-a/unsloth.rql"
)

BF16_POS = BF16_RQL.with_suffix(".positions.jsonl")
GGML_POS = GGML_RQL.with_suffix(".positions.jsonl")
UNSLOTH_POS = UNSLOTH_RQL.with_suffix(".positions.jsonl")

EXPECTED_MODEL_SHA = {
    BF16_MODEL:
        "47d6654f04449d367b84da24187597a76cbf66a31e07a6665d29844f2161bf77",

    GGML_MODEL:
        "ab27b9bfa375a178d6cba48f3ad892b94b7739659dcc7aae8058ce0ffed6b328",

    UNSLOTH_MODEL:
        "f6f851777709861056efcdad3af01da38b31223a3ba26e61a4f8bf3a2195813a",
}

EXPECTED_BF16_RQL_SHA = (
    "f3e7253312f222ac28b44c31ba60c4e9866af0bc9a36566d88a1af69f7417a19"
)

CORPUS_SHA = (
    "7110c8d1f3657bb63bc6b26eca3945b346ea74d11721f1ba4985fec91323e4ac"
)

LLAMA_COMMIT = "70adb1b4cea5ee39f867792c78dc59320921eda7"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fixed_string(b):
    return b.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def parse_rql_header(path):
    with open(path, "rb") as f:
        b = f.read(HEADER)

    if len(b) != HEADER:
        raise RuntimeError(f"short RQL header: {path}")

    if b[:8] != b"RQLOGIT\0":
        raise RuntimeError(f"bad RQL magic: {path}")

    schema, header_size, vocab, dtype = struct.unpack_from("<IIII", b, 8)
    positions = struct.unpack_from("<Q", b, 24)[0]

    model_sha = b[32:64].hex()
    token_sha = b[64:96].hex()

    llama_commit = fixed_string(b[96:137])
    model_id = fixed_string(b[137:265])

    vals = struct.unpack_from("<14I", b, 265)

    gpu_layers = struct.unpack("<i", struct.pack("<I", vals[5]))[0]

    return {
        "schema_version": schema,
        "header_size": header_size,
        "vocab_size": vocab,
        "dtype": dtype,
        "positions": positions,
        "model_sha256": model_sha,
        "token_sha256": token_sha,
        "llama_cpp_commit": llama_commit,
        "model_id": model_id,
        "n_ctx": vals[0],
        "n_batch": vals[1],
        "n_ubatch": vals[2],
        "kv_k_type": vals[3],
        "kv_v_type": vals[4],
        "gpu_layers": gpu_layers,
        "flash_attention": vals[6],
        "tokenizer_add_special": vals[7],
        "tokenizer_parse_special": vals[8],
        "sampling": vals[9],
        "reserved": list(vals[10:14]),
    }


def validate_header(label, path, expected_model_sha):
    if path.stat().st_size != EXPECTED_RQL_BYTES:
        raise RuntimeError(
            f"{label}: RQL size {path.stat().st_size} "
            f"!= {EXPECTED_RQL_BYTES}"
        )

    h = parse_rql_header(path)

    expected = {
        "schema_version": 2,
        "header_size": 321,
        "vocab_size": V,
        "dtype": 1,
        "positions": N,
        "model_sha256": expected_model_sha,
        "token_sha256": CORPUS_SHA,
        "llama_cpp_commit": LLAMA_COMMIT,
        "n_ctx": 4096,
        "n_batch": 512,
        "n_ubatch": 512,
        "kv_k_type": 1,
        "kv_v_type": 1,
        "gpu_layers": -1,
        "flash_attention": 1,
        "tokenizer_add_special": 1,
        "tokenizer_parse_special": 0,
        "sampling": 0,
        "reserved": [0, 0, 0, 0],
    }

    bad = {
        k: {"got": h[k], "expected": v}
        for k, v in expected.items()
        if h[k] != v
    }

    if bad:
        raise RuntimeError(f"{label}: RQL contract mismatch: {bad}")

    return h


def load_positions(path):
    rows = [json.loads(x) for x in open(path)]
    if len(rows) != N:
        raise RuntimeError(f"{path}: expected {N} rows, got {len(rows)}")
    return rows


STATIC_KEYS = (
    "global_scored_row_index",
    "segment_id",
    "domain",
    "input_position",
    "target_position",
    "local_input_position",
    "target_local_position",
    "input_token_id",
    "target_token_id",
)


def verify_position_alignment(ref, other, label):
    for i, (a, b) in enumerate(zip(ref, other)):
        for k in STATIC_KEYS:
            if a[k] != b[k]:
                raise RuntimeError(
                    f"{label}: position mismatch row={i} "
                    f"field={k} ref={a[k]!r} other={b[k]!r}"
                )


def qstats(x):
    if x.size == 0:
        return {
            "n": 0,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
            "count_margin_ge_1": 0,
            "count_margin_ge_2": 0,
            "count_margin_ge_4": 0,
            "count_margin_ge_8": 0,
            "count_margin_ge_16": 0,
        }

    return {
        "n": int(x.size),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
        "count_margin_ge_1": int(np.count_nonzero(x >= 1)),
        "count_margin_ge_2": int(np.count_nonzero(x >= 2)),
        "count_margin_ge_4": int(np.count_nonzero(x >= 4)),
        "count_margin_ge_8": int(np.count_nonzero(x >= 8)),
        "count_margin_ge_16": int(np.count_nonzero(x >= 16)),
    }


print("============================================================")
print("1. INPUT INTEGRITY")
print("============================================================")

for p, expected in EXPECTED_MODEL_SHA.items():
    got = sha256(p)
    print(f"{p}")
    print(f"  sha256={got}")
    if got != expected:
        raise RuntimeError(f"model SHA mismatch: {p}")

bf16_rql_sha = sha256(BF16_RQL)
ggml_rql_sha = sha256(GGML_RQL)
unsloth_rql_sha = sha256(UNSLOTH_RQL)

print()
print("BF16_RQL_SHA256:", bf16_rql_sha)
print("GGML_RQL_SHA256:", ggml_rql_sha)
print("UNSLOTH_RQL_SHA256:", unsloth_rql_sha)

if bf16_rql_sha != EXPECTED_BF16_RQL_SHA:
    raise RuntimeError("canonical BF16 RQL SHA mismatch")

headers = {
    "bf16": validate_header(
        "bf16", BF16_RQL, EXPECTED_MODEL_SHA[BF16_MODEL]
    ),
    "ggml-org": validate_header(
        "ggml-org", GGML_RQL, EXPECTED_MODEL_SHA[GGML_MODEL]
    ),
    "unsloth": validate_header(
        "unsloth", UNSLOTH_RQL, EXPECTED_MODEL_SHA[UNSLOTH_MODEL]
    ),
}

print("RQL_EXECUTION_CONTRACTS: PASS")

print()
print("============================================================")
print("2. EXACT POSITION ALIGNMENT")
print("============================================================")

rp = load_positions(BF16_POS)
ap = load_positions(GGML_POS)
bp = load_positions(UNSLOTH_POS)

verify_position_alignment(rp, ap, "ggml-org")
verify_position_alignment(rp, bp, "unsloth")

print("BF16_vs_GGML_POSITION_IDENTITY: PASS")
print("BF16_vs_UNSLOTH_POSITION_IDENTITY: PASS")

segments = np.array([x["segment_id"] for x in rp], dtype=np.int32)
domains = np.array([x["domain"] for x in rp], dtype=object)
targets = np.array([x["target_token_id"] for x in rp], dtype=np.int64)

ref_top1 = np.array([x["top1_token_id"] for x in rp], dtype=np.int32)
ref_top2 = np.array([x["top2_token_id"] for x in rp], dtype=np.int32)
ggml_top1 = np.array([x["top1_token_id"] for x in ap], dtype=np.int32)
unsloth_top1 = np.array(
    [x["top1_token_id"] for x in bp], dtype=np.int32
)

unique_segments = np.unique(segments)

if unique_segments.size != 128:
    raise RuntimeError("expected exactly 128 segments")

for sid in unique_segments:
    if np.count_nonzero(segments == sid) != 512:
        raise RuntimeError(f"segment {sid}: not 512 positions")

print("PAIRED_128_SEGMENT_GEOMETRY: PASS")

print()
print("============================================================")
print("3. FULL-VOCAB BF16-RELATIVE ANALYSIS")
print("============================================================")

R = np.memmap(
    BF16_RQL,
    dtype="<f4",
    mode="r",
    offset=HEADER,
    shape=(N, V),
)

A = np.memmap(
    GGML_RQL,
    dtype="<f4",
    mode="r",
    offset=HEADER,
    shape=(N, V),
)

B = np.memmap(
    UNSLOTH_RQL,
    dtype="<f4",
    mode="r",
    offset=HEADER,
    shape=(N, V),
)

kl_a = np.empty(N, dtype=np.float64)
kl_b = np.empty(N, dtype=np.float64)

dnll_a = np.empty(N, dtype=np.float64)
dnll_b = np.empty(N, dtype=np.float64)

p_ref_target = np.empty(N, dtype=np.float64)
p_a_target = np.empty(N, dtype=np.float64)
p_b_target = np.empty(N, dtype=np.float64)

bf16_margin = np.empty(N, dtype=np.float64)

t0 = time.perf_counter()

for s in range(0, N, CHUNK):
    e = min(N, s + CHUNK)
    n = e - s
    idx = np.arange(n)

    r = np.array(R[s:e], dtype=np.float64, copy=True)
    a = np.array(A[s:e], dtype=np.float64, copy=True)
    b = np.array(B[s:e], dtype=np.float64, copy=True)

    if not (
        np.isfinite(r).all()
        and np.isfinite(a).all()
        and np.isfinite(b).all()
    ):
        raise RuntimeError(f"non-finite logits in rows {s}:{e}")

    mr = np.max(r, axis=1)
    er = np.exp(r - mr[:, None])
    sr = np.sum(er, axis=1, dtype=np.float64)
    lzr = mr + np.log(sr)

    ma = np.max(a, axis=1)
    lza = ma + np.log(
        np.sum(
            np.exp(a - ma[:, None]),
            axis=1,
            dtype=np.float64,
        )
    )

    mb = np.max(b, axis=1)
    lzb = mb + np.log(
        np.sum(
            np.exp(b - mb[:, None]),
            axis=1,
            dtype=np.float64,
        )
    )

    pref = er / sr[:, None]

    ka = (
        np.sum(pref * (r - a), axis=1, dtype=np.float64)
        + lza
        - lzr
    )

    kb = (
        np.sum(pref * (r - b), axis=1, dtype=np.float64)
        + lzb
        - lzr
    )

    if np.min(ka) < -1e-9 or np.min(kb) < -1e-9:
        raise RuntimeError(
            f"numerically invalid negative KL rows {s}:{e}: "
            f"{np.min(ka)}, {np.min(kb)}"
        )

    kl_a[s:e] = ka
    kl_b[s:e] = kb

    tt = targets[s:e]

    logpr = r[idx, tt] - lzr
    logpa = a[idx, tt] - lza
    logpb = b[idx, tt] - lzb

    p_ref_target[s:e] = np.exp(logpr)
    p_a_target[s:e] = np.exp(logpa)
    p_b_target[s:e] = np.exp(logpb)

    dnll_a[s:e] = logpr - logpa
    dnll_b[s:e] = logpr - logpb

    t1 = ref_top1[s:e]
    t2 = ref_top2[s:e]

    bf16_margin[s:e] = (
        r[idx, t1] - r[idx, t2]
    )

    if e % 2048 == 0 or e == N:
        elapsed = time.perf_counter() - t0
        print(
            f"processed={e}/{N} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )


for name, arr in {
    "kl_ggml": kl_a,
    "kl_unsloth": kl_b,
    "dnll_ggml": dnll_a,
    "dnll_unsloth": dnll_b,
    "p_ref_target": p_ref_target,
    "p_ggml_target": p_a_target,
    "p_unsloth_target": p_b_target,
    "bf16_margin": bf16_margin,
}.items():
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite derived metric: {name}")


flip_a = ggml_top1 != ref_top1
flip_b = unsloth_top1 != ref_top1


def artifact_metrics(kl, dnll, qp, flip):
    hc = {}

    for tau in (1, 2, 4):
        eligible = bf16_margin >= tau
        denom = int(np.count_nonzero(eligible))
        num = int(np.count_nonzero(flip & eligible))

        hc[str(tau)] = {
            "eligible_positions": denom,
            "flips": num,
            "hcdf_rate": float(num / denom) if denom else None,
        }

    return {
        "mean_kl_bf16_to_artifact": float(np.mean(kl)),
        "median_kl": float(np.quantile(kl, 0.50)),
        "p90_kl": float(np.quantile(kl, 0.90)),
        "p99_kl": float(np.quantile(kl, 0.99)),

        "mean_delta_nll": float(np.mean(dnll)),
        "ppl_ratio": float(math.exp(float(np.mean(dnll)))),

        "target_probability_pearson_r":
            float(np.corrcoef(p_ref_target, qp)[0, 1]),

        "target_probability_rms_difference":
            float(np.sqrt(np.mean((qp - p_ref_target) ** 2))),

        "target_probability_mae":
            float(np.mean(np.abs(qp - p_ref_target))),

        "same_top1_rate":
            float(1.0 - np.mean(flip)),

        "changed_top1_count":
            int(np.count_nonzero(flip)),

        "flip_rate":
            float(np.mean(flip)),

        "hcdf": hc,

        "max_flipped_bf16_margin":
            float(np.max(bf16_margin[flip]))
            if np.any(flip)
            else None,
    }


metrics_a = artifact_metrics(
    kl_a, dnll_a, p_a_target, flip_a
)

metrics_b = artifact_metrics(
    kl_b, dnll_b, p_b_target, flip_b
)

print()
print("============================================================")
print("4. PAIRED SEGMENT BOOTSTRAP")
print("============================================================")

segment_rows = []
segment_delta = []

for sid in unique_segments:
    m = segments == sid

    a_mean = float(np.mean(kl_a[m]))
    b_mean = float(np.mean(kl_b[m]))
    delta = a_mean - b_mean

    ds = sorted(set(domains[m].tolist()))
    if len(ds) != 1:
        raise RuntimeError(f"segment {sid} spans multiple domains")

    segment_rows.append({
        "segment_id": int(sid),
        "domain": ds[0],
        "positions": int(np.count_nonzero(m)),
        "ggml_org_mean_kl": a_mean,
        "unsloth_mean_kl": b_mean,
        "delta_kl_ggml_minus_unsloth": delta,
    })

    segment_delta.append(delta)

segment_delta = np.array(segment_delta, dtype=np.float64)

rng = np.random.default_rng(BOOTSTRAP_SEED)

draws = rng.integers(
    0,
    len(segment_delta),
    size=(BOOTSTRAP_REPS, len(segment_delta)),
)

boot = np.mean(segment_delta[draws], axis=1)

ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

delta_point = (
    metrics_a["mean_kl_bf16_to_artifact"]
    - metrics_b["mean_kl_bf16_to_artifact"]
)

if ci_high < 0:
    rank_verdict = "GGML_ORG_LOWER_KL"
    winner = "ggml-org"

elif ci_low > 0:
    rank_verdict = "UNSLOTH_LOWER_KL"
    winner = "unsloth"

else:
    rank_verdict = "PAIRED_CI_INCLUDES_ZERO"
    winner = "joint-tier-1"

point_estimate_lower = (
    "ggml-org"
    if metrics_a["mean_kl_bf16_to_artifact"]
       < metrics_b["mean_kl_bf16_to_artifact"]
    else "unsloth"
    if metrics_b["mean_kl_bf16_to_artifact"]
       < metrics_a["mean_kl_bf16_to_artifact"]
    else "exact-tie"
)

paired = {
    "definition":
        "delta_KL = mean_KL_ggml_org - mean_KL_unsloth",

    "point_estimate_delta_kl":
        float(delta_point),

    "bootstrap_seed":
        BOOTSTRAP_SEED,

    "bootstrap_replicates":
        BOOTSTRAP_REPS,

    "resampling_unit":
        "segment",

    "segment_count":
        128,

    "positions_per_segment":
        512,

    "ci_method":
        "paired nonparametric percentile 95% bootstrap",

    "ci_95_low":
        float(ci_low),

    "ci_95_high":
        float(ci_high),

    "point_estimate_lower_kl":
        point_estimate_lower,

    "rank_verdict":
        rank_verdict,

    "open_artifact_rank_winner":
        winner,
}


print()
print("============================================================")
print("5. EXCLUSIVE FAILURE FORENSICS")
print("============================================================")

both_stable = ~flip_a & ~flip_b
a_only = flip_a & ~flip_b
b_only = ~flip_a & flip_b

both_flip_same = (
    flip_a
    & flip_b
    & (ggml_top1 == unsloth_top1)
)

both_flip_different = (
    flip_a
    & flip_b
    & (ggml_top1 != unsloth_top1)
)

categories = {
    "BOTH_STABLE": int(np.count_nonzero(both_stable)),
    "GGML_ONLY_FLIP": int(np.count_nonzero(a_only)),
    "UNSLOTH_ONLY_FLIP": int(np.count_nonzero(b_only)),
    "BOTH_FLIP_SAME_ALTERNATE":
        int(np.count_nonzero(both_flip_same)),
    "BOTH_FLIP_DIFFERENT_ALTERNATE":
        int(np.count_nonzero(both_flip_different)),
}

if sum(categories.values()) != N:
    raise RuntimeError("exclusive failure categories do not partition positions")

forensics = {
    "categories": categories,

    "ggml_only_flip_bf16_margin":
        qstats(bf16_margin[a_only]),

    "unsloth_only_flip_bf16_margin":
        qstats(bf16_margin[b_only]),
}


print()
print("============================================================")
print("6. DOMAIN METRICS")
print("============================================================")

domain_rows = []

for d in sorted(set(domains.tolist())):
    m = domains == d

    domain_rows.append({
        "domain": d,
        "positions": int(np.count_nonzero(m)),
        "ggml_org_mean_kl": float(np.mean(kl_a[m])),
        "unsloth_mean_kl": float(np.mean(kl_b[m])),
        "delta_kl_ggml_minus_unsloth":
            float(np.mean(kl_a[m]) - np.mean(kl_b[m])),
        "ggml_org_flip_rate": float(np.mean(flip_a[m])),
        "unsloth_flip_rate": float(np.mean(flip_b[m])),
    })


print()
print("============================================================")
print("7. FREEZE OUTPUTS")
print("============================================================")

np.savez_compressed(
    OUT / "per-position.npz",

    segment_id=segments,
    target_token_id=targets,

    bf16_margin=bf16_margin,

    kl_ggml_org=kl_a,
    kl_unsloth=kl_b,

    delta_nll_ggml_org=dnll_a,
    delta_nll_unsloth=dnll_b,

    target_probability_bf16=p_ref_target,
    target_probability_ggml_org=p_a_target,
    target_probability_unsloth=p_b_target,

    top1_bf16=ref_top1,
    top1_ggml_org=ggml_top1,
    top1_unsloth=unsloth_top1,

    flip_ggml_org=flip_a,
    flip_unsloth=flip_b,
)


with open(OUT / "segment-metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=segment_rows[0].keys())
    w.writeheader()
    w.writerows(segment_rows)


with open(OUT / "domain-metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=domain_rows[0].keys())
    w.writeheader()
    w.writerows(domain_rows)


script_sha = sha256(Path(__file__))

result = {
    "schema":
        "ramgpt-qwen3-4b-q4-k-m-open-artifact-rank-v1",

    "schema_version": 1,

    "status": "PASS",

    "comparison_class":
        "OPEN_ARTIFACT_DOWNLOAD_RANK",

    "strict_recipe_rank_eligible": False,

    "strict_recipe_rank_ineligibility_reason":
        "Unsloth explicitly records importance-matrix provenance; "
        "ggml-org does not expose equivalent recipe provenance. "
        "This is therefore not a same-recipe comparison.",

    "primary_fidelity_rule":
        "mean KL(BF16 || artifact), lower is better",

    "weighted_composite":
        None,

    "benchmark": {
        "positions": N,
        "segments": 128,
        "positions_per_segment": 512,
        "vocab_size": V,
        "full_vocab_fp32_logits": True,
        "corpus_sha256": CORPUS_SHA,
        "llama_cpp_commit": LLAMA_COMMIT,
    },

    "inputs": {
        "bf16_model": {
            "path": str(BF16_MODEL),
            "sha256": EXPECTED_MODEL_SHA[BF16_MODEL],
            "bytes": BF16_MODEL.stat().st_size,
        },

        "ggml_org_model": {
            "path": str(GGML_MODEL),
            "sha256": EXPECTED_MODEL_SHA[GGML_MODEL],
            "bytes": GGML_MODEL.stat().st_size,
            "provenance_grade":
                "B_MODEL_MATCHED_SOURCE_OR_RECIPE_UNCERTAIN",
        },

        "unsloth_model": {
            "path": str(UNSLOTH_MODEL),
            "sha256": EXPECTED_MODEL_SHA[UNSLOTH_MODEL],
            "bytes": UNSLOTH_MODEL.stat().st_size,
            "provenance_grade":
                "C_RECIPE_VARIANT_EXPLICIT_IMATRIX",
        },

        "bf16_rql": {
            "path": str(BF16_RQL),
            "sha256": bf16_rql_sha,
            "bytes": BF16_RQL.stat().st_size,
        },

        "ggml_org_rql": {
            "path": str(GGML_RQL),
            "sha256": ggml_rql_sha,
            "bytes": GGML_RQL.stat().st_size,
        },

        "unsloth_rql": {
            "path": str(UNSLOTH_RQL),
            "sha256": unsloth_rql_sha,
            "bytes": UNSLOTH_RQL.stat().st_size,
        },
    },

    "artifacts": {
        "ggml-org": metrics_a,
        "unsloth": metrics_b,
    },

    "paired_primary_comparison": paired,

    "exclusive_failure_forensics": forensics,

    "analysis_script": {
        "path": str(Path(__file__)),
        "sha256": script_sha,
    },
}

with open(OUT / "canonical-results.json", "w") as f:
    json.dump(
        result,
        f,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    f.write("\n")


with open(OUT / "exclusive-failure-forensics.json", "w") as f:
    json.dump(
        forensics,
        f,
        indent=2,
        sort_keys=True,
    )
    f.write("\n")


print("analysis_script_sha256:", script_sha)
print("canonical_results_sha256:", sha256(OUT / "canonical-results.json"))
print("per_position_sha256:", sha256(OUT / "per-position.npz"))

print()
print("============================================================")
print("PRIMARY FIDELITY")
print("============================================================")

print(
    "GGML_ORG_MEAN_KL:",
    f"{metrics_a['mean_kl_bf16_to_artifact']:.12g}",
)

print(
    "UNSLOTH_MEAN_KL:",
    f"{metrics_b['mean_kl_bf16_to_artifact']:.12g}",
)

print(
    "DELTA_KL_GGML_MINUS_UNSLOTH:",
    f"{delta_point:.12g}",
)

print(
    "PAIRED_BOOTSTRAP_95_CI:",
    f"[{ci_low:.12g}, {ci_high:.12g}]",
)

print(
    "OPEN_ARTIFACT_RANK_VERDICT:",
    rank_verdict,
)

print(
    "OPEN_ARTIFACT_RANK_WINNER:",
    winner,
)

print("STRICT_RECIPE_RANK: INELIGIBLE")

print()
print("============================================================")
print("DECISION STABILITY")
print("============================================================")

for label, m in (
    ("GGML_ORG", metrics_a),
    ("UNSLOTH", metrics_b),
):
    print(
        label,
        "flips=",
        m["changed_top1_count"],
        "flip_rate=",
        f"{m['flip_rate']:.9g}",
        "HCDFR@1=",
        f"{m['hcdf']['1']['hcdf_rate']:.9g}",
        "HCDFR@2=",
        f"{m['hcdf']['2']['hcdf_rate']:.9g}",
        "HCDFR@4=",
        f"{m['hcdf']['4']['hcdf_rate']:.9g}",
        "max_flipped_margin=",
        m["max_flipped_bf16_margin"],
    )

print()
print("============================================================")
print("EXCLUSIVE FAILURE COUNTS")
print("============================================================")

for k, v in categories.items():
    print(f"{k}: {v}")

print()
print("ANALYSIS_STATUS: PASS")
