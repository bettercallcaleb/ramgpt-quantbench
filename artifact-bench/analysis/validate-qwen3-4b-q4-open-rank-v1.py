#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import numpy as np

H = 321
N = 65536
V = 151936
CHUNK = 16

BF16 = Path(
    "artifact-bench/results/qwen3-4b-bf16-reference/run-a/bf16.rql"
)

GGML = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "ggml-org/run-a/ggml-org.rql"
)

UNSLOTH = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "unsloth/run-a/unsloth.rql"
)

POS = Path(
    "artifact-bench/results/qwen3-4b-bf16-reference/"
    "run-a/bf16.positions.jsonl"
)

CANON = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "open-artifact-rank-v1/canonical-results.json"
)

OUT = Path(
    "artifact-bench/results/qwen3-4b-q4-k-m/"
    "open-artifact-rank-v1/independent-validation.json"
)

SEED = 20260823
B = 10000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            x = f.read(8 * 1024 * 1024)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


canonical = json.loads(CANON.read_text())

positions = [
    json.loads(x)
    for x in POS.read_text().splitlines()
]

if len(positions) != N:
    raise RuntimeError("position count mismatch")

segments = np.array(
    [x["segment_id"] for x in positions],
    dtype=np.int32,
)

ref_top1 = np.array(
    [x["top1_token_id"] for x in positions],
    dtype=np.int32,
)

R = np.memmap(
    BF16,
    dtype="<f4",
    mode="r",
    offset=H,
    shape=(N, V),
)

A = np.memmap(
    GGML,
    dtype="<f4",
    mode="r",
    offset=H,
    shape=(N, V),
)

BMODEL = np.memmap(
    UNSLOTH,
    dtype="<f4",
    mode="r",
    offset=H,
    shape=(N, V),
)

kl_a = np.empty(N, dtype=np.float64)
kl_b = np.empty(N, dtype=np.float64)

flip_a = np.empty(N, dtype=bool)
flip_b = np.empty(N, dtype=bool)

margin = np.empty(N, dtype=np.float64)

for s in range(0, N, CHUNK):
    e = min(N, s + CHUNK)

    r = np.asarray(R[s:e], dtype=np.float64)
    a = np.asarray(A[s:e], dtype=np.float64)
    b = np.asarray(BMODEL[s:e], dtype=np.float64)

    # Independent logsumexp implementation:
    # repeated binary logaddexp reduction rather than
    # the canonical max-shift + exp + sum path.
    lzr = np.logaddexp.reduce(r, axis=1)
    lza = np.logaddexp.reduce(a, axis=1)
    lzb = np.logaddexp.reduce(b, axis=1)

    logp = r - lzr[:, None]
    p = np.exp(logp)

    logqa = a - lza[:, None]
    logqb = b - lzb[:, None]

    # Direct KL definition:
    # sum p * (log p - log q)
    kl_a[s:e] = np.einsum(
        "ij,ij->i",
        p,
        logp - logqa,
        dtype=np.float64,
    )

    kl_b[s:e] = np.einsum(
        "ij,ij->i",
        p,
        logp - logqb,
        dtype=np.float64,
    )

    ta = np.argmax(a, axis=1)
    tb = np.argmax(b, axis=1)

    flip_a[s:e] = ta != ref_top1[s:e]
    flip_b[s:e] = tb != ref_top1[s:e]

    # Independent top-two extraction from raw BF16 rows.
    two = np.argpartition(r, -2, axis=1)[:, -2:]
    vals = np.take_along_axis(r, two, axis=1)
    vals.sort(axis=1)

    margin[s:e] = vals[:, 1] - vals[:, 0]

    if e % 2048 == 0 or e == N:
        print(f"validated={e}/{N}", flush=True)


mean_a = float(np.mean(kl_a))
mean_b = float(np.mean(kl_b))
delta = mean_a - mean_b


# Independent paired segment reduction.
sids = np.unique(segments)

segment_delta = np.array([
    float(
        np.mean(kl_a[segments == sid])
        - np.mean(kl_b[segments == sid])
    )
    for sid in sids
])


rng = np.random.default_rng(SEED)

boot = np.empty(10000, dtype=np.float64)

for i in range(10000):
    sampled = rng.integers(
        0,
        len(segment_delta),
        len(segment_delta),
    )

    boot[i] = np.mean(segment_delta[sampled])


lo, hi = np.quantile(
    boot,
    [0.025, 0.975],
)


canonical_a = canonical[
    "artifacts"
]["ggml-org"]["mean_kl_bf16_to_artifact"]

canonical_b = canonical[
    "artifacts"
]["unsloth"]["mean_kl_bf16_to_artifact"]

canonical_pair = canonical[
    "paired_primary_comparison"
]

checks = {
    "ggml_mean_kl":
        abs(mean_a - canonical_a) < 1e-10,

    "unsloth_mean_kl":
        abs(mean_b - canonical_b) < 1e-10,

    "delta":
        abs(
            delta
            - canonical_pair["point_estimate_delta_kl"]
        ) < 1e-10,

    "bootstrap_ci_low":
        abs(
            float(lo)
            - canonical_pair["ci_95_low"]
        ) < 1e-10,

    "bootstrap_ci_high":
        abs(
            float(hi)
            - canonical_pair["ci_95_high"]
        ) < 1e-10,

    "ggml_flip_count":
        int(np.count_nonzero(flip_a))
        == canonical["artifacts"]["ggml-org"][
            "changed_top1_count"
        ],

    "unsloth_flip_count":
        int(np.count_nonzero(flip_b))
        == canonical["artifacts"]["unsloth"][
            "changed_top1_count"
        ],

    "winner":
        lo > 0,
}


result = {
    "schema":
        "ramgpt-qwen3-4b-q4-open-rank-independent-validation-v1",

    "implementation":
        "Independent exhaustive NumPy raw-RQL path using "
        "np.logaddexp.reduce and direct sum p*(logp-logq).",

    "rows_validated": N,
    "vocab_size": V,

    "ggml_org_mean_kl": mean_a,
    "unsloth_mean_kl": mean_b,

    "delta_kl_ggml_minus_unsloth": delta,

    "paired_bootstrap_95_ci": [
        float(lo),
        float(hi),
    ],

    "ggml_org_flip_count":
        int(np.count_nonzero(flip_a)),

    "unsloth_flip_count":
        int(np.count_nonzero(flip_b)),

    "checks": checks,

    "status":
        "PASS" if all(checks.values()) else "FAIL",
}


OUT.write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
        default=lambda o: o.item()
        if isinstance(o, np.generic)
        else (_ for _ in ()).throw(
            TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
        ),
    ) + "\n"
)


print()
print("============================================================")
print("INDEPENDENT VALIDATION")
print("============================================================")

print("GGML_ORG_MEAN_KL:", mean_a)
print("UNSLOTH_MEAN_KL:", mean_b)
print("DELTA_KL:", delta)

print(
    "PAIRED_BOOTSTRAP_95_CI:",
    [float(lo), float(hi)],
)

print(
    "GGML_ORG_FLIPS:",
    int(np.count_nonzero(flip_a)),
)

print(
    "UNSLOTH_FLIPS:",
    int(np.count_nonzero(flip_b)),
)

for k, v in checks.items():
    print(
        f"{k}:",
        "PASS" if v else "FAIL",
    )

print()
print(
    "INDEPENDENT_VALIDATION:",
    result["status"],
)

print(
    "independent_validation_sha256:",
    sha(OUT),
)

raise SystemExit(
    result["status"] != "PASS"
)
