# RAMGPT QuantBench

### Measure the artifact. Not the label.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](pyproject.toml)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-70adb1b-orange.svg)](third_party/LLAMA_CPP_COMMIT)

Two GGUF files can both be called `Q4_K_M`. They can come from the same canonical model, be almost the same size, and run in the same inference engine.

**They are not necessarily the same model anymore.**

RAMGPT QuantBench measures how far each artifact moves from a pinned native BF16 reference across tens of thousands of scored token positions and the full output vocabulary. No vibe checks, cherry-picked prompts, or publisher reputation—just reproducible behavioral divergence under a versioned execution contract.

> `Q4_K_M` is a storage label. It is not a fidelity guarantee.

The unit of evaluation is the artifact—not the publisher, quantization brand, or filename.

```text
                    Canonical Model
                         BF16
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Artifact A       Artifact B       Artifact C
        Q4_K_M           Q4_K_M           Q4_K_M
          └────────────────┼────────────────┘
                           ▼
                  RAMGPT QuantBench
                           │
              full-vocabulary comparison
                           ▼
                Behavioral Fidelity Rank
```

A quantization label tells you how the file is packaged. QuantBench measures what happened to the model.

## Why QuantBench Exists

Nominally identical artifacts may differ in tensor types and assignments, auxiliary tensors, metadata, tokenizer behavior, or other artifact properties. The useful question is not “Which publisher is best?” It is: **Which exact artifact is closest to the pinned reference under this measurement contract?**

## What QuantBench Measures

The reference is a native BF16 artifact captured twice under a pinned runtime and execution geometry. Mean KL divergence, `KL(BF16 || artifact)`, is the primary fidelity rank. The system also records ΔNLL, perplexity ratio, target-token probability correlation, RMS and MAE probability displacement, top-1 flips, HCDFR, RCFC, recovery analysis, and paired-segment bootstrap uncertainty where applicable.

> QuantBench compares probability distributions, not just generated strings.

Two models may choose the same visible next token while assigning materially different probability mass across the rest of the vocabulary. Full-vocabulary logits expose that difference.

## A Result Should Look Like Evidence

The frozen Qwen3.8-27B Open Artifact Rank V1 result reports:

| Rank | Artifact | Mean KL ↓ | Top-1 flips |
|---|---|---:|---:|
| 1 | bartowski | 0.010784 | 3,298 |
| 2 | Unsloth legacy standard Q4_K_M | 0.012181 | 3,542 |
| 3 | ggml-org | 0.012310 | 3,650 |
| 4 | mradermacher-static | 0.023989 | 4,697 |

Under the pinned contract, the tested bartowski artifact was behaviorally closest to the native BF16 reference among these four tested artifacts. This is not a publisher-wide or quantization-algorithm claim. The frozen analysis does not establish that the Unsloth-versus-ggml-org difference is statistically distinguished.

The tested Unsloth artifact is the **legacy standard Q4_K_M artifact**, not the current UD-Q4_K_M artifact.

## Artifact-Level Forensics

QuantBench verifies source and SHA-256 identity, BF16 reference identity, tensor topology and shapes, quantized tensor types, tokenizer metadata and behavior, runtime revision, execution geometry, reference determinism, and frozen-result integrity.

- `EXACT_GGUF_TOPOLOGY` means the artifact and reference have matching GGUF tensor names and shapes.
- `SCORED_PATH_TOPOLOGY_EQUIVALENCE` means a structural difference is outside the pinned scored next-token graph and has been explicitly reviewed. In the Qwen3.8 result, the ggml-org artifact omitted the trailing Qwen35 NextN/MTP auxiliary block; the pinned ordinary llama.cpp scored graph excludes that auxiliary path. This is not exact topology.
- `SCORED_TOKENIZATION_EQUIVALENCE` means metadata differs but frozen behavioral checks establish equivalent tokenization under the benchmark corpus contract. This qualification applies to the tested legacy Unsloth artifact.

## Reproducibility Is Part of the Result

```text
RESULT
  ├── artifact and reference identity
  ├── corpus and tokenizer behavior
  ├── runtime commit and execution geometry
  ├── scored positions and distributions
  ├── uncertainty and forensic evidence
  └── immutable hashes
```

> Reproducibility is not metadata attached to a benchmark. It is part of the benchmark.

## The QuantBench Contract

Every result is qualified by a versioned execution contract. Qwen3.8-27B used `QuantBench-V1-REFNGL24`:

```text
reference_gpu_layers = 24
artifact_gpu_layers  = -1
placement_policy     = reference_partial_offload_artifact_full_gpu
```

The placement was asymmetric. QuantBench records that asymmetry because placement can contribute numerical effects; it does not pretend the effect is impossible. Interpret the result only under this pinned contract.

## Quick Start

Requirements: Linux, Python 3.10+, CMake 3.16+, a C++17 compiler, and Git. CUDA is optional.

```bash
git clone https://github.com/bettercallcaleb/ramgpt-quantbench.git
cd ramgpt-quantbench
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt

commit="$(cat third_party/LLAMA_CPP_COMMIT)"
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout "$commit"

cmake -S . -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
python -m unittest discover -s artifact-bench/tests -p 'test_*.py'
python artifact-bench/tools/validate_schemas.py
```

The Python suite and schema validator are the lightweight installation smoke test; they do not download model weights or claim a scientific benchmark result.

The supported artifact-comparison interface is the orchestrator script. Its dry run resolves metadata and estimates disk requirements but does not download model bodies, run inference, or delete files:

```bash
python artifact-bench/run-artifact-comparison.py \
  --reference https://huggingface.co/Qwen/Qwen3-4B \
  --artifact a=https://huggingface.co/ORG_A/REPO/resolve/REV/a.gguf \
  --artifact b=https://huggingface.co/ORG_B/REPO/resolve/REV/b.gguf \
  --quantization-label Q4_K_M \
  --cleanup aggressive \
  --dry-run
```

Remove `--dry-run` only after reviewing the plan, disk budget, model licenses, and immutable URLs. Outputs are written beneath `artifact-bench/runs/<benchmark-id>/`. See `--help` for resume, runtime, build, and placement controls. Heavyweight reproduction requires the declared model artifacts and suitable compute.

Inspect a frozen package with:

```bash
python -m json.tool artifact-bench/public/qwen3-8-27b-q4-k-m-open-artifact-rank-v1/verified-result.json | less
sha256sum artifact-bench/public/qwen3-8-27b-q4-k-m-open-artifact-rank-v1/*
```

## Architecture

```text
Canonical HF Model → Native BF16 Reference → deterministic captures A/B
                                             │
                                             ▼
                                      Reference Freeze
                                             │
                         ┌───────────────────┼───────────────────┐
                         ▼                   ▼                   ▼
                    Artifact A          Artifact B          Artifact N
                         ▼                   ▼                   ▼
                      preflight + full-vocabulary capture under contract
                                             │
                                             ▼
                       analysis → decision forensics → validation
                                             │
                                             ▼
                              immutable public evidence package
```

## Repository Layout

| Path | Purpose |
|---|---|
| `cpp/ramgpt-logit-probe/` | RQL/RQSEG formats, capture, comparison, metrics, and inspection tools |
| `artifact-bench/` | Orchestrator, model adapter, protocols, schemas, analysis, and tests |
| `artifact-bench/public/` | Immutable, allowlisted public evidence packages |
| `data/fidelity-v2-candidate/` | Versioned evaluation corpus, token payload, and provenance |
| `config/` | Execution contracts |
| `docs/` | Format and validation specifications |
| `third_party/LLAMA_CPP_COMMIT` | Required upstream llama.cpp revision; source is not vendored |

## Verification Model

`RAMGPT_QUANTBENCH_VERIFIED` means an artifact/result satisfied the applicable versioned QuantBench verification contract. It does **not** mean RAMGPT recommends the artifact for every workload. `RAMGPT_RECOMMENDED` is a separate designation and is not awarded by this repository.

## Quantization Production Is Out of Scope

QuantBench is an evaluation system, not a quantization-production toolkit. This repository does not contain private optimization recipes, calibration strategies, importance-matrix tuning, tensor-specific allocation heuristics, or UD-class production pipelines. That boundary does not reduce the openness of the measurement methodology.

> QuantBench does not produce the artifact. It measures the artifact.

## Published Benchmarks

Immutable evidence included here:

- [Qwen3-4B Q4_K_M Open Artifact Rank V1](artifact-bench/public/qwen3-4b-q4-k-m-open-artifact-rank-v1/)
- [Qwen3.8-27B Q4_K_M Open Artifact Rank V1](artifact-bench/public/qwen3-8-27b-q4-k-m-open-artifact-rank-v1/)
- [Qwen3.8-27B bartowski vs Unsloth UD-Q4_K_XL V1](artifact-bench/public/qwen3-8-27b-bartowski-vs-unsloth-ud-q4-v1/)

The public project page is [ramgpt.org/quantbench](https://ramgpt.org/quantbench/).

## Scientific Interpretation

QuantBench measures behavioral fidelity to a reference distribution. It does not directly measure downstream usefulness, writing quality, factuality, safety, latency, throughput, memory efficiency, or universal model quality. Lower KL under one contract means closer behavior to that reference distribution under that contract—nothing broader.

## Contributing

Contributions to model-family compatibility, runtime validation, forensic checks, statistical methods, reproducibility, benchmark submissions, tests, and documentation are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Changes must not silently weaken comparability or rewrite frozen evidence.

## Citation

Until a formal publication is available, cite the software repository:

```bibtex
@software{ramgpt_quantbench,
  title  = {RAMGPT QuantBench},
  author = {{RAMGPT}},
  url    = {https://github.com/bettercallcaleb/ramgpt-quantbench},
  year   = {2026}
}
```

## License

Original project source is licensed under [Apache-2.0](LICENSE). llama.cpp and model artifacts are external dependencies with their own licenses and are not redistributed here. Frozen evidence describes third-party artifacts but does not include their model weights.

RAMGPT QuantBench is developed as part of RAMGPT's work on reproducible local-model measurement and artifact forensics.

https://ramgpt.org/quantbench/
