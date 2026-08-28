# Open-Source Release Audit

This repository was assembled as a new allowlisted source tree. The development repository and its Git metadata were not copied.

## Included

- RQL/RQSEG capture, parsing, comparison, metrics, and deterministic validation
- artifact orchestration, provenance, topology, tokenizer, and execution-contract checks
- statistical analysis, schemas, verification and freeze tooling, tests, and documentation
- the versioned evaluation corpus and explicitly curated public evidence packages

## Excluded

- quantization-production optimization and private experimental research
- model weights, raw full-vocabulary captures, caches, conversions, build trees, binaries, and logs
- raw run directories except files individually named by public release manifests
- credentials, machine-local configuration, private notes, and unrelated manuscript material
- vendored third-party repositories; llama.cpp is fetched upstream at the recorded commit

Frozen public files were copied byte-for-byte. Manifest-selected files were verified by SHA-256; no frozen file was sanitized or reformatted.

## Release Checks

Before the root commit, the candidate tree is checked for secrets and credentials with multiple regular-expression scans, filename inspection, local paths and identities, suspicious binaries, and every file above 20 MB. Python tests, CTest, schema validation, import/compile checks, release-manifest hashes, and README smoke commands are run against the candidate tree. After the root commit, tracked content and the complete fresh history are scanned again.

The final command results, counts, largest-file result, and publication verification are recorded in the release report accompanying the initial public release.

## Boundary Review

The private-IP review asks whether the repository could reconstruct a private high-fidelity quant-production process. The open-science review asks whether an independent researcher can inspect and reproduce evaluation of an existing artifact. Publication proceeds only when the first answer is no and the second is yes.
