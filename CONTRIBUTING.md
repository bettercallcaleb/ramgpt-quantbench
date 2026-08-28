# Contributing

QuantBench changes must preserve artifact-level framing, versioned execution contracts, and immutable evidence.

Before opening a change, run:

```bash
ctest --test-dir build --output-on-failure
python -m unittest discover -s artifact-bench/tests -p 'test_*.py'
python artifact-bench/tools/validate_schemas.py
```

New model-family adapters should fail closed on unsupported topology or tokenizer behavior. New statistical methods need deterministic fixtures and explicit uncertainty units. Benchmark submissions must bind artifact, reference, corpus, runtime, contract, and result hashes.

Never edit a frozen result in place. Corrections require a new version with explicit derivation or supersession metadata. Do not submit model weights, raw full-vocabulary captures, credentials, private datasets, or quantization-production optimization recipes.
