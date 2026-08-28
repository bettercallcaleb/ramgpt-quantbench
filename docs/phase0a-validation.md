# Phase 0A validation

- llama.cpp: `70adb1b4cea5ee39f867792c78dc59320921eda7`
- API audit: [phase0a-llama-api.md](phase0a-llama-api.md)
- CPU build: `cmake -S . -B build -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`
- CUDA build: `cmake -S . -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cuda -j`
- Tests: `ctest --test-dir build --output-on-failure` — passed 1/1 on the CPU build; `serialization_changed_values = 0`.

The unit suite covers token serialization and stable SHA-256, RQL headers, bitwise FP32 round trips, top-1/top-2 and margin, comparator identity, truncation, vocabulary mismatch, and corpus-hash mismatch. The expected round-trip line is `serialization_changed_values = 0`.

No compatible GGUF was assumed or downloaded. After a CUDA build, run:

```bash
RAMGPT_BUILD_DIR=build-cuda scripts/run_phase0a_validation.sh /absolute/path/model-bf16.gguf /absolute/path/small-corpus.txt 4096 -1
```

This creates one frozen corpus, captures runs A and B, and reports repeatability. The script currently supports one independent segment whose full token count must fit the context. It has no burn-in option, so all positions except the terminal token are scored. To approximate the requested first validation, use a corpus of roughly 512 tokens; true 8-segment semantics are deferred rather than silently concatenated.

Reproducibility caveats: GPU kernels and scheduling may yield small run-to-run differences; BF16 describes model weights, while this probe explicitly configures F16 KV and captures llama.cpp's returned FP32 logits. The RQL `model_id` is llama.cpp's model description, not a full model-file hash. Split GGUF hashing and multi-segment corpus schemas remain future format revisions.
