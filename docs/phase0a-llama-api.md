# Phase 0A llama.cpp API audit

Pinned revision: `70adb1b4cea5ee39f867792c78dc59320921eda7`.

The public declarations are in `third_party/llama.cpp/include/llama.h`. Models are loaded with `llama_model_default_params` and `llama_model_load_from_file`; `llama_model_params::n_gpu_layers` controls device offload. Contexts use `llama_context_default_params` and `llama_init_from_model`. `llama_context_params::type_k` and `type_v` select KV types; Phase 0A explicitly uses F16 KV, matching the upstream default, and does not claim the KV cache itself is BF16. `llama_model_get_vocab` plus `llama_vocab_n_tokens` gives vocabulary size. Tokenization uses the public `llama_tokenize` sizing call followed by its filling call.

Manual evaluation uses `llama_batch_init`. Every row sets `token`, absolute zero-based `pos`, one sequence ID (`0`), and `logits`. A nonzero `logits[i]` requests an output for that input row. `llama_decode` evaluates the batch, and `llama_get_logits` returns requested rows contiguously in request order, each containing `llama_vocab_n_tokens(vocab)` C `float` values. Its implementation is in `src/llama-context.cpp`; batch allocation is in `src/llama-batch.cpp`. `llama_memory_clear(llama_get_memory(ctx), true)` is the current reset API, though a fresh process/context makes a reset unnecessary for the single segment implemented here.

`tools/perplexity/perplexity.cpp` is the upstream reference. Its `perplexity`/`perplexity_v2` paths build explicit batches, call `llama_decode`, copy full rows from `llama_get_logits` or address them with `llama_get_logits_ith`, and score `tokens[j + 1]` from the logits produced at input `j`.

## Exact token semantics

`ramgpt-tokenize` calls `llama_tokenize(..., add_special=true, parse_special=false)` once. Consequently, model-configured BOS is included when appropriate and the resulting token file records it explicitly. The probe never adds, removes, or retokenizes BOS. Frozen tokens `t0 ... tn` produce `n` scored rows: the row generated while evaluating `ti` has `input_position=i`, `target_position=i+1`, and target `t(i+1)`. The final input has no successor and is deliberately unscored. No other position is skipped or burned in in the current single-segment implementation.

Positions are absolute within sequence 0 and KV state is retained across decode batches. Runs use a fresh context. Independent segments are not silently concatenated; version 1 token files contain exactly one segment and are rejected when they exceed the selected context.

## Formats

Token files are little-endian: 8-byte `RQTOKEN\0` magic, uint32 version, uint32 flags, uint64 count, then count signed int32 IDs. RQL files use `RQLOGIT\0`, version and fixed header size, vocabulary size, dtype (`1` = IEEE-754 binary32), uint64 scored-row count, raw 32-byte token-file SHA-256, fixed model description and llama commit fields, then row-major native FP32 bytes. Phase 0A supports mainstream little-endian IEEE-754 hosts. The fixed header and contiguous rows support sequential reads and mmap.
