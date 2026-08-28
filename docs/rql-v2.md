# RQL v2 and the execution contract

RQL v2 retains one little-endian FP32 full-vocabulary logit row per scored position. It adds a deterministic, fixed-size header binding those rows to their source model, frozen tokens, and execution-critical inference settings. Timestamps and machine provenance are deliberately excluded from the binary and written to `<run>.execution.json`.

## Binary layout

The magic remains `RQLOGIT\0`. All integers are little-endian. Fixed strings are NUL-terminated and zero-padded; values that do not fit are rejected. The header is exactly 321 bytes.

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | magic |
| 8 | 4 | schema version (`2`) |
| 12 | 4 | header size (`321`) |
| 16 | 4 | vocabulary size |
| 20 | 4 | logits dtype (`1` = FP32) |
| 24 | 8 | number of positions |
| 32 | 32 | model SHA-256 |
| 64 | 32 | frozen token-file SHA-256 |
| 96 | 41 | llama.cpp commit |
| 137 | 128 | model identity/description |
| 265 | 4 each | `n_ctx`, `n_batch`, `n_ubatch` |
| 277 | 4 each | K and V types (`1` = F16) |
| 285 | 4 | signed GPU layers |
| 289 | 4 | resolved Flash Attention (`0` off, `1` on) |
| 293 | 4 each | tokenizer add-special and parse-special booleans |
| 301 | 4 | sampling (`0` = none) |
| 305 | 16 | four reserved zero words |

Logit data starts at byte 321. Readers reject unknown versions, incorrect sizes, nonzero reserved fields, invalid execution geometry, unsupported dtype/KV/sampling values, and truncated data. RQL v1's validated 233-byte layout remains readable and existing files are not rewritten.

Basic comparison compatibility requires identical vocabulary size, frozen-token SHA-256, and position count. Strict comparison additionally requires two v2 inputs and equality of model SHA-256, llama.cpp commit, context/batch/ubatch, K/V types, GPU layers, resolved Flash Attention, tokenizer behavior, and sampling. Errors name the mismatched field. Quantized-to-reference comparison necessarily uses a different model SHA, but enforces the v2 reference's execution geometry against the live quantized run.

## Flash Attention and provenance

The pinned llama.cpp public API accepts `AUTO`, `ENABLED`, or `DISABLED`, but exposes no public getter for the state resolved from `AUTO`. RQL v2 therefore forbids ambiguous AUTO capture: `ramgpt-logit-probe` requires `--flash-attention enabled|disabled`, applies that exact context parameter, and records the same successfully-created state. It never infers a resolved state from a default.

The execution sidecar records the timestamp, executable/build identity, paths and hashes, execution settings, command line, visible CUDA mapping, and GPU/CUDA metadata. Hardware values are sourced from `RAMGPT_GPU_NAME`, `RAMGPT_GPU_UUID`, `RAMGPT_GPU_PCI_ID`, `RAMGPT_CUDA_DRIVER_VERSION`, and `RAMGPT_CUDA_RUNTIME_VERSION`; unavailable values are explicit JSON `null`, never guessed.

An official primary reference invocation must explicitly supply tokenizer behavior and Flash Attention:

```bash
CUDA_VISIBLE_DEVICES=0 build-cuda/ramgpt-logit-probe \
  --model models/Qwen3-8B-BF16.gguf --tokens data/phase0a.tokens \
  --output results/reference --context-size 4096 --batch-size 512 \
  --gpu-layers -1 --rql-version 2 --flash-attention enabled \
  --tokenizer-add-special true --tokenizer-parse-special false
```

Use `ramgpt-compare-reference A.rql B.rql --strict` for strict contract checking.
