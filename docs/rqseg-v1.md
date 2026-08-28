# RQSEG v1 segmented token format

RQSEG v1 is the deterministic frozen-token container for the fidelity-v1 corpus. All integers are little-endian. The fixed 256-byte header contains:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 8 | `RQSEG\0\0\0` magic |
| 8 | 4 | version (`1`) |
| 12 | 4 | header size (`256`) |
| 16 | 4 each | segment count, tokens/segment, burn-in tokens, scored tokens/segment |
| 32 | 4 | tokenizer flags: bit 0 add-special, bit 1 parse-special |
| 36 | 32 | tokenizer/model GGUF SHA-256 |
| 68 | 32 | SHA-256 of the ordered raw token payload |
| 100 | 41 | pinned llama.cpp commit |
| 141 | 115 | tokenizer/model identity |

Each segment record contains a uint32 segment ID, a 32-byte NUL-padded domain, the SHA-256 of that segment's raw token payload, and exactly `tokens_per_segment` signed int32 token IDs. Records are ordered by segment ID. The canonical payload checksum hashes all record token arrays in that order, without record metadata.

For fidelity-v1 the dimensions are 128 × 768, with 256 burn-in target tokens and 512 scored target tokens per segment. Logits after input position `p` predict target `p+1`; therefore scored inputs are 255–766 and targets are 256–767. `ramgpt-logit-probe --segmented-corpus` creates and destroys a llama context for every segment. `ramgpt-quant-compare --segmented-corpus` follows the same boundary and verifies that the RQL v2 frozen-corpus SHA-256 matches the exact RQSEG file.

The RQL v2 `frozen_token_file_sha256` field stores the SHA-256 of the complete RQSEG file. This binds reference rows to both token IDs and segment/domain metadata while preserving the existing v2 compatibility rule.
