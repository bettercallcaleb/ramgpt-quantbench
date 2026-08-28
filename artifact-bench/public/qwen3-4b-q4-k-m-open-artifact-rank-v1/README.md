# Qwen3-4B Q4_K_M — RAMGPT QuantBench Open Artifact Rank V1

**RAMGPT_QUANTBENCH_VERIFIED**

## Winner

**Unsloth**

| Rank | Artifact | Mean KL ↓ | BF16-relative top-1 flips |
|---:|---|---:|---:|
| 1 | Unsloth Q4_K_M | 0.0385423074764 | 5,699 |
| 2 | ggml-org Q4_K_M | 0.0539642979728 | 6,437 |

The tested Unsloth Q4_K_M artifact produced
28.58%
lower mean KL divergence from the pinned native BF16 reference than
the tested ggml-org artifact.

The paired 128-segment bootstrap interval for

`mean_KL_ggml-org - mean_KL_Unsloth`

is

`[0.013704046109,
0.017244240664]`.

Because the complete interval is above zero, the frozen verdict is:

**UNSLOTH_LOWER_KL**

## Important qualification

This is an **Open Artifact / Download Rank**.

It is not a controlled same-recipe quantization comparison.

The Unsloth artifact explicitly records importance-matrix provenance,
while the tested ggml-org artifact does not expose equivalent recipe
provenance.

Therefore:

**STRICT_RECIPE_RANK: INELIGIBLE**

No **RAMGPT_RECOMMENDED** designation is awarded.

## What "better" means here

The ranking measures behavioral fidelity to the pinned native BF16
reference under the versioned RAMGPT QuantBench execution contract.

It does not claim universal superiority in downstream capability,
accuracy, generation quality, speed, memory use, or every workload.

## Verification

- 65,536 aligned scored positions
- 128 independent segments
- 151,936-token vocabulary
- Full-vocabulary FP32 logits
- Native BF16 canonical reference
- Two complete BF16 captures bitwise identical
- Primary full-RQL analysis: PASS
- Independent raw-RQL numerical validation: PASS
- 10,000 paired segment bootstrap replicates
- Weighted composite: none

## Frozen evidence

`verified-result.json`

`6a4624c40f412087c5de23ef15c3479cb406cf68f8835cfd2ecf89c1c2bea4cd`

`result-card.md`

`55fa093ec7804b50a743a84a6dcd0cdf14b82136bbdcc6a32846a3d5d8b1f03a`

`freeze-manifest.json`

`34500999556dcc3037c93f49c95d74c4411d1f5e31de9045cb363b9e63244d9c`
