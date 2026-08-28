# fidelity-v2 source-generation protocol v1

Status: source generation pending because no external text-generation capability is connected in the current environment.

## Isolation

Each `source-plan.json` entry is one independent generation request. A request receives only the system constraints below and that entry's segment ID, domain, topic, document genre/structure, and language. It must not receive any earlier generated document, response, or source excerpt. Generation is source creation only; the model has no role in auditing, similarity measurement, token selection, acceptance, inference, or metrics.

Use a fresh model conversation/request for every document. Set no conversational history. Save the response exactly once as UTF-8 after mechanically normalizing line endings to LF and ensuring one terminal newline. Do not editorially revise a response in place. If it fails a source gate, mark that provenance record rejected and issue a new independent request with a new attempt ID and output filename.

## Per-document request

System instruction (`fidelity-v2-source-generation-v1`):

> Create one original, coherent source document for RAMGPT QuantBench's distributional-fidelity corpus. This request is independent; do not create a variant of a template, rewrite another text, translate another corpus document, or substitute names and numbers into a shared example. Follow the supplied domain, unique topic, and unique document genre/structure. Target approximately 1,200–1,800 Qwen3 model tokens of substantive material. Avoid filler, generic introduction/conclusion boilerplate, stock phrases, self-referential AI language, mention of synthetic data or benchmarks, repetition of this prompt, and unnecessary Markdown headings. Maintain natural local coherence and vary sentence/record length, punctuation, vocabulary, and organization as appropriate. Return only the document.

The user payload is a single JSON object:

```json
{
  "prompt_version": "fidelity-v2-source-generation-v1",
  "segment_id": 0,
  "domain": "english_prose",
  "topic": "winter_lighthouse_supply_run",
  "document_genre_or_structure": "first-person logistical narrative",
  "language_or_programming_language": "English"
}
```

Additional domain instruction is selected exactly once from the following list:

- `english_prose`: Use realistic syntax, named entities, numbers, dates where locally appropriate, varied paragraph and sentence lengths, and the specified prose form.
- `chinese_prose`: Write independently authored natural Chinese with suitable Chinese punctuation and discourse structure. Do not translate or echo an English topic.
- `mixed_en_zh`: Write a genuinely bilingual context with natural switching inside discourse. Do not alternate translations and do not reuse monolingual source passages.
- `code`: Return a complete, coherent artifact in the specified language and program category, including realistic validation, errors, data structures, and comments where appropriate. Do not wrap it in explanatory prose or Markdown fences.
- `structured_data`: Return a coherent artifact in the specified format and unique operational model. Use format-appropriate nesting, types, escaping, identifiers, timestamps, nulls, or records without borrowing a common schema.
- `math_symbolic`: Develop the specified subject with coherent notation, derivations or proofs, varied equations, and explanatory prose. Do not enumerate a stock equation inventory.
- `technical_it`: Produce the specified technical document form with concrete systems reasoning, failure modes, constraints, and operational detail. Avoid generic best-practice lists.
- `dialogue_instruction`: Use the specified scenario, participants, and turn structure. Make turns responsive and locally specific; avoid generic reusable instruction lines.

## Capture and provenance

For each response, record provider/model identity when available, request identifier when available, generation date, attempt ID, plan fields, filename, response byte count, and SHA-256 in `source-provenance.json`. Generation metadata is nondeterministic and remains outside the deterministic manifest and RQSEG.

Accepted source filenames use `NNN_<domain>_<topic>.txt`. Rejected attempts use a separate `rejected/` provenance path and are never silently overwritten. The frozen accepted source tree is the reproducibility boundary.

## Downstream deterministic gate

After all 128 accepted source files exist, external generation stops. The pinned Qwen3 tokenizer tokenizes each frozen file once with `add_special=true`, `parse_special=false`. A deterministic hash-derived start selects one contiguous 768-token window. No padding or decode/re-encode loop is permitted. Candidate assembly and the strengthened audit then operate only on frozen bytes and token IDs.
