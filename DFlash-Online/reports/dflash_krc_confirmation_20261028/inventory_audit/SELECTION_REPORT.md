# K1/KRC confirmation panel preparation

The **64 requests are fixed before generation**, using selection seed **20261028**. The panel is new to the current O/K/KC/KRC study. All 64 questions have verified exposure in older, unrelated project experiments; that exposure is permitted and is disclosed per request. This is not a globally unused GSM8K panel.

No model forward, training, generation or dataset download occurred. The executor still binds the unchanged K1/KRC checkpoint identities into its final execution seal.

## Source and preprocessing

- Dataset: `openai/gsm8k`, configuration `main`, test split, revision `740312add88f781978c0658806c59bc2815b9866`. Existing local Parquet files were verified; only the question column was loaded.
- Tokenizer: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`; the exact existing local artifacts were verified. The retained author template helper is unchanged.
- Original question text is unchanged. Append the existing instruction: `Please reason step by step, and put your final answer within \boxed{}.` Use one user message, thinking disabled and the same generation prompt. No special tokens are added during the final encode.
- Prompt lengths: **53–168 tokens**. All selected requests satisfy the unchanged context check with the 2,048-token output allowance and block-size-16 reserve inside 4,096 tokens. No truncation or replacement was needed.

## Exclusions and identity checks

| Current-study inventory | Unique requests | Exact inherited prompt reconstructions |
|---|---:|---:|
| Original event-1 feedback | 128 | 128 |
| Continuation events 2–4 feedback | 384 | 384 |
| Original development panel | 64 | 64 |
| Earlier O/K confirmation panel | 64 | 64 |
| Total | 640 | 640 |

The two feedback inventories contain **512 unique training questions**. The development and earlier confirmation panels contain **128 unique test questions**. Both canonical source IDs and NFKC/whitespace-normalized question text are excluded; normalized test questions are also deduplicated. The 512 training texts introduce no additional exact normalized duplicates in this test split, leaving **1,191 eligible test questions** after 128 exclusions.

Thirty-five retained training, qualification and local-probe receipts were checked for additional feedback identities. Their IDs are subsets of the declared feedback pools. Four early native startup-diagnostic inputs were checked by exact saved prompt IDs and are subsets of the original development panel. Generation correctness gates use requests already inside the declared panels. The complete evidence paths, identities and content bindings are in `CURRENT_STUDY_INVENTORY.json`; no required identity evidence was missing.

## Deterministic selection

Traverse pinned test rows in ascending source order. Apply the exclusions and normalized-text deduplication, shuffle the eligible source rows once with Python `random.Random(20261028)`, then take the first 64. Selection uses no source answers, generated outputs, response lengths, call counts, latency or correctness outcomes. Older unrelated exposure is annotated only after this ordering is fixed.

Three CPU checks pass: source-ID and normalized-text exclusion/deduplication, deterministic seed behavior without changing global RNG, and failure rather than substitution when too few eligible rows remain. Token reconstruction additionally verifies all 640 inherited prompts, including the continuation and prior confirmation panels.

## Fixed artifacts

- `NEW_PANEL.json`: executable request IDs, source IDs, rendered token IDs, order, stop IDs and cap; no canonical outputs yet.
- `SELECTED_QUESTIONS.json`: complete original questions and rendered prompt text for private review.
- `SELECTION_AUDIT.json`: all eligible/excluded source rows, reasons, selected order, exact pins and historical-exposure annotations.
- `CURRENT_STUDY_INVENTORY.json`: all current-study source identities and qualification/probe evidence.
- `audit_inventory.py`, `prepare_panel.py`, `test_prepare_panel.py`: reproducible preparation and focused checks.

Panel SHA256: `e5684b9ec7b7f9e1321e007ae1f2d282980f23da688c2de017d7fdf7cc15c07c`.

This panel tests the fixed K1/KRC checkpoints on new inputs within the current study. It does not test new training seeds, feedback windows or a globally fresh benchmark.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_confirmation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
