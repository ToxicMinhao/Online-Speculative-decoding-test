## Material Passport

- Origin Skill: ARS academic-research-suite 3.23.1, experiment-agent.
- Mode: bounded source inventory and deterministic data preparation.
- Date: 2026-10-27.
- Verification Status: ANALYZED; local source/identity checks and tokenizer reconstruction completed, no model experiment executed.
- Scope: pinned GSM8K data and current O/K study exclusions; no remote operation, GPU computation, model loading, training or generation.

The original requirement of 64 globally unused GSM8K test questions cannot be met. This audit verifies actual completed execution of at least **1,311 of the 1,319** pinned test questions. At most eight remain before any additional history check; those eight are not certified unused. The user then explicitly approved “Use 64 requests new to the current O/K study (Recommended).” Under that amendment, a deterministic 64-request panel is prepared and excludes every current feedback, development and qualification question. **All 64 selected questions were executed in older unrelated project work.** This fact is recorded per request and must accompany the confirmation result.

### Actual-use evidence, kept separate from allocation

The audit reads the pinned source question column and maps historical opaque request IDs to official rows by unique NFKC/whitespace-normalized question equality. The mapping is bijective for all 1,319 test rows and all 7,473 train rows. Neither benchmark answer column is loaded from Parquet.

| Verified completed source | Identity/completion evidence | Unique test questions | New exclusions at this audit step |
|---|---|---:|---:|
| AdaFlash serving reference, `gpu0_ar` | `RESULT.json`: passed, benchmark exit 0; `REQUESTS.jsonl`: 1,024 timed request IDs matching the manifest, plus a duplicate warmup | 1,024 | 1,024 |
| Objective-check `code_s0_buffered_gsm8k` | `RESULT.json`: passed, count/completed 276; 276 `EVALUATION.jsonl` IDs exactly match the manifest; output file SHA-256 matches receipt | 276 | 276 |
| October 22 projected-target GSM8K B8 | Eight returned F records match selected source IDs and exact prompt IDs, have nonempty response sequences and termination metadata | 8 | 8 |
| Async passes, `final_one_pass` | Passed result and completed inference receipt; 256 response source IDs match request manifest exactly | 256 | 3 |
| **Verified union** | Duplicate use is counted once | **1,311** | **1,311** |

The three additional async-final source rows are 39, 119 and 229. The October 22 additional rows are 411, 435, 469, 543, 676, 1010, 1286 and 1287. The eight remaining rows after these verified executions are **270, 418, 438, 725, 799, 833, 1113 and 1302**. This is an upper bound on possible globally unused questions, not a fresh-question list or a new selection.

The earlier conservative exclusion audit covered 1,031 evidence files / 722 unique contents and excluded 1,304 test rows before October 22. It includes reserved/allocated inputs. In particular, row 270 is supported by an inherited D212 selected-data manifest; this audit does not establish its actual execution. After adding the October 22 B8, that reserved-inclusive audit leaves seven rows. The strict **actual-use upper bound is eight**, and the reserved-inclusive figure is not substituted for it. Merely downloading, mapping, normalizing or scanning the full source file is not counted as running a question.

Earlier workspace prose said all 1,319 official test prompts had appeared in evaluations. This bounded audit does not independently establish that stronger statement and does not rely on it. The verified 1,311 is already sufficient to establish the shortfall. No reduced panel, substitute dataset or historical-use exception was applied until the user explicitly amended freshness.

### Current-study exclusions

The authoritative current-study manifest is the sealed October 25 one-event inventory, reused unchanged in the completed O/C/K/A campaign. Exact train/evaluation records in the O/C/K/A campaign manifest match that inventory. C/K/A completed records each contain the same 128 unique feedback request IDs over 256 presentations. The qualification receipt contains zero generations; every training qualification window uses `train_00072` and `train_00121`, both within those 128 feedback requests. Generation qualification uses the first eight of the existing 64 development requests, inside the original evaluation budget.

The current exclusion set is therefore:

- 128 distinct official training-source questions used for feedback, including the training qualification inputs.
- 64 distinct official test-source development questions, including the eight generation qualification requests.
- Exact NFKC/whitespace-normalized question matches to either set, plus duplicate normalized questions within the candidate test source.

The test source has 1,319 unique normalized questions. The 64 development rows are excluded; no additional test question duplicates a current feedback question. That leaves **1,255 eligible test rows**. Exact current-study source IDs, native opaque IDs and question/prompt hashes are retained in [CURRENT_STUDY_SOURCE_IDS.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/CURRENT_STUDY_SOURCE_IDS.json). The broader historical exposure inventory is annotation, not an exclusion rule, under the approved amendment.

### Selection and exact preprocessing

Selection starts with eligible official test-row indices in ascending order, applies `random.Random(20261027).shuffle`, and takes the first 64. Selection does not use answers, generated text, correctness, acceptance, verification counts, latency, prior method gains or any model-based criterion. Historical exposure is annotated after this fixed selection, and no question is replaced because of its historical outcome or exposure status.

Dataset: `openai/gsm8k`, configuration `main`, revision `740312add88f781978c0658806c59bc2815b9866`.

| File | Rows | SHA-256 |
|---|---:|---|
| `main/test-00000-of-00001.parquet` | 1,319 | `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` |
| `main/train-00000-of-00001.parquet` | 7,473 | `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d` |

Serving preserves the original source question, followed by the exact inherited evaluation suffix:

```text
\nPlease reason step by step, and put your final answer within \boxed{}.
```

This denotes an actual leading newline, not the two literal characters `\` and `n`. Rendering uses one user message, no explicit system message, the existing author `apply_chat_template`/`_reasoning_kwargs` helpers, reasoning off / `enable_thinking=False`, generation prompt enabled, and tokenizer encoding with `add_special_tokens=False`. The tokenizer is local `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218`. Its config, template, vocabulary and generation-config hashes are bound in the preparation receipt. No model weights are loaded.

All **128 inherited training prompt-ID sequences** and **64 inherited development prompt-ID sequences** were reconstructed exactly before the new panel was emitted. Training uses the original question without the evaluation suffix, as in the retained manifest. This checks the actual inherited formatting and tokenizer behavior, not just a similar-looking prompt template.

The new prompt lengths are **64–151 tokens**. Each request has `max_new_tokens=2048`, stop IDs `[151645,151643]`, and fits the existing 4,096-token context with block-size allowance. No truncation or length-driven reselection occurred. Requests contain no precomputed output tokens or finish reason: those must come from the new O/K evaluation.

### Prepared artifacts and limits

[NEW_PANEL.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/NEW_PANEL.json) is a flat list of 64 requests. Its SHA-256 is **`f4dfc911a28bf9555739883deeca2881423a901e10ba1b8be3e0ed2d2416c489`**. A deterministic rebuild reproduces this hash. The requests carry panel order, official source row, source ID, exact prompt IDs, stop IDs, output cap and older-execution evidence. Fifty-four selected questions are supported by the older 1,024-request serving reference, and ten by the disjoint 276-request objective-check evaluation.

[SELECTION_AUDIT.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/SELECTION_AUDIT.json) binds the amendment, exclusions, seed, selected order, source files, tokenizer, exact reconstruction results and panel hash. [SELECTED_QUESTIONS.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/SELECTED_QUESTIONS.json) contains only source questions and rendered prompts for review, with no answers or model outputs. [INVENTORY_AUDIT.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/INVENTORY_AUDIT.json) preserves the verified historical-use sets and supporting file hashes.

Both preparation scripts compiled and ran on CPU with offline tokenizer loading. [audit_inventory.py](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/audit_inventory.py) verifies identity/completion evidence; [prepare_panel.py](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/prepare_panel.py) reconstructs the old prompts and selects/renders the new panel. Their logs are retained beside them. This preparation emits no scientific generations and does not itself validate native GPU execution, change checkpoints, or launch the experiment.

“New” here means new to the current O/K study, exactly as amended. It does not mean globally unseen project data, unseen model pretraining data, a new training seed, or a proof against semantic paraphrase overlap. The historical inventory is bounded to retained accessible evidence; the completed-use lower bound and the current-study exclusion set are stated separately. Stage 1/project None is unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_fresh_confirmation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
