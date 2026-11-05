## Material Passport

- Origin Skill: ARS academic-research-suite 3.23.1, experiment-agent run workflow, bounded local data preparation.
- Date: 2026-10-27.
- Verification Status: VERIFIED for deterministic selection and exact prompt reconstruction through repeated local preparation; no model experiment executed.
- Scope: prepare the next three 128-request feedback windows for the authorized continuation of retained O/K to four total events.
- Scientific generations/model forwards/training steps: 0/0/0. No network or remote operation.

The next 384 eligible requests are exactly original training indices **128–511**. No candidate needed exclusion. They preserve the original chronological order, source identities and prompt token IDs. Both fixed 64-request evaluation panels are excluded by normalized question identity and exact prompt tokens. There is no data-preparation ambiguity or blocker.

| Continuation event | Original dataset indices, zero-based | Request IDs | Count |
|---|---|---|---:|
| event_002 | 128–255 | train_00128–train_00255 | 128 |
| event_003 | 256–383 | train_00256–train_00383 | 128 |
| event_004 | 384–511 | train_00384–train_00511 | 128 |

### Original stream and exclusion evidence

The authoritative complete stream is the 7,473-request [October 23 sealed manifest](/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/sealed/MANIFEST.json), SHA-256 `764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4`, checked against its retained seal. Its first 128 dictionaries exactly equal the [one-event manifest](/home/minhao2/EAGLE_SPEC/research/dflash_osd_one_event_20261025/sealed/MANIFEST.json), including input IDs, prompt tokens, output cap, event identifiers and collection flags. All 7,473 dataset indices, one-based source lines and opaque source identities match [TRAIN_REFERENCE.jsonl](/home/minhao2/EAGLE_SPEC/research/dflash_online_osd_rerun_20261023/preparation/TRAIN_REFERENCE.jsonl).

The original training stream uses official GSM8K training-row order; it is not shuffled. Preparation independently renders **all 7,473** official training questions and obtains exactly the stored prompt token IDs at every index. Thus the continuation order is verified across the complete source, rather than inferred from the initial 128 questions.

The development panel identities come from [CURRENT_STUDY_SOURCE_IDS.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/CURRENT_STUDY_SOURCE_IDS.json); the second panel is the unchanged [NEW_PANEL.json](/home/minhao2/EAGLE_SPEC/research/dflash_first_error_fresh_confirmation_20261027/inventory_audit/NEW_PANEL.json), bound to its selection-audit digest. Each contains 64 distinct official test rows, and the panels are disjoint. Selection scans the original training stream after index 127, skips a candidate if its NFKC/whitespace-normalized question matches either panel or earlier training, then keeps the next 384. The selected questions and prompt tokens are all unique, do not duplicate the initial 128 feedback inputs, and overlap neither evaluation panel. No new shuffle or selection seed is introduced.

### Pinned preprocessing and prior exposure

Data: `openai/gsm8k`, `main`, revision `740312add88f781978c0658806c59bc2815b9866`. Only the question column is read from the pinned source Parquet files. Train SHA-256: `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`. Test SHA-256: `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.

Tokenizer: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, loaded only from the existing local snapshot. Each unchanged source question is the sole user message, with no explicit system message or reasoning suffix. The template uses `enable_thinking=False`, `add_generation_prompt=True`, `tokenize=False`, followed by `encode(add_special_tokens=False)`. Runnable tokens are copied from the original sealed stream after independent exact reconstruction. Tokenizer/config file hashes and library versions are recorded in the selection audit.

Output cap remains 2,048 or EOS, stop IDs `[151645, 151643]`; training context remains 2,272 and serving context 4,096. Selected prompts span **37–183 tokens**, so every prompt fits the original context policy without truncation or changing the output cap.

**All 384 selected requests have verified older project execution.** This is permitted by the current authorization and is recorded per request; these are new feedback inputs for the current O/K trajectory, not globally unused questions. After fixing selection, the script checks older request identity, prompt hash and completion existence in [ONLINE_ROWS.json](/home/minhao2/EAGLE_SPEC/research/adaflash_reproduction_20260930/online_analysis_001/ONLINE_ROWS.json), against the [passed original OSD completion receipt](/home/minhao2/EAGLE_SPEC/research/adaflash_reproduction_20260930/online_analysis_001/receipts/online_reference_001/osd/RESULT.json), which reports all 7,473 chronological requests completed. Generated output content, answer correctness, draft acceptance, verification-call counts and latency do not determine inclusion or order. No older feedback response or feature is put in these manifests; the continuation runner must collect the newly authorized feedback separately.

### Deliverables and checks

- [CONTINUATION_REQUESTS.json](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/CONTINUATION_REQUESTS.json): flat list of 384 request dictionaries; SHA-256 `9831173ab0e86768b02de31d6193ba7cb8b92dc1ee03a22d0c056589cc9c8047`.
- [WINDOWS.json](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/WINDOWS.json): three records containing `event_index`, `event_id`, `request_count` and `requests`. Individual `WINDOW_002.json`, `WINDOW_003.json` and `WINDOW_004.json` files each hold the corresponding flat 128-request list.
- [SELECTED_IDENTITIES.json](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/SELECTED_IDENTITIES.json): source identity, normalized question hash, prompt hash and older-execution evidence per selected request.
- [SELECTION_AUDIT.json](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/SELECTION_AUDIT.json): complete selection rule, both exclusion panels, source/tokenizer pins, input/output hashes and zero-work accounting.
- [prepare_windows.py](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/prepare_windows.py): reproducible local preparation; existing artifacts are accepted only if byte-identical, and differing existing artifacts cause a failure.
- [CPU_TESTS.log](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/CPU_TESTS.log): all five focused tests pass, covering original ordering, both-panel exclusions, old/new duplicates, Unicode/whitespace equivalence, explicit exhaustion and replacement refusal.
- [PREPARATION.log](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/PREPARATION.log) and [REPRODUCTION.log](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/data_preparation/REPRODUCTION.log): initial preparation and a second complete identical preparation.

Reproduce from this directory:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 /home/minhao2/.venvs/dflash-specforge/bin/python prepare_windows.py
```

All writes are confined to this new data-preparation directory. Existing source, manifests, model artifacts and results are unchanged. This preparation does not validate resume state, sampling/anchor continuity or GPU execution; those remain the continuation driver's separate checks. Stage 1/project None remains unchanged.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
