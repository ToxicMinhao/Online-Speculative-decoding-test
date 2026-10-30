## Material Passport

- Origin Skill: ARS experiment-agent; Mode: preparation validation.
- Origin Date: 2026-10-30; Verification Status: ANALYZED.
- Version Label: ok_coding_preparation_v1.
- Scope: CPU-only implementation and self-check by `kw_eval`; parent agent performs separate review. No model weights, generation, target forward, backward or optimizer operation was used.

The fixed source and exclusions provide sufficient eligible requests. `preparation_001` now seals **512 training prompts, 128 evaluation prompts and 4,608 evaluation slots**. Data binding is `6def4832d12698fc2733e91c427db77558f3fe50e7f51063e9c2a11f9bc961af`. This completes the input preparation gate, not source publication, storage admission, initialization or serving qualification.

The retained source is `research/dflash_baseline_learning_curve_20261020/execution_001/data/source_artifacts/codealpaca_train.parquet`: **3,339,837 bytes**, SHA-256 `5a2fa0c510d42843292c583f357086abb59f95796e0f55727b97e449aa93bed9`, **20,022 original rows**. Its acquisition receipt, source_paths.json and prepared/source_bindings.json agree on `sahil2801/CodeAlpaca-20k`, revision `3bd2de8099ee8a140653ad9b9df783554d5efabb`, repository path `default/train/0000.parquet`. Only the parquet's instruction and input columns were read. All 20,022 assembled prompts and source positions exactly match the retained legacy `codealpaca:<index>` lookup catalog; that full catalog does not count as an allocation.

The raw-row partition is unchanged: **[0,18,998)** for training and **[18,998,20,022)** for evaluation. Selection uses source-order eligibility followed by isolated Python Random seeds 20261030/20261031, respectively. Training seeds 17/29 play no role in selecting data.

| Reservoir | Raw rows | Historical exclusions | Other exclusions | Eligible | Selected |
|---|---:|---:|---:|---:|---:|
| Training | 18,998 | 5,007 | 0 | 13,991 | 512 |
| Evaluation | 1,024 | 690 | 0 | 334 | 128 |

The retained approved prior-exclusion audit contributes 5,681 source rows. Its SHA-256 is `3f8c96f9c0710d0bcf611fea7bc2fe02bdd0d5e6e2ea21c09a233c79f1474cb5`; all ten underlying first-evidence file hashes still match its recorded history. The separately sealed 2,000-row September17 candidate pool is an actual reserved allocation and remains excluded. This differs from the complete 20,022-row source catalog, which is lookup-only.

The extended audit inspected 36,973 selection-carrier paths, representing 35,457 distinct contents. It includes manifests, panels, selections, requests, prompts, probes, reservations, windows, continuations and plans; output/answer/performance fields are ignored. Later October22/23 CodeAlpaca selections add exactly **16** new rows: 296, 3423, 7798, 7846, 8596, 12372, 18430, 18545, 19213, 19239, 19298, 19323, 19478, 19617, 19638 and 19827. They are retained in the approved projected-target manifest and the later online-OSD manifest. The final historical union is **5,697** rows. `HISTORY_INVENTORY.json` contains scan paths/hashes/matches; `EXCLUSIONS.json` binds each excluded source row to its reasons and evidence. No historical performance entered selection.

No NFC/whitespace duplicate key occurs within or across the exact source reservoirs. Normalization is used only for matching; prompts preserve source indentation/text. Every surviving candidate is rendered with the pinned Qwen3 tokenizer and a single user message, no system message, no-thinking generation prefix. All **14,325** candidate tokenizations exactly agree with the retained author helper. Their maximum lengths are 216 training/131 evaluation tokens, so the fixed 224-token cap removes no further surviving candidates. Selected maxima are **170/95**. Stop IDs remain `[151645,151643]`; the full 2,048-token output allowance remains available.

Ten focused CPU tests passed: answer non-use, invalid input and cap boundaries, source-order deduplication, cross-reservoir duplicate removal, normalization without text rewriting, historical-user/source matching, catalog versus reservation distinction, isolated selection RNG, explicit shortfall reporting, and exact schedule balance. Final self-checks verify all ten sealed artifact hashes, exact ordered selection, 640 disjoint source/prompt identities, window reconstruction and agreement with the campaign agent's separate data-binding/request/schedule contract. Each F/O/K method occupies each position once per request; pair precedence is 192/192 per cell. Preparation output and compact check results are in `PREPARATION_CPU.log` and `PREPARATION_REVIEW.json`; the ten-test execution was captured in this agent’s tool receipt.

Freshness means disjoint from the inventoried retained local project requests, including reservations. It does not establish absence from pretraining, inaccessible history or semantic near-duplicates. No outcome-dependent replacement or population relaxation occurred. Scientific source publication and complete resource admission remain required before any model work.


Implementation and CPU admission source: `5cf1c7e2e7c4a4d08d2225b58c56488578112daf`. No GPU scientific execution occurred.
