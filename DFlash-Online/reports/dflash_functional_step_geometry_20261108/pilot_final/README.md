# Final FSG pilot evidence index

Scientific execution: `6440e8ff5af1a845f830d65c884f6711d7e4a687`. These additive CPU analyses do not alter the executed implementation or its original measurements. The result is a valid negative for the tested FSG version; see [the complete report](../PILOT_REPORT.md). Verification status **ANALYZED**: retained native evidence and independently coded arithmetic were checked; no GPU replication was performed for this review.

## Endpoints and origins

Each endpoint directory contains `requests.csv` (256 paired requests), `repeats.csv` (1,536 physical executions), `methods.csv`, `comparisons.csv`, `sweeps.csv`, `SUMMARY.json`, `PAIRED.json`, `VALIDITY.json`, `RAW_AUDIT.json` and `SOURCE_CHECK.json`.

- [GSM8K per-request](endpoint_gsm8k/requests.csv), [all repeats](endpoint_gsm8k/repeats.csv), [comparisons](endpoint_gsm8k/comparisons.csv), [repeat totals](endpoint_gsm8k/sweeps.csv).
- [MathQA per-request](endpoint_mathqa/requests.csv), [all repeats](endpoint_mathqa/repeats.csv), [comparisons](endpoint_mathqa/comparisons.csv), [repeat totals](endpoint_mathqa/sweeps.csv).
- Authenticated original counts: [GSM8K KPLUS4](origin/gsm8k_KPLUS4_COUNTS.csv), [MathQA KPLUS4](origin/mathqa_KPLUS4_COUNTS.csv). Origin timing is not compared.
- [Independent endpoint arithmetic review](reviews/NUMERICAL_REVIEW.md), [completion and authenticity](reviews/COMPLETION_AND_AUTHENTICITY.json), [final report review](reviews/FINAL_REPORT_REVIEW.md).

The original tables use reduction = `100*(1−candidate/reference)`; negative reduction is worse. The main narrative uses **increase**, its negative, for readability and reverses both confidence bounds correctly. Calls use one validated trajectory after repeat consistency. Times average the two sweeps within request and then sum. Repeats do not multiply the request sample size. Matched accepted proposals and native `progress−1` acceptance are separate counters; committed progress excludes the initial seed. All capped and unfavorable requests remain included.

## Production and costs

- [Production, numerical and cost review](production/PRODUCTION_COST_WORK_REVIEW.md).
- [GSM8K all windows](production/gsm8k/PRODUCTION_WINDOWS.csv), [activity](production/gsm8k/ACTIVITY_SUMMARY.csv), [event costs](production/gsm8k/EVENT_COSTS.csv), [checkpoint identities](production/gsm8k/CHECKPOINT_IDENTITIES.json).
- [MathQA all windows](production/mathqa/PRODUCTION_WINDOWS.csv), [activity](production/mathqa/ACTIVITY_SUMMARY.csv), [event costs](production/mathqa/EVENT_COSTS.csv), [checkpoint identities](production/mathqa/CHECKPOINT_IDENTITIES.json).
- [Work totals](production/work/WORK_TOTALS.csv), [work by scope](production/work/WORK_BY_SCOPE.csv), [physical counters](production/work/PHYSICAL_COUNTERS.json), [cost components](production/work/COST_COMPONENTS.csv), [shared feedback costs](production/work/SHARED_FEEDBACK_COSTS.csv), [state continuity](production/work/STATE_CONTINUITY.json).
- [Geometry traces](production/scientific_trace/GEOMETRY_TRACE.csv), [per-response geometry](production/scientific_trace/RESPONSE_GEOMETRY_TRACE.csv), [distributions](production/scientific_trace/GEOMETRY_DISTRIBUTIONS.csv), [ordinary presentations](production/scientific_trace/ORDINARY_PRESENTATIONS.csv), [exposure](production/scientific_trace/EXPOSURE_BY_EVENT.csv), [FP32 aggregate reconciliation](production/scientific_trace/EXPOSURE_RECONCILIATION.csv).

Intrinsic training is event training minus the measured diagnostic observer. Realization/probe/hook times are nested; do not sum them again. Serving, checkpoint activation, loading, startup, preparation and publication remain separate. Peak allocation is the measured event peak, potentially including diagnostics/export. All cost units are seconds unless explicitly named otherwise; bytes and decimal GB differ from GiB.

## Diagnostics

- [Full interpretation](diagnostics/FINDINGS.md), [main native/FSG comparison](diagnostics/main_comparison.csv).
- [All 320 pair-candidate rows](diagnostics/combined_window_pair_candidates.csv), [640 response-candidate rows](diagnostics/combined_response_candidates.csv), [80 event-role aggregates](diagnostics/combined_event_pair_candidates.csv).
- [192 same-state contrasts](diagnostics/combined_same_state_contrasts.csv), [48 event contrasts](diagnostics/combined_event_contrast_summary.csv).
- [16-window work table](diagnostics/diagnostic_work_by_window.csv), [work audit](diagnostics/ALL_WORK.json), [CPU tests](diagnostics/CPU_TESTS.log), [production cross-review](production/DIAGNOSTIC_CROSS_REVIEW.json).

A is the fitting pair; D is disjoint buffered training data, **not held-out evaluation**. Prefix/error counts are overlapping sampled occurrences, not independent tokens or requests. Pair protected KL averages two response values; teacher-role KL pools numerator/denominator. Same-state scalar scores are pre-guard local candidates, not the trained SCALAR arm. Candidate identities use construction keys, not the historical erroneous `raw_fsg` description. No diagnostic alternative is selected as a successor.

## Reproduction and provenance

[INPUT_SOURCE_MANIFEST.json](INPUT_SOURCE_MANIFEST.json) contains snapshot/input hashes, execution binding, source verification and all six final checkpoint identities. [COMPLETION.json](COMPLETION.json) retains final supervisor/controller/postflight evidence. [Original automated recommendation](AUTOMATED_RECOMMENDATION.json) remains unchanged: its generic mixed fallback has no negative-result branch. The main report applies the original handoff's explicit valid-negative rule.

Private evidence locations:

- A800 original: `/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/pilot_r1_001`.
- Fenrir GSM8K snapshot: `/home/minhao2/EAGLE_SPEC/research/dflash_fsg_gsm8k_interim_20261109/evidence/pilot`.
- Fenrir final MathQA/global snapshot: `/home/minhao2/EAGLE_SPEC/research/dflash_functional_step_geometry_20261108/final_review_001/evidence/pilot`.

Private raw prompts/token arrays, features and model/optimizer tensors are excluded from Git. Compact records retain exact validity evidence. Event-5 bulk tensors were retired after successor authentication; their recorded hashes do **not** imply present tensor-level replay or continuation. Final event-6 recovery/exports remain retained on A800. All original failed qualification and preparation records and the earlier GSM8K report remain preserved.

Reusable CPU entry points under `analysis/` take explicit paths:

```bash
python analysis/endpoint.py --code /path/to/DFlash-Online --evidence /path/to/gsm8k/pilot --dataset gsm8k --output /new/gsm_endpoint
python analysis/endpoint.py --code /path/to/DFlash-Online --evidence /path/to/final/pilot --dataset mathqa --output /new/math_endpoint
python analysis/analyze_production.py --dataset mathqa --evidence /path/to/final/pilot --output /new/math_production
python analysis/diagnostics.py --dataset mathqa --evidence /path/to/final/pilot --output /new/math_diagnostics
python analysis/audit_work_costs.py --gsm8k /path/to/gsm8k/pilot --mathqa /path/to/final/pilot --output /new/work
python analysis/export_scientific_trace.py --gsm8k /path/to/gsm8k/pilot --mathqa /path/to/final/pilot --output /new/trace
python -m unittest discover -s analysis -p 'test_*diagnostics.py'
```

Endpoint requires NumPy and the pinned repository's standard-library-only validation/statistics modules; other primary analyzers use Python's standard library. The separately coded `reproduce_mathqa.py` and `verify_receipts.py` retain this private workspace's explicit paths as provenance; adapt only paths to replay elsewhere. No command above loads models or runs inference. Source integrity checks prevent using modified scientific helpers silently.

Review costs are CPU/file processing only; original model-work counts are unchanged. Minor analyzer schema/FP32 aggregation assumptions were corrected with failed CPU logs preserved in `production/`; no experiment tolerance, measurement, state or scientific code changed. Same-team author overlap is disclosed. Ten focused diagnostic tests pass; all earlier GSM8K scientific CSVs remain unchanged.
