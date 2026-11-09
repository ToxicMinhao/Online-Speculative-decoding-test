# GSM8K interim evidence package

CPU-only analysis of finalized event-6 GSM8K results and events-5–6 training records. Execution remains `6440e8ff5af1a845f830d65c884f6711d7e4a687`; no experimental Python file was changed. The publication commit is the Git commit introducing this directory, distinct from the execution commit. MathQA and the full two-dataset review remain pending.

[Main report](../GSM8K_INTERIM_EVIDENCE_REPORT.md) · [Source/input manifest](INPUT_SOURCE_MANIFEST.json).

## Endpoint

- [methods.csv](endpoint/methods.csv): three methods; pooled counters, within-request timing means, native/cache components, output/cap totals.
- [comparisons.csv](endpoint/comparisons.csv): three prespecified contrasts, original reduction sign and correctly transformed increase sign, paired intervals, request wins/regressions/ties and gross calls saved/added.
- [requests.csv](endpoint/requests.csv): all 256 paired requests, one deterministic trajectory each.
- [repeats.csv](endpoint/repeats.csv): all 1,536 executions, including validity and canonical termination; token sequences remain private.
- [sweeps.csv](endpoint/sweeps.csv): six method/sweep totals.
- [SUMMARY.json](endpoint/SUMMARY.json), [VALIDITY.json](endpoint/VALIDITY.json): unchanged original statistical implementation and structural validity.
- [PUBLIC_COPY_CHECK.json](endpoint/PUBLIC_COPY_CHECK.json): published analyzer reproduces all scientific files byte-for-byte.

`accepted_draft_tokens` means the number of actually committed matching proposals, clipped at the saved stopping boundary. `native_accepted_draft_tokens` is the runtime's committed-progress-minus-one counter per block. `committed_progress` includes matching proposals plus any committed verifier token, excludes the request's initial seed, and is not interchangeable with either accepted counter. Timings use the original complete-request boundary; native/cache measurements are included components, not additional expenses.

## Production and diagnostic tables

- [Production review](production/PRODUCTION_REVIEW.md), [all 512 filtered windows](production/PRODUCTION_WINDOWS.csv), [activity](production/ACTIVITY_SUMMARY.csv), [108 per-event/combined distributions](production/SCALAR_DISTRIBUTIONS.csv), [full summaries](production/PRODUCTION_SUMMARY.json).
- [Six event costs](production/EVENT_COSTS.csv), [cost summaries and scopes](production/TRAINING_COST_SUMMARY.json), [checkpoint identities](production/CHECKPOINT_IDENTITIES.json), [portable replay check](production/REUSABLE_REPLAY_CHECK.json). Nested costs must not be summed.
- [Diagnostic findings](diagnostics/FINDINGS.md), [A–F hypotheses](diagnostics/HYPOTHESES.csv), [160 window/pair/candidate rows](diagnostics/window_pair_candidates.csv), [320 response rows](diagnostics/response_candidates.csv), [40 event/pair/candidate summaries](diagnostics/event_pair_candidates.csv), [96 same-state contrasts](diagnostics/same_state_contrasts.csv), [24 event contrast summaries](diagnostics/event_contrast_summary.csv), [portable replay check](diagnostics/PUBLIC_COPY_CHECK.json).

Diagnostic denominators remain separate: useful repairs / original first-error blocks; retained repairs / native useful repairs; lost prefix occurrences / old matching-prefix occurrences. Protected old-draft KL averages two separately normalized response values. Teacher KL pools the saved role-specific numerator/denominator. All sampled-block occurrences retain overlaps; they are not independent requests or distinct output positions. Empty or missing values remain explicitly empty/inapplicable, never replaced with a success-indicating zero.

Raw FSG is the unadjusted `a=1` BF16 endpoint; feasible FSG is after R1 norm feasibility but before its final KL guard; committed FSG is the actual endpoint. Same-state scalar/uniform/expanded scores are counterfactual measurements, with their own guard outcomes. They do not constitute independently trained arms. Uniform and expanded raw variants are separately retained. Pair D is buffered training data.

## Origin and review

- [KPLUS4 validation](origin/ORIGIN_VALIDATION.json), [paired count table](origin/KPLUS4_COUNTS.csv), [origin counters/progress](origin/ORIGIN_PER_REQUEST.csv). Historical times are excluded.
- [Independent CPU numerical review](reviews/NUMERICAL_REVIEW.md) and [source/cross-review](reviews/CROSS_REVIEW.md). Reviewers are members of the implementation team, with authorship overlap; independent arithmetic is not an external model replication. All 11 ARS statistical-fallacy checks are addressed.

## CPU reproduction

Keep the private snapshot outside Git. Its archive SHA256 is `0fcca11bc286dea8454eef08fa7acadde428737e0dea2fd143c024b91c66e99e`; input-file identities are in the manifest. The evidence directory must preserve paths relative to the recorded pilot root (for example `sealed/PLAN.json` and `campaign_001/evaluation/gsm8k`). The endpoint analyzer authenticates the original executed Python source. NumPy is required; it imports the retained archive validator but constructs no model or serving process.

From the `DFlash-Online` directory, using the qualified local analysis environment:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B reports/dflash_functional_step_geometry_20261108/gsm8k_interim/analysis/endpoint.py \
  --code "$PWD" --evidence /path/to/private/pilot-snapshot --output /path/to/new-endpoint-analysis
python -B reports/dflash_functional_step_geometry_20261108/gsm8k_interim/analysis/production.py \
  --evidence /path/to/private/pilot-snapshot --output /path/to/new-production-analysis
python -B reports/dflash_functional_step_geometry_20261108/gsm8k_interim/analysis/diagnostics.py \
  --evidence /path/to/private/pilot-snapshot --output /path/to/new-diagnostic-analysis
python -B -m unittest discover \
  -s reports/dflash_functional_step_geometry_20261108/gsm8k_interim/analysis -p 'test_diagnostics.py'
```

The private analysis used `/home/minhao2/.venvs/dflash-specforge/bin/python`. Portability checks reproduced every scientific CSV and all scientific JSON fields; only execution-cost fields change between CPU runs. Six synthetic diagnostic-analysis tests pass. No training/model tests or new GPU work were added.

## Analysis costs and preserved limits

Recorded CPU work: snapshot assembly 3.9627 s; retrieval verification 2.2117 s; endpoint 2.9274 s plus portable replay 2.9260 s; independent endpoint/origin arithmetic 2.415 s; final production assembly 0.3170 s plus portable replay 0.3141 s (one earlier successful table pass also retained privately); three diagnostic assemblies total 0.5082 s. Test, source-inspection, other file-checking and narrative-review time are not all instrumented and these figures are not a total personnel/campaign cost. Primary endpoint peak host RSS 806,936 KiB; diagnostic peak 63,120 KiB.

The 41.78 MB compact snapshot was transferred while MathQA was training, outside live timed serving. Original raw artifacts, historical qualification failures and the full-pilot report destination are preserved. No model/recovery tensors or private prompts/output tokens are included here. Zero additional model loads, feedback, teacher preparation, optimizer steps, forwards/backwards, serving generations and diagnostic windows. No new experiment, retuning, pause or restart was performed.
