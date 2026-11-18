# PR53 final result review

Material Passport: ARS experiment validation; retained native records; CPU-only
independent reduction and interpretation. Verification Status: **ANALYZED**.
This review is an assistant-team audit, not external peer review or a new model
reproduction.

The fixed campaign is complete. Independent reconstruction passes **2,321,962
checks with zero errors**. All 3,072 scored executions, 96 final warmups and 516
canonical reference records are present and agree on exact tokens and termination.
Native count consistency, balanced order, fixed checkpoint identities and the
10,000 paired bootstrap draws match the executed summaries. The independent
reductions were saved before those production summaries were opened.

| C versus native K+ | GSM8K | MathQA |
|---|---:|---:|
| Verification calls, native / C | 11,251 / 11,231 | 17,952 / 17,935 |
| Net calls saved | 20 | 17 |
| Request time reduction | +0.097536% | +0.084990% |
| Paired 95% interval | −0.330483% to +0.524245% | −0.263230% to +0.436518% |
| Native accepted-proposal gain | +0.208931% | +0.109677% |
| Paired 95% interval | −0.280138% to +0.706011% | −0.299371% to +0.526406% |
| Request call wins / regressions / ties | 95 / 85 / 76 | 91 / 90 / 75 |
| Gross calls saved / added | 153 / 133 | 179 / 162 |

Both datasets have favorable but small point estimates. **Neither demonstrates
an incremental C advantage:** every prespecified interval crosses zero and is
`DIRECTION_UNRESOLVED`. The timing intervals remain compatible with C being
0.330483% slower on GSM8K or 0.263230% slower on MathQA. Crossing zero does not
certify equivalence, non-inferiority or performance preservation. No 0.5% gate or
S comparator applies to this handoff.

The original qualification commit is `31b36fd27976fb46bb6754d8b8b7e769c3d6e5f6`;
actual production is `0dc9538e8f445faa8bc7471dc9f82d885e98d529`. The original
resource stop remains preserved. Only elapsed time was subsequently relaxed;
qualification and four initial warmup references were reused. The original clock
was retained, not reset. The production phase took 9,162.809139 seconds and all
occupied GPU time including original qualification was 12,018.066354 seconds.
Work totals are exact; no failed, unknown or pending work remains.

## Eleven-category interpretation audit

All **11/11** categories were examined.

1. **Simpson's paradox:** datasets and their predeclared block/repeat views remain
   separate. No cross-dataset pooling or favorable-block selection supports the
   conclusion. All three timing points are slightly positive in each dataset,
   which does not make their request intervals positive.
2. **Ecological fallacy:** aggregate workload savings are not treated as benefits
   for every request. The reported wins/regressions/ties show mixed individual
   outcomes.
3. **Berkson's paradox:** panels were selected from input eligibility before
   endpoint generation. GSM8K is new to this allocation-C family with older
   exposure disclosed; MathQA is unexposed only within the enumerated inventories.
   Neither supports claims about universal or pretraining freshness.
4. **Collider bias:** no selection by generated length, speed, success or capped
   termination. The one GSM8K and nine MathQA capped requests remain in analysis.
5. **Base-rate neglect:** all paired denominators, gross saved/added counts and
   ties are included. No zero-call requests occurred. Native acceptance is
   progress minus calls, distinguished from stop-clipped matching counts.
6. **Regression to the mean:** the contemporaneous native control is used; earlier
   favorable C screens are not a fresh control or evidence of a learning curve.
   Changed continuation RNG and changed inputs prevent attributing differences
   from historical screens to either factor alone.
7. **Survivorship bias:** all planned endpoint executions pass raw validity;
   none was dropped or replaced because of outcomes. Missing/invalid execution
   would have blocked this full comparison.
8. **Look-elsewhere effect:** both datasets and all prespecified metric directions
   are reported. Intervals are individual paired request intervals, not a
   simultaneous familywise guarantee. Calls and acceptance are mathematically
   linked under identical outputs and do not multiply independent evidence.
9. **Garden of forking paths:** fixed panels, ordering, mechanisms, seeds, ratio
   reductions and intervals were retained. The documented elapsed extension is
   an operational change; it did not change the scientific test or erase the
   original stop. No alternative subset or threshold was chosen after results.
10. **Correlation versus causation:** these are controlled endpoint comparisons
    of the prescribed recipes. The arms' own-feedback trajectories can differ;
    this does not isolate confidence, position shape or coefficient allocation
    as the causal explanation, nor establish training-inclusive speedup.
11. **Reverse causality:** training preceded final evaluation; evaluation outputs
    were not training feedback. Three timing repeats are not three independent
    training seeds. Uncertainty remains conditional on these fixed endpoints
    and inputs.

## Storage and publication review

The total live-allocation cap and shared-volume reserve pass. A separate
compact-evidence allowance was exceeded: **1,032,792,335 logical bytes** versus
1,000,000,000, an excess of **32,792,335 bytes (3.2792335%)**. The archive is
94,891,583 bytes; the authenticated local delivery occupies 1,070,616,576 bytes
after inode deduplication. These are distinct quantities, and local allocation
is not an A800 allocation measurement. The evidence must remain preserved;
compression or deletion cannot retroactively make the raw allowance compliant.

The reviewed draft reports this deviation, distinguishes original and resumed
execution, preserves historical results and correctly states the negative
incremental finding. Its numeric table agrees with the independently computed
values to the displayed precision. No scientific interpretation blocker was
found. Scalar-only audit output is prepared separately for publication; private
raw prompts, responses, token arrays, absolute checkpoint paths and consumed-file
inventories are not included in that copy.

Native K+ remains primary. C remains a candidate for contribution assessment with
performance preservation unresolved under the adopted policy. No automatic
tuning, continuation, new panel or method promotion follows from completion.
