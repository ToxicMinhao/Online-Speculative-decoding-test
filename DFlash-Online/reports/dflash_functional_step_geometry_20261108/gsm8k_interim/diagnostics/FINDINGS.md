# GSM8K: eight fixed same-state diagnostic windows

## Material Passport

ARS experiment-agent validate; **ANALYZED**, not an independent model replication. Execution commit `6440e8ff5af1a845f830d65c884f6711d7e4a687`; original events 5/6, windows 1/32/65/128. All eight finalized records, original A presentation/anchor bindings, four disjoint A/D identities per window, candidate coverage, repair identities, fixed teacher supports and pooled protected-KL arithmetic pass. Results contain 160 window/pair/candidate rows and 320 response/candidate rows. No prompts, model tensors or tokens enter these tables.

No new model load, forward, backward, optimizer step, feedback, teacher computation or serving generation was performed. Pure JSON arithmetic took about 0.17 seconds for the first complete analysis; six synthetic CPU tests passed. Source and analysis inputs are recorded separately. Author and source reviewer belong to the execution team; this is not external review.

The **strongest local evidence is suppressed repair/progress, with little or inconsistent useful-prefix preservation**. Committed FSG equals the norm-feasible candidate in all eight sampled windows. Relative to each window's ordinary native update from the same FSG pre-state, it reduces net matching-prefix gain in 7/8 A pairs and 6/8 D pairs (one D improvement, one D tie). These local observations are consistent with the unfavorable endpoint result, but cannot assign a percentage of endpoint latency or calls to any cause.

## Main comparison and denominators

All entries are sums of sampled-block occurrences over four fixed windows per event/role. The same output token can appear in overlapping blocks. They are neither distinct output tokens nor independent test requests. A is the fitting pair; D is buffered training data, not held-out evaluation.

| Event / role | Blocks | Old first errors | Old matching-prefix occurrences | Useful repairs native → FSG | Old prefix lost native → FSG | New earlier errors native → FSG | Net prefix gain native → FSG |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5 / A | 2,329 | 2,013 | 15,496 | 415 → 381 | 64 → 58 | 16 → 16 | +1,154 → +1,016 |
| 5 / D | 1,854 | 1,359 | 15,158 | 73 → 39 | 147 → 166 | 45 → 50 | +44 → −59 |
| 6 / A | 1,945 | 1,475 | 15,661 | 357 → 355 | 22 → 24 | 9 → 11 | +1,194 → +1,174 |
| 6 / D | 1,873 | 1,406 | 15,004 | 44 → 40 | 164 → 162 | 54 → 56 | −10 → −33 |

Native repair identities matter: event-5 FSG A retains 351/415 native repairs, loses 64 and adds 30; D retains 30/73, loses 43 and adds nine. Event-6 A retains 342/357, loses 15 and adds 13; D retains 34/44, loses ten and adds six. Event5/window65 contributes −93 of the −138 A net-progress difference and −71 of the −103 D difference; all other fixed windows remain included.

Protected old-draft KL is a two-response average, with the original within-response protection weights. The mean of the four pair values is:

| Event / role | Native protected KL | FSG protected KL | Teacher KL on prefix: native → FSG | Teacher KL on first error: native → FSG |
|---|---:|---:|---:|---:|
| 5 / A | 0.0606046 | 0.0419718 | 0.0812455 → 0.0842524 | 1.41601 → 1.44792 |
| 5 / D | 0.00532416 | 0.00248619 | 0.0671848 → 0.0693900 | 1.46451 → 1.48355 |
| 6 / A | 0.0862537 | 0.0658900 | 0.0637096 → 0.0641483 | 1.32100 → 1.32842 |
| 6 / D | 0.00800008 | 0.00607068 | 0.0839477 → 0.0840169 | 1.69069 → 1.69011 |

Teacher KL is pooled from its recorded numerator/denominator on each original fixed role. Its exact denominators and occurrences are in the complete tables; it is not an average of response means. Lower old-draft KL is partly imposed by FSG's gate and does not establish useful preservation. Event-6 D slightly better first-error teacher KL alongside worse net progress is also retained.

## Hypotheses A–F

**A — Over-preservation / suppressed learning.** Evidence supports a narrower statement than “FSG successfully preserves useful prefixes”: it reduces A protected drift in 8/8 windows while reducing A net progress in 7/8. Useful repairs and net gain both fall with lower KL in four of eight A and four of eight D windows. Old-prefix loss improves in some windows, but is unchanged or worse in others; event-5 D actually loses 19 more old-prefix occurrences. The local pattern is consistent with suppressing useful changes while optimizing the drift surrogate. It does not establish that every suppressed direction was useful or that useful behavior was preserved.

**B — Fitting versus off-fit transfer.** FSG still improves A over its pre-state in all eight windows; D net progress is negative in five and positive in three. However, this is not a clean “improved A, failed D” explanation for FSG's added mechanism: compared with the same-state ordinary update, FSG is worse on A in 7/8 windows and on D in 6/8. Only event6/window65 has better A net gain but worse D relative to native. Thus the record supports weak off-fit behavior together with damage to fitting progress; it does not isolate reference overfitting as the cause.

**C — Directional harm versus scalar size.** The same-state scalar counterfactual matches the feasible FSG norm to a maximum recorded relative error of `2.4268e-12`. Scalar preserves all native useful repair identities in these windows; its aggregate net progress is +1,166/+44 on event5 A/D and +1,194/−10 on event6 A/D. FSG is worse than that scalar in A net progress in 7/8 and D in 6/8. This supports harmful redirection on these measured states rather than attributing the local difference simply to a smaller overall update norm. It does not prove a full-trajectory causal decomposition. The scalar scores are **pre-guard local counterfactuals**, not the independently trained SCALAR arm: its local A guard accepts two and rejects six candidates (three numerical ambiguity, three increased risk), whereas FSG's committed candidate is accepted in all eight. A rejected diagnostic scalar would guard to native; its recorded candidate scores must not be mislabeled as a committed scalar trajectory.

**D — BF16 realization effects.** Raw-to-feasible scaling changes predictions despite only small changes in the total realized norm. Feasible-minus-raw net progress is +26/+21 for A in events5/6, but −18/−21 for D. It improves A net gain in six windows, worsens one and ties one; it worsens D in five and improves three. Therefore realization is consequential and mixed. Yet raw FSG already underperforms native in aggregate A and D in each event (raw net A/D: +990/−41 in event5 and +1,153/−12 in event6), so feasibility scaling alone cannot explain the local deficiency. Raw a=1 exceeds the strict native norm bound in these windows and is diagnostic, not a deployable accepted result.

**E — Protection weighting.** The uniform-prefix counterfactual improves D net gain over feasible FSG in four of eight windows and worsens it in four. Event5/6 D net gains are −42/−25 versus FSG −59/−33; useful repairs are 40/43 versus39/40. These limited local improvements are not only KL changes, but both event-level D results remain worse than native +44/−10. A net gain worsens by32/34, and uniform candidate guards/outcomes differ. This is mixed evidence, not support for adopting uniform weights.

**F — Sketch expressiveness.** Expanded-sketch feasible scores improve D net gain over FSG in five windows and worsen it in three; event5/6 D net gains are −36/−25, still below native +44/−10. D useful repairs are35/40 versus FSG39/40, so better aggregate net progress is not uniformly more repair. A net gain is lower than FSG by10/3. Counterfactual norms/effective direction also change; this does not isolate rank as the cause or justify selecting another variant. Raw and feasible expanded scores, as well as guard outcomes, are retained separately.

## Complete evidence and limits

- `results_001/window_pair_candidates.csv`: all160 rows, including raw/feasible/committed and candidate guard outcomes.
- `response_candidates.csv`: all320 response rows, role-specific teacher numerators, denominators and occurrence counts.
- `event_pair_candidates.csv`: all40 event/role/candidate summaries.
- `same_state_contrasts.csv` and `event_contrast_summary.csv`: all96 window contrasts and24 event/role summaries; no favorable-window selection.
- `INPUTS.json` and `SUMMARY.json`: consumed identities, exact checks, work, costs and missing-evidence status.

All four-response candidate sets and original identities are intact; no unavailable metric was substituted with zero. Large state snapshots and native tensors were not loaded. Parent-owned retrieval verifies the stable compact snapshot; retained runtime restoration records support state validity. No new model computation was used to independently reproduce predictions.

ARS interpretation coverage: all11 categories are addressed—dataset separation (Simpson), occurrence/request/seed units (ecological), fixed selected windows (Berkson), no accepted-window filtering (collider), all denominators (base rates), fixed endpoints/windows (regression to mean), complete invalid/missing accounting (survivorship), all candidates (look-elsewhere), unchanged settings (forking paths), local controlled contrasts versus endpoint cause (causality), and pre-state-before-update ordering versus later associations (reverse causality). No significance is manufactured from eight windows or overlapping blocks. MathQA and the full pilot conclusion remain outside this GSM8K diagnostic report.
