# Independent MathQA endpoint and completion review

## Material Passport

ARS experiment-agent validate; **ANALYZED**. This is an independent arithmetic implementation within the same research team, with prior implementation/test authorship overlap. It is not an external review or independent GPU reproduction. Source execution `6440e8ff5af1a845f830d65c884f6711d7e4a687`; binding `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`. No model/state tensors were loaded; zero remote/model/optimizer/generation work.

## Verdict

**MathQA endpoint evidence passes the independent audit and also shows a clear FSG regression against both controls.** The saved point estimates and all paired-bootstrap intervals reproduce exactly. SCALAR's small difference from KPLUS remains unresolved, not equivalent. These results agree in direction with the already-audited GSM8K endpoint; do not pool the datasets to obscure their separate effects.

The supervisor completed successfully at **2026-11-09 23:02:04 UTC**, with exit0, all18 tasks in the prescribed order (four shared collections,12 training events,two evaluations), no interrupted attempt or automatic retry, and successful15,122-file source/input postflight. This verifies control-plane completion; the separate work/numerical/state audits establish scientific work accounting and implementation fidelity.

## Reproduction and authenticity

`reproduce_mathqa.py` independently reconstructs pairing, native count sums, repeat means, aggregate ratios, signs and bootstrap. It reuses only the qualified raw-record validator, not the production statistics or reduction functions. All1,536 native records validate against the exact MathQA prompts, canonical output/termination, checkpoint identities and serving contracts. All3×256 repeat pairs have identical deterministic counters. Each arm has9 capped requests and124,961 output tokens; no invalid/incomplete request is silently omitted.

Every one of the **1,550 consumed new-snapshot input files** was rehashed and matched the finalized snapshot manifest. The MathQA snapshot contains6,345files/454,719,553B and was authenticated separately by the parent retrieval check. Origin records use their separately retained PR32 retrieval manifest and byte identities. Scientific checkpoint weights are not loaded or recomputed in this CPU review; their exact export identities are bound to the retained runtime receipts.

The independent arithmetic took **3.3133s**. Subsequent receipt/source review and writing time are not included in that timer. Raw inputs remain unchanged; all analysis outputs are additive under `final_review_001/endpoint_review`.

## Counts and timing

Times sum the two-repeat mean within each request. Counts use one validated trajectory per request.

| Method | Calls | Complete-request sum s | Mean/request s | Matched accepted/call | Committed progress/call |
|---|---:|---:|---:|---:|---:|
| KPLUS6 |16,782|292.830921527|1.143870787|6.443212966|7.430878322|
| FSG6 |17,667|307.871610601|1.202623479|6.070923190|7.058640403|
| SCALAR6 |16,776|292.950915087|1.144339512|6.446471149|7.433536004|

All arms commit124,705 output-progress tokens (124,961 output tokens minus256 initial seeds). Matched accepted totals KPLUS/FSG/SCALAR are108,130/107,255/108,146. Runtime native accepted totals are107,923/107,038/107,929. These counters are separate; stopping boundaries can make them differ.

| Contrast | Call increase %,95% interval | Time increase %,95% interval | Call wins/regressions/ties | Time wins/regressions/ties |
|---|---|---|---|---|
|FSG/KPLUS|5.273507 [4.614075,5.947773]|5.136305 [4.491144,5.808094]|33/206/17|42/214/0|
|FSG/SCALAR|5.311159 [4.589706,6.035964]|5.093241 [4.394412,5.802902]|36/200/20|50/206/0|
|SCALAR/KPLUS|−0.035753 [−0.371501,0.285050]|0.040977 [−0.297862,0.383270]|90/94/72|125/131/0|

Positive increase means worse. The machine tables also preserve the original reduction convention with negated point and reversed/negated interval endpoints. Gross calls saved/added: FSG/KPLUS66/951, FSG/SCALAR75/966, SCALAR/KPLUS159/153.

Request-time sweep totals KPLUS/FSG/SCALAR are292.705288/307.899698/292.954548s and292.956555/307.843523/292.947283s. Both show the FSG slowdown. Native-generation mean-sums are287.553184/302.379071/287.478056s; cache-maintenance mean-sums0.127348/0.127303/0.125118s. Components are inside request latency and are not a full kernel-time decomposition.

Bootstrap exactly uses10,000 paired-request draws, PCG64 seed20261108, linear percentiles2.5/97.5, and one shared request-index matrix across every comparison and metric. Repeats are averaged first. These are unadjusted intervals for this reused development panel and one continuing training seed; they do not establish seed robustness or population-wide generalization.

## Authenticated common-origin counts

All512 retained MathQA KPLUS4 repeat records authenticate against the original PR32 retrieval manifest. Current/prior prompts, canonical outputs, termination, serving configuration, model/runtime pins and origin export identity match. Exact origin weights:2,097,259,104B, SHA256`931e46620d8fc930ea0c22fcdf6af30dd16e16445f8141474ecfa4e48746dff7`.

KPLUS4 used **17,074 calls**. Relative to that origin:

| Endpoint | Call reduction % | Wins/regressions/ties | Gross saved/added |
|---|---:|---|---|
|KPLUS6|1.710203|139/83/34|509/217|
|FSG6|−3.473117|64/169/23|197/790|
|SCALAR6|1.745344|140/84/32|499/201|

Thus FSG actually deteriorates from the common starting checkpoint on MathQA, as it did on GSM8K, rather than merely improving less than KPLUS. This is drafting-efficiency deterioration under identical final tokens, not a target-answer-quality claim. **No historical latency is used.**

## Interpretation and safeguards

The original handoff's success condition is not met: FSG must improve event6 latency over both controls on both datasets, with supported intervals and useful progress. Its two-dataset endpoint direction is instead clearly unfavorable. Combine this endpoint evidence with the independently reviewed nontrivial-update/state/numerical receipts before labeling the implementation a valid negative. Do not reinterpret `statistics.recommend()`'s generic `inconclusive_or_mixed_pilot` fallback as evidence that the clear FSG/controls regressions are statistically unresolved; the source has no dedicated negative branch.

The unresolved SCALAR/KPLUS contrast does not establish equivalence or prove a beneficial smaller-step mechanism. Local protection/repair diagnostics can explain plausible associations, not uniquely identify the cause of endpoint harm. No longer run, variant or precision change follows automatically.

All11 ARS safeguards checked: separate dataset strata (Simpson); paired requests versus occurrences/seeds (ecological); fixed reused panels/diagnostic selection (Berkson); no conditional exclusion by guard/activity (collider); all denominators/base rates; fixed endpoints rather than favorable step selection (regression-to-mean); complete validity/missingness (survivorship); unadjusted multiple contrasts; unchanged prespecified analysis (forking paths); endpoint versus mechanism-cause distinction; and preserved temporal pre/post ordering. No scientific-source or input changes were needed.

Outputs: `ENDPOINT_INDEPENDENT.json`, `PER_REQUEST.csv`, `ALL_REPEATS.csv`, `COMPARISONS.csv`, `KPLUS4_COUNTS.csv`, `INPUT_IDENTITIES.json`, `COMPLETION_AND_AUTHENTICITY.json`, `RUN.log` and `COMPLETION.log`.
