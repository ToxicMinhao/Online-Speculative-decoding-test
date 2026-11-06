# Paper-defined AdaFlash-OPD versus K+ and ordinary OSD

## Material Passport

ARS experiment-agent run -> validate; status **valid_complete**. This is analysis of retained native records, not independent GPU replication.

Execution commit: `681336c5e5c5caa0aa67560e46ffe83ce81e9351`.

Configuration binding: `794063a2982b66c35be61d20e1b0ebcd1188a2d289c175b75d27568701855067`. Exact checkpoint paths, configuration paths and model/runtime pin locations appear in SUMMARY.json and the sealed source manifest.

Planned new training: A58 only, with 58 own-feedback events, 7,424 Adam steps and 14,848 response presentations. O58 and K+58 are retained controls measured freshly together with A58.

A uses 0.8 target-argmax CE + 0.2 entry-wise upper-clipped reverse KL at temperature 1. Each vocabulary contribution is capped at 0.01 before summation; negative entries remain unchanged. All valid proposals receive loss with the matched OSD sampler, position decay and response normalization. There is no K+ prefix/error mask and no adaptive-length head.

This implements the paper formula. The pinned released trainer instead normalizes summed vocabulary divergence before applying a position-level cap; it is not the loss tested here.

| Measurement | O58 | K+58 | A58 |
|---|---:|---:|---:|
| Verification calls | 44730 | 43267 | 45083 |
| Output tokens | 314847 | 314847 | 314847 |
| Complete-request seconds | 810.0691609550267 | 783.7418368533254 | 816.6397623233497 |
| Native-generation seconds | 757.8192295885334 | 734.9236338815342 | 766.1509236519536 |
| Cache-maintenance seconds | 37.876085744549826 | 35.452836548909545 | 36.02615935293337 |
| Accepted draft tokens / verification | 6.037044489157165 | 6.275498647930293 | 5.9818113257769 |
| Committed output tokens / verification | 7.0159400849541695 | 7.253172163542653 | 6.961005256970477 |

**KPLUS_vs_A:** complete-request reduction 4.028450% [95% paired CI [3.3288757870379886, 4.734649874868704]]; call reduction 4.028126%. Request call wins/regressions/ties 687/195/142; gross calls saved/added 2181/365.

**KPLUS_vs_O:** complete-request reduction 3.250009% [95% paired CI [2.567766073213097, 3.9416157844095734]]; call reduction 3.270736%. Request call wins/regressions/ties 633/250/141; gross calls saved/added 1917/454.

**A_vs_O:** complete-request reduction -0.811116% [95% paired CI [-1.3882000046228749, -0.2446954701126817]]; call reduction -0.789180%. Request call wins/regressions/ties 298/468/258; gross calls saved/added 500/853.

Primary decision: **favors_KPLUS**. The sign of K+/A complete-request-time interval controls this decision; call counts and native-generation time are separate outcomes.

Time totals sum each request's three-repeat mean. Count totals use one validated trajectory per request after repeat consistency. PCG64 seed 20261102, 10,000 paired-request bootstrap draws, linear 2.5/97.5 percentiles. Repeats are not independent requests.

A training/serving worker wall: 31265.087 seconds. TRAINING.json and EVENT_COSTS.csv keep training, teacher preparation, feature persistence, activation and publication costs separate; nested timers must not be added. TRAINING_EXPOSURE.csv records clipping activity and actual anchors/positions once per original forward.

LOSS_QUALIFICATION.json separately records the synthetic loss/gradient checks (12 tiny loss forwards/backwards on success, zero model forwards and zero optimizer steps). Four inherited native correctness completions are separate from timed serving.

Limitations: one training seed, a previously used train-disjoint panel, endpoint serving latency. Arms collected their own feedback; this is a recipe comparison rather than identical-feature loss isolation. Faster endpoint serving does not establish training-inclusive or asynchronous deployment speedup. No tuning, extra arm, adaptive length or follow-up run is authorized.
