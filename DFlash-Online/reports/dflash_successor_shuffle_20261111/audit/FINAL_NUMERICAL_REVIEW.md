# Final numerical and interpretation review

ARS 3.24.2 experiment-agent / validate. Verification status: **ANALYZED**. Independent scalar arithmetic and compact-record checks; no native model rerun. Same-team implementation-author overlap is disclosed.

**TRUE supports informative successor-weight placement for this fork and permutation. Incremental verifier benefit versus K and all complete-request latency benefits remain unresolved.**

| Contrast | Call reduction [unadjusted 95% CI] | Complete-request time reduction [95% CI] |
|---|---|---|
| TRUE vs SHUFFLED | +0.595289% [+0.170964,+1.015408] | +1.245038% [−0.140408,+2.562571] |
| TRUE vs K | +0.363196% [−0.071936,+0.784491] | +0.357830% [−0.979443,+1.653258] |
| SHUFFLED vs K | −0.233483% [−0.653537,+0.195408] | −0.898393% [−2.274756,+0.451523] |

TRUE, SHUFFLED and K use 11,522, 11,591 and 11,564 calls. TRUE/SHUFFLED call wins/losses/ties are 111/78/67, with 189 gross calls saved and 120 added. Both timing sweeps favor TRUE (+1.949924%, +0.531610%), but the complete-time interval includes zero. The first-round prefix difference nets −14 tokens across all 256 requests; the separate matched-parent diagnostic does not identify the cause of the full-request gain.

The audit directly reads 1,536 raw measurement rows. All 52,224 raw/CSV fields, 129 report numerical fields, 12 bootstrap intervals, sweep totals and 161 publications match exactly. It uses one verified call count and a mean of two times per request, with 10,000 joint paired-request resamples (PCG64, seed 20261111). The separate root raw-validity audit reports 1,536 exact output/termination matches and native sidecars, 69,354 checked verification blocks, and exact repeat-count/first-block equality. No result-audit mismatch or protocol adjustment occurred.

Main work reconciles 256 new Adam updates and 512 presentations (128/256 per new arm), with zero new main TRUE updates. Qualification's three disposable updates and startup work remain separate. All 256 manipulation rows reconcile 18,452 changed bits versus 18,416.406527 uniform expected changes, conserving recorded class/mass invariants and producing a distinct export. Actuation alone does not establish efficacy.

Current runtime is 3,330.467275 s (55.507788 min). Fresh intrinsic training is K 328.064734 s versus SHUFFLED 400.984953 s: +72.920219 s (+22.227387%). This is an observed arm difference, not pure assignment cost; there is no fresh TRUE training comparison. Extra projection time of 50.372672 s and assignment time of 8.139749 s are nested within SHUFFLED training. The [cost-scope table](/home/minhao2/EAGLE_SPEC/research/dflash_successor_shuffle_20261111/analysis/COST_SCOPE.md) and [CSV](/home/minhao2/EAGLE_SPEC/research/dflash_successor_shuffle_20261111/analysis/COST_SCOPE.csv) separate all clocks. No training-inclusive speedup is demonstrated.

The complete 11-category fallacy review and scalar results are in [NUMERICAL_RESULT_NOTE.md](/home/minhao2/EAGLE_SPEC/research/dflash_successor_shuffle_20261111/analysis/NUMERICAL_RESULT_NOTE.md). Bounds: one continuing training seed, one fixed permutation and one prespecified 256-request panel. Intervals are unadjusted; crossing zero does not establish equivalence. This supports the incumbent placement rule against its shuffled control, not a new method advantage. The bounded run is finished; no automatic extension or stage advance.
