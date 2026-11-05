# KP v1 final interpretation review

**KP v1 does not meet the prespecified useful-improvement criterion.** It saves 56 calls versus K overall, but the effect is small and changes direction between the two fitted seeds. The simpler lower-learning-rate control, K_HALF, gives the better serving result at lower measured training cost.

Execution: `d10521a1aa7d256bb75b37a6aae07586ae7cc8b3`. This is an ARS review of retained records and tables, not new GPU execution. The reviewer authored parts of the reporting code; separate raw-statistics reconstruction belongs to the objective-review agent. Original reports, code and seals were not changed.

## What the endpoint shows

| Comparison | Calls reduced | Paired 95% interval | Request time reduced |
|---|---:|---:|---:|
| KP / K | +0.3435% | [+0.0182, +0.6919]% | +0.1770% |
| KP / F | +0.3252% | [-0.6386, +1.2365]% | +0.3620% |
| KP / K_HALF | -2.4078% | [-3.2861, -1.6281]% | -2.2350% |
| K_HALF / K | +2.6866% | [+1.9233, +3.5338]% | +2.3592% |
| K_HALF / F | +2.6687% | [+1.8411, +3.5566]% | +2.5402% |
| K / F | -0.0184% | [-1.0284, +0.8982]% | +0.1853% |

KP/K has a positive combined lower bound, but its **+0.3435%** point estimate misses the **0.5%** target. Seed 17 is **−0.0491%** and seed 29 **+0.7350%**. KP/F also fails the required positive lower bound. Thus three required conditions fail. The positive pooled interval must not be described as supported useful improvement. KP/K request-time uncertainty also spans zero (−0.3119% to +0.6729%).

K_HALF improves over K by **2.3464% / 3.0258%** and over F by **2.4663% / 2.8712%** in seeds 17/29. KP is worse than K_HALF in both seeds. Their combined interval is wholly below −0.5 percentage points, so this is **not practical equivalence**. Secondary intervals remain descriptive and unadjusted.

The sample is **128 paired request identities**, each carrying both fixed fitted seeds and all methods. The 256 seed/request outcomes supply descriptive win/loss counts; they are not 256 independent request samples. Three timing repeats are repeated measurements, not additional requests. Bootstrap intervals condition on this panel and these two trajectories; they do not establish training-seed robustness.

## Gain retention and cost

Relative to F, K saves 369 calls on improved outcomes and adds 372 on harmed outcomes. KP retains **301/369 (81.6%)** of those same gains, loses 68 and adds 91 different gains. It prevents 112 harmful calls but adds 79, leaving a net **56-call improvement over K**: `91 − 68 + 112 − 79 = 56`. Seed 17 adds four calls; seed 29 saves 60.

K_HALF retains less of K’s original gain (**261/369; 70.7%**) but adds 299 gains, prevents 271 harmful calls and introduces only 24. Its net improvement is **438 calls**: `299 − 108 + 271 − 24 = 438`. Higher retention alone therefore does not make KP better. These identities describe endpoint differences; they do not prove a preservation mechanism.

Across both seeds, KP training takes **3695.3 s** versus **2983.4 s** for K_HALF: **711.9 s extra (+23.86%)**. Peak allocated memory rises from about **19.29 to 21.24 GiB**. Reference and extra-loss subtimers are already included in training; setup is included in loading. Original K costs are historical and unmatched. No training-inclusive or asynchronous deployment speedup is established.

The protection term was active on **151,802 / 3,470,177 protected occurrences (4.37%)**, with 112,870 currently incorrect occurrences and 6,461 wrong ties. This rules out an entirely dormant term, but neither these changing training samples nor loss values measure a matched-K preservation counterfactual.

## Validity and ARS fallacy scan — 11/11

The tables contain all **3,072 valid executions**, 256 paired outcomes and 4,096 original training-response observations. Repeat-consistent counts, paired timing means and all 512 candidate gain/harm identities cross-check. All 2,048 main optimizer steps completed with no unknown work; qualification is separately charged. No evaluation output hit the cap. Native token equality is retained execution/audit evidence, not something re-executed by this review.

| Check | Interpretation boundary |
|---|---|
| Simpson’s paradox | Opposite seed directions are visible; pooling must not hide them. |
| Ecological fallacy | 128 joint units; no every-request or population-seed claim. |
| Berkson’s paradox | Restricted short-prompt panel limits scope; no spurious association is established. |
| Collider bias | No restriction to activated, short, improved or completed-only subsets. |
| Base-rate neglect | Explicit outcome, protected-occurrence and cap denominators are retained. |
| Regression to the mean | No extreme-result subset or historical-versus-fresh timing claim. |
| Survivorship bias | All planned executions and regressions remain visible; no missing cells. |
| Look-elsewhere effect | All six comparisons reported; secondary intervals unadjusted. |
| Garden of forking paths | Fixed panel, constants and endpoint; no result-driven tuning. |
| Correlation versus causation | Endpoint accounting does not identify the causal mechanism. |
| Reverse causality | Sealed inputs precede outcomes; later associations remain descriptive. |

Scope remains four-event adaptation to fixed, shared O-generated feedback, greedy decoding and the selected coding population. It does not test independent online deployments, domain switches, stochastic decoding or an O comparison on this panel.

**One recommendation:** Defer KP v1 and provisionally prefer the already tested K_HALF recipe for this fixed coding setup; do not initiate further tuning or experiments from this result.

Sources: [original measured report](MEASUREMENTS_REPORT.md), [summary and intervals](SUMMARY.json), [paired requests](PAIRED_REQUESTS.csv), [all repeats](ALL_REPEATS.csv), [compact event/pass training aggregates](TRAINING_EVENT_PASS_SUMMARY.json), and [cost scopes](COST_SUMMARY.json). The full training aggregates, fixed handoff and prewritten checklist remain in the private evidence tree named by [provenance](PROVENANCE.json); the compact training file publishes only the event/pass subset. Exact original source identities and full checks are in [the review JSON](FINAL_INTERPRETATION_REVIEW.json). Only this paragraph's publication links and scope labels differ from the original retained review.
