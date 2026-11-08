# K+ role-dependent loss early-adaptation pilot

## Material Passport

ARS experiment-agent run -> validate; status **valid_complete**. All six trajectories and both planned checkpoints are reported. The audit recalculates retained native evidence on CPU; it is not independent GPU replication.

Execution commit: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`; seal `72d52c0f3554c184d9ffd5b51990586fd24d8062cfd8113d3603288ca607bea2`.

KPLUS retains its qualified forward KL. MIX uses the selected-position 0.8 teacher-argmax CE / 0.2 released-position-clipped reverse-KL mixture. ROLE uses forward KL on matching prefixes and the mixture at the first error. Selection, redistributed weights, native BF16 training and serving are held fixed. Each arm collects its own feedback from an independent released initialization.

This tests early adaptation after four events, not the ranking after a 58-event campaign or the full AdaFlash system. Both development panels were previously used. Stage 1 / project None remains unchanged.

## gsm8k: event 2

| Arm | Verification calls | Request seconds | Native generation seconds | Matched / call | Progress / call | Cap requests |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS | 12307 | 222.950972 | 209.145851 | 5.539449 | 6.519623 | 1 |
| MIX | 12314 | 223.772519 | 208.582243 | 5.535488 | 6.515917 | 1 |
| ROLE | 12315 | 221.876236 | 208.384074 | 5.535120 | 6.515388 | 1 |

| Contrast | Request-time reduction % [95% CI] | Call reduction % [95% CI] | Call wins / regressions / ties | Time wins / regressions / ties |
|---|---:|---:|---:|---:|
| ROLE_vs_KPLUS | 0.482050 [-0.7079254888639164, 1.6590712153807852] | -0.065004 [-0.6128588784641059, 0.49488851862159894] | 100 / 113 / 43 | 123 / 133 / 0 |
| MIX_vs_KPLUS | -0.368488 [-1.7155765720901939, 0.9242315620749915] | -0.056878 [-0.7208747731347287, 0.5957035297558001] | 103 / 106 / 47 | 123 / 133 / 0 |
| ROLE_vs_MIX | 0.847415 [-0.32211299035483476, 2.0214783104406777] | -0.008121 [-0.6446529634151738, 0.6256032045366545] | 112 / 100 / 44 | 135 / 121 / 0 |

## gsm8k: event 4

| Arm | Verification calls | Request seconds | Native generation seconds | Matched / call | Progress / call | Cap requests |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS | 12116 | 220.050538 | 205.423940 | 5.642209 | 6.622400 | 1 |
| MIX | 12014 | 218.656423 | 204.177865 | 5.698851 | 6.678625 | 1 |
| ROLE | 12134 | 221.359521 | 205.839502 | 5.632603 | 6.612576 | 1 |

| Contrast | Request-time reduction % [95% CI] | Call reduction % [95% CI] | Call wins / regressions / ties | Time wins / regressions / ties |
|---|---:|---:|---:|---:|
| ROLE_vs_KPLUS | -0.594856 [-1.6329396500081517, 0.4045893460396848] | -0.148564 [-0.7525680729016326, 0.4295546433294054] | 103 / 100 / 53 | 113 / 143 / 0 |
| MIX_vs_KPLUS | 0.633543 [-0.5310965584260752, 1.779601795117744] | 0.841862 [0.13902353048527966, 1.5365655764155681] | 122 / 83 / 51 | 139 / 117 / 0 |
| ROLE_vs_MIX | -1.236231 [-2.2876041827941016, -0.17763088140546907] | -0.998835 [-1.6814580461073658, -0.3319946761368568] | 84 / 133 / 39 | 108 / 148 / 0 |

## mathqa: event 2

| Arm | Verification calls | Request seconds | Native generation seconds | Matched / call | Progress / call | Cap requests |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS | 17243 | 312.578166 | 297.560325 | 6.246535 | 7.232210 | 9 |
| MIX | 17309 | 314.247945 | 297.985558 | 6.218730 | 7.204633 | 9 |
| ROLE | 17317 | 314.044887 | 298.071044 | 6.215280 | 7.201305 | 9 |

| Contrast | Request-time reduction % [95% CI] | Call reduction % [95% CI] | Call wins / regressions / ties | Time wins / regressions / ties |
|---|---:|---:|---:|---:|
| ROLE_vs_KPLUS | -0.469233 [-1.4367869204854455, 0.46243276410696493] | -0.429160 [-0.9031960691428875, 0.031054844153163598] | 86 / 118 / 52 | 129 / 127 / 0 |
| MIX_vs_KPLUS | -0.534195 [-1.6847127346449946, 0.5611917540669561] | -0.382764 [-0.96143400345782, 0.18435085034553816] | 95 / 113 / 48 | 116 / 140 / 0 |
| ROLE_vs_MIX | 0.064617 [-0.9145175995207777, 1.073822360715211] | -0.046219 [-0.5658610830478322, 0.47868170315071673] | 109 / 104 / 43 | 128 / 128 / 0 |

## mathqa: event 4

| Arm | Verification calls | Request seconds | Native generation seconds | Matched / call | Progress / call | Cap requests |
|---|---:|---:|---:|---:|---:|---:|
| KPLUS | 17074 | 308.482511 | 294.518147 | 6.316856 | 7.303795 | 9 |
| MIX | 17227 | 311.653758 | 296.184096 | 6.251640 | 7.238927 | 9 |
| ROLE | 17158 | 309.686315 | 294.766128 | 6.280336 | 7.268038 | 9 |

| Contrast | Request-time reduction % [95% CI] | Call reduction % [95% CI] | Call wins / regressions / ties | Time wins / regressions / ties |
|---|---:|---:|---:|---:|
| ROLE_vs_KPLUS | -0.390234 [-1.2420785115672421, 0.4694125121489888] | -0.491976 [-1.000060575832813, 0.006049736449033609] | 99 / 110 / 47 | 115 / 141 / 0 |
| MIX_vs_KPLUS | -1.028015 [-1.9303312177884224, -0.19220215845017402] | -0.896099 [-1.4423946901449214, -0.3761362201343827] | 95 / 118 / 43 | 111 / 145 / 0 |
| ROLE_vs_MIX | 0.631291 [-0.310381785345708, 1.5932606399140323] | 0.400534 [-0.14587294467868106, 0.9397978444385736] | 114 / 100 / 42 | 126 / 130 / 0 |

## Fixed-context explanation

| Dataset / arm | Initial / event2 / event4 prefix tokens | Event2 / event4 useful repairs | Broken prefixes event2 / event4 | Persistent repairs | Event2 gained / persisted / lost tokens |
|---|---:|---:|---:|---:|---:|
| gsm8k/KPLUS | 819 / 846 / 869 | 31 / 34 | 25 / 20 | 20 | 109 / 64 / 45 |
| gsm8k/MIX | 819 / 845 / 875 | 29 / 31 | 29 / 25 | 20 | 133 / 95 / 38 |
| gsm8k/ROLE | 819 / 826 / 839 | 28 / 32 | 25 / 24 | 20 | 94 / 60 / 34 |
| mathqa/KPLUS | 992 / 1041 / 1074 | 28 / 34 | 21 / 20 | 22 | 117 / 89 / 28 |
| mathqa/MIX | 992 / 1014 / 1021 | 34 / 36 | 26 / 26 | 26 | 122 / 97 / 25 |
| mathqa/ROLE | 992 / 1042 / 1045 | 32 / 30 | 21 / 20 | 20 | 111 / 77 / 34 |

A useful repair retains the entire earlier canonical prefix. Diagnostics use identical fixed contexts, target features, anchors and masks; RNG and model modes are restored. They are local explanations, not serving speedup estimates or proof of causality. TRAINING.json retains additive role masses, selected fractions, D/Z, priority fractions, component losses, cap counts, teacher/canonical discrepancies and numerical counters; ratios pool raw numerators and denominators. EVENT_COSTS.csv retains pre-clipping norms and clipping counts.

## Work and costs

| Trajectory | Events complete | Work status | Optimizer windows | Clipped windows |
|---|---:|---|---:|---:|
| gsm8k/KPLUS | 4 | passed | 512 | 0 |
| gsm8k/MIX | 4 | passed | 512 | 128 |
| gsm8k/ROLE | 4 | passed | 512 | 0 |
| mathqa/KPLUS | 4 | passed | 512 | 0 |
| mathqa/MIX | 4 | passed | 512 | 22 |
| mathqa/ROLE | 4 | passed | 512 | 0 |

The fixed production budget is 3,072 feedback generations, 24 events, 3,072 Adam steps, 6,144 response forward/backwards and 4,608 timed completions. At most 512 missing references, 4 disposable feedback generations, 8 warmup/qualification completions and 6 disposable steps remain separate. Fixed-context diagnostics cost at most 224 draft response forwards and 32 response feature preparations. Target prefill, native verification replay and terminal forwards are counted separately in SUMMARY.json; preparation count is not physical-forward count. Startup/compilation, activation, teacher preparation, persistence and nested timers remain separate. Unknown, invalid and unattempted work stays in original status records and SUMMARY.json.

Exact canonical output and termination, including all cap cases, are required for every timed execution. Counts use one repeat after consistency validation. Event 4 timings average the two repeats within request; event2 has one repeat. The bootstrap uses 10,000 paired-request draws, NumPy PCG64 seed 20261107, ratio of summed measurements and linear 2.5/97.5 percentiles. ROLE/KPLUS event4 is primary separately for each dataset; event2 and multiple secondary intervals are descriptive. There is no pooled cross-dataset efficacy claim.

## Decision and limits

Inspect the fixed role/gradient diagnostics to locate the dataset difference before a separately agreed revision.

One training seed and reused development inputs limit inference. Matched online recipes do not imply identical feedback features. No training-inclusive or asynchronous deployment speedup is established. The pilot stops here; a longer run or another variant needs a separate research decision.
