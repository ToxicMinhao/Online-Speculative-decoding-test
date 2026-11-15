# Native K+ correct-prefix loss investigation

## Material Passport

ARS-Codex 3.24.2 experiment-agent; scalar analysis of retained execution records.
Execution commit: `2549f02d9b1edba8fb86f8e91a497e68c224d586`. Status: **completed**.

Only native N advances the event. E withholds current correct-prefix gradients; C withholds current first-error gradients. P is the static pre-update model; H executes Adam with zero current gradients, so persistent moments and weight decay can move it.
Displayed full losses retain their native value; equal displayed loss does not imply equal gradients or updates.

**Decision: LOCAL_PREFIX_WITHHOLDING_SCREEN_PASSED.**

| Comparison | Mean proposal difference per fixed block | Conditional paired 95% interval |
|---|---:|---:|
| E_N (primary) | +0.015625 | [+0.002441, +0.028809] |
| E_H (descriptive) | +0.012695 | [-0.005371, +0.028809] |
| C_H (descriptive) | -0.000977 | [-0.015869, +0.014404] |
| C_N (descriptive) | +0.001953 | [-0.013672, +0.018066] |
| H_P (descriptive) | +0.012939 | [-0.003424, +0.030273] |

The primary estimate equally averages starts within request/window, all sixteen windows within request, and the 64 requests. Whole-request resampling retains all correlated starts, windows and states; the 20,480 outcomes are not independent samples.

| State | Independent acquired positions | Independent lost positions | Independent net | Fitting net |
|---|---:|---:|---:|---:|
| N | 177 | 136 | 41 | 2809 |
| E | 194 | 89 | 105 | 1966 |
| C | 163 | 114 | 49 | 889 |
| H | 154 | 101 | 53 | 24 |

Acquisition/loss totals count prefix positions. Blocker repairs and earlier breaks use block/window denominators and are retained separately. Lower losses can improve net acceptance despite fewer acquisitions; that is not evidence of stronger repair learning.
Source observations are fitting-block occurrences with overlapping anchors, not independent confirmation.
Source denominators (one count per paired anchor, not per state): 8454 sampled slots, 8454 kept anchors, 8454 eligible blocks, 0 zero-supervision slots (0 kept but empty), and 123450 scored proposal occurrences. Padding is preserved in raw records but excluded from eligible exposure.

## Interpretation limits

This is a reused development panel and one native trajectory. Local one-step branches cannot be summed into an alternative training trajectory or a serving speedup. Removing a role also changes gradient magnitude, clipping and interaction with Adam history; this screen does not isolate directional causation. Intervals containing zero are unresolved, not equivalence. C/H remain explanatory controls, not replacement winners.
A positive primary screen permits considering a separately designed scale control and short continuation; it does not authorize them. A null or negative result ends this screen without tuning or sample expansion.

Work and state/displacement receipts are retained separately; nested timers must not be summed as independent costs.
