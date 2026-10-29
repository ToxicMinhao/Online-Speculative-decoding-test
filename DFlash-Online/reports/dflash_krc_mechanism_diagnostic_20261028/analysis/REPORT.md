# KRC mechanism diagnostic

Execution source: `c9461345189b91d241ff21590027500e6ac7dab4`. Included candidate slots: **80/80** across the fixed event-2/event-4 sample.

**Recommendation: inconclusive.** Machine summaries are descriptive. A final recommendation requires explicit interpretation of all branches of the original handoff decision map; no new numeric decision threshold is imposed.

Only the original R_FULL trajectory continued. Candidates were temporary; no serving generation or replacement training arm was run.

## Four research questions

**1. Does correction realize its fitted-objective prediction?**

- Event 2, R_FULL: 10 windows; actual minus realized-linear prediction median/range -1.08476e-05 / [-6.31442e-05, 7.56336e-05]. Hinge decreased in 8, increased in 1; inactive ideals remain null.
- Event 2, R_HALF: 10 windows; actual minus realized-linear prediction median/range -8.98702e-07 / [-9.92031e-05, 6.66138e-05]. Hinge decreased in 6, increased in 4; inactive ideals remain null.
- Event 2, D_FULL: 10 windows; actual minus realized-linear prediction median/range 5.0991e-06 / [-2.46602e-05, 0.000108823]. Hinge decreased in 8, increased in 2; inactive ideals remain null.
- Event 4, R_FULL: 10 windows; actual minus realized-linear prediction median/range -8.35282e-06 / [-8.42855e-05, 2.6089e-05]. Hinge decreased in 8, increased in 2; inactive ideals remain null.
- Event 4, R_HALF: 10 windows; actual minus realized-linear prediction median/range -6.77216e-06 / [-6.02066e-05, 3.58582e-05]. Hinge decreased in 7, increased in 2; inactive ideals remain null.
- Event 4, D_FULL: 10 windows; actual minus realized-linear prediction median/range 4.70837e-07 / [-5.86657e-05, 3.73732e-05]. Hinge decreased in 7, increased in 3; inactive ideals remain null.

Proposed-versus-realized differences describe rounding of the actual displacement. The remaining actual-versus-linearized residual does not uniquely establish nonlinearity.

**2. Does lower hinge restore native correct prefixes?** The table keeps fitted and cross-role prefix effects beside protected errors; lower V alone is not a success criterion.

| Event | Pass (zero based) | Candidate | Role | Prefix Δ | Gross gain/loss | Mean Δ per valid block | Windows +/−/= | K gain lost | Protected wrong ties |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | 0 | D_FULL | A | +20 | 104/84 | 0.00734484 | 2/2/1 | 69 | 3 |
| 2 | 0 | D_FULL | D | +5 | 85/80 | 0.00186916 | 3/2/0 | 38 | 16 |
| 2 | 0 | D_FULL | R | -3 | 55/58 | -0.00112317 | 2/3/0 | 40 | 30 |
| 2 | 0 | R_FULL | A | +19 | 87/68 | 0.0069776 | 4/1/0 | 43 | 5 |
| 2 | 0 | R_FULL | D | -25 | 40/65 | -0.00934579 | 1/3/1 | 34 | 15 |
| 2 | 0 | R_FULL | R | -29 | 47/76 | -0.0108574 | 0/4/1 | 45 | 33 |
| 2 | 0 | R_HALF | A | -1 | 76/77 | -0.000367242 | 2/2/1 | 60 | 2 |
| 2 | 0 | R_HALF | D | -14 | 46/60 | -0.00523364 | 3/2/0 | 32 | 17 |
| 2 | 0 | R_HALF | R | -21 | 55/76 | -0.00786222 | 1/4/0 | 53 | 34 |
| 2 | 1 | D_FULL | A | +31 | 56/25 | 0.0132365 | 4/1/0 | 25 | 2 |
| 2 | 1 | D_FULL | D | +28 | 69/41 | 0.0119048 | 4/1/0 | 10 | 24 |
| 2 | 1 | D_FULL | R | +9 | 43/34 | 0.00390117 | 4/1/0 | 10 | 17 |
| 2 | 1 | R_FULL | A | +22 | 57/35 | 0.00939368 | 4/1/0 | 34 | 4 |
| 2 | 1 | R_FULL | D | +17 | 65/48 | 0.00722789 | 4/1/0 | 13 | 23 |
| 2 | 1 | R_FULL | R | +15 | 68/53 | 0.00650195 | 3/2/0 | 23 | 16 |
| 2 | 1 | R_HALF | A | +52 | 78/26 | 0.0222032 | 5/0/0 | 25 | 4 |
| 2 | 1 | R_HALF | D | +20 | 73/53 | 0.0085034 | 4/1/0 | 13 | 25 |
| 2 | 1 | R_HALF | R | -17 | 43/60 | -0.00736888 | 3/2/0 | 36 | 15 |
| 4 | 0 | D_FULL | A | -7 | 41/48 | -0.00281237 | 3/2/0 | 46 | 4 |
| 4 | 0 | D_FULL | D | -4 | 50/54 | -0.00163733 | 2/3/0 | 17 | 24 |
| 4 | 0 | D_FULL | R | +8 | 36/28 | 0.00299738 | 2/3/0 | 23 | 12 |
| 4 | 0 | R_FULL | A | -2 | 41/43 | -0.000803536 | 2/2/1 | 39 | 3 |
| 4 | 0 | R_FULL | D | -21 | 39/60 | -0.00859599 | 1/4/0 | 26 | 25 |
| 4 | 0 | R_FULL | R | +41 | 81/40 | 0.0153616 | 3/2/0 | 20 | 13 |
| 4 | 0 | R_HALF | A | -22 | 43/65 | -0.00883889 | 1/4/0 | 62 | 3 |
| 4 | 0 | R_HALF | D | -26 | 49/75 | -0.0106427 | 1/3/1 | 44 | 24 |
| 4 | 0 | R_HALF | R | +5 | 68/63 | 0.00187336 | 3/2/0 | 32 | 12 |
| 4 | 1 | D_FULL | A | +2 | 47/45 | 0.00110072 | 3/2/0 | 26 | 5 |
| 4 | 1 | D_FULL | D | -11 | 28/39 | -0.00570244 | 1/4/0 | 15 | 18 |
| 4 | 1 | D_FULL | R | -11 | 75/86 | -0.00483304 | 2/3/0 | 45 | 21 |
| 4 | 1 | R_FULL | A | -8 | 42/50 | -0.00440286 | 2/1/2 | 30 | 5 |
| 4 | 1 | R_FULL | D | +6 | 50/44 | 0.00311042 | 3/2/0 | 7 | 14 |
| 4 | 1 | R_FULL | R | +46 | 124/78 | 0.0202109 | 4/1/0 | 31 | 24 |
| 4 | 1 | R_HALF | A | +30 | 50/20 | 0.0165107 | 4/1/0 | 12 | 3 |
| 4 | 1 | R_HALF | D | +12 | 28/16 | 0.00622084 | 3/2/0 | 2 | 16 |
| 4 | 1 | R_HALF | R | +22 | 92/70 | 0.00966608 | 3/2/0 | 54 | 11 |

**3. Does correction undo K learning or transfer?** K gained-prefix retention and loss, original-prefix loss, and matched first-error repair identities are retained in CANDIDATE_ROLE.csv. A and D are evaluated at the same restored post-K state. Positive and negative windows remain separately visible in WINDOW_SUMMARY.csv.

**4. How sensitive is correction to the defining reference?** R_FULL and D_FULL use their own fitted hinge and natural step length. D is fitted data for D_FULL, so own-pair benefit does not establish a better reference policy.

- Event 2: g_R/g_D cosine median/range 0.0317299 / [-0.0453778, 0.233606], defined in 10/10 windows.
- Event 2, scored pair R: R_FULL/D_FULL prefix changes -14/+6 over 10 paired windows; see pass/window distributions before interpreting transfer.
- Event 2, scored pair D: R_FULL/D_FULL prefix changes -8/+33 over 10 paired windows; see pass/window distributions before interpreting transfer.
- Event 4: g_R/g_D cosine median/range 0.0298788 / [-0.024246, 0.117531], defined in 10/10 windows.
- Event 4, scored pair R: R_FULL/D_FULL prefix changes +87/-3 over 10 paired windows; see pass/window distributions before interpreting transfer.
- Event 4, scored pair D: R_FULL/D_FULL prefix changes -15/-15 over 10 paired windows; see pass/window distributions before interpreting transfer.

PREDICTIONS.csv retains proposed/realized norms; WINDOW_SUMMARY.csv retains gradient alignment, BF16 aliases and paired A/R/D outcomes.

## Oracle opportunity

| Event | Role | First-only gain | All-protected gain | First blocked by wrong tie: recovered gain |
|---:|---|---:|---:|---:|
| 2 | A | 89 | 89 | 22 |
| 2 | R | 291 | 301 | 97 |
| 2 | D | 253 | 258 | 99 |
| 4 | A | 56 | 56 | 39 |
| 4 | R | 290 | 298 | 83 |
| 4 | D | 175 | 182 | 82 |

These are hypothetical token replacements, not attainable parameter updates or a serving-improvement bound. ORACLES.csv retains every window, including zero opportunities.

## Validity and work

Decoded endpoint gates: event 2: passed; event 4: passed.
Recorded campaign wall time: 3106.17 s. Core response forwards/backwards: 2136/996; ordinary Adam updates: 256. Separate startup forwards/backwards/Adam: 4/4/0.
Nested timing components must not be added to campaign/event time. COSTS_WORK.json preserves numeric phase costs and internal-work accounting. Missing evidence is not replaced by a budget-derived measurement.
The gamma_n response-reduction bound is an analysis consistency check, not a relaxed scientific replay or restoration tolerance. WINDOW_SUMMARY.csv retains each role’s observed reconstruction discrepancy and bound; CANDIDATE_ROLE.csv retains their maxima. D gradient/no-grad scalar, margin and prediction differences are descriptive and do not add a D equality gate.

## Limits and interpretation

- Windows and overlapping anchors share trajectories; they are not independent training replications.
- D is buffered training data, not unseen or statistically independent data; D is fitted data for D_FULL.
- Changing R to D changes both direction and natural step length; this is not a norm-matched direction test.
- Different parent states and feedback prevent an isolated training-age interpretation of event 2 versus event 4.
- Token-replacement oracles describe local opportunity; they do not bound total algorithm or serving improvement.
- Actual-minus-linearized residuals combine possible nonsmooth/nonlinear and finite-precision effects; causes are not uniquely identified.
- No significance tests, serving-speed claims, optimal-dose claims or training-inclusive acceleration claims are made.
- Earlier KRC1 probes already showed local non-monotonicity, including increased R hinge at window 64.

Earlier non-monotonicity source: [KRC1 report, local probes](../../dflash_krc_event_20261027/REPORT.md).

## Statistical fallacy scan

11/11 reviewed as descriptive design/claim checks; this is not a statistical certification.

- Simpson paradox: Mixed phase directions: D_FULL/A, D_FULL/D, D_FULL/R, R_FULL/A, R_FULL/D, R_FULL/R, R_HALF/A, R_HALF/D, R_HALF/R
- Ecological fallacy: Block occurrences and window summaries are not generalized to requests, seeds or users.
- Berkson paradox: Fixed training-buffer sample, not a population sample; sampling/selection limits remain.
- Collider bias: No post-outcome covariate adjustment or favorable-window selection.
- Base rate neglect: Fixed valid-block denominators, zero opportunities and gross losses retained.
- Regression to the mean: Windows fixed before outcomes; no extreme-state selection.
- Survivorship bias: 0 excluded candidate slots remain visible with reasons; no silent survivor-only coverage claim.
- Look-elsewhere effect: All fixed candidates/roles/windows reported; no significance testing.
- Garden of forking paths: Fixed sampling/doses/reference offsets; no outcome-driven tuning or replacement windows.
- Correlation versus causation: Same-state interventions support local contrasts only; endpoint/age/serving mechanisms remain unisolated.
- Reverse causality: Pre-K membership precedes K/candidates; no observational temporal-causality claim.

## Material Passport

- Mode: ARS experiment-agent validate; compact evidence reconstructed without GPU replication.
- Status: ANALYZED. Scientific replay validity is conditional on the recorded endpoint and isolation receipts.
- Stage 1 / project None unchanged. Any proposed next intervention requires a separately authorized test.
- Public files contain scalar summaries only; native token/margin/anchor arrays remain in private run artifacts.
