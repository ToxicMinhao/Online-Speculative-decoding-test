# All 16 fixed functional-geometry diagnostic windows

## Material Passport

ARS experiment-agent validate; **ANALYZED**, not an independent model replication. Scientific execution `6440e8ff5af1a845f830d65c884f6711d7e4a687`, binding `8ec5fed0d4ed9b40dab7c26d5089cd42b01f357ab0379193da42baa93f0b3dc5`. GSM8K and MathQA events 5/6, fixed windows 1/32/65/128, are all included. The same-team analyst helped implement the execution and analyzer; this is disclosed implementation-team evidence review.

All **16 windows / 320 window-role-candidate rows / 640 response-candidate rows** pass identity, role, fixed anchor, support, repair-accounting and restoration checks. Each window has the two fitting responses A and two distinct buffered responses D. D is off-fit training data, not held-out evaluation. Committed FSG equals its feasible candidate in all 16 sampled windows, with a genuine accepted guard decision.

The strongest common pattern is **less useful repair/progress, with limited or inconsistent improvement in preservation of old correct prefixes**. FSG reduces net prefix progress versus the ordinary update from the same pre-state in **7/8 fitting A windows on each dataset**, **6/8 GSM8K D windows**, and **5/8 MathQA D windows**. These observations concern controlled local predictions; they neither establish a unique cause of the endpoint result nor convert to a percentage of time saved.

## Local progress and its denominators

Each row pools four fixed windows and eight response occurrences. Blocks can overlap, so counts are sampled occurrences, not independent output tokens or requests. Native is the ordinary candidate from the same FSG pre-state, not the separately trained KPLUS trajectory.

| Dataset / event / role | Blocks | Old first errors | Old correct-prefix occurrences | Useful repairs native → FSG | Old-prefix loss native → FSG | New earlier errors native → FSG | Net prefix gain native → FSG |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K / 5 / A | 2,329 | 2,013 | 15,496 | 415 → 381 | 64 → 58 | 16 → 16 | +1,154 → +1,016 |
| GSM8K / 5 / D | 1,854 | 1,359 | 15,158 | 73 → 39 | 147 → 166 | 45 → 50 | +44 → −59 |
| GSM8K / 6 / A | 1,945 | 1,475 | 15,661 | 357 → 355 | 22 → 24 | 9 → 11 | +1,194 → +1,174 |
| GSM8K / 6 / D | 1,873 | 1,406 | 15,004 | 44 → 40 | 164 → 162 | 54 → 56 | −10 → −33 |
| MathQA / 5 / A | 3,309 | 2,606 | 25,244 | 459 → 456 | 102 → 100 | 29 → 29 | +1,369 → +1,322 |
| MathQA / 5 / D | 3,011 | 2,142 | 26,702 | 118 → 81 | 201 → 242 | 57 → 72 | +123 → 0 |
| MathQA / 6 / A | 2,675 | 1,929 | 21,961 | 434 → 415 | 77 → 66 | 22 → 24 | +1,338 → +1,243 |
| MathQA / 6 / D | 3,679 | 2,651 | 31,737 | 79 → 78 | 216 → 218 | 72 → 87 | −28 → −37 |

All repair identities (retained/lost/new), native-gain identities, wrong ties and exact teacher-role denominators remain in the response and pair tables. No available metric is replaced with zero.

## Interpretation of the prespecified hypotheses

**A — Over-preservation / suppressed learning.** Lower drift to the old draft is not itself useful preservation. In both datasets, FSG lowers A protected old-draft KL in all eight sampled windows yet reduces A net progress in seven. MathQA event-5 D is particularly clear: 37 fewer repairs, 41 more lost old-prefix occurrences and 123 fewer net prefix occurrences than native. Event-6 A trades 11 fewer lost occurrences for 19 fewer repairs and 95 fewer net occurrences. GSM8K shows the same broad suppressed-progress pattern, with mixed preservation outcomes. The supported statement is that these redirections frequently suppress useful local changes; the drift gate alone does not establish preserved usefulness.

**B — Fitting versus off-fit transfer.** A continues to improve over its pre-state, but the added FSG mechanism generally reduces that improvement relative to native. This is not a clean pattern of “good fitting benefit that only fails to transfer”: fitting progress itself is reduced in 7/8 windows per dataset. D is mixed and often weak or negative. Its small fixed sample and training-buffer status do not identify a unique generalization cause.

**C — Direction versus scalar size.** The same-state scalar matches feasible-FSG norms to maximum relative errors of **2.4268e−12 (GSM8K)** and **1.7651e−12 (MathQA)**. FSG has worse A/D net progress than these scalar candidates in the same 7/8 and 6/8 GSM8K, and 7/8 and 5/8 MathQA windows. On MathQA the scalar preserves the native useful-repair identities and event-role net gains exactly; GSM8K also preserves native repairs, with event-5 A net gain 12 higher. This supports local directional harm beyond simply shrinking the total norm. These scalar scores are **pre-guard counterfactuals**, distinct from the independently trained SCALAR arm. The local scalar guard accepts 2/8 GSM8K cases (three ambiguity and three increased-risk fallbacks) and 1/8 MathQA cases (seven ambiguity fallbacks). Rejected candidates are not committed scalar trajectories. No full-trajectory causal decomposition is claimed.

**D — BF16 realization.** Every raw a=1 diagnostic exceeds the strict native norm bound and is not an accepted deployable candidate. Feasibility scaling changes predictions and has mixed effects. On GSM8K, feasible-minus-raw net gains are +26/+21 for fitting A in events 5/6, but −18/−21 for D. On MathQA they are +27/+11 for A and −14/+16 for D, with four improvements and four regressions across the eight windows of each role. Raw FSG already falls below native in every event-role aggregate on both datasets; feasibility alone therefore does not explain the deficiency. Raw and feasible scores remain separate, and the historical raw free-text label error is not used for candidate identity.

**E — Protection weighting.** Uniform-prefix weighting improves D net progress over feasible FSG in 4/8 GSM8K and 5/8 MathQA windows. It is mixed on A and does not establish a replacement. The favorable MathQA exception is retained explicitly: event-6 D net progress is **0**, compared with **−37 FSG and −28 native**. It still repairs fewer first errors than native (72 versus 79), and its same event's A net progress is 1,244 versus native 1,338. MathQA event-5 D remains below native (+43 versus +123); GSM8K D remains below native in both events. No variant is selected from these diagnostics.

**F — Sketch expressiveness.** The expanded feasible sketch improves D net progress over FSG in 5/8 windows on each dataset. Yet its event-level D results remain below native: GSM8K −36/−25 versus +44/−10; MathQA +83/−34 versus +123/−28. A effects are mixed: MathQA −9/+50 relative to FSG, GSM8K −10/−3. Repair totals do not improve consistently. The candidate changes the effective direction and finite-precision realization as well as sketch expressiveness; these records do not isolate rank as a cause or justify adopting a new method.

Protected old-draft KL uses the original within-response protection weights and then the average of the two response quantities. Event tables report the mean of four such pair values. Teacher-prefix and first-error KL use summed recorded numerators and denominators on fixed original roles, not an average of response means. These objective measurements remain separate from useful-prefix and native endpoint outcomes.

## Work, costs and reproducibility

The 16 original ledgers contain exactly **640 additional response forwards and 64 backwards, zero extra Adam steps**, with zero failed, unknown or omitted diagnostic work. Each dataset contributes 320/32. Recorded physical calls are 84,790 vocabulary-head calls and 5,192 checkpoint proxies; these are not extra response-equivalent counts.

Recorded diagnostic observer time is **1,182.471941 s GSM8K + 1,353.064693 s MathQA = 2,535.536634 s**. It is nested in original FSG training/hook time. Subtract it once to isolate intrinsic adaptation; do not add it again or sum overlapping nested component clocks.

This audit adds **zero model loads, model forwards/backwards, optimizer steps, teacher computation or generation**. The two first full JSON analyses took about 0.381 CPU seconds combined; work auditing and report assembly are separate lightweight CPU operations. Ten focused synthetic tests pass. The dataset parameter defaults to GSM8K; all five prior GSM8K scientific CSVs remain byte-identical, verified in `GSM8K_PRESERVATION.json`. No large tensor snapshots were loaded; state validity rests on authenticated runtime restoration receipts rather than independent prediction regeneration.

Complete deliverables: combined five scientific CSVs (320 pair, 640 response, 80 event-pair, 192 same-state-contrast and 48 event-contrast rows), separate original dataset tables, `main_comparison.csv`, `diagnostic_work_by_window.csv`, `ALL_WORK.json`, per-dataset work/input receipts, tests and analyzer source. No window, candidate or unfavorable outcome is excluded. Main endpoint inference is reported separately by the parent analysis.

## ARS interpretation safeguards

All 11 categories were checked: (1) Simpson—datasets/events remain stratified; (2) ecological—occurrences are not requests or seeds; (3) Berkson—fixed windows, no improvement-based selection; (4) collider—include all candidates, not accepted-only scores; (5) base rates—blocks, errors, prefix and teacher denominators retained; (6) regression to the mean—fixed windows/endpoints unchanged; (7) survivorship—all16 windows present, failure/unknown counts explicit; (8) look-elsewhere—all alternatives disclosed without selecting one; (9) forking paths—same scoring and supports, no tuning; (10) causality—local contrasts do not prove endpoint cause; (11) reverse causality—pre-state supports fixed before candidate evaluation. No significance claim is manufactured from overlapping blocks or eight windows per dataset. This completes the bounded diagnostic analysis, not a new mechanism decision.
