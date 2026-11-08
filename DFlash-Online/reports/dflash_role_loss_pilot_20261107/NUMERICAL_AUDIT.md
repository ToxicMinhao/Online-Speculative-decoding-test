# Role-loss pilot numerical audit

## Material Passport

- Origin Skill: ARS experiment-agent 3.24.2
- Mode: validate
- Verification Status: ANALYZED
- Version: role_loss_numerical_audit_v1
- Execution commit: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`
- Same-team independently written CPU aggregation; no new GPU execution or independent training replication.

**PASSED: 104,406 checks; 0 discrepancies.**

The four-event pilot provides no observed advantage for ROLE over K+ on either dataset. Its endpoint point estimates are slightly worse for both calls and latency, and all four corresponding primary confidence intervals include zero. This is failure to demonstrate the intended benefit, not proof of equivalence or universal harm. MIX improves GSM8K call count but regresses MathQA; the split does not deliver a consistent advantage.

## Endpoint results

Positive reductions favor the candidate. Intervals are unadjusted paired-request percentile intervals from 10,000 PCG64 draws, seed 20261107.

| Dataset | Contrast | Call reduction %,95%CI | Request-time reduction %,95%CI | Call wins/regressions/ties |
|---|---|---:|---:|---:|
| gsm8k | ROLE_vs_KPLUS | -0.1486 [-0.7526,+0.4296] | -0.5949 [-1.6329,+0.4046] | 103/100/53 |
| gsm8k | MIX_vs_KPLUS | +0.8419 [+0.1390,+1.5366] | +0.6335 [-0.5311,+1.7796] | 122/83/51 |
| gsm8k | ROLE_vs_MIX | -0.9988 [-1.6815,-0.3320] | -1.2362 [-2.2876,-0.1776] | 84/133/39 |
| mathqa | ROLE_vs_KPLUS | -0.4920 [-1.0001,+0.0060] | -0.3902 [-1.2421,+0.4694] | 99/110/47 |
| mathqa | MIX_vs_KPLUS | -0.8961 [-1.4424,-0.3761] | -1.0280 [-1.9303,-0.1922] | 95/118/43 |
| mathqa | ROLE_vs_MIX | +0.4005 [-0.1459,+0.9398] | +0.6313 [-0.3104,+1.5933] | 114/100/42 |

## Costs

Timers are nested. Training is included inside update-call time; do not add them. These are measured own-trajectory costs, not equal-feedback microbenchmarks or training-inclusive serving speedups.

| Dataset / arm | Training seconds | Training overhead vs K+ | Whole update calls seconds | Peak allocated GB | Clipped windows / 512 |
|---|---:|---:|---:|---:|---:|
| gsm8k/KPLUS | 1367.688 | +0.00% | 1569.307 | 20.674 | 0/512 |
| gsm8k/MIX | 1548.740 | +13.24% | 1756.801 | 20.719 | 128/512 |
| gsm8k/ROLE | 1491.915 | +9.08% | 1696.041 | 20.674 | 0/512 |
| mathqa/KPLUS | 1963.830 | +0.00% | 2218.906 | 20.733 | 0/512 |
| mathqa/MIX | 2229.812 | +13.54% | 2487.517 | 20.816 | 22/512 |
| mathqa/ROLE | 2128.170 | +8.37% | 2386.585 | 20.771 | 0/512 |

## Fixed-context evidence

All six diagnostic trajectories contain 16 requests × 8 anchors. Prefix lengths were independently rescored from the 14 raw diagnostic prediction snapshots, including EOS and boundary rules; recomputed per-anchor totals match the report. The 48 original response/window observation files also match pooled training counters and losses. ROLE gains 20 prefix tokens versus 50 for K+ on GSM8K, and 53 versus 82 on MathQA relative to the same released reference. ROLE does not show a superior local net-prefix improvement in this sample. On MathQA, MIX repairs 36 original first errors versus 34 for K+, yet breaks 26 prefixes versus 20 and gains 29 prefix tokens versus 82. More repair events therefore do not by themselves establish improved overall progress. These 128 anchors per dataset are nested in 16 requests, not 128 independent conversations or timing samples.

## Limits

All 4,608 measurement rows have valid metadata and consistent deterministic counts; the raw-token/output-gate audit is a separate root audit. This audit reaggregates the saved counters; it does not re-execute native inference or rederive model gradients. All 12 contrasts across both stages are recalculated; all 48 metric confidence intervals and repeat totals match. Own-feedback trajectories mean a change in objective can change later feedback/features; no pure loss-only causal attribution is made. One seed, four early events and reused development panels do not establish long-run ranking or robust unseen-workload gains. No additional run or algorithm change is authorized by this audit.

## Fallacy scan: 11/11 checked

| Item | Assessment | Detail |
|---|---|---|
| Simpson paradox | CHECKED | Dataset results remain separate; no pooled cross-dataset efficacy conclusion. |
| Ecological fallacy | CHECKED | Per-request wins and regressions are retained; aggregate gains are not attributed to every request. |
| Berkson selection | CAUTION | Input-only deterministic selection avoids output-based filtering, but panels are reused development data, not a fresh population sample. |
| Collider bias | CHECKED | No outcome-conditioned covariates or subgroup selection; all requested valid completions retained. |
| Base-rate neglect | CHECKED | 256-request denominators, gross saved/added calls and all ties are explicit; this is not a screening accuracy analysis. |
| Regression to mean | CHECKED | Fixed inputs and contemporaneous controls, without selection for extreme prior outcomes; early-to-late latency is not used as a learning curve. |
| Survivorship bias | CHECKED | All six trajectories, both endpoints and all 4,608 evaluation rows complete; no unfavorable rows omitted. |
| Look-elsewhere effect | CAUTION | Two primary dataset-specific endpoint contrasts and secondary contrasts/metrics are all shown. Intervals are unadjusted; no familywise confirmation claim. |
| Forking paths | CHECKED | Fixed code/panels/arms/events/bootstrap before outcomes; no best-intermediate selection or post-result tuning. |
| Correlation/causation | CAUTION | Controlled online recipes differ in own generated feedback and resulting features. Local diagnostics do not isolate a loss mechanism or prove why native calls changed. |
| Reverse causality | CHECKED | Training precedes serving; diagnostic/evaluation feedback never trains a model. Direction in this bounded experiment is temporally fixed. |
