# Role-loss pilot: training, accounting and diagnostic audit

## Material Passport

- Origin: ARS experiment-agent, validate mode.
- Verification Status: **ANALYZED**.
- Scientific execution commit: `cdb640543d8fe3a43c8e63af4ccaaa002e09a599`.
- Audit: separately written read-only Python calculation, without importing production validation or reduction functions.
- Provenance: native same-team worker with shared project context; no independent GPU replication.

**Result: PASSED, 576,496 assertions across 239 source files; 0 discrepancies.** Input hashes and per-category counts are retained in `WORK_AUDIT.json`; the executable audit is `work_audit.py`.

## Completed work

| Item | Observed |
|---|---:|
| Independent four-event trajectories | 6 |
| Events | 24 |
| Own-feedback responses | 3,072 |
| Original response presentations / forwards / backwards | 6,144 / 6,144 / 6,144 |
| Native AdamW steps | 3,072 |
| Additional inherited correctness generations | 8 |
| Fixed-context draft-response forwards | 224 |
| Fixed-context response feature preparations | 32 |
| Physical target prefill / native verification replay / terminal forwards for those preparations | 32 / 1,838 / 32 |

All six initialization receipts assert exact released weights, empty Adam, and original seed-17 RNG, with no source-checkpoint weights loaded into the candidate to manufacture equality. Every event retains 128 distinct own-served responses, two separately shuffled passes, 256 original response observations and 128 accumulation windows. Global steps and presentations progress continuously to 512/1,024 per trajectory. The sampled anchors are unique, sorted, eligible, and number `min(512, response_length−1)` in every presentation. All loss/diagnostic values and observed gradient norms are finite.

Every feedback record uses its own arm’s preceding event checkpoint, preserves its sealed prompt token IDs, and remains disjoint from the evaluation panel by source ID and normalized stem. All 24 boundaries have exact live/saved-state receipts, all 24 native publications report all 58 source tensors equal, accumulation is empty, and the inherited 7,424-step horizon remains unchanged. Boundary checks add no optimizer/model work.

**Observed pairing:** all 512 feedback outputs and termination states match across all arm pairs within each dataset; all 1,024 presentation-order and anchor bindings also match within each dataset. This was observed after own-feedback execution, not enforced by shared feedback. It does not prove native captured target hidden features were identical.

## Exposure and loss behavior

| Dataset / arm | Sampled anchors | Selected / available occurrences | Selected fraction | Prefix weight fraction | Pooled D/Z | Priority-2 errors |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k/KPLUS | 264,848 | 2,216,221 / 3,865,887 | 57.328% | 91.499% | 1.401021 | 30.908% |
| gsm8k/MIX | 264,848 | 2,217,514 / 3,865,887 | 57.361% | 91.517% | 1.400247 | 28.079% |
| gsm8k/ROLE | 264,848 | 2,230,234 / 3,865,887 | 57.690% | 91.614% | 1.394751 | 30.568% |
| mathqa/KPLUS | 383,056 | 3,479,056 / 5,649,919 | 61.577% | 92.663% | 1.344880 | 32.261% |
| mathqa/MIX | 383,056 | 3,480,222 / 5,649,919 | 61.598% | 92.670% | 1.344302 | 29.172% |
| mathqa/ROLE | 383,056 | 3,489,772 / 5,649,919 | 61.767% | 92.723% | 1.341289 | 31.876% |

Occurrences include overlap between sampled blocks; they are not distinct tokens. Role weights are pooled from original response forwards, never from duplicate accumulated metrics. First-error weight mass satisfies the declared tolerance for every presentation. The objective has one draft forward per response and two head calls per recorded projection chunk because of checkpoint recomputation; no extra production backbone or vocabulary pass was added by the new objective.

| Dataset / arm | Mean preclip norm | Max preclip norm | Clipped steps / 512 | First-error RKL cap | Prefix RKL cap |
|---|---:|---:|---:|---:|---:|
| gsm8k/KPLUS | 0.403433 | 0.964844 | 0 | Not evaluated | Not evaluated |
| gsm8k/MIX | 0.904839 | 1.671875 | 128 | 54,501/204,803 (26.611%) | 6,920/2,012,711 (0.344%) |
| gsm8k/ROLE | 0.452953 | 0.921875 | 0 | 55,091/204,158 (26.984%) | Not evaluated |
| mathqa/KPLUS | 0.269614 | 0.617188 | 0 | Not evaluated | Not evaluated |
| mathqa/MIX | 0.705433 | 1.500000 | 22 | 27,035/275,618 (9.809%) | 1,918/3,204,604 (0.060%) |
| mathqa/ROLE | 0.343470 | 0.683594 | 0 | 25,592/275,174 (9.300%) | Not evaluated |

The cap operates on the released-repository response-normalized reverse-KL rows. ROLE has no prefix RKL branch; that entry is not zero observed cap incidence. KPLUS does not evaluate the mixture branch. MIX/ROLE report zero native draft-probability zeros and zero nonfinite entries in every response. Teacher-argmax/canonical differences are rare but retained: GSM8K MIX 6/2,217,514 and ROLE 4/2,230,234 selected occurrences; MathQA MIX 47/3,480,222 and ROLE 41/3,489,772. These diagnostics do not establish why an endpoint comparison differs.

## Training costs

| Dataset / arm | Four-event training time (s) | Versus KPLUS | Peak allocated / reserved (GB) |
|---|---:|---:|---:|
| gsm8k/KPLUS | 1367.688 | +0.00% | 20.674 / 24.088 |
| gsm8k/MIX | 1548.740 | +13.24% | 20.719 / 23.744 |
| gsm8k/ROLE | 1491.915 | +9.08% | 20.674 / 23.943 |
| mathqa/KPLUS | 1963.830 | +0.00% | 20.733 / 23.425 |
| mathqa/MIX | 2229.812 | +13.54% | 20.816 / 23.306 |
| mathqa/ROLE | 2128.170 | +8.37% | 20.771 / 23.429 |

These are measured serial trajectory training costs, not equal-compute controls or per-kernel attribution. Initialization, feedback serving/capture, teacher preparation, persistence, exports, activation and fixed-context diagnostics remain separate. Nested timers must not be added to training time again. The underlying cost fields are retained in the original `EVENT_COSTS.csv`/`TRAINING.json` and this audit.

## Fixed-context reconstruction

| Dataset / arm | Initial → event2 → event4 prefix tokens | Event2 / event4 repairs | Event2 / event4 broken prefixes | Event2 repairs retained |
|---|---:|---:|---:|---:|
| gsm8k/KPLUS | 819 → 846 → 869 | 31 / 34 | 25 / 20 | 20 |
| gsm8k/MIX | 819 → 845 → 875 | 29 / 31 | 29 / 25 | 20 |
| gsm8k/ROLE | 819 → 826 → 839 | 28 / 32 | 25 / 24 | 20 |
| mathqa/KPLUS | 992 → 1041 → 1074 | 28 / 34 | 21 / 20 | 22 |
| mathqa/MIX | 992 → 1014 → 1021 | 34 / 36 | 26 / 26 | 26 |
| mathqa/ROLE | 992 → 1042 → 1045 | 32 / 30 | 21 / 20 | 20 |

The audit recalculated these counts directly from saved labels, validity masks and argmax predictions at every selected block. All three arms and both endpoints have identical prompt/response token digests, target-conditioning identities, anchor IDs, validity masks and weights for these diagnostics. Every snapshot is marked eval/no-backward, with saved RNG/mode restoration receipts. Each dataset uses 16 requests × 8 anchors =128 anchors. There are 108 original first-error opportunities among 1,874 valid proposals on GSM8K; MathQA has 102 original first-error opportunities among 1,861 valid proposals. A repair requires an extended continuous prefix, so later positional matches behind an earlier error are not credited.

ROLE has less total local prefix progress than KPLUS at event4 on both datasets (839 versus869;1,045 versus1,074). MIX has more GSM8K local progress but less MathQA progress. These sampled local changes are consistent with the mixed serving result; they do not quantify total time saved or prove a causal mechanism.

## Retention and limits

- All 3,072 prepared training feature tensors, 18 superseded recovery states and12 nonselected exports were retired by the admitted policy after success receipts. Selected event2/event4 exports and final complete recovery states remain on A800. Retired tensors are not archived; exact historical tensor-level replay is not available from this compact mirror.
- This audit reads compact source records and checks consistency independently. Exact weight, optimizer/RNG and export equality remains based on the original native execution receipts; no fresh tensor loading or GPU rerun was performed.
- Counts and finite metrics validate the performed work within the retained evidence. They do not establish training-seed robustness, a full58-event ranking, or a training-inclusive speedup.
- Endpoint confidence intervals and the full11-category interpretation scan are reported separately by the numerical auditor. This accounting audit does not select a follow-up experiment.
