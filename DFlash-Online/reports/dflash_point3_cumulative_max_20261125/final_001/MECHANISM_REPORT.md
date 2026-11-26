# Point3 mechanism and adaptation-cost audit

Execution: `6f7f3a5d1769a630e4ec852205eb06ca55047341`. Verification status: **ANALYZED** from retained native records, with an independent CPU recomputation of scalar quantities and discrete proposal metrics. No new model forward, backward, optimizer step or generation was performed.

The MAX substitution changes the priorities and the trained endpoint, but these local measurements do not establish a serving benefit. Relative to SUM, MAX gains four matching-prefix occurrences on the newly trained witness set and eight on the earlier training set. These small net gains coexist with more damaged original-prefix positions. Native serving is a separate outcome in the main report.

## Fixed comparison and denominators

The working control C is **native C + K+PAIR**. SUM and MAX preserve that inner teaching recipe and the C-referenced sensitivity construction. Only the outer assessment changes:

\[
R_{\mathrm{SUM}}=\sum_{j=1}^{K}(K-j+1)\phi(m_j),\qquad
R_{\mathrm{MAX}}=\sum_{k=1}^{K}\max_{j\le k}\phi(m_j),\quad
\phi(m)=\operatorname{softplus}(-m)/\log2.
\]

Every endpoint uses the same retained parent origins in these readouts. Each set contains eight responses with eight saved blocks each: 64 blocks, not 64 independent responses. `new_buffer` belongs to the completed event's training buffer; `parent_training` was trained earlier. Neither is held-out serving evidence. Both assessment values come from the same native margins. Block assessments are averaged within each response over nonempty valid blocks, then responses receive equal weight. All 64 blocks in each set are nonempty. The bound `fixed-K shortfall ≤ MAX ≤ SUM` holds in every audited block; the numerical inequality between MAX and SUM is structural, not an improvement claim.

Matching-prefix length includes every valid consecutive canonical match, including a valid EOS. A useful repair requires repairing the original first error while preserving every earlier original match. Raw repairs need not satisfy that condition. Damage counts original-prefix positions and affected blocks. Followers are the original consecutive matches after the first error, through fixed K; surviving followers remain individually correct, while reachable followers are actually inside the new consecutive prefix.

| TRAIN witness set | Endpoint | Prefix occurrences | Fixed-K shortfall | Raw/useful repairs | Damaged positions/blocks | Surviving/reachable followers | SUM | MAX |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| New buffer | Parent | 488 | 104 | 0/0 | 0/0 | 60/0 | 6.844945 | 4.739249 |
| New buffer | C | 656 | 35 | 33/33 | 3/2 | 58/39 | 3.516551 | 2.428703 |
| New buffer | SUM | 670 | 31 | 35/35 | 2/1 | 59/40 | 3.156065 | 2.183847 |
| New buffer | MAX | 674 | 24 | 36/34 | 4/3 | 59/51 | 3.261790 | 2.232500 |
| Earlier training | Parent | 676 | 64 | 0/0 | 0/0 | 33/0 | 4.628858 | 2.796869 |
| Earlier training | C | 675 | 75 | 10/9 | 14/10 | 30/12 | 6.587701 | 3.772752 |
| Earlier training | SUM | 681 | 69 | 10/9 | 12/8 | 30/12 | 6.650569 | 3.809046 |
| Earlier training | MAX | 689 | 62 | 10/9 | 13/8 | 30/12 | 6.625829 | 3.761107 |

The new-buffer origins contain 488 protected-prefix positions, 44 first-error blocks and 60 followers over 914 valid proposals. Earlier-training origins contain 676 protected-prefix positions, 31 first-error blocks and 33 followers over 922 valid proposals. Assessment columns above use matched native FP32 contributions reduced in FP64; the original FP32 reductions are retained separately in [WITNESS_SUMMARY.csv](WITNESS_SUMMARY.csv).

MAX versus SUM is mixed locally. On the new buffer, MAX obtains four more prefix occurrences and seven fewer shortfall occurrences, but one fewer useful repair and two more damaged positions. Its two assessments are also worse than SUM by 0.105725 and 0.048653. On earlier training, MAX obtains eight more prefix occurrences and seven fewer shortfall occurrences, with equal useful repairs and one more damaged position; its two assessments improve by 0.024741 and 0.047939. Prefix totals and fixed-K shortfall need not move by the same amount because progress beyond the fixed origin K contributes only to the former. These outcomes do not demonstrate a uniform preservation advantage or validate a surrogate as a serving predictor.

All endpoint, response and block records, including regressions, are available in [WITNESS_RESPONSES.csv](WITNESS_RESPONSES.csv), [WITNESS_BLOCKS.csv](WITNESS_BLOCKS.csv), [WITNESS_PAIRED_SUMMARY.csv](WITNESS_PAIRED_SUMMARY.csv), [WITNESS_PAIRED_RESPONSE_CHANGES.csv](WITNESS_PAIRED_RESPONSE_CHANGES.csv) and [WITNESS_PAIRED_BLOCK_CHANGES.csv](WITNESS_PAIRED_BLOCK_CHANGES.csv). Paired assessment differences subtract matched represented contributions before FP64 reduction.

## Sparse allocation measurements

In this table P means protected correct prefix, E the first error, and F the correct followers. The machine-readable records call that follower role `C`; it is unrelated to the comparator arm C. Each multiplicity counts how many cumulative-prefix terms select that role. Distinct counts count positions selected at least once.

| Arm/window | Blocks | P/E/F active positions | P/E/F multiplicities | Half-L1 coefficient movement vs beta/current C | Native preclip norm |
|---|---:|---:|---:|---:|---:|
| SUM/1 | 821 | 1992/679/12 | 5311/1339/15 | 117.824/180.104 | 0.427734 |
| SUM/64 | 740 | 1731/638/13 | 4474/1327/36 | 103.850/161.725 | 0.361328 |
| SUM/128 | 765 | 2247/473/5 | 7336/1336/16 | 108.394/176.540 | 0.298828 |
| MAX/1 | 821 | 1992/679/12 | 5311/1339/15 | 116.010/175.405 | 0.425781 |
| MAX/64 | 740 | 1748/641/17 | 4474/1330/44 | 101.362/158.557 | 0.367188 |
| MAX/128 | 765 | 2248/457/9 | 7426/1275/15 | 108.420/174.320 | 0.306641 |

Both arms log the cumulative-maximum branches for comparison. **SUM does not use only these bottlenecks**: its outer gradient remains the triangular additive gradient. MAX does use the selected branches. At its three logged windows, follower multiplicities are respectively 15/6665, 44/5848 and 15/8716 (0.225%, 0.752%, 0.172%). Thus correct followers rarely determine MAX's active obstruction in these windows, although they still contribute to the number of assessed prefixes and the unchanged inner teaching. This is a descriptive concentration measurement, not proof that follower sparsity caused an endpoint outcome.

Window 1 has common origins and identical branch counts across arms, but already different centered scores and coefficient movements. Later windows have each arm's own evolved roles, so their counters are not matched-state interventions. Across all 128 windows per arm, every isolated native-C reference performed two forwards/backwards, zero Adam steps and zero committed presentations, with restored RNG and cleared gradients. All coefficient seals and recorded pool-mass checks pass. None of the 128 native updates per arm required clipping. Tie counts, reference branches, zero-centered positions, score ranges, beta mass and predicted alignment remain in [SPARSE_DIAGNOSTICS.csv](SPARSE_DIAGNOSTICS.csv); predicted alignment is not an observed improvement.

## Adaptation cost and physical work

| Measurement | SUM | MAX |
|---|---:|---:|
| Training timer, including scoring | 2397.967 s | 2466.660 s |
| Scoring, nested inside training | 1830.708 s | 1901.680 s |
| Native C reference, nested inside scoring | 564.577 s | 563.488 s |
| Candidate-training timer residual, training minus scoring | 567.260 s | 564.980 s |
| Recovery persistence | 19.535 s | 20.619 s |
| Serving export | 14.950 s | 15.224 s |
| Disposable startup | 17.714 s | 20.390 s |
| Event wall before commit | 2443.622 s | 2513.612 s |
| Whole arm child-process wall | 2519.363 s | 2592.784 s |
| Whole-process peak GPU allocated/reserved | 39.681/51.831 GB | 39.681/51.831 GB |

These scopes overlap and must not be summed. Whole-process peaks include startup. Process wall also includes construction, verification and witness work; no separately isolated loading or final-witness timer is available. The recorded objective subtimers are nested instrumentation, not a full kernel-time decomposition. Host-memory peak is not supplied by these arm receipts.

MAX's training timer is 68.692 seconds, or 2.8646%, longer than SUM's in this serial execution. That is an observed run-cost difference, not an isolated causal measurement of the MAX operator. Retained PR60 C training took 564.545 seconds: SUM and MAX are respectively 4.2476× and 4.3693× that historical timer, adding approximately 30.56 and 31.70 minutes. The C measurement is historical context, not a fresh matched overhead comparison. See [TRAINING_COSTS.csv](TRAINING_COSTS.csv), [TRAINING_COST_COMPARISONS.csv](TRAINING_COST_COMPARISONS.csv) and [HISTORICAL_C_COST.json](HISTORICAL_C_COST.json).

Each arm commits 256 response presentations and 128 Adam steps. Physical event instrumentation records 1280 draft forwards, 6400 layer forwards, 153270 vocabulary-head calls and 47160 objective chunks, including the separately counted isolated native-C work and recorded checkpoint recomputation. This is much more work than the committed-presentation count. The reference alone contributes 256 forwards/backwards and no Adam calls. Per arm, startup adds two disposable response forwards/backwards with zero Adam, and final TRAIN witnesses add 16 forward-only response evaluations. Preview/optimizer-VJP and native response-work counters are preserved in [PHYSICAL_WORK.json](PHYSICAL_WORK.json) and [PHYSICAL_ACCOUNTING.json](PHYSICAL_ACCOUNTING.json); they overlap the physical counters and must not be added again.

The mean of the 256 separately normalized response observations is FKL 0.223008/0.223019 and PAIR 0.016978/0.016876 for SUM/MAX. These describe each arm's evolving training trajectory, not a fixed held-out loss comparison.

## Audit scope and interpretation limits

The independent audit makes 26,785 checks across 519 source files: common origins/anchors, every native matching-prefix and repair/damage/follower metric, earliest-minimum branches and multiplicities, both assessment formulas and reductions, fixed-K bounds, complete per-window work, reference isolation and recorded coefficient mass conservation. [MECHANISM_AUDIT.json](MECHANISM_AUDIT.json) records its scope. Full model logits, gradient tensors and vector contractions are not retained in these compact records and were not independently reconstructed. Strongest-vocabulary-rival selection remains native receipt evidence; the audit verifies the arithmetic downstream of the saved margin.

ARS statistical-fallacy coverage: **11/11 checked**. The two TRAIN strata and per-response/block changes remain separate (Simpson's paradox/ecological inference); origins and role eligibility are fixed independently of endpoint outcomes (Berkson/collider selection); explicit original-position and error denominators are reported (base rates); the common parent and all three endpoints remain visible (regression to the mean); no incomplete or unfavorable block is dropped (survivorship); all prespecified sparse windows, endpoints and both assessment values are retained (look-elsewhere/forking paths); local trajectory changes are not asserted to cause serving improvement (causality/reverse causality). No local statistical significance, training-seed robustness, or held-out generalization claim is made.

The evidence supports a narrower conclusion: the changed outer assessment affects allocations, and the two trained variants have small, mixed differences in the fixed TRAIN witnesses at substantial additional adaptation cost. The contemporaneous serving comparison, not these diagnostics, determines whether either configuration should replace C.
