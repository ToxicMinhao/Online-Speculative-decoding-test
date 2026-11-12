# Recent repair history predicts additional rejection risk on the matched population

## Material Passport

- Status: **complete and audited; stopped after this report**.
- Mode: bounded experiment execution and statistical validation using ARS-Codex 3.24.2.
- Scientific execution: `dea3fe81dea957416a4f7c7188b60f44a7ced52c`; report-only validator repair: `aaa196191ae81151a1d5cb061c96e33f3d1d8c5d`.
- Verification: separate CPU reductions of retained evidence; no independent GPU rerun.
- No training, optimizer steps, new arm or latency benchmark.

Among **235 matched repaired starts across 107 requests**, newly repaired tokens became K+6's first rejection at the measured slot **28.94%** of the time, versus **15.74%** for established tokens with similar K+5 confidence. The difference is **+13.19 percentage points**, with a request-cluster 95% interval of **[+5.76, +20.72] points**.

The lower bound exceeds the prespecified +5-point threshold and the support/balance conditions pass. The raw audits confirm the receipts. This supports **recent repair history as additional prognostic information beyond the matched current margin**. The lower bound clears the threshold by only 0.76 points; the interval should remain visible. This does not demonstrate a successful retention method, a causal effect or a speedup.

## What happened at the measured position

All 235 pairs remain in the denominator, including earlier breaks.

| K+6 outcome | Recently repaired | Established |
|---|---:|---:|
| First rejection exactly at j | 68/235 — 28.94% | 37/235 — 15.74% |
| First rejection before j | 13/235 — 5.53% | 14/235 — 5.96% |
| Accepts through j | 154/235 — 65.53% | 184/235 — 78.30% |
| Direct proposal at j is canonical, even if an earlier slot failed | 162/235 — 68.94% | 189/235 — 80.43% |

Earlier-break risk differs by **−0.43 points**; direct canonical agreement at j is **11.49 points lower** for recent repairs. The excess exact-slot rejection therefore is not merely an excess of failing before the position is reached. Among the earlier-break cases, j itself remains correct in 8/13 repaired cases and 5/14 established cases. These are descriptive decompositions of the same pairs, not additional independent tests.

## Matching reduces the gap without removing it

On the **same 235 repaired starts**, before imposing the margin match, the weighted structural comparator risk is **5.84%**, giving a gap of **+23.09 points [17.20, 29.02]**. Each repair compares with the mean of all its same-request, same-j, same-a5 established starts.

Margin matching reduces that point gap by **9.90 points** to +13.19. This is consistent with current confidence explaining part of the discovery association, while repair history retains additional information. It is not a causal decomposition, and the 42.87% relative attenuation must not be described as a proven percentage “explained by confidence.”

## Strong measured balance, limited overlap

The 192-request panel supplies 8,647 valid K+5 starts and **1,397 repaired starts**. Of these, 1,190 have a structural comparator, and **235 match** the fixed 0.25 caliper and zero-tie rule. The estimate therefore covers **16.82% of all repaired starts**, not all repairs. The other 1,162 (83.18%) remain outside the matched result: 207 lack structural comparators and 955 fail the margin/zero-tie requirement.

There are **223 distinct comparator starts**, reused at most twice. Pair weights remain one per repaired start. All fixed conditions pass: 235≥100 repaired starts, 107≥40 requests, and absolute SMD **0.02390≤0.10**. Mean native K+5 margins are **1.1351 versus 1.1644**. Paired absolute-margin difference median/p95/maximum is **0.125/0.25/0.25**.

| Native margin summary | Matched repaired | Matched established | Unmatched repaired |
|---|---:|---:|---:|
| q25 / median / q75 | 0.375 / 0.75 / 1.4375 | 0.375 / 0.75 / 1.5 | 0.25 / 0.625 / 1.375 |
| q05 / q95 | 0.125 / 3.6625 | 0.125 / 3.6625 | 0 / 3.25 |
| Minimum / maximum | 0 / 7.125 | 0 / 7.375 | 0 / 12 |

Unmatched repaired counts by proposal slot j:

| j | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Count | 110 | 163 | 179 | 118 | 95 | 85 | 69 | 58 | 54 | 48 | 47 | 47 | 31 | 24 | 34 |

Full distributions and exclusion reasons remain in `SUMMARY.json` and `REPAIR_COVERAGE.csv`. Good margin balance does not establish balance in unmeasured token difficulty or hidden-state properties.

## Validity, duration and report-only repair

The native receipts report all **192 parent walks, 8,647 neutral K+5 blocks, 8,647 K+4 blocks and 1,343 required K+6 blocks completed**, with no missing matched outcomes or failed/unknown native operations. All **32 fixed hash-selected accepted score rows** pass the full-vocabulary margin check. Occupied time was **82.73 minutes on physical A800 GPU0**, within the three-hour ceiling and the initial 1–2-hour estimate.

- Publication tables, outcome joins and interval reduction: **330,739 checks passed** in `REPORT_TABLE_AUDIT.json`, without importing the production analyzer. All primary counts, risks, paired delta and bootstrap endpoints reproduce exactly. Independent raw label/matching reconstruction: **455,467 checks passed with zero discrepancies**; all 235 pairs and both clustered intervals reproduce.
- Native proposal/commit/clipping/cache reconstruction audit: **1,536,925 checks passed with zero discrepancies**, including all 192 parent traces, 8,647 neutral replays and 709,054 cache-prefix tensor comparisons.
- QA sample and full-vocabulary re-reduction: **32/32 passed**, with fixed hash sampling from 51,005 accepted margins and independent full-vocabulary reduction.
- Original file identities, postflight and retrieval integrity: **passed**: inputs/checkpoints/source unchanged; all 66,707 retrieved files (670,031,768 bytes) match retained identities.

The native diagnostic completed before the report builder exited with a seal-validation error. JSON had converted integer histogram keys into strings, changing their sorted hashing order at depths 10–15. The narrow CPU repair restores that declared v1 field type only for validation; it does not rewrite the original seal, pairs or outcomes. Preserve the original `EXIT=1` and failure records alongside the repaired publication. The actual retained seal has now been checked: its original 7,025,107 bytes still match file SHA256 `f7bd53131c9358d4bbab59a83be12a299744f8313fd86247590780ac0e4a7678`, and restoring only the typed histogram keys reproduces the original payload fingerprint `bd9629be436d0016979ab8cae2f736e7479c3a3374db5ac5a091f3a7b2648cf8`. The original `EXIT=1` and report-builder traceback remain intact.

The rebuilt local report agrees with native `ANALYSIS.json` in every result field except two structural aggregate floats. They differ by 2.78×10⁻¹⁷ and 3.47×10⁻¹⁸ (one and two binary64 units): native values exactly match an independent `math.fsum`, while local values exactly match ordered Python 3.11 summation. Primary estimates, interval endpoints, pair records, decisions and displayed precision are unchanged. Both versions remain recorded; no numerical tolerance was widened.

## Supported decision and limits

The prespecified request-cluster bootstrap uses 10,000 PCG64 draws with seed 20261115; all pairs and reused controls move together. The receipt reports zero undefined draws. This interval concerns request variation for **one fixed checkpoint transition and training seed**. The panel is disjoint from the excluded training/discovery requests but has prior PR39 exposure; it is not globally unseen.

The audited result supports considering a separately designed selective-retention test. It does not show that protecting these positions improves new repairs, accepted progress or complete-request serving. The limited matched coverage and residual confounding remain material limits. No result authorizes new training; this diagnostic ends with the report.

Same-team authorship overlap is disclosed: this interpretation reviewer authored the production analysis/report code. The raw reducer uses separate formulas without production-analysis imports; its author also prepared the panel. The native-accounting auditor authored the native observer. These checks share executor-team authorship; no independent GPU replication is claimed.

## Protocol and identities

The panel contains the first 192 eligible requests in original PR39 TRUE/KPLUS5 order. All 256 original requests passed the required identity, normalized-question and rendered-token exclusions; the last 64 were outside the fixed size limit. There is no shortfall. The sealed exclusion inventories cover all events 1–6 feedback, PR42 discovery inputs and all 64 PR34/40 GSM8K discovery requests; none overlaps the selected 192. Prior PR39 serving and retained older-project exposure are disclosed in `PANEL_MANIFEST.json`. Selection did not use outputs, lengths, performance or reference availability.

Each model replays the same K+5 canonical history for a given start. Each repaired/control pair uses different starts within the same request, the same one-based slot and the same K+5 accepted length. The fixed 0.25 caliper, zero-with-zero ties, replacement and hash tie-break remain unchanged. Labels, margins, pairs and all structural pools were sealed before K+6 loading. The ledger plus pinned control flow establishes this order; publication ledger rows do not separately timestamp checkpoint paths.

Serving stayed Qwen3-8B / five-layer DFlash, native BF16/FA3 greedy, block 16 (seed plus 15 proposals), concurrency 1, context 4,096, cap 2,048/EOS. All 192 parent walks reproduce retained canonical output and native traces. Margins came from original native BF16 head scores before greedy choice, with FP32 scalar subtraction; no extra head forward or layout change was used.

| Identity | SHA256 |
|---|---|
| K+4, step 512 | `4f3010e961bd7dcf35fa100486c355ffbc7629d816dda6e2b8fa001878255701` |
| K+5, step 640 | `9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795` |
| K+6, step 768 | `8c257620e44226c5c0469e5886224ed5d2dcf9fa2a94eddbdd10d7e36c697560` |
| Public panel manifest | `78b8f3f1f4f463d8beb40802eb14ead4cb6460a54955e0d2e14aa31ac2a16a36` |
| Private exact-token panel | `aa558fd2d7c14bc1044dc108731f8a0b7a9009993bd23b22cddebdc6cfd5cf46` |

Dataset revision: `740312add88f781978c0658806c59bc2815b9866`. The original pre-K+6 file was sealed at **2026-11-12 22:25:51 UTC**; its file/payload hashes above remain unchanged. Native `ANALYSIS.json` is retained as `NATIVE_ANALYSIS.json`; rebuilt tables do not overwrite that receipt.

## Work, duration and storage

| Recorded work | Completed |
|---|---:|
| K+5 complete parent walks | 192 |
| Neutral K+5 / historical K+4 blocks | 8,647 / 8,647 |
| Unique K+6 outcome blocks | 1,343 |
| Measured draft-block operations | 27,284 |
| Startup draft-block / target-forward operations | 51 / 48 |
| Total draft-block / target-forward operations | 27,335 / 27,524 |
| Cache-context forwards | 551,021 |
| Initialization/checkpoint publications | 386 |
| Optimizer steps / backward calls / timed endpoints | 0 / 0 / 0 |

There are **zero failed, unknown or unattempted prescribed native operations**. These are nested work counts, not independent response samples. Unused conservative allowances are not unattempted planned work. The original wrapper's report-only exit 1 is separately preserved.

Physical A800 GPU0 UUID: `GPU-9986735a-3e68-d1bd-5998-9634ef24341e`. Occupied time was **4,963.502 seconds**, including startup/postflight, within 10,800 seconds. Supervisor start/end: **November 12 21:20:50–22:44:01 UTC**, or **83m11s elapsed**, including admission/reporting. The estimate was 80–110 GPU minutes. Implementation, transfer and CPU audit time are separate. Both GPUs were idle at closeout.

Initial disk free space was 619.524 GB; after evidence packaging it was 618.343 GB. All 66,707 compact/private records were hash-verified on Fenrir; the 73,705,280-byte archive has SHA256 `e151c524ee1f92d0166f4f80c135475da68a5e6bf41887b39a2e2e4dc86b055f`. No artifacts were deleted. Private prompts, canonical/proposal tokens, traces and full-vocabulary tensors remain outside Git.

The narrow report repair passed 68 focused/inherited CPU tests; the independently coded reduction also passed its two synthetic/tamper fixtures. Original failure evidence and failed intermediate audit receipts are retained. No native work was rerun and no analysis threshold was widened.

## Statistical review and deliverables

ARS review covered **11/11 categories**: request heterogeneity, ecological inference, selection/Berkson bias, collider adjustment, base rates, regression to the mean, survivorship, look-elsewhere effects, forking paths, correlation/causation and reverse causality. Matched-population selection, observational margin adjustment, limited overlap, one seed and one transition remain explicit limitations. Checklist coverage does not prove absence of residual bias.

- `MATCHED_PAIRS.csv`, `STRUCTURAL_OUTCOMES.csv`, `PER_REQUEST.csv`: paired outcomes and denominators.
- `ALL_STARTS.csv`, `K5_MARGINS.csv`, `REPAIR_COVERAGE.csv`, `STRUCTURAL_POOLS.csv`, `COMPARATOR_REUSE.csv`: labels, scalar margins, coverage and reuse.
- `PRE_K6.json`, `PRE_K6_SEAL.json`, `PANEL_MANIFEST.json`, `EXCLUSIONS.json`, `INPUT_SEAL.json`, `PLAN_PUBLIC.json`: original bindings and token-redacted configuration (private plan hash retained).
- `WORK_ACCOUNTING.csv`, `COSTS.csv`, `PARITY.csv`, `POSTFLIGHT.json`: execution completeness and validity.
- `INDEPENDENT_AUDIT.json`, `NATIVE_ACCOUNTING_AUDIT.json`, `REPORT_TABLE_AUDIT.json` and `validation/`: separate retained-evidence reductions.
- `REPORT_REPAIR.json`, `ORIGINAL_REPORT_FAILURE.log`: preserved report failure and narrow repair.

This publication completes the diagnostic. No intervention, extra panel or subsequent experiment was started.
