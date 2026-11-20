# CODE storage admission review

**The fixed 80 GB additional-study cap does not admit the current complete-run forecast.** The training peak is **86.5669 GB**, a **6.5669 GB deficit**. No GPU qualification, training, or generation has run. The original cap and all scientific settings remain unchanged.

| Phase component | Training peak, GB | Final-evaluation peak, GB |
|---|---:|---:|
| Required states/exports/features, including 5% tensor margin | 74.0669 | 49.3946 |
| Retained training compact evidence and event packaging | 9.0000 | 9.0000 |
| Fresh final raw evidence | — | 8.0000 |
| Simultaneous final packaging copy | — | 8.0000 |
| Compiler growth | 2.0000 | 2.0000 |
| Source | 0.5000 | 0.5000 |
| Workspace | 1.0000 | 1.0000 |
| **Total forecast** | **86.5669** | **77.8946** |

GB here is decimal. The training tensor calculation retains seven complete-state copies (including atomic overlap), four serving exports, and the full current feedback window. Each recovery state is bounded at 6,791,777,312 bytes by the separate metadata review; each export is 2,097,259,104 bytes. The sealed worst 128-request window has 293,039 prompt-plus-output rows when every response reaches the unchanged 2,048-token cap. Including 4,097 replacement rows at 49,164 bytes/row gives **14,608,394,304 bytes** of feature overlap. No short-response assumption enters this feature calculation. This review conservatively reserves4,097replacement rows; the formal plan can use the sealed maximum sequence3,397plus one terminal row, reducing the forecast by about0.0361GB. That does not resolve the80GBdeficit.

Lossless packaging is useful but does not justify assuming arbitrarily small compact evidence. The authenticated historical C58 compact files occupy **7.0526 GB**, including native traces. Three historical event packages compress to 9.55–9.85 MB each, but those packages omit native request files. Historical event58 generated 32,393 output tokens across128responses, only about253tokens/response. Simply doubling its compressed event size would silently depend on shorter outputs than the cap.

The CPU-only serialization sensitivity uses full2,048-token responses. At seven committed tokens per verification, its three trace copies project to **6.1470 GB** over174events. Adding 1.7134 GB for174largest historical event packages, and1.1396GB for indexes, receipts, current loose evidence and packaging overlap, gives the explicit9GB working compact forecast. Some evidence is intentionally counted twice. This is **not an absolute worst-case bound**: compression and verification counts depend on the eventual responses; the one-token-progress stress needs37.08GB for traces alone. These are synthetic serialization measurements, not scientific generations or predictions of CODE model quality. Runtime component gates must retain failures and stop if measured growth invalidates admission.

The final phase retains six recoveries and three exports, with no feedback feature pool. Its separate8GB raw and8GB packaging-overlap forecasts exceed the historical PR55 compact allocation of2.793GB for10,552generations, but remain forecasts rather than arbitrary full-cap guarantees.

The last read-only A800 measurement was493.8136GB free. This86.5669GB forecast plus the unchanged400GB shared reserve requires486.5669GB, leaving7.2467GB at that measurement. A fresh admission check is required because other users can consume space. The blocked resource is the study's80GB allowance, not current physical exhaustion.

A scoped **90GB** cap would leave3.4331GB above this working forecast; it has **not been authorized or applied**. It would not change algorithms, panels, work budgets, the400GB reserve, or historical retention. No historical checkpoints are proposed for deletion. Exact evidence and formulas are in [STORAGE_ADMISSION_REVIEW.json](STORAGE_ADMISSION_REVIEW.json), [COMPACT_STORAGE_REVIEW.json](COMPACT_STORAGE_REVIEW.json), and [COMPACT_FULL_CAP_SENSITIVITY.json](COMPACT_FULL_CAP_SENSITIVITY.json).
