# Independent audit of the one-event U/P experiment

## Material passport

- Mode: ARS experiment validation; offline compact-artifact audit.
- Evidence: retrieved `raw/campaign_001`, original sealed manifest and O receipts, final retention receipt and execution seal.
- Result: **passed; no accounting or native-validity inconsistency found**.
- Machine-readable audit: [FINAL_AUDIT.json](FINAL_AUDIT.json).
- No GPU replay, new generation, retraining or tuning was performed for this audit. The auditor has five passing CPU fixture tests, including a full 768-record synthetic case and deliberately invalid report/status/binding cases.

## Execution and validity

Both U and P completed one event with **256 response presentations and 128 optimizer steps**, giving **512 presentations and 256 main steps**. Each arm's complete presentation order, feature identities, sampled anchors, keep masks, learning rates and step boundaries match the retained O schedule. Each arm used **65,096 anchors and 949,560 supervised-position occurrences**. Exposure is independently reconstructed from anchors and checked against optimizer-boundary metrics; intermediate accumulation metrics are not counted twice.

The single process warm-up used two presentations and one discarded optimizer step. Its initial-state gate and both fresh U/P initial-state gates pass. Warm-up state was not inherited. The reused qualification's 27 gate digests and receipt, runner, seal and pinned-initial-state digests agree with the execution seal. The historical seven-trial qualification is not charged as new work.

All **768 replays** cover **64 requests × four arms × three rotating repeats**. All requests remain in the shared valid cohort. Independently reconstructed **37,098 native blocks** agree with canonical tokens, exact prompt IDs, EOS/cap termination, accepted prefixes, committed tokens and verification counts. F/O verification counts reproduce their historical first-event values. Repeated counts are consistent, published weights and cache contracts validate, and replay feedback/training counts are zero. No work is unattempted or of unknown completion.

U/P reference-prefix summary metrics agree at every presentation. Each arm reports **429,510 protected-position occurrences** and reference prefix weight **251,420.569641**. P's coefficient mass equals that weight; U's mass is **251,420.567932**, within the existing numerical check. These are summary checks, not an independent replay of every reference mask; full-block alignment and mask construction were established by the reused qualification and reviewed code.

## Independently recomputed comparisons

Each request contributes one verification count. Latency uses its mean of three fresh repeats; aggregate reductions are ratios of summed measurements. The repeated runs are not additional independent requests.

| Method | Verification calls | Summed mean request latency | Calls per 1,000 output tokens |
|---|---:|---:|---:|
| F | 3,140 | 54.069934 s | 160.638461 |
| O | 3,075 | 52.814925 s | 157.313143 |
| U | 3,082 | 53.012440 s | 157.671254 |
| P | 3,069 | 52.807258 s | 157.006190 |

All methods produce the same **19,547 output tokens** over the 64 requests.

| Comparison | Call reduction | Request-time reduction | Call improvements / regressions / ties |
|---|---:|---:|---:|
| P vs O — primary | +0.195122% | +0.014517% | 24 / 25 / 15 |
| P vs U | +0.421804% | +0.387045% | 28 / 22 / 14 |
| P vs F | +2.261146% | +2.335264% | 36 / 19 / 9 |
| O vs F | +2.070064% | +2.321084% | 40 / 19 / 5 |

P saves 45 calls on improving requests but adds 39 on regressing requests relative to O: **six calls saved overall**. Its summed mean latency advantage over O is **0.007667 s across the entire panel**. The three repeat-level P/O time reductions are **−0.017648%, +0.028324%, +0.032866%**. This provides no meaningful practical advantage over ordinary OSD in this screen. P's improvement over frozen is largely also achieved by O. This statement is descriptive; no significance or generalization claim is made.

## Cost and retention

| New work | U | P |
|---|---:|---:|
| Training presentations | 382.763 s | 381.627 s |
| Checkpoint save | 19.501 s | 19.196 s |
| Export | 14.319 s | 14.445 s |
| Complete update event | 419.419 s | 417.917 s |
| Model loading | 19.081 s | 18.409 s |
| Controlled initialization | 7.977 s | 7.895 s |
| Reference work, already inside event | 67.446 s | 65.698 s |
| Peak allocated memory | 29.329 GB | 29.254 GB |

New feedback generation and teacher-feature preparation are zero because retained artifacts were reused. This does not mean their historical costs were zero. Teacher-head computation during training remains included in training time. Reference costs are nested and are not added again to the update-event total. Initialization receipts and all duplicated cost receipts agree.

The discarded process warm-up took **47.756 s**, backend loading **20.009 s**, and evaluation **1,896.182 s**. Evaluation includes **1,234.589 s** of checkpoint activation across 705 publication receipts. Those repeated benchmark switches are diagnostic costs, excluded from inference latency; they are not deployment update costs. The disjoint measured phase sum is **2,854.643 s**, below the **2,860.061 s** campaign wall time, with no nested-cost double counting.

The post-exit retention audit reports **107 source files, 132 retained O artifacts and 32 qualification dependencies unchanged**, exit code zero, no deletion and **193.222 GB** free. This independent audit checks that receipt against the seal; it does not rehash remote model tensors locally.

## Interpretation boundaries

- The result concerns this one-event, one-seed development panel. It does not establish multi-event performance or reject all prefix-ranking mechanisms.
- The historical failed lambda-zero gate remains part of the evidence. Controlled qualification demonstrates one-window equivalence under the declared startup; it does **not** retroactively establish exact equality to the complete historical 128-step O trajectory. O is the retained immutable baseline.
- Request latency is performance with a checkpoint available. The roughly 418-second P update is not repaid by this panel's serving savings; this is not an adaptation-inclusive acceleration result.
- No additional experiment is needed to state the immediate finding: **the tested P modification did not deliver a useful improvement over O**. The audit authorizes no automatic tuning, additional update or campaign.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
