# One-event O/C/K/A results

## Material Passport

- Mode: ARS experiment-agent validate; verification status **ANALYZED**. Completed native measurements and retained evidence were audited; no new training or native rerun was performed during this review.
- Source: `raw/campaign_001`; supervisor exited 0, 2026-10-26 22:52:49–23:47:41 UTC (**54m52s**).
- Scope: three new one-event arms, 384 main optimizer steps; 768 fresh four-arm evaluations. Stage 1/project None unchanged.

## Finding

**First-error selection gave a modest improvement under both loss choices. Replacing forward KL with canonical CE did not help drafting in this screen. K was the best of the four arms.** This is a result on one trained seed and the reused 64-request GSM8K development panel, not a validated general algorithm or an adaptation-inclusive speedup.

## Setup and validity

The pinned Qwen3-8B / DFlash-b16 pair and DFlash-Online infrastructure were unchanged. Each C/K/A arm began from released draft weights, fresh BF16 AdamW and the qualified seed-17 full RNG. All reused the same 128 frozen-generated feedback responses/features, exact 256-presentation schedule, two passes and 128 optimizer steps. Constant learning rate 3e-4, weight decay 0.01, response batch 1 with accumulation 2, up to 512 anchors per response, eight-block memory chunks, decay gamma 7, BF16 probability arithmetic and Flex attention remained fixed. Only supervised positions receive loss; full-block attention remains intact.

All arms used the same 64 held-out development inputs, greedy block size 16, 2,048 output tokens or EOS, native SGLang/FA3 and the pinned cache policy. Training sequence limit was 2,272 and serving limit 4,096. Three fresh repeats used rotating method order, capture disabled and no concurrent training or artifact transfer. Evaluation inputs never trained these arms.

**All 768 outputs and termination records matched retained F, with no invalid or incomplete requests.** Fresh O call counts matched retained O. The independent audit reconstructed 36,786 native verification blocks; source/configuration/controls were preserved (117 source files, 10 execution files, 32 dependencies, 144 retained artifacts). The seven-window GPU qualification passed before training, including exact all-selected K/O and A/C comparisons and current-mask/gradient checks. Its seven temporary steps, 14 presentations and 30 bounded logit-gradient checks are separate from main work.

## Available-checkpoint serving results

Calls use one validated trajectory per method/request; timing is the sum of each request’s mean over three repeats. All methods produced the same 19,547 output tokens. Positive reduction means improvement. Training, loading and activation are excluded from these request timings.

| Arm | Training objective/positions | Verification calls | Calls / 1,000 output tokens | Sum of mean request times |
|---|---|---:|---:|---:|
| O | Forward KL; all valid proposals (retained OSD) | 3,075 | 157.313 | 52.777 s |
| C | Canonical CE; all valid proposals | 3,102 | 158.694 | 53.258 s |
| K | Forward KL; current matching prefix plus first error | 3,027 | 154.858 | 52.021 s |
| A | Canonical CE; current matching prefix plus first error | 3,058 | 156.443 | 52.456 s |

| Planned comparison | Question | Call reduction | Request-time reduction | Requests: fewer / more / tied calls |
|---|---|---:|---:|---:|
| K versus O | Selection with KL | +1.561% | +1.433% | 33 / 21 / 10 |
| A versus C | Selection with CE | +1.418% | +1.505% | 35 / 17 / 12 |
| C versus O | CE versus KL, all proposals | -0.878% | -0.911% | 14 / 30 / 20 |
| A versus K | CE versus KL, selected prefix | -1.024% | -0.837% | 21 / 31 / 12 |

K saved 92 calls on improving requests and added 44 on regressions, for **48 net calls saved** versus O. Its summed request-time saving was **0.756 s** over the panel. A saved 44 net calls and 0.802 s versus C. Both selection comparisons had the same favorable aggregate direction in all three timing repeats; call counts were identical across repeats. The repeats assess timing stability for these checkpoints, not independent training seeds.

A descriptive 10,000-resample paired-request bootstrap (seed 20261027) places the K/O call reduction at 0.202–2.876% and time reduction at 0.109–2.691% (95% percentile ranges). A/C call reduction spans −0.062–2.832%. These summarize sensitivity to the mix of the 64 observed requests, not training-seed robustness, unseen-workload coverage or confirmatory significance. All four comparison directions also survive leaving out any single request. Full calculations and limits are in RESULT_INTERPRETATION.md.

Historical F used 3,140 calls on these inputs: O/C/K/A reduce that count by 2.070% / 1.210% / 3.599% / 2.611%. F was not timed freshly in this campaign, so no F latency comparison is made.

## Exposure and update costs

Each new arm saw the same 65,096 sampled blocks and 949,560 available supervised-position occurrences across its 256 presentations. Occurrences include overlap and repeat passes; they are not unique tokens. Only optimizer-boundary accumulation rows are summed.

| Arm | Selected occurrences | Fraction of available occurrences | Training loop | Complete update event | Loading + fresh-state check | Peak allocated GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| C | 949,560 | 100.00% | 209.963 s | 241.748 s | 23.638 s | 20.473 GB |
| K | 534,063 | 56.24% | 325.094 s | 356.563 s | 23.516 s | 20.472 GB |
| A | 542,343 | 57.12% | 213.625 s | 245.658 s | 23.556 s | 20.473 GB |

Complete event time already includes the training loop, checkpoint persistence, export and commit overhead. Selection timers (C/K/A: 9.550/13.072/13.267 s) and K’s teacher-head time (103.525 s) are nested, not added again. CE arms skip teacher vocabulary-probability projection, but still consume the retained target features. O was reused, so this is not a fresh O-versus-K training-cost benchmark.

There was no new feedback collection or teacher-feature preparation; their earlier costs are not zero or erased by reuse. Qualification cost 642.040 s including setup (639.096 s inside its driver). Evaluation backend loading cost 21.257 s. All benchmark replay/switching work cost 1,625.394 s, including 975.354 s of activation and separately logged 9.068 s of weight validation. These repeated checkpoint switches are experiment overhead, not a deployment latency claim. The inner campaign timer was 3,209.824 s; the supervisor’s 54m52s also includes preflight and remote CPU tests.

**No adaptation-inclusive speedup is established by this panel.** The reported serving benefit assumes the checkpoint is available. A deployment comparison must charge the applicable update/preparation/publication costs for each alternative, rather than treating retained O’s update as free.

## Interpretation limits

- Selection helps in this particular one-event comparison under both KL and CE. The combined mask and selected-weight normalization were tested together; this does not isolate adaptivity, early-position emphasis, or normalization as the cause.
- A/K compares the loss choice under the same dynamic selection rule, not identical realized masks. Their evolving predictions select different positions: A supervises 8,280 more occurrences than K. Fewer supervised positions do not imply fewer transformer computations.
- Later positions still participate in full attention. The implemented exclusion is from direct prediction loss, not from all contextual influence.
- K still worsens 21 of 64 requests versus O. Its 1.56% call and 1.43% time gains are modest; they do not demonstrate robust gains across seeds, domains, multi-event sessions or long conversations.
- Raw training losses have different objectives and selected supports, so lower recorded loss is not used to rank methods.
- The development panel was already inspected in earlier project experiments. All four planned contrasts and every regression are retained. Three timing repeats are not 192 independent requests.
- Startup qualification establishes checked-window equivalence and fresh state, not numerical identity to the full historical 128-step O training trajectory; the original startup failure and its review remain preserved.

The evidence supports a narrow statement: **keeping forward KL and selecting the current matching prefix plus its first error was the best tested choice on this panel.** It does not yet explain the underlying failure of full-block training or establish the final mechanism. No additional run, tuning or mechanism change was performed.

## Evidence

- [Independent audit](FINAL_AUDIT.md) and [machine-readable audit](FINAL_AUDIT.json).
- [Interpretation and request-resampling sensitivity](RESULT_INTERPRETATION.md).
- [Per-request results](raw/campaign_001/evaluation/report/PER_REQUEST.csv), [all timing repeats](raw/campaign_001/evaluation/report/ALL_REPEATS.csv), [repeat totals](raw/campaign_001/evaluation/report/REPEAT_TOTALS.csv).
- [Costs](raw/campaign_001/COSTS.json), [qualification](QUALIFICATION_REPORT.md), [source/artifact preservation](raw/POSTFLIGHT.json), [retrieval verification](RETRIEVAL.json).


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_first_error_execution_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
