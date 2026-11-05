# Checkpoint-only projected-target fitting diagnostic

October 22, 2026 · ARS experiment-agent · Status: executed and independently checked.

**The projected-target checkpoint did not fit its own fixed target better than released DFlash.** Its KL is higher on all 32 saved layouts, covering all 16 source responses. This corrects the unverified premise that “KL improved but drafting dropped.” The current evidence instead shows failure to obtain a useful model fit under the declared eight-step recipe. This does not establish that the target is unattainable or that another loss direction would fix the problem.

## Scope and controls

Completed on A800 GPU 0 using only retained GSM8K and CodeAlpaca artifacts: eight source responses per group, both original anchor layouts per response, 64 blocks per layout, and F/C/P on every layout. Total: **96 checkpoint forward evaluations / 6,144 logical blocks; zero optimizer steps, zero new generated answers**. ShareGPT and held-out B were not evaluated. The stopped generation campaign remains stopped.

F is released Qwen3-8B-DFlash-b16, C is its final ordinary hard-label CE checkpoint, and P is its final projected-target checkpoint. Each group uses its own saved C/P exports. Qwen3-8B teacher features and verified labels are unchanged. Models/head remain BF16 with original FP32 objective arithmetic; eager attention, block size 16, gamma 7, eight-block vocabulary chunks and original masks are preserved. There is no full target transformer execution or feedback collection.

The existing SpecForge block forward is reused with exact saved anchors; no sampler is called. Every F/C/P prediction has identical conditioning, labels and weights. F supplies the common q0 and r, and defines correct/wrong membership for all arms. Only response positions receive weight; slot zero, prompt positions and out-of-range positions remain excluded.

The first F evaluation reproduces the original first P objective before any training **exactly: 0.9344481825828552**. This passes the predeclared 2e-5 absolute plus 2e-5 relative tolerance. Strict export tensor equality, canonical token equality, anchor equality, all exposure receipts, finite metrics and unchanged forward RNG checks pass. Eighteen focused tests pass. An independent calculation from raw position rows reproduces aggregate KL within 2.92e-8.

## Actual target fit

Numbers below use the original weighted loss denominator for each response-layout, then average layouts equally. Both passes are retained. Lower KL(r || q) is better for the proposed objective; it is not C's hard-label training objective.

| Dataset | F KL | C KL | P KL | F matching prefix | C matching prefix | P matching prefix |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 0.1842 | 1.5245 | 1.2266 | 5.752 | 6.570 | 5.466 |
| CodeAlpaca | 0.1485 | 1.5153 | 1.1333 | 6.259 | 7.819 | 6.298 |
| Combined | **0.1664** | **1.5199** | **1.1799** | **6.005** | **7.195** | **5.882** |

The matching prefix is the contiguous sequence of correct sampled-block predictions before the first mismatch, starting after the known anchor. It includes a correctly matched EOS. It is **not** native accepted progress or a serving-time result. Full matches reaching a saved cap boundary are flagged as censored; no unseen tokens are credited.

The fixed target entropy is 1.1702 overall. Soft CE is 1.3365 for F, 2.6900 for C and 2.3501 for P. Subtracting the common entropy reveals the actual KL; the larger P value is not an entropy artifact. Tiny negative FP32 KL residuals on preserved F-correct rows remain in the raw evidence instead of being clamped.

P is closer to r than C on 14 of 16 source responses, but **farther from r than F on every response and every individual layout**. C's higher KL should not be called a failure to minimize its own loss: it was trained against hard verified tokens, not r. Its better matching prefixes also show why lower r-based KL across differently trained checkpoints is not sufficient to rank useful drafting.

## Repairs and damage coexist

There are 29,928 sampled supervised position exposures per checkpoint: 13,633 initially wrong and 16,295 initially correct. These cover 4,197 distinct sampled source-token positions. Overlapping blocks and two passes repeat positions; these are not independent tokens or examples.

| Final checkpoint | Repairs / initially wrong | Damage / initially correct | Net additional correct exposures |
|---|---:|---:|---:|
| C | 5,561 / 13,633 (40.79%) | 3,201 / 16,295 (19.64%) | +2,360 |
| P | 4,007 / 13,633 (29.39%) | 3,762 / 16,295 (23.09%) | +245 |

Thus P does learn some useful argmax changes. It also damages many previously correct decisions. A small positive net token count does not ensure a longer accepted prefix: later repairs can be blocked by an earlier error.

At the 2,048 matched block layouts, P versus F produces 714 longer prefixes, 787 shorter prefixes and 547 ties. Averaged over each response's two layouts, eight responses improve and eight regress. P's mean prefix falls from 6.005 to 5.882 overall. It is lower than C on 15 of 16 source responses.

The fixed correct/wrong partitions both show worse r-based fit after P training. This is not merely an aggregate cancellation where successful fitting on initially wrong rows hides damage on correct ones. Partition numerators, conditional means and contributions to the original denominator are retained separately in SUMMARY.json and layouts.csv.

## What this resolves, and what it does not

The earlier qualified native source counts were F=812, C=775 and P=890 for identical outputs; held-out counts were F=901, C=1,023 and P=987. Those are historical measurements, not newly generated or retimed here. This diagnostic supplies missing evidence about the actual checkpoints: P's source-fitting premise is not established, and its aggregate sampled-prefix regression is consistent with its unfavorable native source result. CodeAlpaca's near-tied/slightly better sampled prefix does not resolve its native trajectory behavior.

The result does not establish that forward KL is generally unsuitable, that the projected target is impossible for this backbone, or that a smaller learning rate will solve the issue. Only final checkpoints were tested; these data cannot identify the first harmful optimizer step or separate step magnitude from sequential response interference and target trade-offs. The projection constructor can be correct while the finite shared-parameter update fails to realize its intended behavior.

**Recommended next decision:** prioritize a bounded check of actual model fitting, retaining the fixed r objective and observing its matched-example loss before and after real optimizer steps, with a predeclared smaller-update control. This would test update scale instead of assuming it is the cause. Adding an AdaFlash-style loss now would alter the optimization and potentially the target/preservation premise simultaneously. No new training or loss modification is authorized or launched by this report.

## Runtime, storage and evidence

Diagnostic driver runtime was **34.89 seconds**, including initial model loading (15.76 s) and group checkpoint loading (2.68 s within the remaining 19.13 s). Implementation, process imports and CPU-test preparation are outside that driver timer. This is analysis cost, not deployment latency. Peak allocated GPU memory was 13.09 GB (12.19 GiB); peak reserved was 13.65 GB. Exit code 0; both A800 GPUs idle afterward.

A800 free storage was 13.474 GB before launch and 13.456 GB afterward; output is about 17 MiB. The 8 GiB reserve and 256 MiB write budget were respected. Fenrir has 85.08 GB free after retrieval. No artifacts were deleted. Original features and exports remain unchanged. Two pre-model launcher/test failures are retained and explained in IMPLEMENTATION_REVIEW.md; they performed no model evaluations.

- [Compact comparison](results/COMPARISON.csv)
- [Validated aggregates and per-response results](results/SUMMARY.json)
- [Every response-layout and fixed partition](results/layouts.csv)
- [Every sampled block and first-mismatch result](results/prefixes.csv)
- [Every supervised exposure, margin and prediction](results/positions.csv)
- [Plan, exact paths and settings](results/PLAN.json)
- [Initial objective qualification](results/QUALIFICATION.json)
- [Completion and memory counters](results/COMPLETION.json)
- [Execution log](FIT_CHECK_001.log)
- [Implementation and independent review](IMPLEMENTATION_REVIEW.md)
- [Diagnostic entry](staging/fit_check.py), [metrics](staging/fit_metrics.py), [aggregation](staging/summarize_fit.py)

The maintained A800 entry is `/data/sources/minhao/DFlash-Online/experiments/projected_target/fit_check.py`. Fenrir DFlash-Research remains unchanged. The bounded comparison is complete and stopped; no campaign or new mechanism has been started.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
