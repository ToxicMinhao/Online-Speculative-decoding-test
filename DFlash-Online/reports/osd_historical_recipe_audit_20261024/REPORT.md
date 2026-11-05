# Historical OSD settings and baseline fairness

## Material Passport

October 24, 2026. ARS experiment-agent validation; **ANALYZED** from preserved
execution configurations, recorded optimizer-step learning rates, source code,
training receipts and completed reports. Three independent scoped source reviews
cover the successful native reference, older Llama session experiments and early
project milestones. No model execution, remote operation, tuning or source change.
This audits the principal experiment lineages, not every historical trial.

## Conclusion

**Yes, the early configurations survive, and the recent negative eight-response
DFlash-Online experiment did not match the successful OSD reference recipe.** Its
forward-KL objective was correctly implemented and matched to the other objective
arms, but several learning settings and the training budget differed materially.
It was not sufficient evidence that a competent persistent OSD baseline fails.

**No, we were not always using that mismatched recipe.** Older author-code Qwen
experiments and qualified Llama OSD experiments already used the successful core
recipe and produced positive as well as negative findings. The latest improvement
cannot be attributed to hyperparameter repairs alone: feedback volume, training
horizon, evaluation sample and runtime also changed.

Here “OSD reference settings” means the pinned AdaFlash repository's forward-KL
training path and our documented successful native execution. It does not mean
that every implementation called OSD has one mandatory hyperparameter vector.
It also does not mean full AdaFlash, which adds a different objective/head.

## Recent Qwen comparison: what actually changed

The following old-screen column comes from its executed GSM8K configuration and
raw training result, not present-day defaults. Each CodeAlpaca and ShareGPT group
used the same old recipe with a fresh released draft/optimizer.

| Setting | Negative DFlash-Online screen, Oct23 | Successful author-code reference, Oct2 | New persistent rerun, Oct24 |
|---|---|---|---|
| Target/draft | Qwen3-8B / DFlash-b16 | Same released revisions | Same released revisions |
| Objective | Forward KL, T1, no hard-label mix | Same | Same |
| LR | Nominal6e-4, eight-step cosine | Constant3e-4 | Constant3e-4 |
| Optimizer arithmetic | FP32 master weights/Adam moments; BF16 model | Direct BF16 AdamW | Direct BF16 AdamW |
| Weight decay | 0 | .01 | .01 |
| Response order | Fixed order each pass | Shuffled each pass | Shuffled each pass |
| Anchors per response presentation | 64 in every recorded presentation | Up to512 | Up to512 |
| Probability/KL arithmetic | FP32 after BF16 head projection | BF16 rows; FP32 weights/reduction | BF16 rows; FP32 weights/reduction |
| Draft training attention | Eager | Flex | Flex |
| Responses per event | 8 | 128 | 128 |
| Events / optimizer steps | 1 / 8 per dataset | 58 / 7,424 | 58 / 7,424 |
| Distinct trained responses | 8 per dataset | 7,424 | 7,424 |
| Response presentations | 16 per dataset | 14,848 | 14,848 |
| Passes / effective response batch | 2 / 2 | 2 / 2 | 2 / 2 |
| Serving | Official DFlash HF/SDPA | SGLang/FA3 | Qualified SGLang/FA3 |
| Output cap | 512 or EOS | 2,048 or EOS | 2,048 or EOS |
| Training sequence | Exact IDs, limit2,048 | Retokenize and truncate/pad to2,048 | Exact IDs, limit2,272, no truncation |
| Held-out GSM8K questions | 8 | 1,024 | 1,024, same reference inputs |
| GSM8K held-out calls | 303→361: **19.14% more** | 50,549→44,556: **11.86% fewer** | 50,584→44,730: **11.57% fewer** |

The old screen's three-dataset pooled increase was9.28%;19.14% is the GSM8K-only
value appropriate to the row above. These are separately trained endpoints and
different old-screen evaluation inputs, not a paired intervention on one setting.

Raw old-screen step learning rates are:
`[6.237397e-4, 6e-4, 5.323951e-4, 4.312173e-4, 3.118698e-4, 1.925224e-4, 9.134456e-5, 2.373968e-5]`.
Thus “6e-4 versus3e-4” alone is an incomplete description. Its mean rate was
3.508536e-4; integer warmup steps were zero. Raw records confirm identical response
order in both passes and64anchors for every presentation in all three groups.

The new GSM8K trajectory has **928 times** the distinct feedback responses and
optimizer steps of that old GSM8K group. This is not evidence that928times the
training is necessary, or that scale alone caused the reversal. None of the old
GSM8K A/B answers hit512tokens, so that cap did not truncate those16examples.
FP32 master updates are not inherently erroneous; they define different updates
from direct BF16 AdamW. The effect of each numerical choice remains unisolated.

Sources: [old executed config](../dflash_online_osd_20261023/execution_001/final/campaign/gsm8k/O/config.yaml),
[old raw steps](../dflash_online_osd_20261023/execution_001/final/campaign/gsm8k/O/trainer/events/update/RESULT.json),
[old executed plan](../dflash_online_osd_20261023/execution_001/final/campaign/PLAN.json),
[old result](../dflash_online_osd_20261023/execution_001/RESULTS.md),
[reference audit](AUTHOR_REFERENCE.md),
[new sealed config](../dflash_online_osd_rerun_20261023/sealed/config.yaml),
[new result](../dflash_online_osd_rerun_20261023/RESULTS.md).

## Does this invalidate earlier comparisons?

Three questions need separate answers:

1. **Were comparison arms treated consistently?** In the recent short objective
   screen, C/P and the OSD follow-up shared feedback, anchors, loss weights,
   optimizer/schedule and training budget; native outputs were checked. That is a
   controlled objective comparison under that recipe. There is no evidence here
   that only OSD was given the worse schedule. Historical ShareGPT P's output
   mismatch remains excluded; it is not retrospectively repaired by this audit.
2. **Was OSD a demonstrated strong baseline for that workload?** Not established
   by matching the other arms or passing correctness checks. The recent short
   screen inherited the projected-target feasibility recipe instead of recovering
   the known successful OSD recipe. Describing its negative result as general
   OSD failure would overstate the evidence. Matching a weak shared recipe does
   not make a method-superiority claim convincing.
3. **Were all old studies subject to that issue?** No. Older native author-path
   studies already used constant3e-4, BF16 AdamW/decay.01,512anchors and Flex.
   Some accumulated gains, while some limited-feedback transfers worsened.
   Those are different workload/budget findings, not all manifestations of the
   newer screen's optimizer defaults.

Short-feedback scenarios can be legitimate when feedback availability and costs
are part of the problem and applied equally. They do not establish behavior after
thousands of feedback responses. Conversely, the new long-run positive result
does not prove that a first update must help, or overturn a paired first-update
regression.

## Historical evidence that must remain visible

| Retained study | Recipe distinction | Result and valid scope |
|---|---|---|
| Sep28 custom aligned adaptation | LR1e-4/FP32-master custom updater; not author OSD reproduction | Released Llama DFlash calls fell12.45%; EAGLE result differed. Ordinary learning was not uniformly negative. |
| Oct2 native Qwen OSD | Constant3e-4/BF16AdamW/decay.01/512anchors;128-response buffers | Already11.86% fewer held-out GSM8K calls. |
| Oct4–5 author-native generalization | Established native recipe, two seeds, same method on math/code | Buffered GSM8K calls fell10.78–11.28%; CodeAlpaca slightly worsened. Dataset dependence survives without the recent recipe mismatch. |
| Oct15 Llama two-update author path | Eight-response events, two passes; qualified author sampling/weighting | U2 saved11.70% request time and16.88% calls versus frozen. Eight-response updates can help. |
| Oct15 Llama per-turn author path | One response/two steps per event, full history, same core recipe | WildChat9.66% time/9.12%calls saved overall;39of110latest updates increased next-request calls. |
| Oct16 first-update LR control | Same native author path, matched feedback/anchors,3e-4vs1e-4 | Next-request calls544F/592high/545low; both rates improved source-response fit. A specific transfer result, not general failure. |
| Oct20 fixed-feedback curve | Author-style optimizer/objective, four sources/eight steps | Held-out calls1,857F/1,901S8; lowerLR3e-5gave1,858. Tiny fixed-pool findings remain valid for that setup. |

Sources and exact implementation bindings are in [historical sessions](HISTORICAL_SESSIONS.md)
and [early milestones](EARLY_SCOPE.md). The earlier capture-block “OSD-style” KL
updater was also genuinely different from the later author path: sampled examples,
teacher conditioning, weighting and reduction changed. The paired comparison
recorded3.34times the supervised exposure for the author path at equal step counts.
This supports careful baseline implementation, not a causal attribution to one
ingredient. Some earlier selector/pass-count studies used AdaFlash's mixed
reverse-KL objective; they must not be relabeled pure forward-KL OSD.

## What matches now, and what still does not

The latest rerun restores the successful reference's principal optimizer,
precision, ordering, anchor-cap and learning-horizon choices. It still uses exact
token preparation, a2,272training limit, newer anchor eligibility, chunked loss
execution, fixed seed17 and synchronous publication. The historical reference
retokenized/truncated/padded to2,048 and trained/published asynchronously. These
remaining differences are disclosed, not assumed harmless or causal.

Even the successful author-code reference was not a literal paper reproduction:
the pinned shell recipe uses two training replicas/global response batch4; our
executed reference used one replica/global batch2, matching the paper's stated
effective batch. This deviation was recorded before execution. Distinguish the
pinned script, actual reference execution and our port.

## Consequence for the research story

Use the defensible statement: **online OSD has demonstrated useful gains, while
particular short-feedback updates and transfers can still be harmful.** Do not
build a replacement's motivation on “OSD cannot learn” or on the recent short
screen alone. Do not describe all numerical and sampling changes as guaranteed
performance-neutral implementation details.

For future mechanism comparisons, the recommendation is a competent OSD control
under the same model, workload, available feedback, evaluation and cost boundary.
The current long-run recipe supplies a working positive reference; it cannot
silently replace the baseline for a different scarce-feedback scenario. A causal
explanation of the latest reversal would require a matched-feedback comparison,
which this audit has not run or newly authorized. Preserve old observations and
narrow unsupported generalizations instead of rewriting the results.

No net online speedup follows from checkpoint-ready gains alone. The current run
incurred5.67hours of blocking adaptation; neither old nor new final-evaluation
ratios should hide that cost. Timing repeats are not independent training seeds.

## Evidence limits

Configurations, source, logs, token-level measurements and compact optimizer/
sampling receipts remain available. Some bulky historical optimizer/feature
payloads were explicitly retired, so not every historical tensor-level replay is
possible. This review did not rerun models or verify every historical trial.
It directly rechecked the recent old-screen recorded learning-rate/order/anchor
facts and relies on the linked source-bound audits for the other lineages.
No causal, universal-success/failure, statistical-significance or exact numerical
equivalence claim is made. No settings, models, experiments or research stage were
changed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_historical_recipe_audit_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
