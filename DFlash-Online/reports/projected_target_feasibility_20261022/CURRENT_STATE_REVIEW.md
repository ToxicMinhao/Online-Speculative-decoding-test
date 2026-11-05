# Review of the consolidated DFlash-Online report

## Material Passport

- Origin: ARS experiment-agent, validate mode.
- Date: October 22, 2026.
- Verification status: **ANALYZED**. This review examines source, retained receipts and saved measurements; it does not rerun model computation.
- Version: current-state review v1.
- Reviewed artifact: [CURRENT_STATE_REPORT.md](CURRENT_STATE_REPORT.md).
- Supporting recomputation: [CURRENT_STATE_TABLES.json](CURRENT_STATE_TABLES.json).

## Review scope

Two separately delegated reviews examined the numerical evidence and the implementation procedure while the report was drafted. Both then read the draft. The primary agent reconciled their findings with the handoff, source reports and current user decisions. This is internal source/evidence review, not external peer review or a new native reproduction.

The numerical review read the raw final campaign records, six training receipts, 24 preparation receipts, original/recovery timing records and saved numerical-diagnostic evidence. It reconstructed all 11,730 retained verification blocks and independently decoded the two saved BF16 final logit vectors without model execution.

The procedure review inspected the executed archive, maintained source archive, online-feedback implementation, pinned SpecForge sampler/attention masks/reducer, configuration, and training receipts. It checked where serving feedback ends, teacher preparation begins, reference targets are calculated and optimizer steps occur.

No SSH session, model forward, generation, training, new precision setting, tolerance change, campaign restart or artifact deletion occurred during report preparation. No raw results or historical reports were rewritten.

## Numerical cross-checks

| Check | Finding |
|---|---|
| Budget accounting | 136/144 generated answers; 16 reused original F + 120 new recovery; eight ShareGPT P-B requests unexecuted. |
| Validity denominators | 135 individually output-valid records, but only 96 records in fully qualified groups. |
| Training accounting | Six events, 96 response presentations, 48 optimizer steps; every event has 16 presentations/eight steps. |
| Source A calls | F/C/P = 812/775/890 across 16 requests. |
| Held-out B calls | F/C/P = 901/1,023/987 across 16 requests. |
| P vs C, B | −3.5191% calls; reduced harm compared with C. |
| P vs F, B | +9.5450% calls; no frozen-baseline improvement. |
| P vs F, A | +9.6059% calls; no useful native source-fitting gain in aggregate. |
| Token accounting | A: 4,251 outputs, 4,235 verification-committed tokens. B: 5,131 outputs, 5,115 verification-committed tokens. The difference is one initial prefill token per request. |
| Accepted proposals | A F/C/P = 3,436/3,474/3,357; B = 4,229/4,103/4,138. These exclude target corrections/bonuses. |
| Termination | A EOS 14/16, cap 2/16; B EOS 13/16, cap 3/16; identical across qualified arms. |
| Trace consistency | Every retained trace reconstructs its own saved answer and acceptance/commit accounting. Only ShareGPT P A07 disagrees with the paired F output. |
| Consumed campaign runner time | 37.984244345 + 1,175.149142241 = 1,213.133386586 s; reused F execution counted once. |
| Request timer total | 455.378843 s over all 136 executions, including the invalid comparison's consumed work. |
| New numerical diagnosis | Saved F logits tags/users = 25.125/25.0; P = 25.125/25.125. P's lower-ID maximum is users. All saved final logits finite. |
| Diagnostic scope | 443 target forwards, zero new answers or optimizer steps, 27.464147 s; separate from campaign costs. |

No arithmetic discrepancy was found. The JSON companion retains exact values, dataset-level aggregates, ratios, gains/regressions/ties, costs and metric definitions.

## Implementation descriptions checked

The report distinguishes the persistent `OnlineController` workflow from this fixed-pool screen. Both use the existing backend/trainer; C and P each start from released weights and a fresh optimizer. They do not sequentially train the same learner or supply their evaluation outputs as new feedback.

It also records the following details explicitly:

- The buffer/feature payload is more than response text, but contains no complete q0/r probability vectors.
- P's pre-event reference lasts across all eight steps. q0/r are recomputed from that reference for the sampled block, without advancing sampling RNG.
- Target y is a verified canonical token; draft q0 is a parallel masked-block prediction with anchor-conditioned inputs. These are different conditioning descriptions.
- Block size 16 consists of one known anchor plus up to 15 supervised predicted slots. The first response token is not a supervised prediction in this author sampling path.
- Distinct sampled blocks can overlap. All recorded presentations use 64 anchors; pass-two draws differ, and C/P corresponding draws match.
- Earlier canonical context is visible, future canonical answer features are not. Bidirectional attention among mask slots is allowed by the pinned author configuration.
- Response losses are individually normalized and averaged across two presentations before one optimizer step. Eight-block objective chunks are a memory setting, not the optimizer update cadence.
- C is hard CE and P is projected soft CE, not the old Llama/AdaFlash forward-KL experiment.
- The unusual first scheduler rate is disclosed. The nominal rate is not represented as a constant actual LR.
- `full_posterior_match` checks all saved argmax IDs, not full probability-vector equality.
- Consolidation CPU tests and byte-identical report reproduction are not described as a GPU execution of the new entry point.

## Corrections incorporated during this review

1. Made the parallel-block conditioning explicit, avoiding the impression that each predicted slot receives its full canonical prefix.
2. Distinguished accepted draft tokens, verification-committed progress and total response tokens, including the initial prefill token.
3. Replaced “requests are independent” with separate prompt/cache execution and an explicit statistical-independence limitation.
4. Replaced “over-adapted control” with “a control whose held-out drafting regresses”; the cause of that regression remains unresolved.
5. Scoped constructor correctness to its observed checks rather than a universal guarantee.
6. Superseded historical “BF16 tie unproved” wording using the later diagnostic, while preserving the original failed gate and ShareGPT exclusion.
7. Corrected explanatory pseudocode to define labels in both arms and pass original reference logits to the constructor, preserving its native argmax-before-softmax rule. Named the draft feature projection and normalization parameters explicitly in the trainable set.

## Interpretation review: 11/11 fallacy checks

| Item | Assessment |
|---|---|
| Simpson reversal | No reversal of the reported A/B aggregate directions across the two qualified datasets; dataset rows remain visible. |
| Ecological inference | Request gains/regressions/ties are reported separately; pooled gains are not attributed to every request. |
| Selection/Berkson bias | Eligibility and historical freshness filters limit the sample; population-wide claims are avoided. |
| Collider bias | No fitted covariate adjustment or inferred correlation mechanism is used. |
| Base-rate neglect | Gain/regression/tie counts include denominators; no diagnostic-classifier accuracy claim is made. |
| Regression to the mean | Requests were not chosen for extreme measured F performance; the same fixed F control is retained. |
| Survivorship | The incomplete third group, its 40 records, missing eight requests and consumed costs remain explicit. Qualified numbers are not labeled full-campaign results. |
| Look-elsewhere | No sweep, best-seed selection or selected favorable request subset supports the conclusion. |
| Forking paths | The replay repair, original failure, strict equality gate and later numerical reclassification are disclosed; settings and denominators are not silently changed. |
| Causal overreach | Source/held-out results do not identify optimizer failure, excessive adaptation, memorization or a general flaw in forward KL. The current ordinary loss is hard CE. |
| Reverse causality | Training precedes adapted evaluation and B is excluded from feedback; inference results are not used to train the evaluated checkpoints. |

No inferential test, p-value, population effect or confidence interval is claimed. The scientific conclusion remains descriptive: one seed, 16 qualified held-out requests, two completed datasets, single instrumented timing observations and one incomplete dataset.

## Verdict and practical limits

The reviewed report supports discussion of the current state without conflating implementation readiness, target-constructor behavior, actual source fitting, transfer and numerical output validity. Its unfavorable results are not hidden by the fact that P beats C on held-out verification counts.

The review does not establish universal correctness, independent statistical replication, a useful projected-target mechanism, or serving speedup. Large feature/model artifacts remain on A800; local review uses compact receipts and saved source except for the separately retained final-logit diagnostic tensors. The current remote artifact inventory was not rechecked during this documentation task.

The full screen remains stopped. Discussion points in the report are proposals for the user's next decision, not instructions to resume, retune or implement another mechanism.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/projected_target_feasibility_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
