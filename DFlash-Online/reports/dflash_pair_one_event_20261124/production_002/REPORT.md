# BASE/PAIR one-event result

## Material Passport

- Status: **completed and audited**; ARS experiment-agent run/validate workflow.
- Scientific execution: `416cb7931028621c09b8fa983e53fc8a7dcebaab`, `execution_002`.
- Native qualification reused: `79b8c076610e9b4d310169b3891cb76ac6d0f9bb`; no new qualification updates.
- Recipe: PR56 `1551e5e5d0dd7736d8441977aa3dea250535da7b`; reviewed PR58/report `e38c942b5f607bb0b816cb05f3535a78c48110ec`.
- Statistical interpretation: **ANALYZED**. Independent CPU reconstruction of retained results passed; this is not an independent GPU replication.
- Population: one seed, one feedback event, 128 held-out OpenCode requests from the existing panel. No experiment expansion or stage change.

**PAIR did not demonstrate a held-out serving advantage.** Its complete-request latency reduction was **0.150%**, with paired 95% confidence interval **[−0.411%, +0.697%]**. Verification-call reduction was **0.248% [−0.325%, +0.804%]**. The fixed training witness modestly favored PAIR, but measured training time increased **8.65%**. This is an unresolved serving result, not evidence of equivalence or universal ineffectiveness.

## What ran

BASE is the complete native C+K+ objective. PAIR adds the already-qualified teacher-conditional KL on each block's current first error, weighted by its final response-wide redistributed coefficient and `1−mu`. Native BF16 full-vocabulary forward KL, current role selection, response normalization, C prefix allocation and accumulation are unchanged. The auxiliary retains FP32 selected-row arithmetic and detached teacher/role/weight quantities. No loss, tolerance, coefficient or kernel was changed for production.

Each arm began independently from released Qwen3-8B/DFlash weights with fresh AdamW and seed 17, generated its own 128 responses to the same sealed training prompts, and trained for two passes: 256 presentations and 128 steps. The learning recipe retained constant LR `3e-4`, weight decay `0.01`, clipping at 1, microbatch 1/accumulation 2, up to 512 anchors, eight-block head chunks, gamma 7, eta 0.5, prefix alpha 0.5 and bound 0.25. The inherited configuration horizon remains 7,424 steps/58 events; the production coordinator enforces the separately sealed **one-event** limit. The final states contain exactly 128 steps and 256 presentations each.

Training used the pinned BF16/FlexAttention path on physical GPU B. Feedback serving used the pinned SGLang/FA3 path on A. Final scoring ran on A with B idle, no concurrent training or artifact transfer, capture disabled, greedy block-16 decoding, 2,048 tokens or EOS, context 4,096 (qualified physical context 4,112), concurrency 1, full cache flushing and checkpoint/helper refresh. Each of 128 held-out requests had three rotating-order repeats per arm. The 24 declared generation warmups are separate from the 768 scored generations. No target-only AR comparison was added.

Fresh-origin, empty-Adam, RNG/startup restoration, final recovery/export equality and native publication checks passed. The independent audit validated all **792** evaluation/warmup outputs and **47,925** verification blocks against retained frozen-F canonical tokens and termination, including repeat-count consistency. There were **zero invalid, unknown or unattempted generations**, and no GPU failure or retry in this production attempt. All held-out requests ended at EOS; none reached the cap. Canonical equality is not a semantic answer-quality assessment.

## Held-out serving

Count totals use one validated trajectory per request. Times sum the three-repeat mean of each request. Positive reductions favor PAIR.

| Measurement | BASE1 | PAIR1 | PAIR reduction, 95% CI |
|---|---:|---:|---:|
| Complete-request time, s — primary | 136.709589 | 136.504564 | **0.14997% [−0.41101, +0.69737]** |
| Native generation time, s | 132.713467 | 132.509288 | 0.15385% [−0.41177, +0.69995] |
| Cache maintenance, s | 0.063865 | 0.063963 | −0.15414% [−2.72951, +2.31156] |
| Verification calls | 7,669 | 7,650 | 0.24775% [−0.32513, +0.80376] |
| Output tokens | 50,226 | 50,226 | Identical |
| Native accepted proposals | 42,429 | 42,448 | — |
| Stop-clipped matching proposals | 42,555 | 42,575 | — |
| Native accepted proposals/call | 5.532534 | 5.548758 | — |
| Stop-clipped matching proposals/call | 5.548963 | 5.565359 | — |
| Committed progress/call | 6.532534 | 6.548758 | — |

PAIR wins on calls for **50 requests**, regresses on **43**, and ties on **35**: 100 gross calls saved and 81 added, giving 19 net saved. Both arms commit 50,098 post-seed tokens; the native acceptance and progress ratios are consequently related to call counts and are not independent confirmations. The stop-clipped trace counter has a different definition and is reported separately. Output lengths range from 144 to 1,314 tokens, mean 392.391, median 359.5.

| Repeat | BASE complete-request total, s | PAIR total, s |
|---|---:|---:|
| 0 | 136.568097 | 136.468831 |
| 1 | 136.746341 | 136.517774 |
| 2 | 136.814330 | 136.527087 |

The prespecified estimator is `100 × (1 − sum(PAIR request means)/sum(BASE request means))`. Intervals use 10,000 paired whole-request bootstrap draws, PCG64 seed 20261124 and linear 2.5/97.5 percentiles. The three timing repeats are averaged before resampling; they are not training replications. Secondary intervals are descriptive and unadjusted. These intervals condition on these fixed checkpoints and requests and do not capture training-seed robustness or every source of run-level timing drift. [Per-request table](PER_REQUEST.csv), [all repeats and warmups](ALL_REPEATS.csv), [full summary and bootstrap ordering](SERVING_SUMMARY.json).

## Fixed-origin training witness

The witness was sealed before training from the first eight BASE feedback responses, with eight fixed anchors each: **64 overlapping block occurrences**, 64 original first errors and 326 old correct-prefix positions. Origin identity, original error/rival and support stay fixed; endpoint rivals are recomputed. At the original error, `M_original = z_y − z_r0` uses the frozen original rival, while `M_current = z_y − max(v != y) z_v` uses each endpoint's strongest noncanonical rival. Both use native logits followed by FP32 subtraction. Each endpoint was measured on the same blocks. This is training-fit evidence, not held-out transfer.

| Witness measurement | BASE1 | PAIR1 |
|---|---:|---:|
| Original error becomes correct / 64 | 49 | 51 |
| Useful repair: error correct and all preceding matches survive / 64 | 43 | 47 |
| Damaged old-prefix positions / 326 | 18 | 16 |
| Blocks with damaged old prefix / 64 | 8 | 5 |
| Total consecutive matching-prefix occurrences | 487 | 494 |
| Net matching-prefix gain from origin | +161 | +168 |
| Mean original-rival margin change / 64 | +2.722656 | +3.031250 |
| Mean strongest-current-rival margin change / 64 | +2.468750 | +2.660156 |
| Current strongest rival changed / 64 | 18 | 22 |
| Original rival beaten but current prediction still wrong / 64 | 1 | 1 |

Useful repairs comprise 42 shared, five PAIR-only, one BASE-only and 16 neither. PAIR has seven more net prefix occurrences, with 13 improved blocks, eight regressed and 43 tied versus BASE. Mean PAIR-minus-BASE margin differences are +0.308594 against the original rival and +0.191406 against the current strongest rival. Native argmax resolves ties; a zero margin alone is not a correctness rule. There were no origin blocks without an error in this particular fixed sample; those paths were covered by CPU fixtures.

The audit reconstructed support, predictions, tie predicates, margin differences and aggregate outcomes from retained observations. Full vocabulary logits were not retained, so it did not independently redo the BF16 projections. No extra model evaluation was run. [Witness summary](WITNESS_SUMMARY.json); private matched blocks remain in the retained execution directory.

## Learning activity and costs

PAIR's auxiliary was positive on **52,646/65,929 repair occurrences (79.85%)**. This counts positive computed auxiliary values, not nonzero gradients. Mean pair mass `mu` was 0.874534 and mean pair-conditional KL was 1.505151 across repair occurrences. The teacher preferred the canonical token on 64,247 occurrences, tied on 1,682 and strictly preferred the rival on zero; no roundoff clamp was used.

Equal-response means were BASE ordinary FKL 0.195892; PAIR ordinary FKL 0.198252, auxiliary 0.014762 and total 0.213014. The auxiliary mean was 7.45% of PAIR's ordinary FKL mean. These are changing-current-role training losses, not a fixed held-out comparison. Both arms had 1,342,742 available supervised proposal occurrences over the two passes; selected counts were 807,408 and 810,687. Mean preclip norms were 0.286102 and 0.298729; neither arm clipped in any of its 128 steps.

| Recorded cost, seconds | BASE | PAIR |
|---|---:|---:|
| Full arm wall time | 1,001.733 | 1,051.829 |
| Trainer loading | 20.213 | 20.543 |
| Serving loading | 20.508 | 21.007 |
| Feedback collection calls | 273.096 | 275.281 |
| Feedback buffer persistence | 58.446 | 59.656 |
| Teacher preparation | 8.206 | 8.434 |
| Prepared-feature persistence | 10.599 | 10.636 |
| Training timer | **510.163** | **554.307** |
| Checkpoint persistence | 18.949 | 20.568 |
| Export | 14.851 | 14.362 |
| Disposable compiler startup | 19.030 | 13.902 |
| Post-event activation | 0.991 | 0.958 |
| Publication-check activation | 0.994 | 0.939 |

These scopes overlap and **must not be added**. PAIR's training timer increases 44.144 seconds, **8.65%**; its full-arm wall increases 5.00%. The serial, single-event comparison does not isolate the auxiliary kernel cost from other trajectory/runtime effects. No additional profiling or optimization was introduced. Witness times were 2.558 seconds at the released origin, 0.476 at BASE1 and 2.074 at PAIR1. Separate boundary-validation time and training allocator peaks were not captured; they remain unavailable. The recorded serving allocator peaks are in [training summary](TRAINING_SUMMARY.json), not substitutes for training memory measurements. Evaluation peak allocated/reserved memory was 57.321/57.634 GB.

## Work, resources and retention

| New work | Completed |
|---|---:|
| Own-feedback generations | 256 |
| Terminal teacher feature forwards | 256 |
| Learning presentations / ordinary backwards | 512 / 512 |
| AdamW steps | 256 |
| Disposable startup response forwards/backwards / Adam | 4 / 4 / 0 |
| Fixed-witness response forwards / head projections | 24 / 24 |
| Scored generations / warmup generations | 768 / 24 |
| Total complete-response generations, including feedback | 1,048 |

The recorded training counters per arm are 256 draft forwards, 1,280 layer forwards, 22,988 objective chunks and 45,976 vocabulary-head calls; startup counters are separate. Three native workers recorded 153 setup forward completions and 51 graph captures in total. These are instrumentation counters, not a detailed hardware-kernel cost attribution. [Work, validity and resources](WORK_AND_RESOURCES.json) preserves the actual counters and their scope.

New production completed in **2,991.870 seconds (49.86 minutes)** and **5,053.874 occupied-card seconds (1.404 card-hours)**, within the 3-hour/5-card-hour limits. Prior qualification remains separately charged at 288.524 elapsed and 554.446 card-seconds. Historical-plus-new charged totals are 3,280.393 elapsed and 5,608.320 card-seconds; the elapsed sum is not a continuous wall-clock interval. The earlier forecast was approximately one hour, conservatively two after storage admission. Neither serving results nor these nested cost receipts establish training-inclusive speedup.

Before launch, exactly eight audited disposable PR51/PR52 qualification tensors/exports were retired after validating 17,556 local compact evidence files and 199 current remote records. Measured space recovered was **33.578627 GB**. The removed tensors are **retired, not archived**; their exact disposable-state reload/replay is no longer available. Production checkpoints/recovery states, important results, PR56/58 dependencies, models, environments and useful caches were preserved. [Exact paths and retained-evidence mapping](CLEANUP.json), [cleanup explanation](CLEANUP.md).

The conservative admitted A800 new-allocation envelope was 52.616 GB, with a separate at-most-1 GB Fenrir study allowance, within the 90 GB aggregate cap; admission projected 406.153 GB free after peak writes. Post-run recorded A800 charged allocation was 25.418 GB, and the subsequent configuration-retrieval check found A800 free space of approximately **432.128 GB**, preserving the **400 GB reserve**. These are admission/boundary observations, not an unrecorded continuous filesystem high-water measurement. Both A800 GPUs were idle at final inspection. No useful cache was purged merely to meet the cleanup target.

All final exports/recoveries, feedback/features, raw outputs, compact native traces, anchors and private matched witness records remain under `/data/sources/minhao/runs/dflash_pair_one_event_20261124/execution_002`. The compact local delivery contains 3,735 authenticated files (548.657 MB), plus three separately authenticated YAML configurations. Original blocked-production and PR58 qualification records remain unchanged. Public tables contain ordinal identifiers and scalar metrics only; prompts, token arrays, tensors and weights are excluded. [Exact identities and configuration](IDENTITIES.json), [independent interpretation review](INTERPRETATION_REVIEW.md).

## Decision

Keep C+K+ as the working method. This fixed one-event screen shows a small favorable source-fit signal but **does not demonstrate that PAIR improves held-out serving enough to justify its extra training work**. It also cannot distinguish conditional-pair pressure from simply increasing repair strength. Do not adopt, tune or extend PAIR automatically; return these results for the next design decision. No additional event or experiment has been started.
