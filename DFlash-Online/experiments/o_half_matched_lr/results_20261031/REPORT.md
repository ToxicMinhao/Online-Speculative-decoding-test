## Material Passport

- Origin Skill: ARS-Codex 3.24.0, experiment-agent.
- Origin Mode: run, followed by validate.
- Origin Date: 2026-11-01 UTC (completed October 31 in America/Edmonton).
- Verification Status: **ANALYZED**. The authorized GPU run completed and its native validity gates passed. Separate calculations and record/source reviews passed; no independent GPU reproduction was performed.
- Version Label: o_half_matched_lr_completed_v1.
- Execution commit: `19ee22faf985e0576d287eaa8c8624b5a44b46f5`; [PR #22](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/22). Later result-publication commits do not identify the executed numerical source.
- Status: **completed, exit 0; stopped after the fixed experiment**.

# Result: K retains a matched-learning-rate advantage on this coding panel

The missing ordinary-OSD control improves at the smaller learning rate, but it does not erase K's advantage. **K_HALF needs 177 fewer verification calls than O_HALF: +1.1034%, with a nominal paired-request 95% interval of [0.4548%, 1.8059%].** Both fitted seeds favor K_HALF. All compared outputs have identical tokens and termination.

| Event-4 objective | LR 3e-4: pooled calls | LR 1.5e-4: pooled calls |
|---|---:|---:|
| Ordinary OSD | 16,379 (O) | 16,042 (O_HALF) |
| K: correct prefix plus first error | 16,303 (K) | **15,865 (K_HALF)** |

Released F uses 16,300 calls. Counts sum the same 128 request identities for each of two fitted seeds; repeated timing executions do not multiply the sample. O and O_HALF are newly evaluated; F, K and K_HALF use audited archived counts after the included compatibility bridge.

**Decision A applies:** the primary point estimate exceeds the prespecified 0.5% research-priority threshold, its interval is above zero, and both seed effects are positive (+1.0703% and +1.1365%). The lower endpoint is below 0.5%, so the interval does **not** establish that the true benefit exceeds 0.5%. This is reused-panel development evidence, not fresh confirmation.

Both lower-LR variants improve over their original-LR counterpart. K/O at the original LR and the objective-by-LR interaction remain uncertain. This supports the implemented K objective at the matched lower LR; it does not isolate prefix selection from its changed normalization, prove a preservation mechanism, or establish a learning-rate optimum.

## Full prespecified measurements


Execution source: `19ee22faf985e0576d287eaa8c8624b5a44b46f5`.

Question: does K retain useful drafting benefit over ordinary OSD at the same LR1.5e-4? Before execution, O_HALF was expected potentially to recover part of ordinary O’s deficit; the relative outcome was unknown. Only O_HALF was trained.

**Decision A: Useful matched-LR advantage for K_HALF.**

This reuses the PR21 development panel that already informed the choice of K_HALF. It is development/ablation evidence, not fresh confirmation. F/K/K_HALF counts are verified archived repeat-zero trajectories; O/O_HALF counts come from new two-repeat evaluation. The compatibility bridge is excluded from the performance sample.

| Scope | Method | LR | Count provenance | Calls | Output tokens | Calls / 1,000 tokens | Accepted tokens | Progress / call |
|---|---|---:|---|---:|---:|---:|---:|---:|
| combined | F | released | archived PR21 | 16300 | 88266 | 184.6691 | 71950 | 5.3994 |
| combined | O | 3e-4 | new two-repeat | 16379 | 88266 | 185.5641 | 71873 | 5.3733 |
| combined | K | 3e-4 | archived PR21 | 16303 | 88266 | 184.7031 | 71948 | 5.3984 |
| combined | O_HALF | 1.5e-4 | new two-repeat | 16042 | 88266 | 181.7461 | 72209 | 5.4862 |
| combined | K_HALF | 1.5e-4 | archived PR21 | 15865 | 88266 | 179.7408 | 72387 | 5.5474 |
| 17 | F | released | archived PR21 | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 17 | O | 3e-4 | new two-repeat | 8177 | 44133 | 185.2809 | 35949 | 5.3816 |
| 17 | K | 3e-4 | archived PR21 | 8140 | 44133 | 184.4425 | 35985 | 5.4060 |
| 17 | O_HALF | 1.5e-4 | new two-repeat | 8035 | 44133 | 182.0633 | 36091 | 5.4767 |
| 17 | K_HALF | 1.5e-4 | archived PR21 | 7949 | 44133 | 180.1147 | 36177 | 5.5359 |
| 29 | F | released | archived PR21 | 8150 | 44133 | 184.6691 | 35975 | 5.3994 |
| 29 | O | 3e-4 | new two-repeat | 8202 | 44133 | 185.8473 | 35924 | 5.3652 |
| 29 | K | 3e-4 | archived PR21 | 8163 | 44133 | 184.9636 | 35963 | 5.3908 |
| 29 | O_HALF | 1.5e-4 | new two-repeat | 8007 | 44133 | 181.4289 | 36118 | 5.4958 |
| 29 | K_HALF | 1.5e-4 | archived PR21 | 7916 | 44133 | 179.3669 | 36210 | 5.5590 |

| Scope | Comparison | Call reduction | Paired 95% interval | Wins / regressions / ties | Gross saved / added |
|---|---|---:|---:|---:|---:|
| combined | K_HALF_vs_O_HALF | +1.1034% | [+0.4548, +1.8059]% | 125 / 82 / 49 | 336 / 159 |
| combined | K_vs_O | +0.4640% | [-0.2227, +1.1112]% | 97 / 112 / 47 | 315 / 239 |
| combined | K_HALF_vs_K | +2.6866% | [+1.9233, +3.5338]% | 175 / 48 / 33 | 570 / 132 |
| combined | O_HALF_vs_O | +2.0575% | [+1.4279, +2.7140]% | 145 / 66 / 45 | 467 / 130 |
| combined | O_vs_F | -0.4847% | [-1.3753, +0.4021]% | 100 / 124 / 32 | 304 / 383 |
| combined | K_vs_F | -0.0184% | [-1.0284, +0.8982]% | 107 / 113 / 36 | 369 / 372 |
| combined | O_HALF_vs_F | +1.5828% | [+0.8581, +2.3211]% | 137 / 73 / 46 | 418 / 160 |
| combined | K_HALF_vs_F | +2.6687% | [+1.8411, +3.5566]% | 154 / 62 / 40 | 560 / 125 |
| 17 | K_HALF_vs_O_HALF | +1.0703% | [+0.3810, +1.8128]% | 63 / 39 / 26 | 160 / 74 |
| 17 | K_vs_O | +0.4525% | [-0.3479, +1.2013]% | 48 / 53 / 27 | 150 / 113 |
| 17 | K_HALF_vs_K | +2.3464% | [+1.4715, +3.2975]% | 86 / 28 / 14 | 267 / 76 |
| 17 | O_HALF_vs_O | +1.7366% | [+1.0655, +2.4142]% | 72 / 34 / 22 | 211 / 69 |
| 17 | O_vs_F | -0.3313% | [-1.2717, +0.6044]% | 51 / 59 / 18 | 159 / 186 |
| 17 | K_vs_F | +0.1227% | [-0.9545, +1.1381]% | 57 / 58 / 13 | 195 / 185 |
| 17 | O_HALF_vs_F | +1.4110% | [+0.6927, +2.1460]% | 67 / 40 / 21 | 200 / 85 |
| 17 | K_HALF_vs_F | +2.4663% | [+1.6011, +3.3875]% | 77 / 33 / 18 | 274 / 73 |
| 29 | K_HALF_vs_O_HALF | +1.1365% | [+0.4068, +1.9180]% | 62 / 43 / 23 | 176 / 85 |
| 29 | K_vs_O | +0.4755% | [-0.4166, +1.3404]% | 49 / 59 / 20 | 165 / 126 |
| 29 | K_HALF_vs_K | +3.0258% | [+2.1274, +3.9901]% | 89 / 20 / 19 | 303 / 56 |
| 29 | O_HALF_vs_O | +2.3775% | [+1.5827, +3.2035]% | 73 / 32 / 23 | 256 / 61 |
| 29 | O_vs_F | -0.6380% | [-1.5722, +0.2822]% | 49 / 65 / 14 | 145 / 197 |
| 29 | K_vs_F | -0.1595% | [-1.2376, +0.8147]% | 50 / 55 / 23 | 174 / 187 |
| 29 | O_HALF_vs_F | +1.7546% | [+0.9553, +2.5633]% | 70 / 33 / 25 | 218 / 75 |
| 29 | K_HALF_vs_F | +2.8712% | [+2.0234, +3.7764]% | 77 / 29 / 22 | 286 / 52 |

The reverse O_HALF/K_HALF reduction is calculated with K_HALF calls as its denominator:

- combined: -1.1157% [-1.8391, -0.4569]%.
- 17: -1.0819% [-1.8463, -0.3824]%.
- 29: -1.1496% [-1.9555, -0.4085]%.

Objective-by-LR interaction uses the common F-call denominator; positive values mean K’s call advantage over O is greater at lower LR. It is not a subtraction of differently normalized reductions.

- combined: +0.6196% [-0.1758, +1.5232]%; numerator 101 calls / F 16300.
- 17: +0.6012% [-0.3467, +1.6207]%; numerator 49 calls / F 8150.
- 29: +0.6380% [-0.3502, +1.7258]%; numerator 52 calls / F 8150.

The 0.5% target is a research-priority point-estimate threshold, not deployment break-even. A positive interval alone does not satisfy that threshold or prove the true effect exceeds it. Practical equivalence requires the entire primary interval within ±0.5 percentage points; the provisional-base decision also requires both seed points within that band.

All intervals resample 128 joint request identities carrying both fixed seeds and all methods, using 10,000 PCG64 seed20261102 draws and linear percentiles. The 256 seed/request outcomes and repeats are not independent samples. Intervals are nominal/descriptive, conditional on these fitted trajectories and reused panel; they do not adjust for earlier K_HALF selection. Secondary intervals are unadjusted.

| Scope / candidate vs comparator | Retained gain | Lost gain | Added gain | Prevented harm | Added harm | Net saved |
|---|---:|---:|---:|---:|---:|---:|
| combined / K_HALF_vs_O_HALF | 309 | 109 | 251 | 85 | 50 | 177 |
| combined / O_HALF_vs_O | 219 | 85 | 199 | 268 | 45 | 337 |
| combined / K_HALF_vs_K | 261 | 108 | 299 | 271 | 24 | 438 |
| 17 / K_HALF_vs_O_HALF | 152 | 48 | 122 | 38 | 26 | 86 |
| 17 / O_HALF_vs_O | 113 | 46 | 87 | 124 | 23 | 142 |
| 17 / K_HALF_vs_K | 135 | 60 | 139 | 128 | 16 | 191 |
| 29 / K_HALF_vs_O_HALF | 157 | 61 | 129 | 47 | 24 | 91 |
| 29 / O_HALF_vs_O | 106 | 39 | 112 | 144 | 22 | 195 |
| 29 / K_HALF_vs_K | 126 | 48 | 160 | 143 | 8 | 247 |

Positive gains and harms are formed within seed/request before pooling. The identity added gain − lost gain + prevented harm − added harm equals net saved calls is checked at every paired outcome. Gross gains/harms, retained fractions and paired intervals are retained in SUMMARY.json. This accounting does not identify gradient conflict, memorization or a preservation mechanism.

## Work and cost

New training budget: 1,024 Adam steps, 2,048 ordinary response presentations/forward-backward equivalents, zero reference/diagnostic passes and zero disposable optimizer qualification steps. New serving budget: 48 compatibility-bridge plus 1,024 O/O_HALF executions = 1,072; all 112 gate/bridge executions are included. Actual completion/failure counters appear in SUMMARY.json.

| O_HALF seed | Valid events | Training seconds | Event wall seconds | Loading/resume seconds | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|
| 17 | 4 | 1423.212 | 1580.863 | 155.751 | 19.234 |
| 29 | 4 | 1419.227 | 1574.747 | 155.637 | 19.234 |

Historical O/K/K_HALF updater costs are unmatched context in COST_SUMMARY.json. Nested forward/head/compile/setup timers are not additive to their parent totals. Response losses are separately normalized observations captured before accumulation merging; window gradient/clipping records are counted once. O and K have different masks/normalization, so their raw losses are not evidence of serving improvement.

**No comparative serving latency or throughput result is supported.** New evaluations batch checkpoints and use two repeats; reused methods were measured previously with three. Request, native-generation, cache, loading and activation times are operational accounting only. No old/new timing pooling, call-count-to-speedup conversion, or training-inclusive speedup is claimed.

## Scope and one next action

Keep K_HALF as the provisional coding base; recommend a separately agreed bounded portability or online-setting check of this fixed base.

This recommendation does not authorize a follow-on experiment or a project-stage change. The fixed O-generated feedback stream is shared replay, not an O_HALF-controlled online deployment. Equal LR neither matches Adam displacement nor isolates K’s selection from its normalization. Lower LR also changes decoupled weight decay. No general learning-rate optimum, convergence, domain-shift/stochastic-decoding benefit or novelty claim follows.

Files: [paired counts](PAIRED_REQUESTS.csv), [all new evaluations](ALL_NEW_EVALUATIONS.csv), [summary](SUMMARY.json), [bridge audit](BRIDGE_REPORT.json), [training audit](TRAINING_AUDIT.json), [costs](COST_SUMMARY.json), [training costs](TRAINING_COSTS.csv), [evaluation accounting](EVALUATION_COSTS.json).


## Execution, conditioning and validity evidence

Only O_HALF was trained: two original per-seed O step-zero states, four continuous events per seed, with own weights, native BF16 Adam moments/counters and restored training RNG. Exact original ordering and sampled anchors were checked against natural draws, not forced from the records. All eight completed boundaries pass saved/live/export equality and clean accumulation checks. Shared original O-generated feedback remains unchanged; this is not a new measurement of independently collected O_HALF feedback.

The unchanged numerical recipe uses Qwen3-8B target/tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218`, released Qwen3-8B-DFlash-b16 revision `9b41424b7109f9c5413454f481b09a82b85333f4`, 58 original trainable tensors, ordinary all-valid-proposal forward KL, BF16 projection/probability/Adam arithmetic and FP32 positional reduction. LR alone changes to 1.5e-4; Adam betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and global clip 1 remain. Each event has two passes, microbatch 1/accumulation 2, 256 presentations and 128 steps. Sampling uses at most 512 anchors, block 16, full-block Flex attention and eight-block projection chunks. Training keeps each response's full prompt context and response-only supervision, maximum length 2,272; no reference model or extra loss is introduced.

Serving preserves pinned SGLang/FA3, greedy BF16, reasoning off, block 16, 2,048 tokens or EOS, context 4,096 and concurrency 1. Capture is disabled; full idle cache flushing, checkpoint refresh and loaded-weight checks apply to every required boundary. All output tokens, termination, native traces and repeat-consistent count fields pass. The 128 distinct outputs have lengths 37–2,042 (mean 344.789), all terminate by EOS and none hits the cap. Each seed contributes the same 44,133 canonical output tokens. All lengths and terminations remain explicit in SUMMARY.json.

| New scientific work | Planned | Completed |
|---|---:|---:|
| O_HALF event jobs | 8 | 8 |
| Adam steps | 1,024 | 1,024 |
| Response presentations / core forwards / backwards | 2,048 each | 2,048 each |
| Bridge generations | 48 | 48 valid |
| Final O/O_HALF generations | 1,024 | 1,024 valid |
| New feedback / target-feature preparation / reference passes | 0 | 0 |
| Extra qualification optimizer steps | 0 | 0 |
| Invalid / unknown / unattempted scientific work | 0 | 0 |

The 48 bridge executions and 64 first-eight final gate executions are already included. The bridge is not part of the performance sample. Preflight audited 12,434 identities and all 3,072 archived executions / 194,145 native blocks. Postflight checked the 2,790-file source/configuration seal and immutable dependencies. The separate raw-record/statistics audit passes 643,740 checks, 1,072 new raw executions and 67,872 native blocks. The training/cost audit passes 76,167 checks, two initial migrations, six own resumes and all 2,048 schedule/anchor bindings. All 3,486 retrieved files / 280,550,166 bytes hash-match.

The three reviews are separate same-team reviews with disclosed authorship overlap. The statistics reviewer also authored production statistics/report code; the interpretation reviewer authored archive/evaluation code. Independently reconstructed arithmetic and raw records are not independent investigators or GPU reruns. Endpoint tensor equality is supported by execution-time decoded checks and receipts; the post-run local review did not independently reload model-sized tensors. All 11 ARS statistical-interpretation categories were checked, with limits retained.

## Exposure and detailed cost accounting

Actual exposure is **554,920 sampled anchors and 8,117,722 supervised proposal occurrences**. There are 201 clipped optimizer windows. Instrumentation records 70,140 original objective chunks plus 70,140 checkpoint-recomputed chunks, and 280,560 vocabulary-head calls. No extra response forward is hidden in these chunk counts. Zero new target-feature preparation does not mean zero teacher-head computation: KL projects retained teacher features through the frozen vocabulary head. Teacher-head work is inside training cost.

| Fresh O_HALF cost, both seeds | Seconds | Scope |
|---|---:|---|
| Training loops | 2,842.439 | Includes nested objective/compile/optimizer/logging work |
| Event time | 3,155.609 | Includes training, save and export |
| Model loading / initialization / own-state validation | 311.388 | Separately recorded |
| Checkpoint save | 160.578 | Inside event time |
| Export | 124.392 | Inside event time |
| Endpoint validation / hashing | 148.383 | Separate worker component |
| Complete workers | 3,621.703 | Includes event/loading/validation |
| Interpreter subprocesses | 3,694.019 | Encloses workers |
| Campaign | 5,157.132 | Training, bridge/evaluation, orchestration |
| Supervisor | 6,022.000 | 1 h 40 m 22 s, includes CPU checks/postflight |

Peak allocation is **20.653 GB / 19.234 GiB**; reservation is **27.856 GB / 25.943 GiB**, maxima across worker processes rather than simultaneous whole-machine residency. The 886.115 s teacher-head timer includes recomputation and overlaps training; host optimizer timers are not a detailed GPU-kernel breakdown. Cold compilation remains inside the recorded initialization/objective scopes, not an inferred standalone kernel cost. Detailed nested counters are preserved in TRAINING_COSTS.csv and COST_SUMMARY.json.

Operational serving sums: the bridge has 52.555 s request time, 51.727 s native generation and 0.187 s cache maintenance; final O/O_HALF evaluations have 1,133.299 / 1,099.244 / 14.677 s respectively. Per-stage loading, activation and weight-validation costs are separate in EVALUATION_COSTS.json. These are accounting totals, **not comparative performance estimates**. Historical O/K/K_HALF updater costs are unmatched context. No training-inclusive speedup is inferred.

## Implementation repairs and preserved failure lineage

Two preparation issues were resolved before scientific execution; neither changed the algorithm, numerical tolerances or budget.

1. JSON transport changed three tuple-valued provider contract fields to lists. The new wrapper now decodes only `dflash_target_layer_ids`, `specforge_step_options` and nested `specforge_model_provenance` back to their declared types before the unchanged strict native validator. Both actual original O initial states passed CPU migration/export/RNG validation (42.365 s), with no model construction or CUDA initialization. All 45 final CPU tests passed locally (48.665 s) and on A800 (44.871 s). Original failure evidence remains.
2. `production_001` passed its CPU tests but correctly stopped at source admission because 35 committed PR21 result-publication files were missing remotely. No model, optimizer or generation work started. Exact committed files were supplied; `production_002` was freshly sealed at the same execution commit. All 2,790 sealed files then matched. A preliminary ad-hoc diagnosis used the wrong size-field name and was superseded by the qualified `DEPLOYMENT_DIAGNOSIS_V2.json`; both records remain private. There was no scientific retry or favorable-outcome selection.

Executed command: `bash /data/sources/minhao/DFlash-Online/experiments/o_half_matched_lr/run_a800.sh /data/sources/minhao/runs/dflash_o_half_matched_lr_20261031/production_002`.

The sole scientific supervisor ran from **2026-10-31 23:54:51 UTC to 2026-11-01 01:35:13 UTC**, exit 0. Source and all original numerical implementations were fixed throughout. MEASUREMENTS_REPORT.md is the unmodified generated report; this delivery adds reviewed explanations and provenance without changing its measurements.

## Storage, evidence and stopping point

The admitted peak-storage budget was 75,394,485,270 bytes, including temporary overlap, compiler allowance, slack and the unchanged reserves; initial free space was about 83.69 GB. No pre-existing experiment artifact was removed or changed. Under the sealed rolling policy, only this run's event-1/event-3 state/export tensors were retired after their immediate successors passed: eight exact tensor paths, 33,559,334,252 logical bytes. They were **retired, not archived**; exact odd-event resume and export loading are lost.

Both genuine initial states and complete event-2/event-4 states/exports for both seeds remain on A800. All event/schedule/anchor/validation/failure records and raw native outputs/traces remain. The final existence/size/hardlink check reports idle GPUs and **45.356 GB free**; it is distinct from endpoint content validation. Compact evidence was retrieved only after serving stopped, without staging an archive on A800. Private prompts, token IDs and full native traces remain outside Git.

The public tables retain all outcomes and repetitions; SUMMARY.json identifies archived versus fresh provenance. Private raw evidence is under `/home/minhao2/EAGLE_SPEC/research/dflash_o_half_matched_lr_20261031/compact_evidence_001`; the authoritative tensor and native run root is `/data/sources/minhao/runs/dflash_o_half_matched_lr_20261031/production_002`. Source/configuration and input bindings are in PROVENANCE.json; exact retirement paths are in RETENTION_AFTER.json.

**One supported next action:** retain K_HALF as the provisional coding base and separately agree one bounded portability or online-setting check of that fixed base. No new controller, KP retuning, follow-on experiment or project-stage change has been executed or selected by this report.

Review evidence: [raw records/statistics](RAW_RECORD_REVIEW.md), [training/cost/retention](FINAL_TRAINING_COST_REVIEW.md), [interpretation/source](FINAL_INTERPRETATION_REVIEW.md). These reviews state their verification scopes explicitly.
