# Independent O/K continuation execution audit

Status: **passed**. Completed campaign: True. Issues: 0; missing files: 0.

The independent audit reconstructed 52,270 verification blocks from exactly 1,152 native response records: 384 shared-feedback responses and 768 evaluation responses. All six own-arm live restores and all matched response/anchor schedules pass. Each arm ends at 512 cumulative steps and 1,024 presentations; the continuation adds 768 total optimizer steps and 1,536 presentations. The separate 384 terminal-feature forwards are accounted for and are not response generations.

| Event | Available proposal occurrences per arm | K selected occurrences | K selected share | K selected weight share |
| --- | ---: | ---: | ---: | ---: |
| 2 | 977,434 | 556,911 | 56.98% | 71.12% |
| 3 | 975,476 | 564,886 | 57.91% | 71.74% |
| 4 | 963,417 | 554,913 | 57.60% | 71.68% |

These are sampled position occurrences, not unique tokens. Totals use optimizer-boundary accumulation rows once. Across the three new events, measured training time is 962.039 s for O and 999.706 s for K; inclusive event time is 1,065.730/1,101.760 s, with loading costs separate. Reduced selected exposure did not produce lower training time in this run.

Both evaluated events preserve every retained-F token and termination in all 384 executions. Native within-arm call counts agree across repeats. The same 64 canonical outputs contain 19,547 tokens, length 146–515, all EOS terminated and no cap hits. At event 2, O/K use 3,059/2,973 calls and 52.616864/51.029653 s summed mean request time; event 4 uses 2,994/2,942 calls and 51.484646/50.517116 s. All three within-event timing-repeat directions favor K in both evaluations. These are same-event serving comparisons with checkpoints available.

The manual final-state supplement passes: both CPU-inspected final snapshots have Adam step 512, exact event 4 exported weights, and identical stored O/K RNG state. Its 14-file identity inventory covers both final restart files and all six event exports. The 185-file pre/postflight checks pass; 152 locally available sealed files were rehashed here, while 33 remote-only dependencies rely on those receipts. The retrieval receipt verifies all 4,001 files totaling 213,932,852 bytes. See `FINAL_AUDIT_SUPPLEMENT.json` and retrieved `FINAL_STATE_AUDIT.json`. This supplement is a manual cross-check of retained CPU inspection evidence, not a new local tensor reload. Supervisor duration is 4,160 s (69m20s), exit 0.

Material Passport: ARS validate; local compact execution evidence; Verification Status: ANALYZED. This audit loads no model or optimizer and performs no inference.

Observed feedback/evaluation/native records: 384/768/1152. Reconstructed native verification blocks: 52270.
Completed optimizer calls reported by trainer: 768; presentations: 1536; arm counters: {'O': 512, 'K': 512}. Interrupted training: False.

Checks cover own-arm complete-state resume receipts, exact live Adam/RNG/counter restoration receipts, causal O-only feedback checkpoint identity, shared immutable feature metadata, matched response/anchor schedules, boundary-only exposure totals, first-error inclusion/count bounds, full native token/termination/call traces, rotation/repeat schedules, included correctness gates, source/runtime pre/postflight, and phase-specific storage admission.

Exact tensor restore is evidenced by sealed runtime receipts, not repeated locally. Sequential source and closure receipts establish the intended training/evaluation order; there is no independently timestamped activity-overlap trace. Missing/incomplete runs cannot pass.

Interpretation limits:

- One continued training trajectory per arm; repeat generations are serving repeats, not independent training seeds.
- Evaluation uses the original64 development inputs and retained F canonical outputs. F provides no fresh latency baseline.
- The next384 training inputs exclude both64 evaluation panels and original128 training inputs. All384 have disclosed older-project exposure.
- Shared O-generated feedback controls this objective comparison; it does not estimate independent K deployment feedback or adaptation-inclusive speedup.
- Natural K masks are audited through bounded native qualification and current-forward aggregate diagnostics; this compact continuation audit cannot reconstruct unsaved training logits.
- Masking and selected-weight normalization change together. Training losses with different selected distributions are not direct efficacy comparisons.

Methodological scan: 11/11 design and accounting checks; numerical direction and efficacy interpretation are in the separate analysis report.

- Simpsons paradox: Keep events and timing repeats separate; do not pool event1/2/4 latency.
- ecological fallacy: Aggregate improvement does not imply every request improves; repeats do not create independent trained seeds.
- Berksons paradox: Original development panel and chronological train selection are fixed; older-project exposure is disclosed.
- collider bias: Validity gate covers all returned rows; no filtering by favorable performance is permitted.
- base rate neglect: Report all64 requests and full384 evaluation slots per evaluated event, with incomplete/unknown work visible.
- regression to mean: Continuation follows an earlier positive screening result; use concurrent O/K within each event, not an unadjusted before/after latency claim.
- survivorship bias: Missing, failed and unknown work prevent a complete passed audit; no replacement or automatic retry.
- look elsewhere effect: Report both prespecified evaluation events2/4 and retained event1 context; no significance testing in this auditor.
- forking paths: Sealed schedule, checkpoints, fixed budgets and unchanged source constrain analysis choices; broader study remains exploratory.
- correlation causation: Controlled objective comparison is conditional on shared O feedback and one trajectory; it does not establish net deployment acceleration or a unique mechanism.
- reverse causality: O checkpoint is bound before feedback collection; own-arm updates precede evaluation by sealed serial orchestration.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_ok_continuation_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.

Historical-source gap: this run archive contains 2 unresolved source-pin occurrences; the missing versions remain explicitly identified.
