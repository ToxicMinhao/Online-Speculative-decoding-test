# PAIR one-event follow-up: implementation complete, production resource-blocked

## Material passport

ARS experiment-agent execution/readiness validation. Evidence: retained PR58 native qualification, new CPU tests, actual A800 CPU preflight and storage/transfer observations. Scientific stage unchanged. **No new native training, feedback, witness or scored generation occurred.** This is a readiness report, not a completed efficacy experiment.

Execution/staging source: `416cb7931028621c09b8fa983e53fc8a7dcebaab`; clean checkout before staging. Reviewed parent: `e38c942b5f607bb0b816cb05f3535a78c48110ec`. Reused qualification: `79b8c076610e9b4d310169b3891cb76ac6d0f9bb`; original PR56 recipe: `1551e5e5d0dd7736d8441977aa3dea250535da7b`. Publication is [PR #59](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/59), stacked on PR #58; no merge. Original PR58 report, plan, clock and artifacts remain intact.

## What is implemented and checked

The coordinator now performs exactly BASE1 then PAIR1, each initialized independently from released weights with fresh Adam and seed 17. Both train on physical B and serve on A. It waits for clean phase exits before the fixed quiet A endpoint evaluation with B idle. It invokes the existing 128-request, three-repeat evaluator and request-level bootstrap; it never invokes another disposable qualification. The unchanged objective, trainer, sampling, normalization, settings and horizon 7,424 are authenticated against the qualified source.

The witness freezes the original first error, canonical token and original winning rival for up to 64 block occurrences in the first 8 BASE responses. Its existing origin/BASE1/PAIR1 readouts now provide both requested margins, full-vocabulary native correctness/ties, rival replacement, useful repairs requiring all preceding matches to survive, prefix damage and full-block progress. No-error blocks retain null margins and remain in preservation/progress totals. This adds no model forward. Both a strict-positive old-rival margin and a tie-aware pair-winner predicate are reported, without substituting either for native full-vocabulary correctness.

**60 unique CPU tests pass:** 50 pilot tests plus 10 unchanged-objective tests. They include stub coordinator sequencing and the full 792-slot evaluation, fixed-origin/tie/EOS/gap behavior, observation-state isolation, evidence authentication and per-filesystem admission. Independent reviews found and resolved three integration issues: storage scans racing with normal temporary-file retirement; inherited import paths selecting unsealed training code; and path aliases diverging from accounted storage. No scientific formula changed. An initial local test invocation used the wrong interpreter/discovery root and failed imports; its log is retained privately, followed by the successful pinned-interpreter runs. These are CPU fixtures, not new native qualification.

Actual A800 preflight completed in 27.048 seconds with authentication passed: **5,062 staged source files, 105 sealed bindings and 20 released model artifacts checked**. Native serving/training pins and the exact training interpreter/source route matched. The new receipt explicitly says `reused_passed`, with zero qualification steps, forwards or generations. Fresh-origin/startup checks remain mandatory at eventual production; old disposable states cannot be loaded as origins.

## Why execution stopped before GPU work

At actual admission, the original data volume had **425.137820 GB free**. The phase-specific all-local envelope is **52.615699 GB** including 5% tensor margin and 4 GB source/log/compiler allowance. Newly staged/accounted files already occupy **0.331878 GB** by `st_blocks`; remaining budgeted writes are **52.283821 GB**. Keeping 400 GB free therefore requires another **27.146001 GB** of headroom. Reflinked source copies are conservatively charged by allocated-block counts; apparent file size is not claimed as newly consumed physical free space.

Fenrir’s new study files occupy 0.342 GB. Allowing a full 1 GB for local source/preparation/delivery makes the combined all-local budget 53.616 GB, below the 90 GB cap. This is a separate cross-host budget check; the runtime guard measures the sealed A800 filesystems. No Fenrir feature-transfer route is admitted. It does not resolve the A800 reserve deficit.

The actual inventory found no suitable second writable durable A800 filesystem. `/` reports the same capacity as `/data` through overlay backing and is not credited as a second disk; `/mnt` is on that overlay. The 17.01 GB root-device exposure is read-only driver-file bind mounts. `/dev/shm` is volatile and not admitted. Both physical A800s are idle with 0 MiB compute allocation.

A genuinely independent feature filesystem would need **32.124135 GB available**, including its 2 GB free reserve. The main-side split budget would be 23.597312 GB. This route is implemented but not admitted because no such existing destination was found. Merely offloading BASE after training is insufficient under current main-volume headroom.

A real 41,161,935-byte retained-feature transfer to Fenrir took 141.583 seconds and hash-matched. At that observed rate, one 14.261 GB full-capacity pool projects to **13.63 hours**, before other transfer/training work. This is a small-file linear estimate, not guaranteed bulk throughput; it does not support admission within 3 hours. Fenrir has capacity, but an adequate transfer-backed execution route has not been demonstrated. The temporary 41.169 MB allocated benchmark copy was removed only after verification; the A800 source remains. **No pre-existing artifact was deleted.**

The handoff explicitly requires completing useful implementation but stopping before new GPU work when no authorized route fits. Accordingly, the new plan is `production_admitted=false`; no `CLOCK.json`, launch or arm directory exists. The original qualification was not repeated as a substitute.

## Work and scientific results

| New work | Planned | Completed |
|---|---:|---:|
| Own-feedback generations / preparation units |256 / 256|0 / 0|
| Learning presentations / backwards |512 / 512|0 / 0|
| Adam steps |256|0|
| Scored generations / declared warmups |768 / 24|0 / 0|
| Startup forwards / backwards / Adam |4 / 4 / 0|0 / 0 / 0|
| Existing witness draft forwards |At most 24|0|

BASE1 and PAIR1 do not yet exist. Latency/call reductions, confidence intervals, margin changes, useful repairs, prefix damage, activity and memory costs are **unavailable**. There are no actual per-request/repeat or matched-block observations for this attempt; none are fabricated. The machine-readable work table records all planned work as unattempted. A resource-blocked run is not a negative PAIR result.

Historical native qualification cost remains 288.524 elapsed seconds and 554.446 occupied card-seconds, with 2 disposable Adam steps and 8 generations. New GPU cost is zero; combined historical-plus-new card cost remains 554.446 seconds. CPU preparation and transfer are reported separately. Existing qualification's positive auxiliary activity establishes execution, not efficacy.

## Duration and next action

Once an admissible storage route exists, the retained-timing forecast is approximately **one hour**, with a **two-hour conservative allowance** for the unchanged event, evaluation and reporting; 3 elapsed hours and 5 card-hours remain hard limits. There is no running-job finish time now. Full-capacity feedback and cold initialization may affect actual duration.

The supported next action is to provide an independent durable feature destination with at least 32.125 GB available, or explicitly authorize a different storage-retention arrangement that creates at least 27.15 GB additional main-volume headroom while retaining 400 GB. Recheck actual capacity and reseal before any launch. Do not tune PAIR, shorten the event, delete protected PR58 evidence or repeat qualification to bypass admission.

## Evidence locations

Public compact evidence is beside this report: `IDENTITIES.json`, `QUALIFICATION_REUSE_SUMMARY.json`, `RESOURCE_ADMISSION.json`, `WORK_AND_ENDPOINTS.json`, `WORK_TABLE.csv`, `COSTS.json`, `AUDIT.json`, CPU logs and independent reviews. Sealed input/canonical tokens, detailed reuse bindings and raw preparation records remain private.

- New A800 preflight: `/data/sources/minhao/runs/dflash_pair_one_event_20261124/execution_001`.
- Staged source: `/data/sources/minhao/pair_one_event_source_20261124_001/DFlash-Online`.
- Preserved old qualification: `/data/sources/minhao/runs/dflash_repair_conditional_kl_20261124/execution_001`.
- Fenrir private evidence: `/home/minhao2/EAGLE_SPEC/research/dflash_pair_one_event_20261124/result_private`.

This ends the bounded readiness attempt. No automatic retry, loss change, additional arm or expanded experiment follows.
