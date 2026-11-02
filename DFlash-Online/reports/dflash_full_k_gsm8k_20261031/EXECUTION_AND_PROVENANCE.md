# Execution, provenance and retained artifacts

Scientific execution commit: `0698eeb2ddd97b9e2ac6953989dc82acbc0a1e34`.
Repository: [Online-Speculative-decoding-test, PR #23](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/pull/23).
The result-publication commit is separate; it does not change the code that ran.

## Fixed execution

The completed run root is `/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002`.
The source checkout was `/data/sources/minhao/DFlash-Online`.
The exact sealed binding is `f5484a60dce5afa0d7ba9056a9373a2f3868ca775e098473d7a981c6dd388989`.
`production_001` was unused preparation and is preserved; the reseal preceded all scientific work.

The launch used these fixed paths and the existing operational supervisor:

```bash
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export FULL_K_TRANSFER_CLEARANCE=/data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002/TRANSFER_CLEARANCE.json
bash experiments/full_k_gsm8k/run_campaign.sh /data/sources/minhao/runs/dflash_full_k_gsm8k_20261031/production_002
```

These are provenance commands for the completed run, not instructions to rerun an occupied directory. The wrapper completed 65 CPU tests, preflight, the fixed campaign, accounting/report construction and postflight. It used the pinned scientific interpreters below, disabled online downloads and retained the qualified compiler and cache settings. There was no automatic scientific retry, no disposable Adam qualification step and no additional efficacy probe.

| Component | Actual setting |
|---|---|
| Target | Qwen3-8B, revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Draft | Qwen3-8B-DFlash-b16, revision `9b41424b7109f9c5413454f481b09a82b85333f4` |
| Training Python | `/data/conda_envs/minhao/dflash_specforge_20261022/bin/python` |
| Training runtime | Python 3.12.15; Torch 2.13.0+cu130; Transformers 5.12.1 |
| Serving Python | `/data/conda_envs/minhao/adaflash_author/bin/python` |
| Serving runtime | Python 3.12.14; Torch 2.9.1+cu130; Transformers 4.57.1; pinned SGLang/FA3 files |
| GPU assignment | Target/native serving on GPU 0; training on GPU 1 |
| Training configuration | [K](configs/config_K.yaml), [K_HALF](configs/config_K_HALF.yaml), [original O recipe](configs/ORIGINAL_CONFIG.yaml) |
| K objective | `osd_first_error`; current correct proposals plus first mismatch; selected-weight normalization; response mean |
| Training horizon | 58 sequential online feedback events per arm; 128 responses/event; two shuffled passes; microbatch 1; accumulation 2; 7,424 Adam steps |
| Serving | Greedy BF16, block 16, 2,048 tokens or EOS, context 4,096, concurrency 1, full per-request cache flush |

The configuration's generic `loss_type: dflash` is not the online objective selector. The published full-horizon wrapper chooses `osd_first_error` explicitly. `online_feedback.passes_per_event: 2` governs the event loop; the generic offline `num_epochs` field does not override it. Existing teacher conditioning, sampling, numerical loss and optimizer implementations are reused.

The original O manifest and config are retained with their source inventory. The full manifest SHA-256 is `764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4`; the original config SHA-256 is `f71fb286701392e0c80af0915be406c3ccf71b2c15b612f9757357bea7fcf081`. The manifest, exact rendered inputs, native token outputs and teacher/anchor metadata are private retained artifacts, not duplicated as request text in Git.

## Retained evidence locations

All paths below are relative to the completed run root unless otherwise stated.

| Evidence | Location |
|---|---|
| Source/config/data identity seal | `sealed/SEAL.json`, `sealed/PLAN.json` and associated sealed files |
| Original F/O input and native-reference inventory | `sealed/` and the original full-O run referenced by the plan |
| Arm progress, exact work and event observations | `campaign_001/arms/{K,K_HALF}/` |
| Initial states and complete final states/exports | Exact paths in [training review receipt](reviews/TRAINING_COST_REVIEW.json), including each arm's initial state and `trainer/events/event_058/` |
| Per-presentation anchors, window losses, selection and exposure | Each arm's retained trainer observation/anchor records |
| Raw online inputs, output tokens, capture metadata and compact native traces | Each arm's online/native artifact directories referenced in its records |
| Intermediate records | `campaign_001/arms/{K,K_HALF}/evaluation/event_{014,029,044}/` |
| All final raw executions and validity | `campaign_001/evaluation/final/` and referenced native trace files |
| Full generated tables, costs and resource records | `final_report/` |
| Transfer exclusion proof | `TRANSFER_CLEARANCE.json`, `campaign_001/TRANSFER_CLEARANCE_WAIT.json`, `ARCHIVE_TRANSPORT_PAUSE.json` |
| Runtime and source validation | `PREFLIGHT.json`, `POSTFLIGHT.json`, `CPU_TESTS.log` |
| Completion | `EXIT_CODE`, `REPORT_EXIT_CODES`, `START_UTC`, `END_UTC` |

The public result package includes all intermediate/final measurement rows, all paired requests, event costs and exposure, resource timers, aggregate accounting and independent review outputs. Private native responses, large model/optimizer tensors and full raw training logs remain outside Git. Exact locations and content identities are retained in the sealed plan, original audit and review receipts.

## Storage operations and limits

Before launch, ten audited, completed historical intermediate optimizer states were retired, freeing 62.931 GB. They were **retired, not archived**. Every associated serving export and initial/final state was retained; only direct optimizer/RNG continuation at those intermediate boundaries was lost. A800's free space rose to 108.154 GB, above the unchanged 97.358 GB admission budget.

The user authorized historical offloading to overlap training/capture. Thus those adaptation costs may include I/O contention. After both training workers exited, the final evaluation waited 90.002 s for verified transfer quiescence; transfers stayed paused throughout timed evaluation. The pause and clearance were bound to the exact run and source commit.

A separate archive-resumption SSH timeout occurred **after** successful campaign exit. The remote sender had resumed; the local processes remained stopped until their original identities were checked and resumed. That copy then exited unsuccessfully. Ten of twelve archive entries are verified; all twelve A800 originals and the incomplete copy remain. There was no source deletion from that offload and no scientific retry. This storage housekeeping issue did not overlap or invalidate final timing. Its original failure and reconciliation receipts remain under the private task directory.

The scientific runner's bounded retention applies only to its own new bulk artifacts after verified successor commitment. Intermediate model/replay tensors removed under that policy cannot be used for later tensor-level reanalysis or arbitrary intermediate resume. Compact outputs, traces, anchors, observations and failure/validity records remain; initial/final complete states and final exports remain available. The independent training review reports the exact retention accounting and the limits of receipt-based state validation.

## Review reproduction scope

The [review sources](reviews/code/) are exact copies of the CPU audit scripts used for this delivery. Their original task paths and preserved audit-only type-check failure are intentional provenance, not a portable new experiment runner. Full input-identity ledgers remain private; [identity index](reviews/PRIVATE_EVIDENCE_INDEX.json) records their locations and hashes. No GPU computation was added for report review.
