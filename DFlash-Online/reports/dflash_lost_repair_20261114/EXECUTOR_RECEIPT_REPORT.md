# K+ lost-repair gate probe

## Material Passport

ARS experiment-agent / run and validate. Verification status: ANALYZED from retained execution receipts; no independent native replication.

**Status: COMPLETE. Gate: FAIL.**
Frozen-checkpoint historical replay and read-only training exposure only. Optimizer steps, candidate training and timed endpoint runs are zero by contract; recorded work is listed below.

Execution commit: `29b6a15c5f5f16135663ea61e3d31f21b3f1d465`.

## Census and token consequences

Scheduled requests: 128. Observed natural starts: 4877. Validated full natural-start denominator: 4877.
Observed H: 252; M: 2147; converse newly passed old blocker: 341.
H / all natural starts: 5.167111%; H / genuine K+6 rejections: 5.926623%.
H contributing requests: 105; largest request share: 3.968254%.
Previously attained then lost prefix occurrences, sum(a5−a6) over H: 764.

H requires the same genuine first rejection under K+4 and K+6, with K+5 passing that slot. M requires all three to first reject there. Known seeds are excluded; native EOS/cap boundaries are retained. Other histories, earlier prefix breaks, terminal/all-correct and missing replays remain in private census records and aggregate counts.

History counts: `{"H": 252, "M": 2147, "all_correct_or_terminal": 625, "other_rejection_history": 1853}`.
H by one-based depth: `{"1": 31, "2": 50, "3": 41, "4": 29, "5": 26, "6": 14, "7": 7, "8": 3, "9": 9, "10": 13, "11": 10, "12": 3, "13": 4, "14": 6, "15": 6}`.
H a5−a6 histogram: `{"1": 105, "2": 52, "3": 20, "4": 22, "5": 11, "6": 10, "7": 8, "8": 8, "9": 5, "10": 3, "11": 2, "12": 4, "13": 1, "14": 1}`.

## Scheduled training exposure

| Stage | Scheduled occurrences | Distinct anchors | Distinct requests |
|---|---:|---:|---:|
| native_replay | 252 | 252 | 105 |
| scheduled | 503 | 252 | 105 |
| same_j | 479 | 240 | 104 |
| comparable_M | 237 | 119 | 69 |
| matched | 229 | 115 | 69 |

Observed response presentations: 256 / 256.
Native H weight before matching: 315.502452; observed native first-error weight: 27646.563011; observed selected weight: 288152.924683.
Full-event native first-error denominator: 27646.563011; full-event selected denominator: 288152.924683. Training evidence: training/RESULT.json.
H weight shares: 1.141200% of native first-error weight; 0.109491% of selected weight.

Support counts each response/start once; weights count every scheduled presentation. Full-response denominators appear once in PRESENTATION_WEIGHTS.csv. NATIVE_WEIGHTS.csv records each sampled occurrence and its actual native coefficient.
The native response-wide FP32 first-error mass is authoritative. The separately summed serialized anchor weights are retained and checked with unchanged native mass tolerances (rtol 2e-6, atol 2e-7); they do not silently replace a missing native denominator.
High means a correct valid immediate successor, low means a wrong valid successor, and ineligible means no valid successor. Ineligible errors keep the native low coefficient and remain in denominators; matching stays within this separate class.

## Paper-only matching gate

H is ordered by SHA256 of compact JSON [20261114, response ID, canonical start]. Each H receives the nearest available M by wrong-minus-canonical margin within the same presentation/response/depth/priority; ties use the same hash. No M is reused within a stratum. No weight changes are applied.

| Screening value | Observed | Threshold | Pass |
|---|---:|---:|---|
| distinct_matched_H_anchors | 115 | 32 | True |
| distinct_matched_H_requests | 69 | 16 | True |
| projected_weight_fraction_all_native_first_error | 0.146918% | 0.250000% | False |

Projected mass Σ[0.25×min(wH,wM)]: 40.617696.
Unmatched H occurrences: 250; reasons: `{"M_exhausted_without_replacement": 8, "no_comparable_M": 242}`.
Margin balance: `{"maximum_absolute_difference": 4.625, "mean_H_margin": 0.8837336244541485, "mean_M_margin": 1.6965065502183405, "mean_absolute_difference": 1.107532751091703, "mean_signed_difference": -0.8127729257641921, "pairs": 229}`.

The 32-anchor, 16-request and 0.25% thresholds screen this scheduled batch only. A pass ends at this report and warrants a separate mechanism discussion; it does not authorize training or establish future-request recovery, acceptance gains or serving speedup.

## Completeness, parity and costs

Execution status: completed; missing replays: 0; invalid replays: 0.
A missing full census or training denominator makes the gate INCOMPLETE. Observed counts are measured numerators, not estimates of missing work. Null ratios are unavailable, never zero.

| Recorded receipt | Value |
|---|---:|
| phase_costs.census.wall_s | 2673.612681567669 |
| phase_costs.census.exit_code | 0 |
| phase_costs.training.wall_s | 150.0979630202055 |
| phase_costs.training.exit_code | 0 |
| work.attempted.model_load | 386 |
| work.attempted.draft_block | 19559 |
| work.attempted.target_call | 19812 |
| work.attempted.feedback_capture | 128 |
| work.attempted.context_forward | 366527 |
| work.attempted.teacher_preparation | 128 |
| work.attempted.neutral_block | 4877 |
| work.attempted.historical_block | 9754 |
| work.attempted.training_forward | 256 |
| work.completed.draft_block | 19559 |
| work.completed.target_call | 19812 |
| work.completed.model_load | 386 |
| work.completed.context_forward | 366527 |
| work.completed.feedback_capture | 128 |
| work.completed.teacher_preparation | 128 |
| work.completed.neutral_block | 4877 |
| work.completed.historical_block | 9754 |
| work.completed.training_forward | 256 |
| work.remaining.feedback_capture | 0 |
| work.remaining.teacher_preparation | 0 |
| work.remaining.training_forward | 0 |
| work.remaining.neutral_block | 257267 |
| work.remaining.historical_block | 514534 |
| work.remaining.draft_block | 1080441 |
| work.gpu_seconds | 2853.5035617239773 |
| work.gpu_seconds_limit | 14400 |
| occupied_gpu_budget.attempted.model_load | 386 |
| occupied_gpu_budget.attempted.draft_block | 19559 |
| occupied_gpu_budget.attempted.target_call | 19812 |
| occupied_gpu_budget.attempted.feedback_capture | 128 |
| occupied_gpu_budget.attempted.context_forward | 366527 |
| occupied_gpu_budget.attempted.teacher_preparation | 128 |
| occupied_gpu_budget.attempted.neutral_block | 4877 |
| occupied_gpu_budget.attempted.historical_block | 9754 |
| occupied_gpu_budget.attempted.training_forward | 256 |
| occupied_gpu_budget.completed.draft_block | 19559 |
| occupied_gpu_budget.completed.target_call | 19812 |
| occupied_gpu_budget.completed.model_load | 386 |
| occupied_gpu_budget.completed.context_forward | 366527 |
| occupied_gpu_budget.completed.feedback_capture | 128 |
| occupied_gpu_budget.completed.teacher_preparation | 128 |
| occupied_gpu_budget.completed.neutral_block | 4877 |
| occupied_gpu_budget.completed.historical_block | 9754 |
| occupied_gpu_budget.completed.training_forward | 256 |
| occupied_gpu_budget.remaining.feedback_capture | 0 |
| occupied_gpu_budget.remaining.teacher_preparation | 0 |
| occupied_gpu_budget.remaining.training_forward | 0 |
| occupied_gpu_budget.remaining.neutral_block | 257267 |
| occupied_gpu_budget.remaining.historical_block | 514534 |
| occupied_gpu_budget.remaining.draft_block | 1080441 |
| occupied_gpu_budget.gpu_seconds | 2853.5035617239773 |
| occupied_gpu_budget.gpu_seconds_limit | 14400 |

Declared aggregate occupied-GPU ceiling: 14400 seconds. Nested phase clocks are not added. This probe contains no timed serving comparison.

Missing evidence: none.
Analysis/read errors: none.

## Source and checkpoint identities

```json
{
  "execution_commit": "29b6a15c5f5f16135663ea61e3d31f21b3f1d465",
  "checkpoint_identities": {
    "K4": {
      "config_identity": {
        "bytes": 1211,
        "sha256": "5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa"
      },
      "export": "/data/sources/minhao/runs/dflash_role_loss_pilot_20261107/production_001/campaign_001/trajectories/gsm8k/KPLUS/trainer/events/event_004/export",
      "weights_identity": {
        "bytes": 2097259104,
        "sha256": "4f3010e961bd7dcf35fa100486c355ffbc7629d816dda6e2b8fa001878255701"
      }
    },
    "K5": {
      "config_identity": {
        "bytes": 1211,
        "sha256": "5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa"
      },
      "export": "/data/sources/minhao/runs/dflash_survival_head_20261110/execution_001/campaign_001/gsm8k/event_005/KPLUS/trainer/events/event_005/export",
      "weights_identity": {
        "bytes": 2097259104,
        "sha256": "9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795"
      }
    },
    "K6": {
      "config_identity": {
        "bytes": 1211,
        "sha256": "5749d96966f9ce7e6a9662c9f482b8aa0d461845fb6b9ce09ac2c08311fab2aa"
      },
      "export": "/data/sources/minhao/runs/dflash_functional_step_geometry_20261108/pilot_r1_001/campaign_001/gsm8k/event_006/KPLUS/trainer/events/event_006/export",
      "weights_identity": {
        "bytes": 2097259104,
        "sha256": "8c257620e44226c5c0469e5886224ed5d2dcf9fa2a94eddbdd10d7e36c697560"
      }
    }
  },
  "manifest_identity": {
    "bytes": 10868762,
    "sha256": "764fa339938a240259949040134581a2231de542b7c799af5a7eb85c5ea1dbd4"
  },
  "source_manifest_path": "/data/sources/minhao/runs/dflash_lost_repair_20261114/execution_001/SOURCE.json",
  "plan_file_identity": null,
  "evidence_file_identities": {
    "census": {
      "bytes": 1649617,
      "sha256": "3fb2376df762591bbb7d1432e698f7275abda27a73b0bafd1fea96bba5848259"
    },
    "training": {
      "bytes": 41608781,
      "sha256": "1245ed7f1e82b0828b9c748c12ad96e3789f5347bfb74ac55b99a0edf9430321"
    },
    "status": {
      "bytes": 160,
      "sha256": "8b32954da9ba60ecc53ec8ae974fbfd59969aa09542fcca3ee30048a2eee9d85"
    },
    "budget": {
      "bytes": 845,
      "sha256": "f1e877e16b38961445ee16bd05bacd97aca1a57e7127eba2f6f2a708b30ee64e"
    },
    "result": {
      "bytes": 1322,
      "sha256": "e1d6a2a4e333a0882d0ae81eb21c5a4ea91788c060658d54436a8ecb9bd6ea97"
    }
  },
  "durable_presentation_file_identities": {}
}
```

Public tables contain identifiers and measurements only. Raw prompts, token arrays, reconstruction receipts and tensors remain in the private run directory. Labels and denominators require the separately recorded independent reduction before final scientific sign-off.

Interpretation boundary: recurrence frequency and training exposure are different questions. Frequent H with little scheduled exposure identifies an exposure gap; coverage lost only at matching identifies weak support for this comparison. Historical fixed-start token differences cannot be converted to complete-request speed or assumed recoverable future gain.
