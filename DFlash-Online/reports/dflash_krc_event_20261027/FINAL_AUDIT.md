# Independent final audit: KRC

**Audit passed with no discrepancies or missing evidence. KRC improves the prespecified native outcome on this panel: 3,027 → 3,005 verification calls, a 0.7268% reduction.** Thirty-one requests improve, 15 regress and 18 tie. This is a modest positive result for one event and one seed on the reused development panel, not a generalization or robustness result.

## Material Passport

- Workflow: ARS experiment-agent reproducibility validation.
- Inputs: sealed source/data schedule, window/training receipts, retained first-step reference and every fresh native record.
- Outputs: [audit code](audit.py), [machine-readable audit](FINAL_AUDIT.json), [correction steps](analysis/CORRECTION_STEPS.csv), [dual-role probe blocks](analysis/DUAL_ROLE_PROBE_BLOCKS.csv).
- Scope: local compact-record analysis only; no model execution, new experiment or scientific code change.

All **1,367 retrieved files** verify against the transfer manifest. Preflight/postflight verify the same **627 sealed files**. The audit independently rebuilds the complete cyclic reference schedule from the original O presentation receipt, including both within-pass wraps, disjoint learning/reference identities and each reference pair's own anchors. No development request enters either training role. All 128 windows match the schedule, fresh initialization receipts and exact optimizer/gradient/RNG/runtime preservation checks.

The audit reconstructs prefix lengths, validity/EOS boundaries, canonical token identities, triangular weights, global two-response denominators and the prescribed FP32 correction coefficients. It verifies the learning-batch first-step reproduction exactly and reconstructs every local repair identity. It also reconstructs **18,096 native verification blocks** across all 384 executions, checking canonical outputs/termination, progress, accepted tokens, checkpoint publication, repeat consistency and pooled measurements. There are no invalid, pending, unknown or unattempted executions.

| Fresh native result | Retained K1 | KRC |
|---|---:|---:|
| Verification calls | 3,027 | 3,005 |
| Accepted draft tokens | 16,513 | 16,535 |
| Identical output tokens | 19,547 | 19,547 |
| Sum of per-request mean latency | 52.275888 s | 51.829177 s |

The net saving is **22 calls**: 44 saved across improving requests and 22 added across regressions. Counts use one validated trajectory per request. Fresh serving latency decreases **0.8545%** with checkpoints available; the three repeat-level reductions are **0.9599%, 0.8456% and 0.7580%**. These repeated timings are not additional independent requests, and they exclude adaptation and checkpoint switching. Historical KC's 3,035 calls are context only; its timings are not pooled with this run.

Execution matches the declared scope: **256 K presentations, 128 Adam steps, 786 core response forwards and 482 backwards** (256 K plus 226 violation backwards). The 18 local diagnostic forwards are included in the 786. There are **126 applied corrections**, all nonzero after BF16 rounding, and two zero-violation no-ops. Startup adds two disposable forward/backwards and zero Adam steps, separately counted. No new feedback, extra qualification optimizer step or baseline retraining occurs.

Learning and reference roles each use **65,096 anchors** and 949,560 available proposal occurrences. Reference protection covers **484,656 prefix occurrences**. The reference pair supplies additional training exposure and work; it is not a held-out test or a compute-matched control against K.

| Step | Role | Lost old-prefix tokens, ordinary candidate → corrected | Net prefix gain, ordinary candidate → corrected | Repairs retained / lost / new |
|---:|---|---:|---:|---:|
| 1 | Learning A | 153 → 155 | +1,740 → +1,746 | 526 / 1 / 4 |
| 64 | Learning A | 3 → 1 | +258 → +254 | 77 / 7 / 3 |
| 128 | Learning A | 13 → 13 | +163 → +165 | 62 / 2 / 1 |
| 1 | Reference R | 151 → 121 | +43 → +88 | 54 / 3 / 5 |
| 64 | Reference R | 6 → 7 | +9 → +14 | 4 / 1 / 2 |
| 128 | Reference R | 26 → 21 | 0 → +12 | 11 / 0 / 4 |

The local results are mixed. At reference step 64, V increases from **0.000011174 to 0.000016761**, and old-prefix loss increases from six to seven. At learning step 64, seven useful repairs are lost and three gained. The method therefore does not establish a strict preservation or nonlinear loss-decrease guarantee. All these are overlapping training-block measurements; later ordinary candidates start from the same current KRC pre-step state, not from historical K64/K128 checkpoints.

The full KRC event costs **674.201 s**, including **640.685 s** in training windows. Intrinsic reference capture, gradients and correction cost **287.090 s**; diagnostic replays cost **6.192 s**; integrity/reproduction checks cost **6.621 s**. Persistence/export cost **17.132/13.611 s**. Peak allocated/reserved memory is **38.722/40.613 GB**. The driver takes **1,358.423 s**, including benchmark work. All 193 benchmark publications cost 280.493 s and are excluded from request latency. These timers are nested and must not be added twice; no fresh matched K training-cost control exists.

The supported conclusion is that this complete KRC recipe produces a **small native improvement over retained K1 in the prescribed screen**. The evidence does not identify whether the gain comes specifically from preservation, the extra reference exposure, or another consequence of the correction trajectory. It does not establish an adaptation-inclusive online speedup, a stable benefit on fresh requests, or permission to tune or extend the run. The experiment has stopped at the agreed report.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
