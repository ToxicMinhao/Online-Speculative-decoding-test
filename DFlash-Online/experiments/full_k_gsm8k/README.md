# Full GSM8K K and K_HALF campaign

This additive runner completes the October 31 fixed handoff. It calls the
existing `online_feedback` inference, conditioning, first-error-selected KL,
sampler, accumulation, optimizer and exporter. It does not implement a new loss.

K starts from released Qwen3-8B-DFlash-b16, empty Adam and seed 17, at constant
LR 3e-4. K_HALF starts independently from the same released weights and RNG,
with only LR 1.5e-4 and run metadata different. Both serve the original 7,473
GSM8K requests in order, collect their own feedback and update after each 128
responses. Each completes 58 events, 7,424 Adam steps and 14,848 presentations;
the last 49 requests are served without training. Retained F/O58 are controls,
not training initializations or shared feedback suppliers.

The model/runtime pins and original full-O manifest are verified before any
model work. The old O feature tensors are not needed or reconstructed. The
two reviewed shared-source differences add the established K objective and
experiment interfaces; original O arithmetic, teacher conditioning and runtime
paths remain unchanged. This campaign adds wrappers and tests only.

## Entrypoints and responsibilities

- `preflight.py`: read-only original-artifact inventory, native trace checks,
  source/model/runtime verification and resource admission.
- `seal.py`: immutable original inputs, exact published source, configuration,
  complete evaluation schedule and storage/work budget.
- `training.py`: persistent native trainer, independent initial states, exact
  own-state recovery, event/response/window observations and RNG isolation.
- `campaign.py`: K then K_HALF; synchronous serve/prepare/train/publish;
  bounded retention and exact work ledger. No automatic scientific retry.
- `evaluation.py`: capture-disabled intermediate/final evaluations, rotating
  method order, native correctness checks and independent raw-record audit.
- `costs.py`: passive training-device peak memory and nested compiler timers.
- `audit.py`: reconstruct training counts, chronology, exposure and costs from
  the actual event and native records.
- `statistics.py` / `report.py`: paired summed ratios, fixed bootstrap and
  complete reporting, including invalid or unfinished work.

Intermediate endpoints are 14/29/44, once per arm on the first 128 original
held-out requests. RNG/state are preserved and no checkpoint selection occurs.
Once both arm workers exit, final serving compares F/O58/K58/K_HALF58 on all
1,024 original requests, three repeats, with 64-request method blocks. The
qualified PR21 schedule maps F/K/K_HALF/KP to F/O/K/K_HALF. Across 16 blocks and
three repeats, every method has each position 12 times and each pair has 24/24
precedence. Every request and publication receives the established full cache
flush. No training or artifact transfers overlap timed final serving.

## Fixed work

| Work, both new arms combined | Count |
|---|---:|
| Training events | 116 |
| Committed Adam steps | 14,848 |
| Ordinary response forwards / backwards | 29,696 / 29,696 |
| Online generations | 14,946 |
| Terminal-feature preparation forwards | 14,848 |
| Intermediate generations | 768 |
| Final generations | 12,288 |
| Inherited correctness generations | 8 |
| All complete generations | 28,010 |

The four inherited correctness generations per arm are one capture-disabled
replay before its first update and updated/frozen/reloaded checks after that
actual update. There are no extra Adam qualification steps, generation warmups,
reference probes or efficacy gates. Null or harmful results continue.

## Execution

Run the inventory using the pinned training interpreter, the original full-O
root and original coding runtime-pin directory. This checks every retained
F/O execution and native trace before constructing the canonical references:

```bash
python -m experiments.full_k_gsm8k.preflight ORIGINAL_FULL_O_ROOT \
  --pins ORIGINAL_CODING_SEALED_DIRECTORY --inventory-output NEW_INVENTORY_DIRECTORY
```

After publishing the clean source and verifying the historical offload, seal
the exact destination. The `--inventory` argument is its `REUSE_AUDIT.json`:

```bash
python -m experiments.full_k_gsm8k.seal \
  --code DFLASH_ONLINE --inventory REUSE_AUDIT_JSON --pins ORIGINAL_CODING_SEALED_DIRECTORY \
  --output LOCAL_NEW_RUN --remote-root A800_NEW_RUN --remote-code /data/sources/minhao/DFlash-Online
bash experiments/full_k_gsm8k/run_campaign.sh A800_NEW_RUN
```

The launcher uses the same qualified environment as full O, completes CPU
checks and capacity admission, runs both streams, then writes the report and
postflight checks. Its 48-hour ceiling is an execution bound, not a learning
criterion. Logs, exit status and all failed work remain available. Never rerun
the launcher in an occupied directory. An explicit `campaign --resume` accepts
only exact own complete event boundaries with no unaccounted scientific work;
it does not repeat an incomplete event or partial final evaluation.

## Storage and interpretation

Admission covers two empty-Adam initials, the completed arm, atomic replacement
of the active arm's complete state/export, coexisting native/prepared feedback,
five-percent tensor headroom, 16 GiB compact evidence, 8 GiB compiler growth,
2 GiB workspace and an 8 GiB reserve. Only verified owned tensors already on
disk receive remaining-write credit; no future deletion receives credit.
Initial/final states, final exports and all compact evidence remain. New
superseded states, prior weights and event features are retired only after
verified successor commitment, with exact receipts and lost replay capability
disclosed. Historical input artifacts are not retired by this runner.

Primary is K_HALF58/O58; K58/O58 is the same-LR full-updater comparison.
K_HALF/O is **not** a matched-LR ablation. Ratios use summed counts and request
mean latency across three repeats. Intervals use 10,000 paired-request draws,
PCG64 seed 20261102 and linear percentiles. They are conditional on seed 17 and
the previously used 1,024-request benchmark. Historical O timing is not reused.
Intermediate curves provide count-only comparisons with F, not faster-learning
claims versus O. Timing is checkpoint-ready serving, not asynchronous or
training-inclusive speedup. Compiler timers nest within training; worker
compilation remains included in loading/serving rather than independently
identified. No aggregate request timer is presented as a kernel breakdown.
