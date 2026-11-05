# Projected-target feasibility screen

This is a small experiment driver around DFlash-Online, not another trainer.
The main reading path is run.py: run() then run_arm().

1. Frozen DFlash generates eight source A and eight held-out B answers per group.
2. native.prepare_feedback() replays the exact saved A verification blocks through
   the frozen target. It checks every saved argmax token, retains canonical-prefix
   features and masks the prompt. It does not compare full probability vectors.
3. C and P each create a fresh SpecForgeEventTrainer from released weights and
   seed 17. They use the same eight A responses, two passes, accumulation two,
   masks, sampled anchors, reduction and eight-step schedule. B never trains.
4. C uses hard CE; P uses the immutable pre-event projected target (c=1.1).
   Both call the existing train_event() and native.publish().
5. Each trained draft replays A and B. Exact input/output/termination equality
   gates comparison. Harmful but valid results continue; correctness failures stop.

The existing OnlineController handles persistent successive serve/update events.
This screen deliberately calls the same backend/trainer directly because F's
feedback pool must be shared by two independent learners. It is a fixed-pool
feasibility test, not a long-lived per-turn deployment measurement.

## Where to inspect

- run.py: experiment sequence and arm initialization/publication/evaluation.
- checks.py: qualification, exposure accounting and this screen's storage budget.
- config.yaml: unchanged pinned training recipe.
- report.py: saved measurements and validity; no model execution.
- ../../online_feedback/training.py: _batch(), _train_passes(), train_event().
- ../../online_feedback/objective.py and projected_target.py: the actual loss.

Qualification observes the reference through train_event(reference_check=...).
Normal online updates leave that argument unset. Pre-step learning rates are
recorded by the shared trainer; the experiment does not replace optimizer.step
or objective lifecycle methods. Timing for checks remains separate in receipts.

## Entry points

From the A800 shell, source the required helper and use the pinned environment:

```bash
source /data/sources/minhao/EAGLE_SPEC/setup/a800/a800_env.sh
cd /data/sources/minhao/DFlash-Online
export PYTHONPATH=$PWD:$PWD/training:$PWD/inference
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m experiments.projected_target.run --help
```

A future authorized run supplies --manifest, --target, --draft and a fresh --output;
--config defaults to the config in this package. Data and outputs remain outside
this source folder. The runner performs its own peak-storage admission. Current
free space does not admit another full campaign. There is no automatic recovery
or special reuse option in this entry point.

To recompute a report without rewriting historical evidence:

```bash
/data/conda_envs/minhao/dflash_specforge_20261022/bin/python -m experiments.projected_target.report SAVED_RUN --output-dir NEW_REPORT_DIRECTORY
```

The screen uses Qwen3-8B / DFlash-b16, BF16 model tensors, FP32 master AdamW,
greedy decoding, block 16, cap 512 or EOS, prompt cap 1024, context limit 2048,
fresh request caches and concurrency one. This is not the historical
Llama/AdaFlash forward-KL recipe. Actual learning rates follow the preserved
upstream eight-step scheduler; they are not eight constant 6e-4 updates.

## Existing evidence and current limits

The October 22 execution stopped at 136/144 generations, after all 48 training
steps. GSM8K and CodeAlpaca supply 96 qualified records. ShareGPT P A07 differs
from frozen at the final cap token; the entire ShareGPT group is excluded from
gain claims and its P restart state is retained. No automatic restart is allowed.
On the 16 qualified held-out B requests, calls F/C/P are 901/1023/987. P reduces
harm versus C but remains worse than frozen. Single instrumented timings do not
establish serving speedup.

Historical runner/report and exact execution archives remain under EAGLE_SPEC/
research/projected_target_feasibility_20261022. That historical runner includes a
one-off pre-training recovery that is intentionally absent here. The maintained
code revision must not be described as the source that executed those old results.
A native backend failure before returning can leave generation completion count
unknown; it stops the run. Pairing failures after return consume the recorded
budget. No retry is inferred from either counter.

## Matched OSD follow-up

`python -m experiments.projected_target.osd_followup` adds the OSD objective to
the existing sealed setup. Supply `--source` with the original campaign root,
the same `--manifest`, `--target`, `--draft`, and a fresh `--output`.

It reuses the original eight A feature files per group, starts each OSD learner
fresh, and matches historical C's anchors, exposure, LR and RNG. Frozen and OSD
each generate all sixteen A/B answers freshly: 96 generations, 24 optimizer steps
and 48 response presentations across three groups. The first complete group
qualifies correctness; valid negative results continue. Historical C/P are not
rerun, and their timings do not enter the new comparison. No extra warmups,
sampling changes or learning-rate tuning are included.

Fresh frozen outputs/counts must reproduce the retained reference, and every OSD
output must match the canonical tokens. Training ends before evaluation. Source
fit A and held-out B are reported separately. Single instrumented timings are
descriptive; these are not repeated deployment-speedup measurements. This entry
does not resume or repair the old stopped P run.

The October 23 follow-up completed all 96 generations and 24 optimizer steps;
all three groups passed native qualification. Pooled source calls decreased
6.08%, driven by ShareGPT, while held-out calls increased 9.28%. Every F/OSD pair
has identical output tokens. The completed comparison is retained under
`/data/sources/minhao/runs/dflash_online_osd_20261023/execution_001/campaign`.
It did not change learning settings or automatically launch another study.
