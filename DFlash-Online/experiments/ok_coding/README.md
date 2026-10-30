# Fixed two-seed O/K coding confirmation

This additive driver implements the October30 handoff without changing O/K
numerical training. Seeds17 and29 each start from the released draft and complete
events1..4. Each seed/event generates its own128-response O feedback pool once,
then runs O and K in separate fresh interpreters. K checks O's natural response
orders and anchors, including event1; it does not force either draw.

Preparation seals512 training prompts and one128-request evaluation panel before
model work. Source answers never enter the inputs. Four cells—17/2,17/4,29/2,29/4—
measure F/O/K with three repeats. The first eight requests of each cell are the
included72-generation correctness gate. First-cell F supplies the globally reused
exact output/termination reference; every F count trajectory stays identical
across cells. Timing uses the prescribed balanced rotation.

The complete budget is1,024 feedback generations,1,024 terminal-teacher preparation
forwards,2,048 native Adam updates,4,096 response presentations and4,608 evaluations.
The5,632 total generations include the288 serving-gate executions. There is no
extra model qualification, warmup, preview or reference generation.

## Source and CPU checks

Run from the DFlash-Online source root using the pinned training environment:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:training:tests python -m unittest discover \
        -s tests -p 'test_ok_coding_*.py' -v

contract.py pins input order, budgets and all4,608 serving slots. training.py adds
genuine empty-Adam event-zero validation and events1..4 continuation gates.
Historical validators and numerical implementations remain unchanged.
evaluation.py reuses qualified native output/trace/publication checks. report.py
reconstructs raw measurements and checks actual durable work ledgers.
CPU fixtures are not a new GPU qualification or an experimental result.

## Sealing and admission

Use prepare.py --help for exact retained-source preparation inputs. Preparation
refuses insufficient eligible pools. The published implementation is bound to
the completed preparation with:

    python -m experiments.ok_coding.seal \
        --code /absolute/path/to/DFlash-Online \
        --prepared /absolute/path/to/preparation_001 \
        --pins /absolute/path/to/qualified/sealed \
        --output /absolute/path/to/local/production_001 \
        --remote-root /data/sources/minhao/runs/dflash_ok_coding_20261030/production_001 \
        --remote-code /data/sources/minhao/DFlash-Online

The seal requires a clean published Git revision. Source, stable per-seed configs,
model/runtime pins, ordered inputs, evaluation schedule and peak storage budget
are bound before model work. Config horizon7424 and native max_events58 remain;
the campaign independently stops after event4/step512.

Before server actions, follow the workspace A800 rules. Full storage admission
includes all eight capture/prepared pools, eight required endpoint state/export
pairs, four initial states, rolling predecessors, atomic overlap, compiler
caches and reserve. No offload or favorable generated lengths are assumed.
A capacity failure blocks launch rather than permitting reduced retention.

    CUDA_VISIBLE_DEVICES='' python -m experiments.ok_coding.preflight /absolute/run/root

Once exact source, environment and complete resources are admitted, invoke the
supervisor with an explicitly recorded wall-time bound (seconds):

    bash experiments/ok_coding/supervise.sh /absolute/run/root WALL_TIME_BOUND_SECONDS

It runs the following wrapper once, with durable start/exit/finish records:

    bash experiments/ok_coding/run_a800.sh /absolute/run/root

The shell performs CPU preflight/tests, then runs the campaign once and records
postflight. Each training job has exactly one CUDA-visible updater device and
six separate, initially empty compiler/kernel cache directories. Pinned model
caches remain unchanged. PID, runtime/compiler options, cache paths and physical
device identity are recorded. Exact boundary checks and nonrandom work observers
do not add model passes or alter restored RNG.

## Retention and explicit recovery

Outputs live under campaign_001/seed_17 and seed_29. Each arm retains its true
initial_state.pt. Every event has an immutable complete-state hard link and
export; only rolling event1/3 tensors/exports may be retired after verified own
successors and the required event2/4 evaluation. Required event2/4 endpoints,
all feedback/captures/prepared features, compact training records and failures
remain. Retired files are deleted, not archived.

Each resume_NNN_ARM directory contains initial or saved/live-resume checks,
exact endpoint checks, identities, startup/cleanup receipts, costs and WORK.jsonl.
Every forward/backward/native Adam action records a durable start before execution
and completion afterward. Interrupted starts stay charged and unknown. Feedback
and terminal preparation use the same accounting.

Recovery is explicit after rechecking source, active jobs and capacity, under the
same pinned environment. Use the launch wrapper's --resume mode, or its exact
environment followed by:

    CUDA_VISIBLE_DEVICES=1 python -u -m experiments.ok_coding.campaign /absolute/run/root --resume

The campaign performs original seal/model/runtime/config, hardware and conservative
full-capacity readmission before constructing a backend. Recovery validates
complete immutable pools, closed ordinary jobs, endpoint identities and raw
completed evaluation cells, then reuses those results. Partial feedback,
preparation, training or evaluation work blocks rather than being regenerated.
Missing rolling boundaries require completed successor-retirement receipts;
missing required endpoints always block. Interrupted retirement with only intent
or progress evidence also fails closed. This is complete-boundary recovery,
not arbitrary interruption repair. Prior failure status and explicit admission
remain under recovery/.

## CPU result reconstruction

After completion or a preserved failure:

    CUDA_VISIBLE_DEVICES='' python -m experiments.ok_coding.report \
        /absolute/run/root /absolute/report/directory --execution-commit FULL_EXECUTION_COMMIT

Outputs include SUMMARY.json, REPORT.md, ALL_REPEATS.csv, REPEAT_TOTALS.csv,
PAIRED_REQUESTS.csv, TRAINING_COSTS.csv and TRAINING_AUDIT.json. Raw-valid completed
cells remain available when another cell is incomplete; an invalid/incomplete
campaign receives no overall efficacy interpretation. Source authorship and CPU
reconstruction do not constitute an independent GPU rerun.

Primary K/O effects are reported separately at event4 for each seed; event2 is
secondary. F is contemporaneous in all four cells. Intervals use10,000 paired
request resamples with PCG64 seed20261030 and linear percentiles. Both seeds reuse
the same draw matrix. Repeats, repeated F cells and seed/request rows are not
independent samples. Costs and scoped memory stay separate from serving latency.
No automatic seed, event5, tuning, convergence or deployment-speedup claim follows.
