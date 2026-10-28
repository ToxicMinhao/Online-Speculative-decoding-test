# DFlash online feedback

A reusable synchronous loop around official DFlash inference and SpecForge
training. The learning copy starts from released DFlash; the target stays fixed.
The default uses upstream weighted **hard-token cross-entropy**. Optional
`osd` teacher forward-KL and `projected` targets share the same training loop.
Choose the objective explicitly; changing it does not change the configured
optimizer, sampler, buffer or update schedule.

## Read the code in this order

| File | What to inspect |
| --- | --- |
| `online_feedback/controller.py` | `serve`, `update`, `evaluate`: the complete online sequence. |
| `online_feedback/native.py` | Generation, preparation of teacher features, publication of new draft weights. |
| `online_feedback/sglang_backend.py` | SGLang/FA3 alternative with the same controller interface. |
| `online_feedback/sglang_hooks.py` | Native target-feature observation and cache-safe weight publication. |
| `online_feedback/training.py` | Model preparation, batch construction, training passes, checkpoint/export. |
| `online_feedback/objective.py` | FP32 hard CE, OSD teacher KL, and projected targets. |
| `online_feedback/projected_target.py` | The small preservation-and-repair target calculation. |
| `online_feedback/buffer.py` | An ordered queue of responses awaiting training. |
| `online_feedback/qualify.py` | Optional real-model integration checks, separate from the reusable loop. |

The central sequence is:

```python
record = controller.serve(prompt_ids, ...)
# The caller appends record['response_ids'] to its conversation history.
event = controller.update(event_id, count=responses_per_update, passes=2)
# The next serve() uses the newly published draft.
```

`update()` reads the oldest queued responses, replays their exact tokens through
the frozen target, trains the draft, exports it, publishes it, and marks those
responses consumed. There is one model/optimizer/scheduler across all events;
starting a new event does not reset learning. `evaluate()` never queues feedback.

## Where learning actually happens

In `training.py`, `_build_runtime()` creates the upstream model, optimizer,
strategy and `TrainerCore`. `_batch()` forms a training batch. `_train_passes()`
contains the readable nested pass/response loop and calls `core.train_step()`.
Step into `training/specforge/training/controller.py` to inspect forward, loss,
backward and the optimizer operation.

Full prompt/history tokens provide context. Only the current response receives
loss. The preparation code checks the saved response against the frozen target's
next-token choices; it never substitutes dataset answers or rejected draft tokens.
Only complete optimizer accumulation windows can be published. Sampling, learning
rate, scheduler and the configured reduction are unchanged by the refactor.

The Transformers backend prepares teacher features by replaying saved native
blocks. The SGLang backend captures them from actual prefill/verification and
separately completes the final emitted token's unused feature row.

## SGLang/FA3 backend on A800

```python
from online_feedback.sglang_backend import SGLangDFlashBackend

backend = SGLangDFlashBackend(
    target_path, released_draft_path,
    worker_python="/data/conda_envs/minhao/adaflash_author/bin/python",
    output_root=fresh_native_directory, serving_gpu="0",
    context_limit=2048, block_size=16,
)
# Pass this backend to the same OnlineController and SpecForgeEventTrainer.
# Close backend.close() and trainer.close() when the session finishes.
```

The SGLang worker has its own Python environment. The trainer retains the current
SpecForge environment and Flex attention; FA3 is the serving/teacher backend.
The inherited `model.target_backend` training config does not select the online
serving implementation. Choose the backend class explicitly. Existing historical
experiment entry points are not silently changed.

`online_feedback.qualify_sglang` runs the bounded infrastructure check with
two responses, two passes and two optimizer steps. It does not launch the
128-response/58-event campaign. Launch it with the project environment helper;
the training process uses GPU 1 and the worker uses GPU 0. The exact launch is
retained with the qualification artifacts under `runs/dflash_online_sglang_20261023`.

Each request starts with empty target/draft caches. Publication reloads the draft,
refreshes SGLang's copied KV helper weights and clears caches. Failed publication
blocks serving. Evaluation disables feature capture and never supplies feedback.
Full tensor equality is checked in qualification, outside ordinary request timing.

## OSD baseline: teacher forward KL

```python
trainer = SpecForgeEventTrainer(
    config_path, output_root, objective="osd", reduction="response_mean",
)
```

This uses AdaFlash's OSD training objective: the frozen target's full distribution,
forward KL, temperature 1 and no hard-label mixture. For prediction position `j`,
the saved normalized final target hidden state at `j-1` passes through the frozen
vocabulary head. The draft predicts at `j`. Prompt/history supplies context;
only sampled positions in the newest response receive loss. No `q0` draft copy
or projected target is used by OSD.

The loss is `sum(weight * KL(teacher || draft)) / (sum(weight) + 1e-6)` per
response. `response_mean` gives equal weight to each response during accumulation;
the existing `upstream_weight_mean` option instead pools weighted numerators and
denominators. Earlier responses are not added to the buffer again by this choice.
`OSDTrainStrategy` always passes teacher features, including when optional
diagnostics are disabled. Teacher tensors/head remain outside optimization.

This is **OSD-style teacher forward-KL under DFlash-Online**, not an exact paper
reproduction or full AdaFlash. The pinned author recipe is
[`507b7aba`](https://github.com/ZinYY/AdaFlash/blob/507b7aba5fbccaa99a400f161e4f2794ba6a388c/recipes/qwen3_osd.sh).
The current OSD recipe is **`configs/osd.yaml`**, selected October 23:
constant LR `3e-4`, direct PyTorch AdamW on BF16 draft weights/moments, weight
decay `0.01`, responses shuffled independently each pass, up to `512` sampled
anchors per response, BF16 probability/KL arithmetic, and Flex training attention.
Position weights, weighted loss reduction and denominator remain FP32. The selected
schedule is **128 responses per event, two passes and 58 events**, with accumulation
two: 128 optimizer steps per event and **7,424 steps total**. `max_steps: 7424`
is the total horizon. The config loader checks that the event schedule agrees.

```python
trainer = SpecForgeEventTrainer(
    "configs/osd.yaml", fresh_training_directory,
    objective="osd", reduction="response_mean",
)
controller = OnlineController(backend, trainer, fresh_loop_directory)
# After serving 128 new responses into this controller:
event = controller.update("event_001")
# Continue with the same trainer/controller for the next 128 responses.
```

The controller reads buffer size, pass count and event limit from the trainer's
recipe. It refuses an update beyond event 58; the existing trainer separately
enforces the total step horizon. Explicit capacity/count/pass arguments still
support the small qualification and historical experiments. Importing this code
or loading its config does not launch a dataset loop. The caller still supplies
requests and decides when to invoke `update()`; no collection policy is changed.

`online_feedback/recipe.py` reads the online options before upstream config
validation. `optimizer.py` wraps direct AdamW without FP32 master weights or a
scheduler. `training.py` records the actual per-pass order, anchors and LR and
preserves optimizer/RNG state. `objective.py` selects the loss arithmetic.

Anchor eligibility, exact saved-token sequence preparation, response-only masks,
chunked vocabulary loss and HF/SDPA serving are unchanged. No old results or
configuration files are rewritten: configs without the online options retain
the previous FP32-master optimizer, FP32 KL and fixed response order. Resuming
an old state under the new recipe is refused; start with a fresh optimizer.
This adopts selected training settings and event budget, not the author's older
sampler, async pipeline, reverse-KL mixture or adaptive length head. CPU component
checks do not qualify full-model Flex execution or establish a learning gain.

Feedback preparation and training remain separately charged. `teacher_head_s`
records teacher projection work, including checkpoint recomputation, **inside**
`training_s`; do not add it again to total cost. Teacher softmax/KL and draft work
also enter training time. The buffer stores features and token IDs, not full
response-by-vocabulary distributions. New head evaluation can differ numerically
from the original native block's logits; exact native posterior equality is not
claimed.

Start reading at `TargetObjectiveModel._chunk_loss()` and `OSDTrainStrategy`,
then follow the existing `_train_passes()` and `TrainerCore`. CPU tests check
formula/gradient alignment, chunking, masks and integration. The October 23 matched
follow-up completed all three groups with 96 native generations and 24 optimizer
steps. All output, publication, anchor/exposure and RNG checks passed. Held-out
calls increased 9.28% under the unchanged prior recipe, despite a source-call gain
on ShareGPT. This qualifies execution for that setup, not general learning benefit.
Results are under `/data/sources/minhao/runs/dflash_online_osd_20261023/execution_001`.

## Projected-target component

Choose the loss when creating the existing trainer; the controller does not change:

```python
trainer = SpecForgeEventTrainer(
    config_path, output_root, objective="projected", reduction="response_mean",
)
# Matched control: objective="hard_ce". Original path: objective="upstream".
```

For one event, `train_event()` first validates the saved response labels against
the frozen target's canonical replay receipt. It then makes one detached copy of
the pre-event draft. Every presentation uses the author's sampled anchors for
both the current draft and this fixed reference. The reference does not consume
training RNG and never joins the optimizer or exported checkpoint.

At each sampled position, `projected_targets()` keeps the reference distribution
when its native argmax is correct. Otherwise it moves probability mass from the
largest competitors until the verified target token has at least 1.1 times each
competitor's probability. Weighted soft CE trains the current draft against that
detached target. The matched control uses hard CE on the same labels, masks,
position weights and denominators. Both calculate probabilities/losses in FP32;
model tensors remain BF16. Sampling, backbone, FP32-master AdamW and schedule are
the existing implementations. Correct-reference rows stay in the objective.

The reference stays fixed across **all responses, passes and optimizer steps in
the event**. It is released after the final backward; the next event captures its
own pre-update draft. Targets are recomputed for bounded chunks from that same
reference, rather than caching response-by-vocabulary tensors. Activation
checkpointing bounds retained loss intermediates and incurs recomputation.

Native argmax uses the lowest token ID on ties. FP32 softmax underflow is retained
as zero; there is no probability floor or post-hoc renormalization. Tests pin mass
and margin absolute tolerance at 2e-6 and gradient/oracle tolerance at 2e-7 absolute,
2e-5 relative. Configuration and objective identity must match when resuming.

Event `objective_work` records reference setup, reference forward, reference head
and projection time. **Setup is outside `training_s`; the other three are already
inside it**, including loss-checkpoint recomputation. Do not add them again.
These component timers synchronize CUDA for attribution; they are diagnostic
training costs, not asynchronous-deployment measurements. The existing event wall
timer includes all of them, plus persistence/export. Publication remains separate.

To inspect this change, put breakpoints at `TargetObjectiveModel.begin_event()`,
`TargetObjectiveModel.forward()`, `projected_targets()` and `_chunk_loss()`.
Continue into the existing `TrainerCore` to inspect backward and AdamW.

The fixed-feedback screen now lives in [experiments/projected_target](experiments/projected_target/README.md).
It reuses this infrastructure. GSM8K and CodeAlpaca passed native correctness;
ShareGPT stopped at an adapted-output mismatch. The screen is incomplete and
has not demonstrated improvement over frozen DFlash. The consolidated driver
has CPU checks; it has not been run as a new GPU campaign. Historical executed
source and results remain separate and unchanged.

## State and checks

This is a single-process experiment, not a concurrent serving service. The queue
uses one readable `state.json`, with exact token lists and explicit event status.
Checkpoints retain optimizer state, counters, sampled anchors and RNG state.
Small shared helpers perform atomic writes and isolate RNG changes from replay,
model loading and export. No file, token or model SHA-256 is computed by the loop.

Checks live at meaningful boundaries: full-context limits, response masks,
target/response alignment, compatible checkpoint publication and exact replay
outputs. A failed update remains visible and blocks automatic retraining. The
caller must resolve the failure explicitly; there is no automatic recovery loop.
Checkpoint paths identify the released draft and per-event exports. Treat completed
exports as fixed files within a run; this runner no longer detects file tampering.

The refactor uses a new buffer/trainer state format. Old run artifacts are not
migrated or overwritten. The old implementation and its recovery path remain in
`/data/sources/minhao/runs/dflash_online_refactor_20261022/before.tar.gz` for inspecting
historical runs. Start new work in a fresh output directory.

## Debugging and measurements

Place your first breakpoint in `qualify.run()` before backend/trainer creation to
inspect the whole setup, or in `controller.update()` to focus on adaptation.
Useful next breakpoints are `native.prepare_feedback()`, `training._batch()`,
`training._train_passes()` and `native.publish()`.

Each inference call creates fresh target and draft caches. Timed generation
includes the optional verification-trace callback, which can add CPU/GPU transfer
overhead. Teacher preparation, training, checkpoint saving, export and activation
are reported separately. Benchmark replays do not modify the feedback queue or
conversation history. Neither instrumented latency nor this infrastructure
fixture is evidence of a scientific speedup.

## Optional native qualification

The existing fixture uses Qwen3-8B / released DFlash-b16, BF16,
seed 17, cap 128 or EOS, block size 16, context limit 2048. Four serving requests
form two two-response update events, with two passes and accumulation two:
four optimizer steps and eight response presentations in total. Three held-out
replays test updated, frozen and reloaded drafts. It checks exact publication and
restart state without putting model scans in normal serving or training.

```bash
bash /data/sources/minhao/DFlash-Online/run_qualification.sh \
  /data/sources/minhao/runs/dflash_online_foundation_20261022/NEW_RUN_DIRECTORY
```

The refactored loop passed this native qualification on October 22: seven
real-model generations, two update events/four optimizer steps, exact publication
and trainer restart, and identical held-out output IDs. The updated draft needed
23 verification calls versus frozen's 19 on that fixture; this is a correctness
pass, not an improvement claim. The run completed in about 3m24s. See
`/data/sources/minhao/runs/dflash_online_gpu_check_20261022/REPORT.md` and
`/data/sources/minhao/runs/dflash_online_foundation_20261022/refactor_qualification_001/RESULT.json`.
Historical interrupted evidence remains separate. A new run requires a fresh
output directory and admission for overlapping checkpoints plus an 8 GiB reserve.

To select OSD, use the same `python -m online_feedback.qualify` entry with
`--objective osd` plus its required `--target`, `--draft`, `--config` and fresh
`--output` paths. Its restart check preserves the selected objective and reduction.
The historical shell launcher continues to choose the default hard-CE path.

## Source boundaries

- Official DFlash: `07ebd93db9f472af339b644bb70221ad8428328a`.
- SpecForge: `1839aae30003e3573d00979eb1edd57347b3578c`.
- `inference/dflash/model.py` retains the earlier optional trace callback and
  Transformers full-attention Qwen3 cache compatibility fix.
- `training/scripts/prepare_hidden_states.py` retains the earlier cache-directory fix.
- This refactor changes neither of those upstream trees, the learning settings,
  nor Fenrir's `/home/minhao2/DFlash-Research` folder.
