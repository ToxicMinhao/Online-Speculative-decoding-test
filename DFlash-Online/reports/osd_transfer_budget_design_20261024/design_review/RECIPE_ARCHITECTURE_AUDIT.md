# Successful-recipe and architecture audit

## Material Passport

- ARS experiment-agent, plan-mode supporting audit; 2026-10-24.
- Scope: read-only inspection of retained successful-source snapshot, sealed configuration and existing local SpecForge source. No GPU, SSH, model run or production-code changes.
- Requested deliverable: exact learning parameters and minimal architecture work for the proposed four-event, shared-first-event comparison.
- Status: source-audited design advice; native compatibility and new branch orchestration remain unqualified.
- Local storage read: Fenrir approximately 78 GiB available. Existing user processes left untouched. No deletions.

## Source scope and identities

Primary archived implementation is `research/dflash_online_osd_rerun_20261023/worktree/`; its online files are enumerated in `sealed/SEAL.json` alongside the successful configuration. Local SpecForge is `/home/minhao2/DFlash-Research/training` at `1839aae30003e3573d00979eb1edd57347b3578c`, the same upstream commit identified by archived `worktree/README.md:291–297`. The only local modified upstream file is the preexisting `scripts/prepare_hidden_states.py`; it was not touched. The archived online source tree does not include the external installed SGLang tree, so launch must compare the actual A800 source/environment to the sealed installation rather than treating this local audit as a native qualification.

Pinned successful trainer versions in `sealed/SEAL.json`: Torch 2.13.0, Transformers 5.12.1, safetensors 0.8.0. The exact online file manifest remains the reference; no source copied or altered by this audit.

## Exact recipe to preserve

| Setting | Successful effective behavior | Source |
|---|---|---|
| Model pair | Frozen Qwen3-8B target and released Qwen3-8B-DFlash-b16; target/draft paths under `/data/sources/minhao/models/qwen3_dflash_foundation_20261022/` | `sealed/config.yaml:1–9`; `REPORT.md:14–24` |
| Trainable weights | Existing draft model parameters that require gradients; target vocabulary head and embeddings remain frozen | `online_feedback/training.py:231–237`; `online_feedback/optimizer.py:18–28`; `online_feedback/objective.py:75–80` |
| Objective entry | `SpecForgeEventTrainer(..., objective='osd', reduction='response_mean')`; neither upstream hard CE nor projected-target mode | `experiments/osd_persistent/run.py:198–200` |
| Objective | Pure forward KL from frozen target to current draft, temperature 1, hard-label mixture 0 | `online_feedback/objective.py:106–121`; `online_feedback/training.py:249–255` |
| Teacher | Native SGLang captured committed-context features and final hidden vectors, not a regenerated answer; teacher probability computed through frozen head at each training forward | `online_feedback/sglang_backend.py:159–230`; `online_feedback/objective.py:111–121` |
| Teacher alignment | Target hidden row at position i−1 supplies target for token i; padded safe indices remain masked | `online_feedback/objective.py:181–186`; SpecForge `algorithms/common/dflash_family_model.py:450–465` |
| Position weights | Offset j=1…15 gets exp(−(j−1)/7); anchor slot 0, prompt positions and outside-response tail receive zero | `online_feedback/objective.py:164–174` |
| Response loss | Sum weighted token KL divided by summed weights plus 1e−6 once per response, not once per chunk | `online_feedback/objective.py:187–200` |
| Across responses | Equal response means across the two-response optimizer window, not global token-weighted aggregation | `online_feedback/training.py:117–149`; SpecForge `training/controller.py:430–467` |
| Precision | BF16 draft weights and Adam moments; BF16 logits/probabilities/log-softmax/KL rows; FP32 position weights and final weighted sums | `online_feedback/optimizer.py:16–33,70–75`; `online_feedback/objective.py:106–143`; sealed recipe |
| Optimizer | Direct torch AdamW, no master parameters; LR constant 3e−4, decay .01, betas (.9,.999), epsilon 1e−8, clip global norm 1 once after accumulated gradients are normalized; zero grads with set_to_none | `online_feedback/optimizer.py:16–50`; `REPORT.md:19`; SpecForge `training/controller.py:455–467` |
| Hidden optimizer defaults | Constructor forwards only lr and weight_decay, so Torch defaults govern betas/eps/amsgrad/maximize/foreach/capturable/differentiable/fused. Recipe snapshot stores all resulting param-group options and rejects mismatched restoration. Preserve installed Torch and inspect saved options before launch; do not silently force a different fused/foreach path | `online_feedback/optimizer.py:26–38,55–75`; `sealed/SEAL.json` |
| Event | 128 completed responses, two passes, batch one response, accumulation two, 256 response presentations and 128 optimizer steps | `sealed/config.yaml`; `online_feedback/recipe.py:29–35`; `experiments/osd_persistent/run.py:111–119` |
| Order and RNG | Seed 17 once when trainer runtime builds, then preserve full RNG during continuation; fresh torch.randperm response order each pass. Anchor draws use author sampler; observation wrapper makes no new draw | `online_feedback/training.py:204,316–323,423–463`; `online_feedback/rng.py:7–27` |
| Anchors | Uniform random ranking without replacement among positions whose anchor token and next token are both supervised; cap min(512, eligible count); sample per response presentation, then sort selected starts chronologically | SpecForge `algorithms/common/dflash_family_model.py:344–384`; `online_feedback/objective.py:218–222` |
| Training block | Block size 16: clean anchor plus 15 predicted positions, overlapping anchors permitted | `online_feedback/objective.py:166–174`; model config; native block agreement check `sglang_backend.py:43–46` |
| Attention and chunking | Flex training attention; vocabulary objective checkpointed in chunks of 8 blocks; chunking does not introduce optimizer steps | `sealed/config.yaml`; `online_feedback/objective.py:187–190` |
| Exact tokens and masks | Saved native prompt+response IDs; prompt-zero/whole-response-one mask; validate complete sequence before upstream normalizer; no decoding/re-tokenization or silent truncation | `online_feedback/training.py:67–114,406–420`; `online_feedback/sglang_backend.py:203–221` |
| Lengths | Output 2048 or EOS; complete training sequence at most 2272; serving admitted context 4096, native engine context set to 4096+16 | `sealed/config.yaml`; `sealed/MANIFEST.json`; `online_feedback/sglang_backend.py:47–55,117–118` |
| Serving | Native SGLang/FA3, BF16, TP1, concurrency1, greedy, block16, seed17, no-thinking Qwen prompts | `online_feedback/sglang_backend.py:47–60`; `experiments/osd_persistent/data.py:98–107`; native worker |
| Caches | Full idle flush before every request and publication; fused KV helper refresh on publication | `online_feedback/sglang_backend.py:148,232–239` |
| Timing | Capture disabled for benchmark generation; trace instrumentation remains active in every method. Complete request latency includes worker round trip and serving work, excludes preceding publication. Publication timed separately. No training may run concurrently | `online_feedback/sglang_backend.py:113–153,232–239`; `experiments/osd_persistent/run.py:64–108,294–299` |

The latest run's 58 events and 7424 steps are campaign budget, not independent per-step optimizer hyperparameters. For the proposed comparison, both branches must use the identical short-campaign config **from initialization**, with `max_events: 4` and `max_steps: 512`. All other effective settings remain unchanged. Do not set 896 as either branch's max_steps: 896 is the total physical work after sharing event1. DirectBF16AdamW has no scheduler (`optimizer.py:9–11,52–53`), so shortening this horizon does not introduce an LR schedule change. `load_event_config` checks the event/horizon product exactly.

## What current APIs already support

1. **Persistent learner and exact branch resume.** `SpecForgeEventTrainer(config_path, output_root, resume_state=..., objective='osd', reduction='response_mean')` builds the author core and restores weights, optimizer, Python/NumPy/CPU/CUDA RNG, global step, response presentations, event history and core microbatch counter. See `training.py:155–191,325–349,466–485`.
2. **Repeated fixed feedback without altering sampler.** `train_event(new_event_id, original_B1_feature_paths, passes=2)` can already reread the same 128 feature files in a later event. `_prepare_event` rejects duplicate responses within one event, but does not prohibit reuse across distinct events (`training.py:367–404`). B1's exact captured teacher features remain fixed; current draft outputs are recomputed. Each new event samples/shuffles normally.
3. **Serving and evaluation separation.** `SGLangDFlashBackend.generate(..., purpose='evaluation')` disables feedback capture, and `publish` switches checkpoint/caches. Both preserve the trainer process RNG (`sglang_backend.py:122–125,234–235`). The evaluator must validate against the already saved S canonical response rather than choosing a newly generated F answer as the reference.
4. **Source-fit diagnostic hook.** `train_event(..., reference_check=callback)` invokes the callback before/after training (`training.py:503–536`). The callback must implement eval/no_grad, identical preselected anchors/teacher/masks/weights at both endpoints, restore the model's train/eval mode and RNG, and avoid optimizer mutation. The existing hook does not enforce those invariants itself.

## Minimal implementation additions and pitfalls

### A thin experiment driver, not a new trainer

Add a scoped `experiments/osd_transfer_budget/` driver with preparation/manifest, branch scheduling, multi-checkpoint paired evaluation and reporting/retention. Reuse unchanged `online_feedback` loss, optimizer, training, capture and publication APIs. A small testable branch-state helper is reasonable; no generic abstraction framework is needed.

The existing `experiments/osd_persistent` path is deliberately guarded for 7473 training requests, 58 events and 1024 final-evaluation requests (`data.py:18–27,158–162`). Its `check_recipe` also hardcodes 58/7424 (`qualify_sglang.py:23–49`), and its evaluator supports only F/O. Do not weaken historical guards globally to make the new design fit. Use a new bounded manifest schema and the same per-step recipe with a separate explicit 4/512 schedule assertion.

### Preserve the first fork, then keep two independent continuations

Train B1 once at step0→128. Before subsequent updates, preserve a committed full restart artifact and S1 export. Both S2 and R2 must restore from that exact artifact into separate output roots. Use **one identical config text** for both branches: `_restore` checks `config_text` byte-for-byte and also checks algorithm contract, horizon, accumulation and numerical optimizer recipe. Changing `run_id` or config output_dir merely to label each branch will cause restore rejection. Branch identity belongs in experiment metadata and the constructor `output_root`, not in two different numerical configs. Never edit state files to bypass these checks.

Prefer serial trainer construction from saved branch state, then `close()`, around each event. Two live trainer objects share process-global RNG, so simply alternating two objects without restoration would intermingle sampling streams. Evaluation and source diagnostics must also preserve RNG. Identical fork RNG does not guarantee identical actual anchor positions after distinct feedback inputs: sequence shapes differ and CUDA draws consume different amounts. This is expected. Preserve per-branch stochastic continuation, rather than resetting seed17 each event or forcing anchors from a different response onto an arm.

Normal sampling still changes across repeated events. R repeats B1 responses/teacher features, **not necessarily the exact same anchor subset**. When a response has <=512 eligible anchors, all eligible starts are selected, although a random ranking is still drawn. Repetition versus unique response coverage is the treatment; it is not an exact repeated-gradient experiment.

### Do not requeue committed B1 records in FeedbackBuffer

`FeedbackBuffer._pending_records` excludes all records reserved by any event (`buffer.py:71–74`); duplicate IDs and nonmonotonic arrival indices are rejected (`83–113`), and `seal_event` consumes pending records only (`132–154`). This is correct serving behavior.

R should use an experiment-level replay manifest referencing B1's immutable prepared paths and source record IDs, then call the public `train_event` with new branch event IDs. It should not mark replay data as new serving data, duplicate IDs under invented request identities, or alter `FeedbackBuffer` to make committed records pending again. Its publication, event costs and metadata can be recorded by the experiment driver, using the same backend publication API.

### Native evaluation needs to restore S after benchmarking

At each boundary, save S's canonical B(k+1) outputs under S_k before consuming that buffer in training. Evaluate all methods on the saved input IDs/canonical output IDs. After rotating method order, explicitly republish S_k before any further canonical serving/capture; otherwise the final benchmark method would silently generate the next online feedback. R's replay outputs never enter any buffer.

Deduplicate identical checkpoint states at k1/k2 by explicit shared-state identity, not by treating coincident equal outputs as checkpoint identity. The original evaluator creates reference outputs from F (`run.py:69–84`); the new evaluator must use S's saved references and retain per-contrast validity, not silently reuse this behavior.

### Retention must be designed for this dependency graph

The old `checked_retire` deletes each event's native/prepared features and the previous export (`retention.py:167–207`), which would destroy B1 replay and current-versus-previous comparisons. Do not call it unchanged.

At minimum retain: immutable B1 prepared features for all R events and source diagnostics; shared first-event restart until both branches fork safely; current full optimizer/RNG state for each branch; current and preceding exports through their final scheduled panel; current S buffer capture/prepared features through its training/source-fit completion; compact outputs, traces, masks, anchor metadata, costs and validity records. Reserve atomic-save overlap and feature duplication explicitly. Existing single-trajectory storage admission is insufficient: it models two states/two exports plus one event's double feature tensors (`retention.py:41–109`), not two branches plus B1 memory. Recalculate at actual A800 paths before launch; no remote capacity claim is made here.

### Avoid changing optimization in the name of fairness

Both arms match initial state, optimizer steps, passes, response presentations and numerical recipe. They naturally differ in input length, sampled anchor count, weighted supervised exposure and time. Do not silently force equal anchor counts, repeat shorter responses more, truncate longer responses or alter gradient denominators to match tokens. Those would change the treatment and depart from the competent recipe. Report these differences and limit the interpretation to replacing repeated feedback with newly available feedback under fixed optimizer work.

## Budget/counter checks to implement

- Shared event:128 steps,256 response presentations, one physical computation.
- S events2–4:384 steps,768 presentations; R events2–4:384 steps,768 presentations.
- Per branch cumulative endpoints:128/256/384/512 steps and256/512/768/1024 presentations.
- Physical total:896 steps and1792 presentations;512 unique S training responses,128 unique R training responses.
-640 canonical generations: B1 released, B2 S1, B3 S2, B4 S3, B5 S4. B5 is never trained.
- Benchmark checkpoints/window:2,4,5,5 ×32 requests×3 repeats=1536 generations;128 distinct scored requests.
- Source diagnostics and publication tensor checks do not add generations or optimizer steps, but their work must be bounded and separately timed.
- Do not copy original first-event qualification generations without accounting: `run.py:130–153,231–251` adds replay generations beyond the proposed2176total. Fold output-preservation/publication gates into the scheduled panel or explicitly revise/seal the budget before execution.

## Focused qualification/tests recommended

1. Fork restoration exactly matches model weights, Adam moments and step counters, RNG and empty accumulation state, before either child event executes.
2. Interleaving branch event order does not mutate the other branch's saved state. Evaluation and fixed-anchor diagnostics preserve RNG/state.
3. B1 replay paths/IDs remain immutable, events have unique branch IDs, and no benchmark record is eligible for training.
4. All128responses appear exactly once per pass; two passes/event; every second response steps;128steps/event and no incomplete accumulation.
5. Source-fit callback keeps fixed objective at before/after and restores training mode/anchor sampler.
6. Benchmark input IDs are identical to saved S inputs, output IDs match saved S outputs, feedback capture is disabled, and every method uses the same cache policy with no overlapping trainer execution.
7. Survival of B1 features and previous exports until their final dependency; peak storage includes branch states and atomic overlap.
8. Stop only on correctness/storage failures, preserving evidence. Null/harmful learning is valid and must not trigger new settings, extra events or additional cases.

No implementation, sampling, seal, GPU qualification or campaign was performed by this audit.

## Follow-up: exact bounded source-fit diagnostics

The existing `reference_check(model, phase)` hook is suitable for a source-panel diagnostic without changing the OSD objective. Use 16 responses per source buffer, selected without performance information using a private `random.Random(17)` instance. Declare whether the same sampled relative indices are used for each buffer or a single private generator advances across B1…B4; do not leave that choice implicit. B1's selected responses/layouts should remain identical for all R events. The result describes these 16 source responses, not every response in the 128-response buffer.

For each selected response, draw the original capped-512 anchor layout exactly once using `trainer._original_sampler`, inside `preserve_rng()` and a separately declared diagnostic seed/state. The original sampler uses `torch.rand(..., device=device)` (`dflash_family_model.py:369`), so CPU-only RNG seeding is insufficient for GPU anchor draws. Seed/restore the pertinent Torch CPU/CUDA generators as well, or replay a saved private diagnostic RNG snapshot. Save the returned anchor positions and keep mask, token/response identities, precise gamma7 position weights, supervised-position count and denominator. Do not invoke `trainer._sample_and_record_anchors` when creating the diagnostic layout: that is the training observation wrapper.

At each callback phase:

1. Save per-module training flags, the current sampler attribute, a copy of `_anchor_observations`, the full process RNG state and a copy of `model.costs`.
2. Enter `torch.no_grad()` and eval mode. For each selected raw feature record, replace `model._sample_anchor_positions` with a fixed-layout callable matching `(seq_len, loss_mask, device, max_valid_anchors=None)`. Return saved int64 anchors and bool keep mask on the requested device. The same record must have exactly the same IDs, masks, teacher vectors and layout before/after.
3. Reuse `trainer.author_strategy.forward_loss(trainer._batch(raw))` (or its same wrapped OSD strategy), **not** `TrainerCore.train_step`. This preserves all four tensors, teacher predecessor alignment, BF16 KL, chunk size8, FP32 weighted sums and denominator epsilon. Read `StepOutput.loss`, its `(loss_sum, denominator)` and supervised-exposure metric; average the 16 per-response losses equally. Do not replace this with a pooled token-weighted loss.
4. Restore sampler, observations, every saved training flag, RNG and original cost counters even on diagnostic failure. Leave parameter requires_grad, gradients, optimizer state and counters unchanged. No backward/optimizer call occurs. Fixed-anchor evaluation must not enter training rows.

`checkpointed_chunk_reduce` automatically skips activation checkpoint recomputation when `torch.is_grad_enabled()` is false, but retains the same eight-block reduction arithmetic (`specforge/core/chunking.py:74–102`). Thus no separate loss implementation is needed. The callback's eval/no_grad mode is a source-fit measurement boundary, not evidence of numerical identity to every training-mode execution; qualify the fixed diagnostic by repeating unchanged weights/layout once within the planned forwards where possible and preserve observed deviations.

**Accounting pitfall:** `TargetObjectiveModel._chunk_loss` increments `model.costs['teacher_head_s']` even under no_grad (`objective.py:111–117`). Snapshot/restore these counters around each callback and record diagnostic teacher work separately. `train_event` begins its wall clock before the callback and reports `event_wall_s_before_commit` afterward (`training.py:511,528–532,575`), so the callback's total span also sits inside enclosing update clocks. Exclude the measured diagnostic span once when reconstructing serving-plus-adaptation cost; do not subtract its teacher sub-timer a second time. Preserve raw inclusive clocks for audit.

The fixed bound is **7 physical events ×16 responses ×2 endpoints =224 source-panel forwards** (event1 shared). These are additional forwards, not generations or optimizer steps. Layout construction does not perform draft forwards. No extra anchors or responses are added when a diagnostic is unfavorable or has zero supervision; record the case. In this qualified stream, fewer than two response tokens already stop the training event rather than silently replacing the response (`training.py:83–85`; `controller.py:78–82`).

### Objective identity during restoration

Full trainer state does not need a separate top-level `objective` field to reject an OSD-to-other-objective restore: the `algorithm_contract` contains `online_target_objective` with `name='osd'`, version1, BF16 loss precision, frozen-target predecessor teacher, temperature1, hard-label mix0 and epsilon1e−6 (`training.py:249–255`). `_restore` compares the entire contract (`334–339`), while `config_text` and reduction are checked separately (`332–333`). Projected/hard-CE contracts use distinct names and settings; upstream lacks the OSD contract. Nevertheless, the experiment must explicitly instantiate `objective='osd'`, `reduction='response_mean'`, assert the objective in status and event receipts, and copy the resolved contract into its sealed recipe. Restoration validation is an integrity guard, not a substitute for declaring the experiment's objective.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_transfer_budget_design_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
