# DFlash two-update experiment — implementation for review

**Baseline: full-weight forward-KL adaptation of DFlash with the target fixed.**
Compare it with the same released frozen DFlash; target-only autoregressive
generation supplies a timing reference. This is an OSD-style DFlash control,
not an exact reproduction of published OSD.

This commit contains implementation and CPU validation. **No new model
generation, GPU qualification, training run, or scientific result is included.**
Dataset artifacts and local review receipts stay outside the repository.

## What is tested

Can two consecutive updates improve later native drafting when work remains
within a task family, alternates every eight requests, or mixes tasks?

Each of development replicas 1–2 has five 32-turn episodes. Replicas 3–4 remain
reserved; this code does not run them or select a confirmation winner.

| Turns | Serving checkpoint | Learning action |
|---|---|---|
| 1–8 | F: released DFlash | Context only |
| 9–16 | F | Collect current-response feedback |
| After 16 | — | U1: train on responses 9–16 |
| 17–24 | U1, fixed | Evaluate, then admit these responses as support |
| After all first-window scores | — | U2: continue U1 weights and Adam state on responses 17–24 |
| 25–32 | U2, fixed | Final evaluation; no third update |

Every episode resets draft, optimizer and conversation. Updates finish before
the next block. No training occurs inside a scored window or timing repeat.

## Response ownership is the loss mask

For response 18, the full causal prefix can contain responses 1–17. They all
receive **zero loss in response 18's examples**, including response 17 from
the same support group. Response 17 receives loss only in its own examples.
The fixed two training passes intentionally revisit those examples; they do
not turn history tokens into additional supervised positions.

Each native block carries the current response ID, input boundary, active
draft checkpoint hash, teacher identity, original positions and explicit slot
ownership. The known seed, padding and terminal/cap exclusions are masked.
Native-eligible positions must not be silently dropped or reassigned.

Teacher rows describe the block's original proposal history. A valid row after
a rejected proposal is not relabeled as if it described the corrected answer.
The existing native replay regenerates student states from current weights;
it does not reuse draft-dependent cached states from older weights.

The full forward-KL objective is `KL(P_target || Q_draft)`: mean eligible
positions within each block, mean eligible blocks within each response, then
mean the two responses in an optimizer step. It uses the complete teacher
distribution, not a top-1-label substitute. Teacher parameters remain frozen;
gradients still pass through its frozen output projection into the draft.

## Fixed learning and serving settings

- Full released DFlash draft parameters; no LoRA, CE, half-strength publication,
  replay of older support, new scheduler, or loss search.
- AdamW: learning rate 0.0003; betas (0.9, 0.999); epsilon 1e-8;
  weight decay 0.01; gradient norm clip 1; training seed 17.
- Two chronological passes; two responses per optimizer step; eight steps
  per update. Ten episodes give 20 updates, 160 optimizer steps and 320
  response presentations. U2 continues Adam counters 9–16, not 1–8 again.
- Qualified frozen Llama-3.1-8B-Instruct target and released DFlash-UltraChat
  model snapshots are checked through exact checkpoint/source hashes.
- Greedy generation; 512 output-token cap with natural EOS; block size 10
  (known seed plus nine proposals). Full input limit 16,384 and native total
  context limit 17,408 are checked independently. No history truncation.
- Qualified cold prefix-cache policy before every request; normal native KV
  reuse within a request. All methods use the same saved complete input.

## Code map

| File | Responsibility |
|---|---|
| `experiment.py` | Chronology, canonical history, fixed-window paired evaluation, retained failures |
| `training.py` | Response-specific masks, unchanged KL, persistent full-weight optimizer |
| `native.py` | Source-bound native serving/capture/publication and training bridge |
| `native_dependencies.json` | Exact hashes for the existing qualified runtime installation |
| `report.py` | Per-episode/block/repeat comparisons and explicit cost boundaries |
| `run.py` | CPU-only plan by default; explicit single-development-episode launch |
| `test_*.py` | Synthetic CPU tests; they are not model-performance experiments |

The installed native runtime remains an external dependency. It is **not
vendored or claimed to be a standalone public reproduction**. Missing or
changed dependencies cause failure instead of substitution. The compatibility
changes needed for active-checkpoint capture and learned-weight publication
are explicit in `native.py`, with source and effective-code hashes recorded.
They preserve the original files. Their integration still requires native GPU
qualification after review; earlier W8 qualification alone does not certify
this new multi-update controller.

## Dataset and CPU review commands

From the repository root, using Python 3.10 or later:

```bash
python datasets/two_update/build_streams.py --selection /path/to/selected_prompts.jsonl --output-dir /path/to/two_update_dataset
python datasets/two_update/build_streams.py --selection /path/to/selected_prompts.jsonl --output-dir /path/to/two_update_dataset --check
python -m unittest discover -s datasets/two_update -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s experiments/dflash_two_update -p 'test_*.py' -v
python experiments/dflash_two_update/run.py plan --selection /path/to/selected_prompts.jsonl --streams /path/to/two_update_dataset/streams.jsonl --out /path/to/new_plan
```

Numerical training tests use an existing PyTorch environment with CUDA disabled.
Without PyTorch, those tests are explicitly skipped; metadata/protocol tests
still run. No model downloads or package installations are performed.

## Later native execution, only after review

Prepare a private runtime config following `runtime_config.example.json`.
It must refer to the qualified workspace, model bindings, Python environment,
runtime hashes, explicitly assigned GPUs and complete peak storage budget.
Source the server's approved environment/cache helper before Python starts.
The bridge uses local snapshots; it does not download models or use SSH.

The training worker needs a separate assigned GPU from serving so the same
trainer and optimizer can remain resident. The serving GPU runs SD and grouped
target-only AR serially. This avoids AR engine restarts on every request.
Do not concurrently launch another episode on those same two GPUs.

```bash
python experiments/dflash_two_update/run.py run --selection /path/to/selected_prompts.jsonl --streams /path/to/two_update_dataset/streams.jsonl --episode R1_STABLE_CODE --runtime-config /path/to/private_runtime_config.json --out /path/to/fresh_episode --allow-gpu-execution
```

Review native mask/output/publication checks on the first admitted episode
before executing the remaining development episodes. Every attempt uses a
fresh output directory; there is no silent restart, checkpoint reset, or
automatic confirmation run. Long inputs that exceed the qualified limit are
retained as failures, not clipped or replaced.

## Evaluation and costs

Canonical serving accumulates its own model-generated answers, preserving
their original token IDs. Source assistant answers are absent. For U2, feedback
comes from U1's actual proposals and target verification, not a new frozen-draft
capture. Both methods then replay the same saved input IDs.

Each window gets three F/U timing pairs in alternating order, counterbalanced
across episodes. AR repeats are grouped afterward. Checkpoints stay fixed;
timing outputs do not enter history or training. There are 160 distinct scored
contexts, 960 SD timing executions and 480 AR reference executions. These
are repeated measurements, not 1,440 independent examples. Canonical serving
and capture-off early frozen references are recorded separately.

An SD output mismatch is written to evidence before the episode stops. It is
never silently removed or fixed by changing verification. AR mismatches remain
visible and disallow an identical-output acceleration interpretation for that
reference. Negative effects and incomplete episodes remain in the report.

Record accepted draft tokens, trace-derived verification progress, calls per
1,000 output tokens, complete request latency/throughput, context/output
lengths, supervised-position counts and both checkpoints. Complete request
latency includes prefill. Total output tokens per verification includes the
initial target token and is named separately from verification-only progress.

Native receipts separate capture/replay, training, export/activation, cache
maintenance and worker loading. Canonical request timing already contains
capture; nested timers are not added again. The observed update-call wait is
included in the accounted episode total, including any lazy initialization
unless separately identified. Initial loading, diagnostic timing repetitions,
AR switching and benchmark-only weight switches are also recorded separately.
The report distinguishes actual experiment wall time from reconstructed
serving-plus-update episode cost; it does not call either an asynchronous
production measurement.

Report each scenario and both windows separately. A U1-versus-U2 raw difference
on different windows cannot isolate U2's incremental effect. Lower training
loss alone, or fewer calls without lower elapsed time, does not establish
acceleration. No efficacy threshold or confirmation activation is invented.

Private outputs: `PLAN.json`, source/config manifests, `EVENTS.jsonl`,
per-turn canonical/scored results, native traces and feedback, U1/U2 weights
and optimizer state, `RESULT.json`, `REPORT.json`. No such artifacts are
included in the public code commit.
