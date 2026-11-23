# Graduate

A readable DFlash implementation in `/home/minhao2/Graduate` on Fenrir.
Our learning method is **K+ with bounded correct-prefix allocation (C)**.
The active A800 campaign and its SGLang runtime are separate and unchanged.

## Step 5A: collect live feedback

Open this folder in Cursor/VS Code. The single F5 entry is
**Graduate: collect live feedback (two requests)**. Start with
`config = load_config(arguments.config)` in [main](src/graduate/__main__.py).
F5 and the terminal run the same application, through the same entry point:

```bash
PYTHONPATH=src \
/home/minhao2/.venvs/dflash-specforge/bin/python -m graduate collect \
  --config configs/fenrir.toml --debug-responses 2
```

The two prompts in [requests.json](configs/requests.json) are local walkthrough
inputs, not a research evaluation panel. They use the actual released models,
BF16, greedy decoding, block 16 and a 2048-token-or-EOS cap with context 4,096.
`justMyCode: true` remains enabled. There are no path-input boxes, generation warmups, optimizer construction or
automatic training. SGLang owns its normal scheduler/model worker processes.
Stop an existing GPU training/debug session before starting collection: the
full target and draft alone require approximately 17.2 GiB of GPU memory.

Read the new flow in this order:

1. [live.py](src/graduate/live.py), `collect_feedback`: read request roles,
   render/tokenize each prompt once, load the model pair, collect and save.
2. [generation.py](src/graduate/generation.py): start the native SGLang engine,
   flush request caches, generate exact token IDs and retrieve the capture.
   SGLang owns DFlash drafting, verification, commits and cache advancement.
3. [native observation hooks](src/graduate/sglang_capture.py) and
   [collection.py](src/graduate/collection.py), `FeedbackCapture`: retain only
   canonical input rows. Rejected draft-tail features never become feedback.
4. `final_token`: after SGLang releases the finished request cache, one explicit
   full-context target forward supplies the final emitted token's missing row.
   Only that row is retained; these logits never replace native response labels.
   Its input rows and cost are recorded separately. No anchors are sampled here.
5. `save`, then [load_feedback](src/graduate/batches.py): write and validate the
   same exact-token JSON and full-response tensor format used by Step 4.

Useful breakpoints in the current F5 process: `config = load_config(...)`,
`with load_generator(...)`, `generated = generator.generate(...)`, and
`load_feedback(record, features_path, layout)`. Step into the last call to inspect
the exact saved tensors. Native capture hooks execute in SGLang workers; this
profile does not attach to subprocesses or step through author kernels.

## Output and data roles

Each launch writes a new directory under `/home/minhao2/Graduate-data/collection`:
rendered inputs/configuration, raw generations/traces, response records, prepared
features and collection costs. The manifest configured under `[feedback]` points
to this launch's trainable responses. A run-local copy preserves its history.
Previous response files are never overwritten. No artifact lives in the source tree.

Requests have `role: feedback` or `role: evaluation`. Evaluation requests produce
outputs/traces only: no feature capture, response record, buffer import or training
manifest entry. Identical normalized text cannot appear in both roles. Separate
IDs do not bypass that check. Dataset answers are not accepted as labels.

An answer with fewer than two tokens is retained with `no_training_anchors`,
excluded from the training manifest and never replaced. Feedback contains:

- Exact prompt and final target-verified response IDs, without retokenization.
- Unshifted selected target layers and final normalized target hidden states.
- A response-only loss mask, including EOS; prompt positions have zero weight.
- Native label/layer binding and an honest runtime identifier.

The target row at `j-1` supplies the teacher distribution for canonical token `j`.
The final feature row completes the existing packet format. Anchor sampling and
current-model masks still belong to the later training forward, not the trace.

## Continue using Step 4 training

After collection, the existing command reads its manifest:

```bash
PYTHONPATH=src /home/minhao2/.venvs/dflash-specforge/bin/python -m graduate train \
  --config configs/fenrir.toml --debug-responses 2
```

Two responses over two shuffled passes give four presentations and two AdamW
steps. Remove `--debug-responses` after collecting 128 distinct trainable responses
for the full 256-presentation/128-step event. Training validates/imports records into
the buffer, samples up to 512 anchors per response presentation, and averages two
separately normalized response losses before clipping and updating.

The reviewed [model](src/graduate/model.py), [objective](src/graduate/objective.py)
and [training loop](src/graduate/training.py) are unchanged. K+ redistribution and
C prefix allocation still pool all eight-block projection chunks before response
normalization. BF16 gradients/Adam moments, constant LR 3e-4, weight decay 0.01,
clipping 1, decay 7, accumulation 2 and seed 17 are unchanged.

Collection and training remain separate commands for this review step. There is
no continuous online event, export/activation or recovery claim yet. Those are 5B.
The existing Step 4 packets remain at `/home/minhao2/Graduate-data/step4-native`.

## Runtime and source choices

The user selected **SGLang/FA3 for both local collection and A800 serving**.
Local collection uses the installed SGLang 0.5.18 DFlashWorkerV2, with FA3 for target
and draft, BF16, synchronous scheduling, concurrency1 and CUDA graphs. It is
qualified separately from A800's older pinned runtime; no cross-machine latency
or bitwise equality claim is made. The local memory fraction is explicitly 0.88
because the inherited A800 0.65 allocation cannot hold both models on a 24 GB card.
Prefill buffers stop at 4,096 and graph batches at 1, matching the supported context
and concurrency instead of reserving unused capacity.

[Official SGLang](https://github.com/sgl-project/sglang) owns model loading,
DFlash kernels and serving. Graduate installs only process-local observation/RPC
wrappers, leaving installed author files untouched. Graph-safe feature capture
retains normalized target hidden states separately from auxiliary DFlash features.
The capture design references our retained DFlash-Online native hooks and
SpecForge's author-supported capture patch, without importing historical runners.

Training still uses [SpecForge](https://github.com/sgl-project/SpecForge), commit
`1839aae30003e3573d00979eb1edd57347b3578c`, installed from
`/home/minhao2/DFlash-Research/training`. It owns DFlash blocks, anchor sampling and
FlexAttention. The shared environment is Torch 2.13.0, Transformers 5.12.1,
safetensors 0.8.0 and NumPy 2.3.5. Exact target/draft snapshot paths are in TOML.
Graduate imports no historical experiment runners or baseline-mode registry.

Capturing existing target features preserves the native conditioning. The
remaining full-context terminal completion is an explicit preparation cost, not
an inference speedup or a hidden zero-cost operation. Per-round
CPU copies are part of feedback collection cost. Evaluation disables those copies
and the final feature forward. Local capture timings include its overhead.

## Validation and remaining work

```bash
PYTHONPATH=src \
/home/minhao2/.venvs/dflash-specforge/bin/python -m unittest discover -s tests -v
```

Tests cover the existing objective/optimizer, canonical feature alignment,
EOS/caps/rejection, cold caches, evaluation exclusion and serialization.

**105 CPU tests pass.** The real two-request SGLang/FA3 CLI and installed-debugger
runs also pass: 60/127 output tokens, 13/15 verification calls, with exact matching
tokens, termination, traces and saved feature tensors. Capture-disabled executions
reproduce both trajectories. Independent BF16 head projection of all 187 saved
teacher positions reproduces every canonical token. Actual worker metadata confirms
FA3 for both models and target graph execution.

Peak worker allocation was 18.95 GiB. Initialization took 75.13 seconds on the first native
run and 49.31 seconds in the corrected debugger run; these are local startup observations,
not latency benchmark results. The complete qualification used 8 generations over
these two requests, 6 full-context terminal completions and zero optimizer steps.
A tokenizer serialization failure before model loading and an initial DAP-harness
failure after generation are preserved separately. The corrected debugger exits 0.

This is bounded local qualification, not a 128-response or maximum-context campaign.
Cursor GUI/editor-focus behavior remains untested; the installed debugger's source
breakpoints, variables and step-over were exercised. GPU execution commit is
`b3c45e41efb632a16ccb418beecb2d9956e91e11` (initial CLI `080eddc`).
Validation records are outside Git in `/home/minhao2/Graduate-data/step5a-validation`.

| Step | Scope |
|---|---|
| 1–4 — reviewed | Shared CLI, configuration, feedback storage, K+/C training |
| 5A — current | Live generation and compatible feedback packets |
| 5B/5C — later | Continuous updates, recovery/publication and evaluation campaigns |
| 6 | Final cleanup and reading guide |
| 7 | Profiling and measured optimizations |

Stop for user review before 5B. No A800 experiment or scientific setting changes
are part of this refactor.
