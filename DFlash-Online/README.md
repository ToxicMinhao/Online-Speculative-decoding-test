# DFlash-Online

DFlash-Online is the working implementation for our DFlash online-adaptation experiments. It combines the released DFlash inference implementation, SpecForge training components, a shared feedback/update loop, and separate runners for bounded comparisons.

The current publication preserves the experimental implementation. It does not introduce a new algorithm, retrain checkpoints or replace historical results with measurements from newer code. The earlier workspace README is retained as historical documentation.

## Where to start

| Path | Purpose |
|---|---|
| [`online_feedback/`](online_feedback/) | Shared serving, feedback preparation, objectives, optimizer/state handling and online updates |
| [`experiments/`](experiments/) | Experiment-specific schedules, qualification, comparisons and reporting |
| [`tests/`](tests/) | CPU correctness tests for the shared loop and experimental components |
| [`configs/`](configs/) | Configuration templates and historical recipes; each report identifies its actual run configuration |
| [`inference/`](inference/) | Pinned DFlash source, with the documented local compatibility changes |
| [`training/`](training/) | Pinned SpecForge source used by the online trainer |
| [`reports/`](reports/) | Results, limitations and links to exact source provenance |
| [`THIRD_PARTY.md`](THIRD_PARTY.md) | Upstream source identities, licenses and attribution |
| [`DEPENDENCIES.md`](DEPENDENCIES.md) | Separate training/serving environments and artifact requirements |

Start with [`online_feedback/controller.py`](online_feedback/controller.py) for the general sequence: serve a request, save its actual response, seal a feedback event, prepare teacher features, update the draft, then publish the checkpoint. Evaluation does not append feedback or advance training. Individual experiments can intentionally reuse a retained feedback pool instead of collecting a new one; their runners make that choice explicit.

## Following one update in the code

1. [`buffer.py`](online_feedback/buffer.py) stores the completed serving records and selects a bounded event.
2. [`sglang_backend.py`](online_feedback/sglang_backend.py), [`sglang_worker.py`](online_feedback/sglang_worker.py) and [`sglang_hooks.py`](online_feedback/sglang_hooks.py) connect the separate native-serving process to inference and feedback preparation.
3. [`training.py`](online_feedback/training.py) uses SpecForge for model construction, sampled blocks, separately normalized response losses, accumulation, AdamW, checkpointing and exact state restoration.
4. [`objective.py`](online_feedback/objective.py) and [`first_error.py`](online_feedback/first_error.py) implement the training losses and K's current-prefix-plus-first-error selection.
5. [`presentation_schedule.py`](online_feedback/presentation_schedule.py) and [`rng.py`](online_feedback/rng.py) support the recorded order, sampled anchors and RNG handling.
6. The selected module under [`experiments/`](experiments/) specifies what is trained, what is compared and which correctness checks must pass.

For the latest correction-removal control, read [`experiments/krc_off/campaign.py`](experiments/krc_off/campaign.py), then [`online_feedback/krc_off.py`](online_feedback/krc_off.py). It restores the complete KRC1 state before disabling correction. Ordinary K then continues for one event. The adapter records existing work; it does not add reference forwards or a replacement optimizer.

## Experimental components

| Name | Change relative to its stated control | Main implementation |
|---|---|---|
| O | Ordinary response-only forward KL | `objective.py`, `training.py` |
| U / P | Uniformly weighted / reference-prefix-weighted ranking auxiliary loss | `ranking.py`, `objective.py` |
| R | Reference-margin-preservation version of P's auxiliary loss | `ranking.py`, `objective.py` |
| C / K / A | Canonical CE / first-error-selected KL / first-error-selected CE | `first_error.py`, `objective.py` |
| G | Prefix-preservation projection of K's native optimizer displacement | `prefix_guard.py` |
| KC | One post-K correction protecting the same learning pair | `prefix_correction.py`, `prefix_correction_event.py` |
| KRC | The same correction protecting a different buffered pair | `reference_schedule.py`, `reference_correction_event.py` |
| KR | Ordinary K updates on both the learning pair and its reference pair | `reference_replay_event.py` |
| KRC2_OFF | Ordinary K continuation from KRC1 with event-2 correction disabled | `krc_continuation.py`, `krc_off.py` |

These names describe implemented experiments, not a set of established improvements. The reports retain harmful and null outcomes. For example, the latest KRC2_OFF comparison is mixed: keeping event-2 correction slightly worsened development call counts and slightly improved the additional panel. That does not establish a universal correction schedule.

## Settings and reproducibility

The recent controlled studies use Qwen3-8B with released DFlash-b16, BF16 training/serving, greedy decoding, block size 16, up to 2,048 output tokens or EOS, serving context 4,096, and concurrency 1. The recent K-family training recipe uses constant learning rate 3e-4, AdamW weight decay 0.01, seed 17, response microbatch 1 with accumulation 2, Flex attention, and up to 512 sampled anchors per response. Its prepared training-sequence limit is 2,272. These are recent-study settings, not claims about every historical run.

A usual one-event comparison trains on 128 retained responses for two passes: 256 response presentations and 128 Adam steps. KR and continuation studies have different explicitly recorded budgets. A reference pass or direct correction is not an additional Adam step, but its exposure and cost are counted separately.

The per-run sealed configuration and execution records are authoritative. Generic recipes are not a substitute for them. Training and native serving use different pinned environments; see [DEPENDENCIES.md](DEPENDENCIES.md).

Each report records the published source commit and its historical-source mapping. Publication is retrospective: a new Git commit identifies preserved source bytes, not a claim that the old run executed from that commit at the time. Older variants remain available through the provenance archive; do not reproduce an old result by assuming today's file has the same implementation.

Model weights, recovery states, teacher-feature tensors, private responses and complete runtime traces remain external artifacts. Their omission from Git is not deletion. The source and compact reports alone do not recreate a retained checkpoint or provide private feedback tensors.

No command in this README launches training or generation. Inspect the relevant report, its pinned inputs and its bounded runner before preparing a new execution.
