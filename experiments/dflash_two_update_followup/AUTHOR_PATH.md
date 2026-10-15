# AdaFlash OSD training path under our session protocol

This module ports the OSD training mechanics from AdaFlash commit
`507b7aba5fbccaa99a400f161e4f2794ba6a388c`. It is an adaptation to the released
Llama-3.1-8B/DFlash-UltraChat pair and the existing two-update session protocol,
not an exact reproduction of the paper's Qwen experiment. CPU differential tests
qualify the port's numerics; they do not qualify GPU performance or scientific
benefit. This change does not authorize an experiment.

## Source trace

All paths below are relative to the pinned AdaFlash repository. The requested
historical `scripts/workers/training_worker_sft.py` path is absent at this pin:
`bin/training_worker_sft.py` is its entry point and delegates to
`pipeline/training/sft.py`.

| Source | Preserved behavior |
| --- | --- |
| `recipes/qwen3_osd.sh` | Teacher temperature 1, alpha 0: forward KL only; no hard-label mixture |
| `scripts/pipeline/run_async_pipeline_osd.sh` | Batch 1, two passes, two microbatches per optimizer step, learning rate 3e-4; context limit adapted explicitly below |
| `pipeline/training/args_common.py` | 512 maximum anchors, loss decay gamma 7 |
| `pipeline/training/utils.py::_encode_text_record` | Supervised reply ownership; padded-tail anchor availability |
| `specforge/core/dflash.py::_sample_anchor_positions` | Random-sort sample without replacement, sorted selected anchors, `min(512, valid_count - 1)` |
| `specforge/core/dflash.py::forward` | Real committed anchor token plus nine masks; strictly preceding target features; bidirectional own-block attention; no cross-block attention; teacher distribution from normalized `h[position - 1]`; anchor excluded; gamma weights; sum divided by sum of weights plus 1e-6 |
| `pipeline/training/setup.py::build_sft_optimizer` and `utils.py::_train_on_buffer` | AdamW defaults, loss divided by two before each backward, gradient norm clipped to 1 before each step |

Pinned source hashes:

- `specforge/core/dflash.py`: `d6a16d8ea8f9401e1e523f227945f228bc40d5feb021baff5fca620386416024`
- `pipeline/training/utils.py`: `83044b64a236426dd4fc4972ef8fe16d9487b3673fa8972c0e36c062420496fc`
- `pipeline/training/args_common.py`: `21d048f06f1bb5f6b52f564cdd5bb82a9ce56e6346d8d846fe85c7c5c1194ee8`
- `recipes/qwen3_osd.sh`: `43a8e5773945014058ed9f4508ac77009517a6fec80d9307a14fb8a848f89296`
- `scripts/pipeline/run_async_pipeline_osd.sh`: `07465f60cc48825098e0b803b15c433c55172480de765a81e93a89dd0fc72164`

## Explicit adaptations

The native bridge supplies an author-compatible DFlash backbone, frozen target
embedding/head and a frozen teacher callback. The callback receives exact token
IDs for the full causal input plus the current committed response. It returns
selected-layer features and normalized final-layer features, each covering every
real token. Earlier assistant answers are conditioning only, including earlier
answers in the same eight-response support group. The mask is rebuilt separately
for each response from its exact input/output boundary; no tokenizer boundary
heuristic, re-rendering or historical group-wide loss is used. The pinned SGLang
teacher calls one full-sequence extend directly; it does not inherit the serving
scheduler's 8,192-token prefill chunks. The bridge retains that author replay
behavior and records its complete cost. Native GPU peak memory and compatibility
still require qualification before any scientific run.

The author worker tokenizes and pads to a fixed maximum length, 2,048 by launcher
default, and can truncate. This port never truncates. It uses the established
31,744 input / 32,768 total envelope and appends nine masked tail positions after
the real sequence. Those nine positions preserve the response anchor support of
an author example with sufficient right padding, including the final response
token; they are not supervised. Padding features are zero placeholders that no
real anchor can attend to, and no synthetic padding is sent through the teacher.
Absolute block positions remain within the qualified envelope. Dropping that
padding would incorrectly exclude the final nine possible response anchors.
The padded length changes the global RNG draw count relative to fixed 2,048 or
32,768 padding; it preserves the sampling distribution, not the original seed's
identical sampled subset across those different shapes.

The sampler retains the author's `valid_count - 1` rule: with response length
`L`, it samples `L - 1` anchors under the 256-token cap. Length zero or one cannot
satisfy that rule. Preflight fails the complete event with the response identity
and exact count before teacher work or optimizer mutation; it never drops a row
or silently substitutes a different sampler. For a two-token response, the
sampler can select only its last token, yielding no supervised non-anchor slot.
That presentation is explicitly recorded with zero loss/gradient and still
counts toward accumulation, matching the author wrapper. The export/report must
retain `zero_supervision_presentations` and the sampled positions.

A device-local generator is seeded 17 once for each episode and advances through
both passes and updates; serving draws cannot perturb it. Each update consumes
eight responses in chronological order twice, two responses per AdamW step,
exactly eight steps. Weights and Adam moments continue into U2, with a required
actual exported-weight hash binding between updates. Every new episode must
construct a fresh trainer from the released weights. No third update is admitted.

All sampled blocks use one backbone forward per response. Only frozen vocabulary
projection and KL rows are chunked (default 64 rows); nonreentrant checkpointing
recomputes those heads during backward to bound vocabulary activation memory.
It never recomputes the backbone per anchor or head chunk. Zero-weight rows are
omitted from projection, leaving the author numerator and denominator unchanged.
The native head dtype is retained, including its softmax/KL arithmetic. The
alpha-zero hard CE branch, which the author evaluates then multiplies by zero,
is omitted. No reverse KL or adaptive length head is used.

## API, evidence and accounting

`PersistentAuthorOSDTrainer(draft, embed_tokens, lm_head, teacher_features, *,
episode_id, target_id, target_identity_sha256, released_checkpoint_sha256,
mask_token_id, attention_backend="flex_attention", projection_chunk=64,
synchronize=None, check_frozen=None)` accepts one complete response per
microbatch. `teacher_features(input_ids)` returns the two detached feature
tensors. The embedding/head must already be frozen and in eval mode;
`check_frozen` additionally verifies the full external teacher. `sdpa` is used in
CPU tests; the production bridge pins the author-compatible attention backend.

Each support record supplies `episode_id`, `record_id`, `turn`, `input_ids`,
`input_token_count`, `output_ids`, `output_cap`, `target_id`,
`target_identity_sha256` and `capture_checkpoint_sha256`. The last field identifies
the serving draft that generated the committed answer. `train_event(event,
responses)` returns compact audit metadata, actual anchors, teacher positions,
slot weights, response boundaries, source hashes, unique and repeated supervised
exposure, and costs. `bind_checkpoint(event, hash)` binds actual exported serving
weights. `save_checkpoint_state(path, include_draft=False)` durably stores the
persistent Adam state and anchor RNG for explicit recovery; no automatic resume,
retry or reset is implemented.

`costs.training_s` is the complete training loop wall time and already includes
`teacher_preparation_s`. `draft_training_and_optimizer_s` is the residual;
those latter two form the nonoverlapping split. Do not add teacher preparation
to `training_s`. `training_event_wall_s` additionally contains preflight/audit and
small event bookkeeping. Per-response forward/loss and backward timers synchronize
the native device, while vocabulary recomputation is charged to backward. Capture,
export/transfer and activation are measured by the native bridge. Raw losses have
this recipe's denominator and must not be compared directly to the old updater's
mean-slot/mean-block/mean-response losses.

Run CPU checks with a CPU-capable torch installation and no CUDA devices exposed:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest experiments.dflash_two_update_followup.test_author_training -v
```

Set `ADAFLASH_SOURCE_ROOT` to the pinned author checkout outside this workspace to
run the source differential test elsewhere. The test verifies the core SHA-256,
imports the actual unchanged file, and stubs only its annotation-only draft-class
import. No author model implementation is substituted for the tested wrapper.
Without that source or torch, the corresponding tests report skips; such a run
alone is not a completed differential qualification.

## Upstream attribution and license

The sampler, attention mask and loss formulas are adapted from AdaFlash's
vendored SpecForge training wrapper, whose file identifies the following license.

MIT License

Copyright (c) 2025 sgl-project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
