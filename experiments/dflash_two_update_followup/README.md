# DFlash two-update follow-up comparison

**The original ten-session campaign is complete. The retained-author replay
extension is awaiting review before GPU execution.** See
[AUTHOR_REPLAY.md](AUTHOR_REPLAY.md) for that bounded extension and
[VALIDATION.md](VALIDATION.md) for the original native qualification.

Two questions are tested with the existing ten R1/R2 development sessions:

1. Does the existing second update improve on the first update when both see
   exactly the same final eight request inputs?
2. Does AdaFlash's committed-response OSD training path outperform the existing
   captured-verification updater under the same two-update session protocol?

R3/R4 remain reserved. This adds no loss, adaptive-length head, model architecture,
or training budget. The full-weight forward-KL controls and fixed target remain.

## Comparison matrix

| Saved original inputs | Methods, each repeated three times |
| --- | --- |
| Turns 17–24 | Frozen (`F`), existing U1 (`C1`), author-path U1 (`A1`) |
| Turns 25–32 | Frozen (`F`), existing U1 (`C1`), existing U2 (`C2`), author-path U2 (`A2`) |

`both` runs this shared matrix once: 168 scores per session, 1,680 overall,
covering 160 distinct saved request contexts. Repeats are timing measurements,
not independent examples. `checkpoints-only` can independently evaluate the
retained F/C1/C2 weights on final inputs with no training (72 scores per session).
The original campaign deliberately omitted author-path U1 on final inputs.
The new `author-replay` mode closes this gap with retained F/A1/A2 checkpoints,
without retraining. It executes 72 fresh scores per episode, 720 overall.
The original `both`, `author-path` and `checkpoints-only` modes retain their
original comparison matrices.

All score calls disable feature capture, training and publication within the
inference timer. They use the identical saved input IDs, unchanged native
verification, cold prefix caches and rotating method order. Publication/cache
maintenance are measured outside request latency. Outputs must match the saved
target-verified answer; a discrepancy is recorded before the comparison stops.
No mismatching requests are silently removed or verification changed.

## New arm's own online trajectory

The new arm starts each session from released weights and a fresh optimizer.
It reads the fixed user-message sequence and appends its own emitted, verified
output IDs. It serves turns 1–16 frozen, then trains U1 on responses 9–16.
It serves 17–24 with U1 fixed and completes early paired evaluation **before**
using its own responses 17–24 for U2. Weights and optimizer persist into U2;
U2 serves 25–32 without further training. Benchmark responses never enter support.

Each training example contains the full causal prefix and one current response.
**Only that current response receives loss.** Earlier answers, including earlier
answers in the same eight-response group, remain masked whenever they recur as
context. No later turn is present in an earlier response's example. Comparisons
between new and original input/output token sequences are retained per turn.
If feedback differs, the result compares complete online procedures, not only
the sampler or position weighting.

## Pinned author mechanism and adaptations

The arm is named **“AdaFlash OSD training path under our session protocol.”**
It is not an exact reproduction of published OSD or AdaFlash results.
[AUTHOR_PATH.md](AUTHOR_PATH.md) explains every ported operation and the
[source manifest](author_sources.json) binds AdaFlash commit
`507b7aba5fbccaa99a400f161e4f2794ba6a388c`.

Preserved: committed-token anchors, selected-layer teacher context, shifted
normalized teacher targets, 512 maximum random anchors, gamma-7 position
weights, weighted forward KL divided by total weight plus 1e-6, temperature 1,
and zero hard-label mixture. Sampled blocks share one batched backbone forward
per response. Only vocabulary-head/loss projections are chunked and recomputed
during backward to bound memory; the backbone is not rerun for each anchor.

Matched session controls: released Llama-3.1-8B-Instruct/DFlash-UltraChat pair,
seed 17, AdamW learning rate 3e-4, betas (0.9, 0.999), epsilon 1e-8, weight decay
0.01, norm clipping 1; two chronological passes, two responses per step, eight
steps per update. No optimizer reset within a session. Block size 10; 256 output
tokens or EOS; input limit 31,744 and total context limit 32,768; 8,192-token
serving prefill chunks. No truncation or silent author-default 2,048 limit.

Nine masked right-tail slots preserve the author's padded-tail anchor support
without padding teacher replay to the full maximum. A response shorter than two
tokens cannot satisfy the author's anchor sampler; it is an explicit recorded
failure, never silently discarded. Actual selected anchors, teacher positions,
weights, response boundaries and supervised exposure are retained.

## Measurement and retained evidence

Report complete inference-only request time (including prefill), throughput,
verification calls per 1,000 output tokens, accepted draft tokens per call and
committed/output progress per call. State that these are **performance with an
available checkpoint**. Raw losses have different normalizations and are not
compared across recipes.

Token-only support serialization is measured after serving. The new arm obtains
teacher features by full causal replay during training; no proposed-block tensor
capture runs during canonical serving or benchmarks. Original capture/preparation
costs remain historical measurements, explicitly separate from fresh paired
inference timing. Teacher replay is included in the new training-loop timer;
export/state saving are separate sub-costs inside the synchronous update call.
Activation includes checkpoint validation and native publication; its internal
timer is nested, not added again. The reconstructed session charges canonical
requests, support preparation, update waits, activation and cache maintenance
once. Benchmark replays and one-time loading remain separate from that session;
actual experiment wall time includes them. Missing timings are unmeasured, not
zero. No complete frozen-session speedup is invented from the partial score windows.

Keep original and new model checkpoints, exact tokens, configuration/source
hashes, anchors, raw timings, output checks and failures. Preserve optimizer
recovery states for active or failed sessions. Under the approved storage policy,
a separate operational controller may retire only a fully completed session's
optimizer files after validation and a verified compact evidence archive; record
the exact removed paths and loss of optimizer continuation. This scientific
entry point performs no cleanup and never overwrites a prior attempt.

## Review and later execution

CPU tests (torch is needed for numerical tests):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s experiments/dflash_two_update_followup -p 'test_*.py' -v
```

The optional pinned-author differential tests use `ADAFLASH_SOURCE_ROOT` pointing
to the exact source tree; missing source/torch produces a reported skip, never a
claim that the differential check passed. A metadata-only plan launches nothing:

```bash
python experiments/dflash_two_update_followup/run.py plan \
  --selection /external/data/selected_prompts.jsonl \
  --streams /external/data/streams.jsonl --out /external/review-plan
```

After review and native qualification, one explicit episode can run:

```bash
python experiments/dflash_two_update_followup/run.py run --comparison both \
  --selection /external/data/selected_prompts.jsonl \
  --streams /external/data/streams.jsonl --episode R1_STABLE_CODE \
  --reference /external/original/R1_STABLE_CODE/RESULT.json.gz \
  --reference-sha256 ORIGINAL_FILE_SHA256 \
  --runtime-config /external/runtime.json --out /external/fresh-run \
  --allow-gpu-execution
```

Use [runtime_config.example.json](runtime_config.example.json) with the qualified
installation; it is not a portable standalone environment. The bridge verifies
the existing runtime pins in `../dflash_two_update/native_dependencies.json`,
the author sources and both code trees. The native GPU qualification for this
teacher/trainer integration has passed; scientific efficacy remains untested.
The inherited storage gate is conservative and includes a reserve and checkpoint
overlap. Review available space and active jobs before any server session.
