# Agreement-filtered OSD: bounded development pilot

Implementation for review. **No GPU qualification or pilot result is claimed.**
The question is whether selecting components of an ordinary OSD update improves
subsequent drafting beyond lowering its learning rate or uniformly shrinking it.
This package implements the October 19 handoff; it does not resume correction
bank v0.2 or introduce a new architecture.

## Declared comparison

Use requests 1–3 of the six already approved conversation streams, in manifest
order (four WildChat and two synthetic PersonaMem-v2). Each learning arm starts
from released DFlash, fresh AdamW and seed 17 for each stream.

| Arm | Update |
|---|---|
| F | Frozen DFlash |
| H | Qualified AdaFlash OSD training path under our session protocol, LR `3e-4` |
| L | Same ordinary update, LR `1e-4` |
| G | H's ordinary displacement filtered by agreement with two raw group gradients |
| N | Its own ordinary displacement uniformly scaled to its own hypothetical filtered norm |

H is the **primary OSD baseline**. G beating L and N while trailing H is a
trade-off, not an overall improvement over OSD. These are existing development
cases, not fresh confirmation. Source fitting or merely matching F is insufficient.

Generate response 1, update, generate response 2, update, generate response 3,
stop. Each arm retains **its own** generated history and optimizer state. Only
the newest answer supplies loss; earlier messages remain causal context. Dataset
assistant answers are unused. Keep the pinned model pair/runtime, full-weight
trainable inventory, forward KL, original weighting/normalization, normal per-pass
anchor sampling, two passes and two optimizer steps per event. The qualified
training path originates at AdaFlash commit
`507b7aba5fbccaa99a400f161e4f2794ba6a388c`; H/L invoke the existing trainer.

Main counts: **72 own-history generations, 48 update events, 96 optimizer steps,
12 common benchmark contexts and 60 scored replays.** F is scored once per
context inside H's serving worker. Arm executions are serial; single-shot timing
is diagnostic and does not support deployment speedup claims.

## Filter and numerical contract

Split the ordered eligible output positions at `floor(n/2)`; an odd extra position
belongs to B. The first answer token is an anchor, not a supervised prediction in
the qualified sampler, so eligible positions run from `response_start + 1` to
`response_end - 1`. Position assignment stays fixed across both passes and every
overlapping occurrence. Anchors are sampled normally each pass. An empty sampled
group or nonfinite value is an explicit correctness failure, never a resampling
or ordinary-update fallback.

Both partial losses use the **original full weighted denominator** and the same
stochastic forward at the same weights. Their raw gradients sum to the ordinary
gradient mathematically. Clip the aggregate once, then call ordinary AdamW once,
including momentum, bias correction and decay. Let `d = theta_proposed - theta`.
G keeps a coordinate exactly when both raw gradients have a nonpositive product
with `d`. The implementation uses comparisons of signs and zeros, **never tiny
products**, to make that decision. Adam moments/counters are never filtered or
reset, including a zero applied write.

N computes `alpha = ||filtered_d|| / ||d||` at **N's own state**, over all trainable
parameters jointly; zero denominator gives zero alpha. It applies `alpha*d` and
never consumes G's weights, mask, gradient, norm or future outcome.

The native recipe has BF16 parameters and Adam moments, without FP32 master
weights. Raw gradient buffers are FP32 copies of native autograd gradients;
promotion cannot recover precision lost in BF16 differentiation. G/N sum these
buffers in FP32 and cast to the native gradient dtype before the unchanged clip.
Signed displacements are FP32 differences of native pre/post weights. Norms and
directional-derivative diagnostics use chunked FP64 reductions. G retains the
exact proposed native value or the exact old value. N's scaled result rounds back
to BF16: report its ideal norm **and actual stored displacement norm**. Do not
claim exact post-rounding direction or norm equality. **N qualifies only when
`abs(actual_norm - intended_norm) / intended_norm <= 0.01` at every optimizer
step.** A zero intended norm requires an exactly zero actual norm; there is no
absolute tolerance that can admit an erased tiny update. This 1% numerical
admission limit is fixed before GPU outcomes and cannot be tuned to performance.
`controls.N_NORM_POLICY` pins it. A failed step preserves the update/checkpoint,
stops qualification or the run, and prevents the clean main campaign from
launching after failed qualification. No alternative precision/update is applied.

Reporting recomputes N's validity from both steps of every ancestral event and
binds that evidence to the evaluated N checkpoint. A later good event cannot
repair an earlier control failure. Missing, duplicate, stale or mismatched evidence
also blocks selective-filter attribution. Native comparisons remain visible as
descriptive measurements; they cannot establish an advantage beyond taking a
smaller step when N fails. Pooled tables disclose qualified-control denominators.

Full constants live in `core.NUMERICS`,
`training.GROUP_PROTOCOL` and `training.EQUIVALENCE_TOLERANCES`.

Split BF16 backwards can differ numerically from a full backward. Qualification
checks the same-forward gradient sum, then compares a complete ordinary two-step
event with a filter-disabled event from the exact same state and feedback.
Weights must be within one native ULP plus `1e-7`, **and** total displacement-error
norm must be at most 10% of the reference displacement norm. A missing nonzero
reference update fails. Moment tolerances are separately pinned; counters,
anchors, exposure, sampler and stochastic RNG must match exactly. Report all
observed errors and zero/nonzero coordinate transitions. These are finite-precision
acceptance limits, not claims of exact native equality. Failure blocks the pilot;
no silent tolerance adjustment is authorized.

Native compatibility fixes the Torch AOTAutograd setting
`torch._functorch.config.donated_buffer=False` before model setup in **every**
H/L/G/N trainer worker. Compiled single-use donated buffers cannot support the
retained graph needed by the two group backwards and qualification's full-loss
backward. The disabled-filter reference trial can otherwise compile a backward
that rejects the subsequent grouped trial. Keeping saved buffers reusable changes
compiler memory reuse, not the loss, BF16 precision, optimizer or tolerances.
Its memory/runtime costs remain measured. `COMPILER_RUNTIME.json`,
`AUTHOR_RUNTIME.json`, update accounting and recovery state record the effective
setting and Torch version; configuration drift fails before another update.

## Qualification and causal evaluation

Before the campaign, deterministic fixtures check numerical corner cases. Run
requests 1–3 of `wildchat_00` for all four learning arms: 12 own-history generations,
8 update events/16 steps and 10 scored replays. G's disabled-filter check adds
four **temporary qualification-only** optimizer steps; exact restoration leaves
zero persistent steps. These costs/counts remain outside the main pilot.
Check masks, shared denominators, gradient sums, moments, exact input/output token
IDs and the existing weight-publication/cache receipts. Performance is not a gate.
On success, restart all six streams from new workers, fresh optimizer/RNG and
separate output directories. No qualification learning enters the campaign.

Common benchmark inputs are the **existing immutable canonical turn-2/3 inputs**
from the completed per-turn pilot. `references.py` binds the complete historical
RESULT files and preserves exact user whitespace, message IDs, token arrays and
source order. The reference bundle is prepared without a tokenizer/model and its
file hash is required at execution. Each arm uses D1 for request 2 and D2 for
request 3, before learning the corresponding response. Replays have capture and
learning disabled and cannot supply training feedback or alter conversation
history. Preserve output mismatches; affected comparisons cannot claim gains.
Every scored row carries measurement-validation status and errors. Reports rerun
the same qualified validator rather than trusting saved flags: impossible counts,
invalid costs, malformed outputs and capture-enabled receipts cannot enter gain
pools. Rejected nonfinite values are retained as explicit lossless JSON markers,
so strict serialization cannot discard their failure evidence.

Serving remains greedy, block size 10, **1,024 output tokens or EOS**, full history,
and the qualified cold-request/weight-switch cache policy. No silent truncation.

## Run interfaces

From repository root, metadata-only preparation:

```bash
python -m experiments.dflash_osd_agreement_filter.run prepare-references \
  --dataset /path/to/conversation_pilot \
  --historical-results-root /path/to/completed/per_turn/attempt_001 \
  --out /path/to/new/reference_package
python -m experiments.dflash_osd_agreement_filter.run plan \
  --dataset /path/to/conversation_pilot \
  --references /path/to/new/reference_package/REFERENCES.json \
  --out /path/to/new/review_plan
python -m pytest -q experiments/dflash_osd_agreement_filter
```

GPU execution remains on review hold. After approval, fill the path placeholders
in `runtime_config.example.json`; model files, author code and external runtime
must match the inherited hashes. Use `run` with the same dataset/references, pinned
`--tokenizer`, `--runtime-config`, `--allow-gpu-execution`, and both
`--reviewed-code-sha256` and `--reviewed-reference-sha256` from the plan receipt.
The controller performs qualification and a clean restart automatically; failed
correctness checks stop execution. It never automatically resumes an interrupted
arm or correction-bank campaign.

## Evidence, cost and retention

Keep source loss/exposure and anchors, every raw score/native trajectory, exact
input/output IDs, arm-specific histories, group membership, filtering fractions,
proposed/applied norms, directional derivatives, numerical receipts, peak memory,
all regressions and errors. Source losses from independently sampled passes are
not a matched-anchor before/after fit test.

Reports provide per-request metrics, comparisons against F and G against H/L/N,
turn-2/3 splits, and WildChat/synthetic PersonaMem separately. Ratios are computed
from summed valid paired measurements. Turn 3 describes accumulated two-update
states; it does not isolate a causal effect of update 2.

Request timing excludes adaptation/publication. Training (including teacher
preparation), export, optimizer-state persistence, activation, cache maintenance
and collection metadata costs remain separate. Native publication time is nested
inside the complete activation timer. G/N perform additional gradient work;
matching optimizer-step counts does not match computation. Benchmark-only work
is separate from own-history serving, and no asynchronous deployment claim is made.

Only one arm's workers/checkpoints are resident. Storage admission reserves eleven
model-sized units plus 512 MiB staging and at least 8 GiB free reserve; subsequent
arms are admitted again. Recovery states remain until their comparisons and
cleanup pass and an owned, hash-verified `ARM_RESULT.json` is durable. Then retire
only that completed arm's D1/D2 weights and Adam states, logging exact paths and
lost replay/continuation capability. Invalid/incomplete arms keep recovery state.
Compact evidence is retained; private data, weights and results never enter Git.

Stop at the declared report, including unfavorable outcomes. No tuning, new arms,
repeated timing campaign or mechanism expansion is included.
