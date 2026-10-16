# Per-turn DFlash adaptation pilot

This experiment measures when learning from a completed conversation response
helps or hurts drafting on the next request. It reuses the **AdaFlash OSD training
path under our session protocol**, not an exact reproduction of either paper.
Implementation is submitted for review. **No GPU pilot results are claimed.**

## Declared protocol

Use the existing pinned Llama-3.1-8B-Instruct/DFlash-UltraChat model pair, native
SGLang runtime, fixed target and full-weight draft parameter set. The qualified
teacher replay, committed-response anchor sampler, gamma-7 position weighting,
forward KL at temperature 1 with no hard-label mixture, and weighted loss
normalization are reused from [the author-path port](../dflash_two_update_followup/AUTHOR_PATH.md).
No adaptive-length head or new training objective is introduced.

Each conversation independently starts at released weights with a fresh AdamW
optimizer and seed 17. AdamW retains learning rate `3e-4`, betas `(0.9,0.999)`,
epsilon `1e-8`, weight decay `0.01` and gradient clipping at norm 1. Each nonfinal
turn is scored before its response is used for training. That response is then
presented twice, with effective response batch size 1 and **one optimizer step
per pass**. Weights, Adam moments and anchor RNG continue between turns.

Only positions in the newest answer receive loss. Earlier questions/answers
remain full causal context, masked from loss. Dataset assistant answers, source
profiles and our annotations never enter serving or supervision. The target's
full causal replay supplies teacher features/targets for the generated response.
This path records committed-token metadata; it does not claim verification
feature reuse or free teacher preparation.

The six approved streams contain 116 requests: four WildChat streams with
20/16/17/23 requests and two synthetic PersonaMem streams with 18/22. Skipping
each final update gives **110 events, 220 steps and 220 response presentations**.
Actual anchors/weighted and unweighted supervised exposure are recorded; the
author sampler selects at most 512 anchors, not every position of a long answer.
This schedule has different optimizer-step exposure from the earlier
eight-response experiments.

## Exact conversation inputs and native serving

- Greedy decoding, block size 10, output cap **1,024 or EOS**.
- Input limit 31,744; total context limit 32,768; reserve nine final speculative
  positions. With the full cap the conservative input ceiling is 31,735.
- No truncation, answer substitution or source-question rewriting.
- Common pinned system framing; source persona/system metadata is excluded.
- Original user text includes its exact outer whitespace, indentation and
  newlines. `framing.py` bypasses only the tokenizer template's trim; all other
  framing remains pinned. The 22 affected selected prompts stay unchanged.
- Prior canonical assistant **token IDs** enter history directly, without
  decoding/re-tokenizing them. EOS/cap finish reasons and lengths are retained.
- Serving retains the qualified 8,192-token prefill chunks and cold-request
  cache policy. Checkpoint publication refreshes affected draft helpers and
  invalidates incompatible caches. This is not a prefix-cache optimization test.

Every timing replay uses the canonical saved input ID list. The native observer
reconciles the supplied IDs; controller checks require exact list and hash
equality. Equivalent decoded text is insufficient. Replays do not supply
training feedback or replace the canonical history.

## Matched checkpoint comparisons

Let D_t be weights after learning response t. On request t, compare released F,
current D_(t-1), and previous D_(t-2) when available. At turn 1 only F is run;
at turn 2 the previous checkpoint equals F and is deduplicated. Run three
capture-disabled timing repeats with rotating method order.

The full campaign contains **990 main timing replays on 116 distinct saved
request contexts**, plus 116 canonical serving calls. Repeats are not independent
examples. Online/F measures accumulated adaptation. Current/previous isolates
the latest update's effect on the same next request; it can regress while both
checkpoints still outperform F.

All output token IDs must match the canonical reference for a valid gain claim.
Raw mismatches are retained, not dropped or repaired by changing verification.
A comparison with any invalid/missing repeat is excluded as a complete paired
context, with excluded coverage visible. The first complete stream must pass
masks, scheduling, checkpoint identity, exact inputs/outputs and worker cleanup
before the remaining five are released automatically. Any subsequent failed
implementation gate also stops the queue; there is no automatic crash retry.

## Bounded regression diagnostics

Select cases in the declared stream/turn order, not by ranking the largest
effect: at most two harmful updates and one beneficial control across the whole
campaign. The rule is fixed before outcomes:

- More verification calls in all three valid pairs: drafting harm; retain a
  possible timing tradeoff rather than claiming slower serving automatically.
- Equal calls in all three pairs, but slower in each: timing-only harm, which
  does not establish degraded token prediction.
- Fewer calls in all pairs with no latency regression, or equal calls and faster
  in every pair: beneficial control, with its metric basis labeled.
- Mixed signs, missing pairs or mismatching outputs: no reproducible case label.

Restore exact pre-update weights, Adam state/counters, anchor RNG and
Python/NumPy/CPU/CUDA RNG. Re-run the same two-step event at `1e-4`, requiring the
same feedback and sampled-anchor/weighting hashes. Then restore and verify the
entire original main state, including tensor identities by content hash, before
continuing. The lower-rate checkpoint is separately named L_t and can never
become the main online draft.

Compare pre-update, original post-update and lower-rate checkpoints on the same
next request and one earlier saved request, each with three rotating repeats.
The earlier request is `max(1,t-1)` for update event t; it is an already-observed
retention probe, not an unseen generalization test. Diagnostic branches add at
most 3 updates/6 steps/54 timed replays, outside the main counts.

## Costs and retained evidence

Complete request latency includes prefill, drafting, target verification and
the common native observer. It excludes checkpoint activation, teacher replay
and draft training. State clearly: **performance with an available checkpoint**.
Activation is the whole synchronous publication call, including validation;
the nested worker-native timer is recorded separately. Cache maintenance is
outside paired request timing and reported separately.

Preserve separate teacher preparation, draft forward/backward/optimizer,
export, recovery-state recording, activation, collection and transfer timers.
Nested timers are not added twice. Trainer initialization is one-time loading.
Reconstructed pilot blocking cost and actual experiment wall time are separate;
the latter includes repeats, diagnostics, state auditing and engine lifecycle.
No asynchronous serving or complete training-inclusive speedup is established
by this pilot alone.

Rolling pre/post optimizer/RNG states and weights stay available until their
next-request comparison completes. Selected case bundles persist. Exact
checkpoint IDs/hashes, input/output IDs, support boundaries, anchors, configuration,
all measurements and failure records remain compact evidence. Unselected bulky
rolling state is retired only after its required comparisons and durable report;
every deletion records exact paths and lost replay/continuation capability.
Failed streams keep recovery evidence. Interrupted-episode automatic resume is
not implemented; do not restart over existing outputs.

Admission checks include peak checkpoint/state/export overlap and an 8 GiB
reserve. Completed case bundles still occupy disk. Before native execution,
confirm actual A800 free space and approved offload capacity. No remote storage
or GPU operations were performed for this implementation review.

An answer shorter than two tokens cannot satisfy the qualified author anchor
sampler: the event stops explicitly with its identity, rather than silently
skipping a required update or changing the sampler. Its legitimate sampled
zero-supervision edge case is recorded without inventing supervision.

## Files and review commands

| File | Responsibility |
|---|---|
| `dataset.py`, `framing.py` | Immutable source package, tokenizer and lossless input checks |
| `experiment.py`, `run.py` | Canonical trajectory, replay matrix, first-stream gate and diagnostics |
| `training.py` | Per-turn schedule, durable state and unchanged borrowed loss |
| `native.py` | Qualified serving/training bridge, publication and bounded retention |
| `report.py` | Raw repeats, per-turn comparisons, conversation reports and case analysis |

The local approved dataset package is required for execution; raw conversation
data, models, credentials and experiment outputs are **not uploaded** with this
code. `dataset.py` pins its manifest, file inventory, sources and selected IDs.
Runtime models and the existing qualified private bridge dependencies must be
installed separately as recorded in the earlier experiment. An example runtime
configuration is provided without credentials or local machine-specific paths.

Metadata-only plan, from the repository root:

```bash
python -B -m experiments.dflash_per_turn.run plan \
  --dataset /path/to/conversation_pilot --out /path/outside/repository/plan
```

CPU/tokenizer qualification (no model execution):

```bash
USE_TORCH=0 USE_TF=0 python -B -m experiments.dflash_per_turn.run qualify-text \
  --dataset /path/to/conversation_pilot --tokenizer /path/to/pinned/tokenizer \
  --out /path/outside/repository/text-check
```

Focused tests, including real artifacts when supplied:

```bash
CUDA_VISIBLE_DEVICES='' USE_TORCH=0 USE_TF=0 \
PER_TURN_DATASET=/path/to/conversation_pilot \
PER_TURN_TOKENIZER=/path/to/pinned/tokenizer \
python -B -m unittest discover -s experiments/dflash_per_turn -t . -v
```

After code review and explicit execution approval, source the approved A800
environment helper and use `run` with `--runtime-config`, `--tokenizer`,
`--allow-gpu-execution` and the exact `--reviewed-code-sha256` from the plan.
The controller runs streams serially so timing is not contaminated by another
stream's training. Initial two-GPU allocation is one serving worker and one
teacher/training worker; updates are synchronous.

Outputs include durable `EVENTS.jsonl`, `PROGRESS.json`, `RESULT.json`,
`CAMPAIGN_STATE.json`, raw repeat CSVs, `per_turn.csv`, `REPORT.md` and `CASES.md`.
WildChat and synthetic PersonaMem results remain separately identifiable.

See [VALIDATION.md](VALIDATION.md) for performed CPU checks and unperformed GPU
qualification. Passing implementation checks does not advance the research stage.
