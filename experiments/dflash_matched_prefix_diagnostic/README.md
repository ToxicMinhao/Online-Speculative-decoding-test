# DFlash matched-prefix acceptance diagnostic

This bounded, read-only diagnostic asks whether adaptation fixes a frozen
draft's first error, introduces an earlier error, or changes probabilities
without extending the correct proposed prefix. It compares each original
group's F, S8 (`3e-4`) and L8 (`3e-5`) at **identical canonical block starts**.
It does not train or measure deployment speedup.

Status: **all six repaired groups completed and independently audited**.
The original cold-prefix failure remains preserved. Original-schedule replay
now passes all-nine frozen equality at all 240 selected locations; the full
720 measured blocks and 25,854 reconstruction target forwards completed with
zero invalid pairs or unknown work. See [VALIDATION.md](VALIDATION.md) for
results, costs and interpretation limits. The bounded diagnostic has stopped;
earlier agreement-filter and correction-bank campaigns remain paused.

## Inputs and fixed selection

Reuse the completed PR9/PR10/PR11 evidence. Bind the sealed sample manifest
`1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`
and the executed PR11 commit
`673e75efb72595f718b41c03fb2c7aa0e71fc40d`. The committed reference bindings
contain hashes and provenance, not private prompts or answers.

- All 24 held-out requests are included.
- One source control is selected per group: sort the four original source
  responses by F output-token length, break ties by original manifest order,
  and take the second entry (lower median). These six controls are illustrative;
  they are not an estimate for all 24 source requests.
- For each request, sort retained F rounds by output offset. With `n > 1`, let
  `k=min(n,8)` and select indices `floor(i*(n-1)/(k-1))`. Keep all available
  rounds when `n <= 1`. Never replace zero-round requests or short terminal
  blocks, select by adapted outcomes, or resample.
- Seal IDs, offsets, seeds, all nine retained F proposals and checkpoint
  bindings before native work. A later run revalidates the sealed plan against
  the original evidence and refuses changed selection.

The upper limit is **30 requests × eight locations × three arms = 720 native
one-block evaluations**. First-group qualification is included. The actual
budget follows the sealed plan. There are no extra model warmups, automatic
retries, complete-response generations, scored-block target verifications,
response-level KL evaluations, optimizer steps or new checkpoints. Historical
target verifications now reconstruct each original cache sequence. Their exact
count is sealed separately: the retained 240-location plan requires **25,854
additional target-verification forwards**, including all three arms. These are
not extra draft measurements, and their runtime is not assumed negligible.

Within each group, evaluate all selected locations under F, then S8, then L8.
This sealed order avoids repeatedly reloading weights; every probe still
rebuilds cold target/draft caches for its own checkpoint. Qualification and
pairing use the same locations regardless of valid positive or negative results.

## Native reconstruction

Preserve the qualified Llama-3.1-8B-Instruct target, UltraChat DFlash, BF16,
concurrency one, greedy head, block size ten and original 32,768-token total
context limit. No training forward substitutes for native drafting.

For prompt `P`, saved canonical response `Y` and output offset `s`:

1. Cold-prefill the **original prompt `P`** in the pinned 8,192-token chunks.
   Require its first target token to match the saved prefill seed.
2. Replay every retained verification round before `s`, with its **original
   seed and nine proposal IDs**, through the native target-verification path.
   No historical draft forward is run. Recompute target predictions, acceptance,
   rejected-KV freeing and committed target-feature materialization using the
   unchanged native verifier; check all target IDs, committed tokens, cache
   increments and next seeds against the saved trace. Never force a saved
   acceptance length or overwrite a mismatched prediction.
3. Require the reconstructed history and next seed to equal `P + Y[:s]`.
   Run exactly one measured native draft block, containing the known seed and
   nine proposals. Inspect the unchanged greedy head and stop before verifying
   **this measured block**. The replay buffer must match the saved history; it
   does not create a new response or supply learning feedback.

The reconstruction schedule and raw event hashes are part of plan schema v2.
Each arm replays the same frozen-trajectory target inputs while materializing
its own checkpoint's draft KV. Every reconstruction forward is counted before
execution, including a failing call. Failed reconstruction receipts are durable
and consume zero measured draft blocks when the failure precedes drafting.

The diagnostic scheduler preserves the pinned source guards and D181 cache/weight
publication setup. It omits D202's whole-response observation hooks, whose active
trace requirement is incompatible with partial replay; native verification and
acceptance functions are unchanged. A startup compatibility receipt records this.

Every arm uses valid caches for its own weights and refreshed checkpoint
helpers. Preserve native masks, absolute positions and target conditioning.
Long-prefix cache indices are copied to `int64`, even when the native pool is
`int32`. Checkpoint/runtime/file identities and read-only state guards remain
explicit. No optimizer is constructed or loaded.

The old cold-prefix path did not reproduce one retained proposal. Original-schedule
replay removes that known execution difference, but native numerical equivalence
still requires GPU qualification; this repair is not proof of the failure cause.
**All nine reconstructed F proposals must match the saved block**, including
unscoreable terminal positions. Failure is retained as invalid evidence; no
location is replaced or rerun to obtain a match.

## Scoring and validity

The **primary correct matching-prefix length includes a correctly matched
EOS**. Also report non-EOS length and a matched-EOS flag. For example, repairing
EOS after three correct non-EOS proposals changes the primary length from
three to four, even though non-EOS length stays three.

Only compare to the available canonical `Y[s:s+9]` and stop at matched EOS.
Exclude the known seed and all correction/bonus tokens. EOS and cap exhaustion
are boundaries, not rejections. An entirely matching cap-truncated available
suffix is **right censored at the saved boundary**; correctness beyond it is
unknown. Report unavailable proposal positions without scoring them.

Record actual native argmax, canonical-token log probability and canonical
margin against the strongest other token from native BF16 logits with FP32
subtraction/reduction. Check inspected logits reproduce native argmax, reject
nonfinite values and retain explicit tie information. A margin of zero does
not specify which tied token the native argmax selected.

For S8/F and L8/F report local prefix gains/losses/ties, changed proposals,
repairs at F's first mismatch, and introduced errors within F's formerly
correct prefix. A later positional repair is a useful prefix extension only
when no earlier error blocks it. After an earlier mismatch, canonical-token
agreement is a positional observation, **not** the target's decision under a
divergent proposal history.

The execution controller preserves raw measurements before checking them;
offline reporting repeats the same validator. Identity, reconstruction,
geometry, nonfinite or state-mutation failures stop the affected group. A failed
first group blocks automatic continuation. After successful first-group
qualification, later independent groups may proceed after an affected group
stops, provided owned-worker cleanup succeeds. All invalid/unattempted locations
remain in denominators; missing evidence is not regenerated. Null/harmful valid
results never block continuation.

## Reports and costs

Keep paired block and position records, a sealed plan, implementation/configuration
hashes, request/group/domain summaries and raw validity evidence. Average sampled
location effects within each request, then equally across requests; also retain
raw counts and valid/selected denominators. Keep source controls and held-out
results separate. Include the retained complete-response call counts as context.

Count original-prompt prefill chunks, reconstruction target forwards, measured
draft blocks, head inspection, loading/activation, cache maintenance, elapsed
wall time and peak memory separately. `reconstruction_target_verify_s` is a
component of `reconstruction_s`, which also includes setup, native commit and
feature materialization. Both are diagnostic timers nested inside block wall
time, not costs to add together. Diagnostic timings
are not deployment latency. Do not sum nested timers twice or extrapolate local
repairs into saved verification calls. Margins/log probabilities do not identify
where the earlier response-level KL reduction occurred.

The durable progress index stores claims/status without copying all histories
at each stage. Exact inputs remain in the sealed plan, raw rows in the event
log and native receipts, and the complete result is saved at termination.

Possible outcomes include few useful decision changes, repairs canceled by new
errors, local prefix improvement, mixed patterns or insufficient valid evidence.
None automatically authorizes a new mechanism, tuning sweep or data expansion.

## Commands

Run from the repository root. Keep outputs outside this public checkout.

```bash
python -m pytest -q experiments/dflash_matched_prefix_diagnostic

python -m experiments.dflash_matched_prefix_diagnostic.run plan \
  --baseline-root /private/pr9/attempt_001 \
  --diagnostic-root /private/pr10/attempt_001 \
  --lower-lr-root /private/pr11/attempt_001 \
  --out /private/matched_prefix/plan_001
```

The plan command is CPU-only. Private token IDs and selection evidence are
written to the private output directory. Paths above are placeholders.
After code review and explicit GPU-execution approval, on the qualified server:

```bash
python -m experiments.dflash_matched_prefix_diagnostic.run run \
  --baseline-root /private/pr9/attempt_001 \
  --diagnostic-root /private/pr10/attempt_001 \
  --lower-lr-root /private/pr11/attempt_001 \
  --sealed-plan /private/matched_prefix/plan_001/PLAN.json \
  --approved-code-sha256 REVIEWED_SOURCE_DIGEST \
  --allow-gpu-execution \
  --out /private/matched_prefix/attempt_001
```

The run requires clean reviewed code, unchanged evidence, matching selection,
all retained checkpoint tensors, the pinned runtime and a storage preflight.
Do not install a substitute runtime or recreate missing weights through training.
Report-only rendering never loads a model:

```bash
python -m experiments.dflash_matched_prefix_diagnostic.run report \
  --result /private/matched_prefix/attempt_001/RESULT.json \
  --out /private/matched_prefix/report_002
```
