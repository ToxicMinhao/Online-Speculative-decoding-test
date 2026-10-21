# DFlash related-instance transfer control

This bounded, read-only diagnostic asks whether retained S8 updates improve
drafting on genuinely new requests closely related to their feedback tasks.
It introduces no training method. Implementation and request preparation are
complete for review; **native execution has not started**.

Reuse released DFlash (F) and each original group's eight-step checkpoint (S8,
physical artifact `U2`, trained at `3e-4`). The frozen target is
Llama-3.1-8B-Instruct and the released draft is the qualified DFlash-UltraChat
checkpoint. Original model, tokenizer, runtime and checkpoint hashes must pass
admission; missing artifacts cannot be replaced or regenerated.

## Fixed work and parent selection

| New work | Per original group | All six groups |
|---|---:|---:|
| New independent requests | 2 | 12 |
| Natural F generations | 2 | 12 |
| Natural own-group S8 generations | 2 | 12 |
| Training / KL evaluations / new checkpoints | 0 | 0 |
| Additional generation warmups / retries | 0 | 0 |

Bind original manifest
`1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`
and the qualified implementation through PR13
`fb4f0b9d57b14f0d1b2af1dd015b5356feb89971`.

Within each group's four source responses, sort by retained F output length,
breaking ties by original manifest order. Select ascending ranks 2 and 3 before
constructing children. This deliberately excludes both extremes and is not
representative sampling.

| Original group | Selected parent suffixes, in order |
|---|---|
| `gsm8k_00` | `01`, `02` |
| `gsm8k_01` | `00`, `01` |
| `codealpaca_00` | `03`, `00` |
| `codealpaca_01` | `00`, `03` |
| `sharegpt_00` | `01`, `03` |
| `sharegpt_01` | `01`, `02` |

## Prompt-only construction and sealing

`select` exports fixed parent IDs, their user texts and a prompt-only catalog of
the original 48 requests. Saved answers, benchmark solutions and adaptation
metrics are absent from that construction view. The loader verifies retained
evidence internally; this does not authorize using its answers to write children.

Each child must preserve language, task family, requested format and approximate
complexity while changing answer-relevant content. For code, new examples alone
are insufficient if the same general implementation still answers the request.
Semantic review must establish a different required answer/behavior. Exact
duplicate guards complement that review; they cannot detect every paraphrase.

Construction records contain author/tool provenance, prior exposure disclosure,
new IDs, parent/group bindings, exact text, change explanations, nine explicit
semantic checks and pre-seal revisions. No blinded-construction claim is made.
All 12 must pass before sealing; an unsuitable fixed parent blocks preparation
with an explicit report rather than replacement or a smaller campaign.

`seal` reencodes all 48 original prompts with the pinned tokenizer, then seals
all child texts and rendered token IDs, parent/checkpoint bindings and execution
order. No parent conversation or answer enters a child's serving input. The
private complete request package is bound in [REQUEST_SET.md](REQUEST_SET.md).
Do not edit it after observing model outputs.

## Native execution and correctness

For each group in manifest order, run **F child 0, F child 1, S8 child 0,
S8 child 1**. All four generations are natural. Fresh F supplies the child's
canonical reference; no historical answer is forced or used as a reference for
new-request generation. Require exact F/S8 output tokens and termination.

Preserve concurrency 1, greedy BF16 decoding, block size 10, 1,024 tokens or EOS,
context limit 32,768, nominal input limit 31,744, effective prompt limit 31,735
(nine reserved proposal positions), and chunked prefill 8,192. Feedback capture
is disabled. Cold caches apply to every request and checkpoint switch, with the
qualified helper refresh. Normal runtime initialization remains intact and its
cost is recorded; there are no additional generation warmups.

The child-only native entry point rejects training, loss-evaluation, export,
historical-replay and arbitrary-checkpoint calls. Historical four-response and
source-replay guards remain unchanged. The wrapper changes request metadata to
the explicit `related_instance` role, not the generation algorithm.

The first complete group qualifies correctness **inside the 24-generation
budget**. Valid null or harmful results continue through the remaining five
unchanged. Identity, nonfinite, output, mutation, persistence or cleanup failures
stop execution without retry. Preserve raw receipts before validation, including
invalid numeric values. Returned-invalid work, unresolved dispatched work and
unattempted work remain separate. A later independent failure does not erase an
earlier valid pair when cleanup verifies unchanged checkpoints. No complete-run
conclusion is allowed from incomplete or inconsistent accounting.

## Storage, timing and outputs

Storage sizing uses the actual nested retained evidence, child tokens, 24 bounded
traces, configuration/result copies, atomic-write overlap, logs and retrieval
staging. It includes no model downloads, checkpoint exports or optimizer states.
Check the actual native and controller destinations, sum writes on shared
filesystems, and retain at least 8 GiB per distinct filesystem. The package does
not delete original evidence. Runtime-cache rebuilds or extra archives need their
own storage admission.

Record initialization/loading, full checkpoint activation, cache maintenance,
fresh request latency and actual campaign wall time with their native scopes.
**Prefill is charged inside request/native-generation time.** The qualified trace
records prefill token work but has no isolated prefill duration; that field is
explicitly unavailable, not inferred from aggregate latency. Nested timers must
not be added twice. One fixed-order measurement per arm/request cannot establish
serving speedup.

`RESULT.json`, `PROGRESS.json`, append-only `EVENTS.jsonl`, raw native receipts
and token trajectories preserve outputs, identities, per-round offsets, accepted
draft tokens excluding target correction/bonus tokens, committed/emitted
progress, lengths, cap/EOS status and failures. Reports contain:

- `PAIRED_REQUESTS.csv`: every selected child, including invalid/unattempted rows.
- `POOLED.csv`: group, dataset and overall summed paired counts and denominators.
- `RETAINED_CONTEXT.csv` / `RETAINED_CONTEXT_POOLED.csv`: original parent-source
  and ordinary held-out results, descriptive only; no historical generation.
- `COSTS.csv`, `SUMMARY.json` and `REPORT.md`: costs, work reconciliation,
  limitations and provenance.

Only after generation, compare child F tokens with its parent's saved answer and
all four source answers seen by its S8. Keep every valid child in the primary
analysis; also report the **non-exact-repeat** subset and its denominator. Token
inequality does not prove semantic novelty or exclude partial memorization.

The primary change is `100 * (sum(F calls) - sum(S8 calls)) / sum(F calls)` over
identical-output valid pairs, never an average of request percentages. Retain
gains, regressions, ties and incomplete cases. Fewer calls on reviewed new
instances, including the non-exact-repeat subset, supports transfer in this
constructed setting. A null result does not identify the cause of prior
regressions or invalidate online adaptation generally. Different historical
request sets are not matched causal comparisons. Twelve constructed requests
and six retained trajectories do not establish broad generalization or real
conversation benefit; any later selected method needs fresh confirmation.

## Commands

Run from the repository root in the qualified environment. Keep private prompts,
responses, runtime paths and artifacts outside this public checkout.

```bash
python -m experiments.dflash_related_instance_transfer.run select \
  --baseline-root /private/original_pr9_attempt --out /private/parent_selection

python -m experiments.dflash_related_instance_transfer.run seal \
  --baseline-root /private/original_pr9_attempt \
  --parent-selection /private/parent_selection/PARENT_SELECTION.json \
  --construction /private/CONSTRUCTION_FINAL.json \
  --tokenizer /private/pinned_tokenizer --out /private/sealed_requests

python -m experiments.dflash_related_instance_transfer.run plan \
  --baseline-root /private/original_pr9_attempt \
  --diagnostic-root /private/original_pr10_attempt \
  --manifest /private/sealed_requests/REQUEST_MANIFEST.json --out /private/plan
```

The retained PR10 evidence supplies parent-source native context; it does not
launch KL evaluations. After review and execution authorization:

```bash
python -m experiments.dflash_related_instance_transfer.run run \
  --baseline-root /private/original_pr9_attempt \
  --diagnostic-root /private/original_pr10_attempt \
  --manifest /private/sealed_requests/REQUEST_MANIFEST.json \
  --sealed-plan /private/plan/PLAN.json --tokenizer /private/pinned_tokenizer \
  --approved-code-sha256 REVIEWED_SOURCE_DIGEST --allow-gpu-execution \
  --out /private/new_attempt --native-root /private/native_artifacts
```

Execution requires clean reviewed code, the exact source digest and complete
sealed assignments. Plan bindings must reflect verified actual artifact paths.
Changing paths cannot silently change requests, checkpoints or scientific scope.
`report --result /private/new_attempt/RESULT.json --out /private/report` reads
saved evidence only. CPU checks do not qualify native GPU integration, memory
fit or output equality.
