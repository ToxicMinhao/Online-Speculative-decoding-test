# Frozen-DFlash correction bank, v0.1

This implements the user-supplied **DFlash correction bank mathematical
specification, v0.1**, dated 19 October 2026. Its PDF SHA-256 is
`453b142466d8c6d8025391cf3d942426460876e34aa5bcb80c3f78c7b8314e12`.
The PDF defines the algorithm; this directory supplies the bounded implementation,
qualification and six-stream comparison. Raw conversations, model weights and the
PDF are not bundled with the public code.

**Review status: code only.** No GPU qualification or correction-bank campaign
result is claimed. Execution is on hold for code review and subsequent execution
approval. CPU checks establish implementation properties, not useful transfer,
native-output preservation or deployment speedup.

## Mechanism and defaults

Released DFlash, its normalization layers and vocabulary projection `E` remain
frozen. For its normalized pre-vocabulary feature `h`, the correction is
`h + U C phi(h)`, followed by the unchanged single vocabulary projection.
Four zero-initialized `64 x 64` matrices contain 16,384 trainable values in total.
There is no bank Adam optimizer, learned projection, domain-label router or
full-backbone update.

| Choice | Pinned implementation |
| --- | --- |
| Output basis `U` | Top 64 right singular vectors of released `E`; float64 row-chunk accumulation of `E.T @ E`, then `scipy.linalg.eigh(driver="evr")`; descending eigenvalue order |
| Basis signs | Largest absolute entry in each column is positive; first index breaks ties |
| Input projection | `PCG64(17)` Gaussian matrix, reduced QR, same sign convention; `phi = V_in.T h / max(1, norm(V_in.T h))` |
| Descriptor projection | Fixed 64-dimensional `PCG64(17)` Gaussian reduced-QR projection, followed by unit normalization |
| Descriptor feature | Final frozen target-normalized hidden state at the last token of the complete current prompt |
| Row center | Mean of the rows of `P = E U`; `kappa = 2 max_v norm(P_v - mean(P))` |
| Radius | `r = 1/kappa`, hence global architectural displacement bound `R = 1` |
| Read | Zero, four singleton supports and six pair supports; coefficients in `[0,1]`, at most two nonzero, effective matrix norm at most `r`; coefficients need not sum to one |
| Kernel / error | Gaussian bandwidth `ell=1`; `beta=1/2`; `epsilon_t = 1e-3 max_{s<t} norm(G_s)` |
| Protection | `rho=1/4`: preserve at least 75% of each recorded positive anchored gain, subject to declared numerical tolerance |
| Write | Two scheduled constrained majorization steps for one owner; damping `lambda = 1e-2 kappa^2 / (4*64)` |
| New-slot penalty | `zeta=0.05` times the best unused trial's predicted improvement |
| Memory | All response summaries, exact sampled-position Fisher factors and protection records retained during each short conversation; no sketching or eviction |

Every request selects a correction using its causal prompt descriptor and earlier
feedback only. The chosen coefficients stay fixed until its answer finishes.
Their sum is compiled into one matrix; per-token computation does not run four
separate heads. Zero corrections return the original hidden tensor without a
cast or addition. Even this fallback incurs descriptor, routing and fusion work,
which remains in request latency.

Feedback is measured at the released reference. It retains the gradient `G`,
exact sampled-position Fisher factors, context descriptor, saved forecast and its
subsequent error. Current-response head optimization uses frozen features and the
compressed teacher moment `M`, rather than repeated backbone backpropagation or
persistent full-vocabulary teacher distributions. Only the newest response
receives loss; prior questions and answers remain causal context.

Writing evaluates every candidate owner independently of the served selection.
Candidate scores include the current head's response loss, its quadratic
majorizer and the declared unused-slot penalty. A positive served-fusion gain is
recorded **before any owner trial**. The selected owner receives two scheduled
steps, each accepted only after independent feasibility and majorizer checks.
Rejected steps keep the old feasible point; feasible zero steps remain no-ops.
Two scheduled steps need not mean two nonzero updates.

Protection records retain their original coefficients, matrix reference,
gradient, covariance and gain budget. Their constraints bound cumulative change
from that original reference, including historical fused corrections. Allowances
are never reset. Positive served gains are labeled transfer evidence; positive
post-write singleton gains are source-fit evidence. They are not interchangeable.
These guarantees concern the fixed-feature weighted loss on recorded anchors;
they do not certify future routing, greedy agreement or latency.

## Numerical implementation and native hooks

The numerical dependencies are **NumPy 2.2.6 and SciPy 1.15.3**, listed in
`requirements-numerical.txt`. Preserve the existing qualified PyTorch, SGLang,
CUDA and author-training environment; this requirements file does not replace
them. No package installation is performed by the controller.

Read support problems use SciPy **SLSQP**, `maxiter=200`, `ftol=1e-12`, with a
positive `0.1` coefficient start to avoid choosing an arbitrary subgradient at
the error-norm apex. Zero is always an explicit candidate. Write candidates use
**trust-constr**, `maxiter=200`, `gtol=1e-9`, `xtol=1e-10` and
`barrier_tol=1e-10`. Hessian products are matrix-free; no `4096 x 4096` dense
Hessian is constructed. Internal coordinate/objective rescaling does not change
the specified optimization problem.

Independent acceptance checks use feasibility absolute/relative tolerances
`1e-9`/`1e-7` and majorizer absolute/relative tolerances `1e-10`/`1e-8`.
Read near-ties use `1e-10 + 1e-8*abs(best_score)`, preferring zero, then smaller
norm and support. Owner ties use absolute tolerance `1e-10`, preferring an
allocated slot and then its index. Numerically empty kernel support is
`Z <= 1e-15`. Solver success alone does not authorize a write. Failures and
nonfinite diagnostics are retained explicitly without accepting nonfinite state.
`core.BankConfig` is the source of these exact constants.

The draft hook is
`DFlashWorker._greedy_sample_from_vocab_parallel_head(hidden_states)`, after
`DFlashModel.norm`. The descriptor hook observes
`DFlashWorker.forward_batch_generation` and its
`logits_output.teacher_hidden_states` at the final prompt token. It waits for
all native prefill chunks, including the existing 8,192-token chunking policy;
no generated-answer feature enters the read decision. No extra target prefill
is introduced solely for the descriptor. Hook receipts and computed projection
artifacts are saved and hashed.
The wrapper copies `ScheduleBatch.extend_lens` and `prefix_lens` before calling
native forward. It must not read `draft_input.ctx_lens` afterwards: the pinned
runtime clears that field when it consumes the target features into draft KV.

Teacher logits use the qualified frozen vocabulary head in its native BF16
computation, followed by float64 normalization and analytical correction-space
calculations. The bank's mathematical objective uses fixed frozen features and
the float64 affine logit correction. Native serving computes the small residual
and addition in FP32, casts back to the original BF16 head input, and calls the
unchanged vocabulary projection. The retention theorem applies to the
mathematical fixed-feature objective, not automatically to these finite-precision
native logits. GPU qualification must check this boundary, output preservation,
compressed/direct gradients and actual retained-anchor losses. A KL reduction
does not itself establish fewer verification calls.

Qualification additionally runs `precision.py` on the same sampled frozen
features for zero, served and fitted matrices, plus the deterministic control
`C = r * I / sqrt(64)`. It compares actual native corrected-head logits with
the FP64 reference, centering logit changes to remove softmax-invariant offsets.
It records each path's loss change against its own zero-correction baseline,
using identical teacher probabilities, along with feature/logit magnitude ratios,
changed coordinates and correction survival after rounding. A magnitude ratio
can exceed one if rounding amplifies a change; it is not a bounded percentage.
These are observations, not a performance or nonzero-survival admission test.
Their cost belongs to qualification. The protection guarantee remains scoped
to the mathematical fixed-feature objective.

## Causal comparison protocol

The inherited approved package supplies 116 ordered requests in six streams:
four WildChat streams (20/16/17/23 requests) and two synthetic PersonaMem streams
(18/22 requests, different persona IDs). The unchanged dataset loader pins
source revisions, selected IDs, manifest and source records. Exact user wording,
outer whitespace, indentation, newlines and source order are preserved. Source
assistant answers, supplied persona profiles and annotations do not enter
serving. The common pinned system framing and generated assistant token IDs do.

Each conversation starts from released DFlash, a fresh OSD optimizer, empty bank
and feedback/protection memory, and seed 17. BANK and OSD produce **their own**
canonical verified answers and accumulate their own histories. Those histories
are never overwritten to force agreement. At every turn:

1. Save all pre-update states; the bank snapshot is taken **before its read**.
2. Generate each learning arm's canonical answer on its own history.
3. Compare frozen, OSD and BANK on the **bank arm's exact saved input IDs**,
   using their states before learning from the current response. Run three
   capture-disabled timing repeats in rotating order. Each bank replay restores
   the pre-read state and recomputes descriptor, routing and fusion.
4. Only after scoring, obtain supervision from each arm's own generated response
   and perform its update. Replays never supply feedback or replace history.
5. Continue persistent state to the next turn. Skip both final writes; final
   bank feedback summarization is retained as analysis-only work.

A final one-token answer has no anchors under the author's valid-count-minus-one
sampler. BANK records an explicit empty analysis diagnostic with no teacher or
backbone call, no RNG consumption and no write. Its unobserved loss gain is null.
This exception applies only to final analysis; OSD's training recipe and the
nonfinal sampler/audit remain unchanged. Invalid or empty serving outputs are
not accepted by this exception.

The main comparison therefore has **116 contexts, 232 canonical executions,
110 update events per learning arm and 1,044 timed executions**. There is no
deduplication of zero-bank turns. OSD makes 220 optimizer steps; BANK schedules
220 owner steps, with additional candidate and reference-gradient work. These
are not compute-matched step counts. Timed repeats are not independent examples.

Frozen and OSD are unchanged external baselines. OSD remains the **AdaFlash OSD
training path under our session protocol**: full-weight DFlash adaptation, fixed
target, committed-response author anchors, target conditioning, forward KL at
temperature 1, no hard-label mixture, gamma-7 position weighting and its qualified
loss reduction. It retains AdamW LR `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`,
weight decay `0.01`, clipping norm 1, two passes and two optimizer steps per newest
response. The bank uses the same sampling/conditioning/relative position weights,
but explicitly normalizes its fixed sampled response weights to sum one as
specified in the PDF. Raw losses with different normalizations are not a direct
baseline comparison. No AdaFlash adaptive-length head or reverse KL is added.

Serving keeps the pinned Llama-3.1-8B-Instruct/DFlash-UltraChat model pair, native
SGLang runtime, greedy decoding, block size 10 and **1,024 output tokens or EOS**.
The qualified input/total limits remain 31,744/32,768; reserving the full output
cap and nine proposal positions gives a conservative input ceiling of 31,735.
There is no silent truncation. Preserve full histories and the established cold
request/cache invalidation policy.

## Qualification and execution gates

The first stream in manifest order is a separate qualification episode. It
checks zero-output preservation, compiled fusion, newest-response masks and
canonical target conditioning, compressed-versus-direct current-response loss
changes/gradients, and direct old-anchor retention bounds. Qualification retains
frozen draft and teacher features so actual historical losses can be recomputed;
checking only stored quadratic constraints is insufficient.
At least one response must provide nonempty sampled measurements; an empty
final diagnostic cannot turn entirely vacuous checks into qualification. The
native-versus-reference precision report is required, but no sign or minimum
size of its measured improvement is required.

The fixed correction space's source-fit change is reported, without requiring a
positive gain. The specified cold-start read must remain frozen on turns 1 and
2, while response 1 still trains a module. Correctness failures stop admission
for repair. Once qualification passes, all six streams restart cleanly,
**including the first stream again**. A performance win is never an admission
criterion, and unfavorable results do not trigger silent retuning.

The separate 20-request qualification adds 180 timed executions and is excluded
from the 1,044 main measurements. Output/input/state mismatches remain in raw
records. A comparison with an invalid repeat is excluded as an entire paired
context; independent valid method pairs remain usable. An implementation gate
failure stops the queue rather than automatically retrying or changing settings.

## Costs, evidence and storage

Complete request latency includes prefill, descriptor extraction, routing,
fusion, drafting and target verification, including zero-bank requests. Capture
is disabled in paired timing. BANK snapshot loading, checkpoint activation and
external cache maintenance are recorded separately. The full publication RPC,
including validation, is charged; its internal native timer is a nested detail.

Serving-only gains describe performance with an available checkpoint. Separate
capture/preparation, teacher replay, compression, all candidate gradients,
solver work, training, export, loading, activation and cache costs are retained.
An outer recurring adaptation timer includes its nested stages exactly once.
Deployment episode totals include each arm's own canonical serving and blocking
adaptation; benchmark replays, qualification and final analysis-only feedback do
not enter them. Restoring BANK's released weights merely because a benchmark
previously loaded OSD is recorded as benchmark method restoration, not as an
intrinsic bank deployment cost. Actual experiment wall time remains separate.
There is no asynchronous deployment claim.

The cost report also reconstructs a frozen reference on the bank trajectory by
averaging the three fresh measurements per saved input and summing over turns,
including external per-request cache maintenance. It excludes artificial
benchmark model switches and has no adaptation cost. This is explicitly a
reconstructed reference, not a separately executed frozen canonical session.
It is available only with complete valid paired coverage; BANK and OSD's own
canonical episode costs may concern different histories.

Retain per-turn pre-read bank snapshots, computed projections, frozen-reference
gradients, exact Fisher factors, teacher moments, immutable protection records,
RNG/anchor metadata, all candidate/step diagnostics and every timing repeat.
Snapshots use content-addressed immutable arrays; offloading must move both JSON
roots and their referenced arrays. OSD restart state stays available until its
paired comparisons finish. Old completed OSD state may be retired only through
the existing receipt-producing policy; the final state and compact evidence
remain. Do not describe retired historical checkpoints as archived or retained.

Storage admission is still required before execution. A read-only guard runs
before qualification can start. It budgets at least 8 GiB for each main bank,
12 GiB for qualification including direct replay features, seven retained final
OSD bundles, OSD checkpoint/optimizer/export overlap, and an additional 8 GiB
free-space reserve. These are
not a proof that the entire six-stream campaign fits the current server. Budget
qualification features, accumulated retained stream artifacts, snapshot-array
growth, transfers and staging for the whole campaign. Verified offloading to the
approved archive host is available; no offload or cleanup occurs automatically
from this code-review task. The example's storage numbers are placeholders for
admission, not measured capacity requirements or permission to launch.
The implemented admission does not assume any future offload or deletion. If it
fails, it stops before model execution; a concrete offload/storage plan must be
reviewed rather than reducing retention allowances to force admission.

Reports retain every gain and regression and pool ratios from summed
measurements. WildChat and synthetic PersonaMem remain separate. Report overall
and actively corrected turns, singleton/pair use, ownership, predicted versus
realized anchored gains, source fit, blocked writes and costs. Distinguish
capacity, preservation, transfer selection, fusion and runtime limitations;
unfavorable results alone do not establish one particular explanation. Additional
baseline campaigns and ablations are outside this first run.

## Files and review commands

| File | Responsibility |
| --- | --- |
| `core.py`, `projections.py` | Fixed coordinates, read optimization, candidate owners and cumulative protection |
| `feedback.py`, `training.py` | Qualified response sampling, frozen objective/compression and direct qualification |
| `precision.py` | Qualification-only native BF16 versus reference logits/losses and rounding-survival measurements |
| `native.py`, `native_hooks.py` | Existing native verifier/runtime, causal descriptor and compiled correction |
| `artifacts.py` | Immutable, hash-bound snapshots and content-addressed arrays |
| `experiment.py`, `run.py` | Own-arm histories, paired pre-update replay, qualification and clean restart |
| `report.py` | Per-turn/raw repeat tables, source pooling, usage and complete costs |

CPU metadata plan from the repository root:

```bash
python -B -m experiments.dflash_correction_bank.run plan \
  --dataset /path/to/approved/conversation_pilot \
  --out /path/outside/repository/bank_plan
```

CPU tests, with the pinned numerical dependencies and an available CPU-capable
qualified PyTorch installation (development test runner: `pytest==8.3.5`):

```bash
CUDA_VISIBLE_DEVICES='' python -B -m pytest experiments/dflash_correction_bank -q
```

Optional `qualify-text` also requires the pinned local tokenizer; it loads no
model or CUDA worker. `run` imports the native backend only after validating
the explicit `--allow-gpu-execution` flag, runtime configuration, tokenizer and
the exact `--reviewed-code-sha256` from the plan. **Do not execute that action
during code review.** After approval, follow the server rules and confirm a
complete storage budget first. The initial runtime uses one serving GPU and one
teacher/training GPU, with synchronous updates and serial timed requests.

Outputs include `PLAN.json`, source/runtime manifests, `EVENTS.jsonl`,
`PROGRESS.json`, `RESULT.json`, `CAMPAIGN_STATE.json`, qualification receipts,
`report/per_turn.csv`, `report/timing_repeats.csv`, `report/bank_decisions.csv`,
`report/REPORT.json` and `report/CONVERSATION_REPORT.md`. These six previously
inspected streams support mechanism development, not a final generalization
claim. Future untouched-stream confirmation is outside this campaign.
