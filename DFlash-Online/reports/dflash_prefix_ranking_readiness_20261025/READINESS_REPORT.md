# DFlash prefix-ranking: executor readiness

## Material Passport

October 25, 2026. ARS experiment-agent, code/artifact inspection and planning.
**Readiness report only**, as explicitly confirmed by the user after the request
and attachment differed about implementation. No production code changed, models
loaded on GPU, training, response generation, or qualification tests started.
Verification status: **ANALYZED** from the live A800 source and CPU-only artifact
inspection. Two internal code reviewers independently inspected the same sources;
this is not GPU or external-review qualification. Stage 1 / project None remains.

**Verdict: READY WITH EXPLICIT CHANGES for implementation; not yet qualified to
execute.** The exact ranking loss is feasible in the current chunked OSD path.
The main required change is moving sampling and frozen-reference evaluation ahead
of the trainable forward. No generic adapter, new optimizer or upstream rewrite
is needed. The first-screen budget below is a proposal, not a selected run.

## A. Current code and exact objective

The live root is `/data/sources/minhao/DFlash-Online`. Paths/line numbers below
refer to its [read-only source snapshot](source_snapshot/). All 17 inspected
`online_feedback` files match the prior qualified source seal. SpecForge is
pinned to `1839aae30003e3573d00979eb1edd57347b3578c`; the active training Python
is `/data/conda_envs/minhao/dflash_specforge_20261022/bin/python`, Torch
2.13.0+cu130. Native serving remains the separately pinned SGLang/FA3 worker in
`/data/conda_envs/minhao/adaflash_author/bin/python`.

| Operation | Exact file and function |
|---|---|
| Sampling | `training/specforge/algorithms/common/dflash_family_model.py:344`, `_sample_anchor_positions`: sample eligible adjacent response-token anchors without replacement, sort chosen anchors |
| Block preparation/forward | Same file `:596`, `_forward_draft_blocks`; seed/masked slots `:394`, positions `:386`; `training/specforge/modeling/draft/dflash_mask.py:109–117` defines visibility |
| Native target preparation | `online_feedback/sglang_backend.py:159`, `prepare_feedback`: captured canonical rows plus an extra terminal-only feature pass |
| Teacher alignment | `dflash_family_model.py:450`, `_aligned_target_hidden`; `online_feedback/objective.py:181`: predecessor target row predicts the current label |
| Draft and vocabulary loss | `online_feedback/objective.py:146`, `TargetObjectiveModel.forward`; `:106`, `_chunk_loss`; `training/specforge/core/chunking.py:15`, `checkpointed_chunk_reduce` |
| Response normalization | `objective.py:164–194`; `online_feedback/training.py:134`, `ReductionAdapter.forward_loss` |
| Accumulation/update | `training/specforge/training/controller.py:427–523`, `TrainerCore.train_step`; `online_feedback/optimizer.py:40`, `DirectBF16AdamW.step` |
| Event lifecycle/export | `online_feedback/training.py:503`, `train_event`; `:466`, `_checkpoint_state`; `:487`, `_export` |
| Publication | `online_feedback/controller.py:86–95`, `update`; `sglang_backend.py:232`, `publish`; `sglang_hooks.py:255–302`, native helper refresh/cache handling |

For response r, anchor a, and offset j=0…15, the current weights are

```
w[a,j] = kept_anchor[a] × in_bounds[a,j] × (j > 0)
         × response_loss_mask[a+j] × exp(-max(j-1, 0)/7)
D_r = sum_all_sampled_positions(w) + 1e-6
L_O,r = sum(w × forward_KL) / D_r
optimizer objective = (L_response_1 + L_response_2) / 2
```

The epsilon is added once per response, after chunk sums. Overlapping blocks
count repeated positions as exposures. This is neither a unique-token denominator
nor division by protected positions. BF16 head projection and BF16 softmax/KL
rows are followed by FP32 weighting/reduction. Accumulation averages independently
normalized response losses; clipping occurs once before each AdamW step.

Preserve that entire path. For P add `sum(w*g*softplus(-margin))/D_r`; for U add
`sum(w*rho*softplus(-margin))/D_r`, with lambda=1 and response-wide detached
`rho=sum(w*g)/sum(w)`. Do not insert epsilon into rho or normalize either loss
by `1+lambda`. Invalid/empty responses retain the existing stop/no-resampling
policy; define rho=0 when its weight sum is zero.

**Configuration discrepancy:** the source template `configs/osd.yaml` still has
training `max_length=2048`. The successful run actually used **2272**, alongside
the 2048 output cap. Use the copied [qualified run config](QUALIFIED_BASELINE_CONFIG.yaml),
not the stale template. Preserve constant LR 3e-4, BF16 AdamW, decay .01,
betas (.9,.999), epsilon 1e-8, clip 1, two shuffled passes, batch one/accumulation
two, seed 17, up to 512 anchors, decay gamma 7, Flex attention and eight-block
loss chunks. Native serving stays greedy b16, concurrency one, full context,
4096 serving capacity, and the existing cold-cache/publication policy.

## B. Snapshot and capacity

`FrozenReference` in `online_feedback/objective.py:48` already deep-copies the
unwrapped draft, freezes/evaluates it, and preserves RNG. Share only the frozen
target `nn.Embedding` and bias-free vocabulary `nn.Linear`, whose ownership is
checked by the wrapper/provider. Do not copy DDP, optimizer, or the target model.
The qualified NO_SHARD backend actually wraps in single-process DDP.

Each draft must recompute its own `fc`, `hidden_norm` and per-layer K/V from the
same detached raw target features. Training supplies no persistent KV cache.
The native serving fused-KV helper is separate and remains governed by publication;
it is not a reference-model cache. Compiled Flex code can be shared, tensor state
cannot. New no-grad compilation/workspace costs are unmeasured.

| Additional/reference item | Tensor-size estimate, not a measured peak |
|---|---:|
| Independent BF16 draft parameters | **1.953 GiB**, 58 tensors / 1,048,626,432 elements |
| Reference hidden output, 512 blocks × 16 × 4096 | 64 MiB |
| Eight-block BF16 vocabulary logits | 37.09 MiB per chunk |
| An FP32 vocabulary-sized chunk, if created | 74.19 MiB |
| Detached prefix lengths | At most 512 bytes as uint8, plus identity metadata |
| Existing raw auxiliary + final teacher features at length 2272 | 106.5 MiB per response; reused, not an added reference copy |

A whole-response vocabulary tensor would be 2.318 GiB and is unnecessary.
Frozen head and embedding each occupy about 1.159 GiB and should be shared.
Reference activations, masks, attention/MLP workspaces, compiler/allocator reserve
and current training peaks are not established by these sizes.

Live capacity inspection found both A800 GPUs idle: **81,153 MiB free per GPU**,
48.40 GB disk free; Fenrir has about 83.57 GB free. This is capacity before model
loading, not measured training headroom. The future layout should retain serving
on GPU0 and training/reference on GPU1. Memory fit must be measured in qualification;
an OOM must be reported before changing the proposed lazy-reference design.

## C. Smallest forward/loss change

The active OSD wrapper exposes full-vocabulary logits for **one eight-block
chunk** through ordinary differentiable PyTorch operations. There is no fused
loss preventing an exact rival gradient.

1. Add an opt-in prefix/uniform ranking option to the OSD wrapper and its
   objective/config/checkpoint contract. Ordinary O keeps its existing no-reference
   branch, loss, strategy, optimizer and cost.
2. Expose fixed-anchor block forwarding locally: sample once, then pass the same
   anchors/keep flags to reference and current forwards. Preserve author positions,
   masks and every masked slot. A small local wrapper/subclass can supply fixed
   anchors without modifying SpecForge's sampler or attention implementation.
3. Run the reference first under no-grad/RNG preservation. Chunk its BF16 head
   projection, reduce to greedy prefix lengths, and release its hidden/logit
   temporaries before building the current autograd graph. The existing projected-
   target path runs current first and retains reference hidden states through the
   checkpointed loss; it cannot be reused unchanged.
4. In each current chunk, exclude y from a detached copy when finding the exact
   strongest rival. Gather y/rival values from the original differentiable logits,
   cast those values to FP32 **before subtraction**, and use stable softplus.
   Preserve BF16 projection and original teacher/KL arithmetic. Pin first-index
   argmax tie behavior. No stored rival or top-k approximation.
5. Compute response-wide rho before the chunk reducer. Reference work must stay
   outside backward recomputation. Release the reference at event end after all
   backwards, then keep normal export/publication. Extend explicit objective
   allowlists and teacher-required strategy selection; do not reuse the unrelated
   projected-target loss.

This requires focused changes in `online_feedback/objective.py`, training/recipe
plumbing, tests, and a small comparison runner. No general framework rewrite is
needed. It has **not** been implemented or runtime-qualified.

## D. Feature reuse, state availability and cost

Reference/current can share raw detached features already on the training device;
no added target-model pass or mandatory raw-feature transfer is required by U/P.
Both recompute weight-dependent conditioning. The baseline already performs one
terminal-feature target pass per completed response; count it for all arms.

Separate snapshot setup, reference draft forward, reference head projection,
current forward/backward (including checkpointed vocabulary recomputation),
teacher preparation, feature I/O, saving/export and complete publication. Count
O's normal cost only; sharing preparation in a component test does not make its
cost disappear from a hypothetical separate deployment. Never add nested timers
again. Timed serving follows completed training, with no background trainer or
artifact transfers.

CPU mmap inspection confirms the final **6,309,079,707-byte** trainer state exists:
58 draft tensors and BF16 Adam slots, all optimizer counters 7424, 14,848 response
presentations, Python/NumPy/Torch/CUDA RNG, and horizon 7424. No scheduler state is
missing: this recipe uses constant LR without a scheduler. However, the existing
strict resume requires identical configuration, objective contract and horizon.
The state is exhausted at 7424/7424. Extending it or changing its loss requires an
explicit state-fork design; file availability does not make that an ordinary resume.

The original event-1 and event-58 feature/native tensors were retired under the
recorded policy. Their JSON receipts and exact serving records survive; neither
event has prepared `.pt` features remaining. Do not claim immediate tensor replay
from those receipts. This motivates fresh feedback capture for the proposed test.

## E. Correctness gate

Focused fixtures and bounded native qualification must check:

- Independent reference storage for all draft parameters/buffers, identical
  initial values, immutability after optimizer steps, fresh reference next event,
  and exclusion from optimizer/export. Preserve current module modes; reference
  `.eval()` also touches shared head/embedding flags, although these concrete
  Linear/Embedding modules are mode-insensitive.
- Same Python/NumPy/CPU/CUDA RNG and subsequent response-order/anchor draws with
  reference work enabled. Sample only once. Fixed seed alone is insufficient.
- Identical IDs, anchors, positions and full masks. Target context visibility is
  **strictly before the anchor**, and all same-block masked slots remain present.
  Perturb inaccessible future target features to check no leakage, with an earlier-
  feature positive control.
- Prefix calculation on offsets **1–15**, stopping at the first mismatch or
  invalid boundary. Include a matching EOS only if baseline-supervised; cap tails
  are boundaries, not invented errors. Seed, padding and post-EOS slots are never
  protected. Do not directly reuse `compute_accept_len`, which treats invalid slots
  as no-ops; use strict valid-and-matched prefix logic.
- Exact rival identities/ties; stable large/small margins; gradients into both
  selected logits; finite loss/parameter gradients; direct versus chunked formulas;
  unchanged denominator/response averaging; detached masks/rho; and per-presentation
  U/P coefficient mass at a shared start. Zero protected mass remains a valid
  observation, not grounds for resampling.
- Lambda=0 mathematical and numerical equivalence to unchanged O: losses,
  gradients, actual Adam displacement/moments/counters and subsequent RNG/sampling.
  Use existing qualified numerical tolerances; do not widen them after a failure.
  Also exercise lambda=1, since a zero-auxiliary check is insufficient.
- Compare same-weight current/reference logits in matched eval/no-grad modes with
  fixed anchors and preserved RNG; restore original per-module flags afterward.
  Do not change baseline training mode. Actual config has zero attention dropout,
  as required by Flex attention.
- Native publication/caches, exact canonical outputs and repeat-consistent counts.
  Preserve discrepancies. Training-context prefix labels are **not** native
  acceptance events; this screen makes no equivalence claim between them.

One inherited evaluation gate must change explicitly: the component-checkpoint
runner expects historical full-trained O counts. The proposed O is a new first-
event checkpoint, so that expectation is invalid. Keep historical **F** count
checks; qualify new O/U/P by output equality, publication and repeat determinism.

## F. One proposed first screen

**Initialization:** released DFlash and fresh identical BF16 AdamW/seed 17 for
each of O, U and P. This avoids an unreviewed adapted-state fork. Keep the original
constant-LR 7424-step horizon and all learning settings, but deliberately stop
each arm after **one full 128-response event**, not a reduced 4/8-response recipe.
This tests first-event utility, not the final benefit of 58 consecutive events.

**Feedback A:** the first 128 requests in the existing sealed GSM8K training
manifest, in its existing order. Generate fresh answers once using released F
and native capture; prepare one shared immutable feature pool. No dataset answers
are labels. These requests were used in earlier project experiments; the newly
initialized arms have not trained on their current feedback before collection.

**Evaluation B:** reuse the already sealed 64-request GSM8K development panel
from the checkpoint-component diagnostic. It was selected without using outcomes.
A/B have no exact prompt-token overlap (128/64 internally distinct); B never
trains any measured arm. Reuse is explicit, not fresh confirmation. Compare
**F, O, U, P** on identical exact B inputs with three rotating capture-disabled
repeats. Keep 2048-or-EOS, full histories/prompts and the qualified runtime/cache
policy. No extra generation warmups.

U/P each snapshot their own pre-event F. Identical feedback, order, anchors,
validity and weights should make their coefficient masses equal in this event;
verify that equality rather than assuming it. This shared-feedback comparison
does not establish a complete online deployment after divergent histories.

| Work | Proposed exact budget |
|---|---:|
| Fresh feedback generations | 128 |
| Baseline terminal-feature target forwards | 128, shared preparation |
| Main training | 3 arms × 256 response presentations = **768** |
| Main optimizer steps | 3 × 128 = **384** |
| Main reference draft forwards | U/P: 2 × 256 = **512** |
| Temporary qualification | O, U(0), P(0), U(1), P(1): two responses each = **10 current forward/backwards, 8 reference forwards, 5 temporary steps** |
| Matched-mode logit qualification | Two current/reference pairs = **4 detached draft forwards** |
| Timed evaluation generations | 64 × 4 × 3 = **768** |
| Included first-eight evaluation gate | **96** of those 768, not extra |
| Total response generations | **896** |
| Total optimizer steps | **389**, of which five are discarded qualification work |
| Total top-level training/qualification draft forwards | **1,302**, excluding serving and vocabulary/backward recomputation |

The temporary trials use the first two A responses, restore the complete same
initial state each time, and never become serving checkpoints. Deterministic
fixtures cover positive ranking gradients even if natural protected mass is zero;
record any unexercised native case without replacing requests. Restart clean
before each main arm. Every main response presentation retains at most 512 anchors
(at most 64 eight-block vocabulary chunks); report actual anchors, supervised
exposures, prefixes and coefficient masses, not these maxima as measured counts.
Native target verification counts depend on generation and are additional to the
128 terminal-feature calls. Vocabulary checkpoint recomputation is separately
counted/costed, not hidden by top-level forward counts.

Correctness precedes efficacy: after implementation fixtures and temporary GPU
checks pass, train the three main arms; qualify the first eight B requests, then
finish all remaining 56 even for null/harmful results. No outcome-dependent
resampling, retuning or favorable-subset reporting.

**Storage admission proposal:** a worst-case 128-response feature copy is 14.29 GB.
The old controller can temporarily retain native and prepared copies together.
Use sequential arms; offload verified native tensors after feature preparation,
and each completed arm's restart state before starting the next arm. Preserve
shared prepared features through all arms and keep all three exports for timing.
With the existing **8 GiB reserve**, 2 GiB work allowance and 8 GiB
compact-evidence allowance, estimated A800 peaks are **47.92 GB** during preparation
and **46.22 GB** during training, including reserve. A complete Fenrir backup of
both feature copies, three states/exports, compact evidence and 8 GiB reserve is
about **70.99 GB**. Current 48.40/83.57 GB capacities nominally support that staged plan,
but A800 admission has only about 0.49 GB above the budget including reserve;
this is not guaranteed future admission. Recheck capacity and stage more data if needed.
Without staging, retaining every copy on A800 can exceed capacity. Verify actual
destinations/bytes before launch; do not delete unique evidence to make it fit.
The old single-arm retirement helper must not retire the shared pool after O.

**Rough duration: 60–90 minutes, proposed two-hour bound**, not a measured candidate
runtime. The qualified first OSD event took 330 s training / 375 s complete update;
the last 768-generation four-checkpoint comparison took 33m41s including frequent
publication. Add two reference paths, qualification, capture, loading and staged
transfers. Candidate compilation/reference/backward overhead remains unknown.
Finish transfers and stop training before timing begins.

## G. Decision and required return from that screen

The candidate is implementable with the explicit sequencing, objective-plumbing,
configuration and evaluation-gate changes above. No source-level gradient-route
blocker was found. **GPU peak memory, same-weight reference numerics, lambda=0
update equivalence and native candidate behavior remain unverified.** The proposed
initialization/data/budget still require user/reviewer selection before execution.

Report calls per 1000 identical output tokens, complete request time, progress,
per-request improvements/regressions, all repeats, separately logged training KL
and auxiliary terms, protected exposure and all adaptation costs. These training
rows are not a fixed-context endpoint fitting comparison; different combined
objectives are not comparable as raw loss values. Use pooled sums and preserve
invalid outputs; do not reinterpret lower training loss as successful serving.

- **P improves over O and U:** evidence for useful prefix allocation in this
  first-event screen; inspect cost before claiming practical online benefit.
- **P beats U but loses to O:** no overall improvement over competent ordinary
  adaptation; report the trade-off.
- **U and P similarly beat O:** ranking pressure may help; prefix selection has
  no demonstrated added value.
- **Neither beats O:** no support for this candidate in this screen. Do not
  manufacture a mechanism conclusion, retune automatically, or generalize that
  result to all 58-event trajectories.

This objective supplies pressure on previously correct sampled prefixes; it
does not guarantee their preservation or future transfer. Its value is an open
experimental question. No novelty or successful-algorithm claim is made.

Evidence: [live artifact/state inspection](ARTIFACT_INSPECTION.json),
[source/capacity receipt](READ_ONLY_INSPECTION.json),
[machine-readable proposed budget](PROPOSED_BUDGET.json),
[qualified configuration](QUALIFIED_BASELINE_CONFIG.yaml), and
[successful-run report](../dflash_online_osd_rerun_20261023/RESULTS.md).


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_readiness_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
