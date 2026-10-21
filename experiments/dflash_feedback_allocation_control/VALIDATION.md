# P8 implementation validation — 2026-10-21

Status: implemented and CPU-validated for review. **No native GPU qualification,
training campaign, new serving measurement, A800 operation, or previous-campaign
restart was performed.** These are correctness checks, not scientific results.

## CPU checks

**736 passed in 37.59 seconds**, with `CUDA_VISIBLE_DEVICES=''`:

```bash
python -m pytest -q \
  experiments/dflash_baseline_learning_curve \
  experiments/dflash_kl_native_diagnostic \
  experiments/dflash_lower_lr_control \
  experiments/dflash_matched_prefix_diagnostic \
  experiments/dflash_feedback_allocation_control
```

This includes **268 new P8 checks** and **468 inherited regression checks**. The
existing local test environment used Torch 2.11.0+cu130 and pytest 8.3.5; GPU access
was disabled. This is not a substitute for the pinned A800 runtime qualification.
No dependency or historical implementation was changed.

Coverage includes:

- Three isolated pools, exact A/B ordering, frozen source-only feedback, original
  group/role/checkpoint bindings, cross-group local-ID collisions, mutation and
  held-out leakage rejection.
- Real small-tensor FP32 and BF16 training compared with manual independently
  normalized loss averaging, one aggregate clip and fresh AdamW. Exactly eight
  steps and sixteen presentations, continued normal sampling, no old optimizer
  state or fabricated intermediate diagnostics.
- Actual post-training source/held-out diagnostic receipts passed through the
  unpatched controller validator after JSON transport. Exact original anchor
  seeds, masks, weights, positions, denominators, zero-supervision handling,
  `eval`/`no_grad`, RNG/mode/gradient-flag preservation. Native rows in these
  integration tests are explicitly synthetic; no model outputs are claimed.
- Native wrapper admission of a real P8 identity, original request ownership,
  training-process quiescence, capture-disabled generation, unchanged context
  envelope/cache behavior, and complete activation accounting. These use stubs
  and do not qualify actual GPU kernels or outputs.
- Durable invalid measurements; partial backward, uncertain optimizer steps,
  uncertain KL/native work, and unattempted-work accounting. Reports revalidate
  every pair and its actual pool checkpoint, exclude invalid gains, and preserve
  zero-supervision cases. Null/harmful synthetic results continue through all
  three pools when correctness passes.
- Source sealing, explicit launch admission, evidence/destination isolation,
  distinct/shared filesystem capacity accounting, reserve enforcement and
  durable capacity failure before model work.

Integration review caught and fixed a seed guard that would have rejected a
normally advanced sampler, incomplete unknown-work accounting, and a report
checkpoint-owner gap. The historical four-response guards remain unchanged.

## Retained-artifact admission (CPU only)

The real retained PR9/PR10 compact evidence was loaded through the unchanged
qualified readers. All three pool bindings and 48 original request comparisons
validate. Both source and held-out anchor contracts were checked before any
optimizer creation. No questions, answers, anchors or historical measurements
were regenerated.

- Original manifest: `1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d`.
- Order: `p8_gsm8k`, `p8_codealpaca`, `p8_sharegpt`.
- Sealed local plan: `dc3aa96c06bfbbe24bd5acaaa38bb17a7cd68a972ff1f274be20369b965e9b09`.
- Implementation/source digest: `bc8c04f48ad9d478f30f1413ad9da7f72ae9a7be3701ad502771a0d8bb7118d8`.
- Scope: **24 optimizer steps, 48 presentations, 48 native generations,
  48 response-level KL evaluations; zero explicit warmups**.

The admission plan is private because its bindings include artifact paths. Its
source manifest records the implementation state. Execution requires the clean
reviewed code digest and a sealed plan for the actual verified artifact paths.

## Revised storage estimate

The earlier approximately 18.05 GB estimate included only 256 MiB of native
evidence per pool and omitted sufficient allowance for repeated full-pool
configurations and controller copies. The actual nested pool configurations are
45,255,678 / 83,798,359 / 89,380,553 bytes. Three copies alone leave less than
0.3 MB of the ShareGPT allowance for its training, KL, generation and result data.

The new artifact-derived budget counts those copies, dynamic evidence, three
final BF16 checkpoints, one checkpoint staging overlap, controller results and
atomic replacement, and one controller retrieval-staging copy. No compression
savings are assumed.

| Destination arrangement | Free capacity required, including reserve |
|---|---:|
| Same filesystem | 25,829,469,994 bytes (25.83 GB) |
| Native filesystem, when separate | 19,513,298,010 bytes (19.51 GB) |
| Controller filesystem, when separate | 14,906,106,576 bytes (14.91 GB) |

The reserve is 8 GiB per distinct filesystem. These are calculated requirements,
**not current A800 free-space readings**. Before training, preflight verifies both
actual destinations and sums writes when they share a filesystem. It does not
remove original evidence. Additional simultaneous archives need a new budget.

## Remaining execution checks

After code review and storage admission, the first complete dataset pool must
qualify native integration, memory fit, exact output/termination preservation,
training accounting and diagnostic equivalence. Its work is inside the fixed
budget. On correctness, finish the other two without retuning, regardless of
benefit or harm. No source, serving or transfer benefit has yet been measured for
P8. Historical F/S8 versus fresh P8 timings cannot establish speedup.
