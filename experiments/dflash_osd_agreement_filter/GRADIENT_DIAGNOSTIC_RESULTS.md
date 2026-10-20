# Failed G gradient-sum qualification: bounded diagnostic

**The original failure reproduced.** The evidence supports finite-precision
non-additivity of separately evaluated BF16 backwards. Repeated backward and
backward ordering did not change any native gradient; the equivalent eager
attention path also failed the same sum tolerance. No objective/grouping
implementation discrepancy was found in the inspected code. The exact operation
where the discrepancy first arises remains unlocalized.

This diagnostic does **not** qualify G or measure whether filtering helps.
The full qualification and main campaign remain paused. No repair was applied.

## Scope and reproduction

- Base: PR8 `4a2e533`; executed diagnostic code: `85d98dc26f7a1e1ce76abe47427d789f933bc7a9`.
- Reused the retained first G response, exact released model, fresh optimizer,
  and saved Python/NumPy/Torch/CUDA/anchor RNG state. No answer generation,
  H/L history regeneration, serving benchmarks or checkpoint publication.
- Preserved BF16, full loss, group membership, shared denominator, anchor sampler,
  clipping/AdamW/LR, all qualification tolerances and `donated_buffer=False`.
- Preserved the original ordinary-reference warm-up and restoration sequence:
  **two temporary ordinary optimizer steps, zero grouped steps**, zero persistent
  updates. Complete model/optimizer/RNG/state hashes matched after reproduction
  and at final restoration; every diagnostic variant restored the failed-pass state.
- Failure: **event 1, pass index 0 (first pass), disabled-filter G trial**.
  First failing tensor in parameter order: `layers.0.self_attn.q_proj.weight`.
- The original per-pass anchor list was not written before the historical crash.
  The draw was reconstructed from the verified retained RNG; it is not claimed
  to have been compared with a nonexistent historical anchor hash. Every new
  variant reproduced exactly the same sampling, position weights and group metadata.
- Frozen teacher features were reconstructed from the saved input/output IDs,
  without regenerating answers. The case has 69 input and 442 output tokens.

The unchanged check is:

```text
abs(float32(gA) + float32(gB) - float32(gFull))
    <= 8 * eps(parameter_dtype) * (abs(gA) + abs(gB) + abs(gFull)) + 1e-8
```

Raw gradients come from native BF16 differentiation; later FP32 promotion does
not recover precision already lost. Diagnostics are durably written before the
original exception, with all 58 parameter summaries and 32 bounded failing-coordinate
samples. Nonfinite values have explicit representations; none occurred here.
Full model-sized gradient tensors were held only in memory, never dumped to disk.

## Error magnitudes

| Quantity | Original compiled A/B/full | Equivalent eager A/B/full |
|---|---:|---:|
| Checked coordinates | 1,048,626,432 | 1,048,626,432 |
| Coordinates beyond unchanged tolerance | 120,111 (0.011454%) | 133,634 (0.012744%) |
| Tensors containing violations | 31 / 58 | 32 / 58 |
| Maximum absolute discrepancy, over all coordinates | 0.0001831055 | 0.0003967285 |
| Maximum error / tolerance | 15.06498 | 15.04946 |
| Full-gradient L2 norm | 1.19870736 | 1.19681664 |
| Sum-discrepancy L2 norm | 0.004121512 | 0.004220428 |
| Discrepancy / full-gradient norm | 0.343830% | 0.352638% |

Original norms: `||gA||=0.94883746`, `||gB||=0.68139319`,
`||gA+gB||=1.19870252`. Casting the sum back to native BF16 gives a
0.382840% relative discrepancy; it does not eliminate the issue.
The maximum absolute discrepancy above is **not** necessarily a rejected
coordinate: the original maximum occurs in a layer-normalization tensor with
zero tolerance violations.

| Original violation location | Coordinates | Share of violations |
|---|---:|---:|
| `fc.weight` | 97,990 | 81.58% |
| Five key-projection weights | 19,104 | 15.91% |
| Five value-projection weights | 2,637 | 2.20% |
| Remaining tensors | 380 | 0.32% |

Among the first 32 failing coordinates in deterministic parameter/index order,
23 have opposite A/B signs and 3 change sign between the sum and full gradient.
This bounded sample is neither random nor worst-case and cannot estimate the
population frequency of cancellation or sign changes.

## Isolation comparisons

| Comparison | Result |
|---|---|
| Full backward twice on one retained graph | Bit-identical across all coordinates |
| Both repeated full gradients versus original full gradient | Bit-identical |
| Full/A/B versus original A/B/full | A, B and full each bit-identical; same 120,111 violations |
| Equivalent eager FlexAttention, A/B/full | Executes successfully; same tolerance still fails |

Eager replaces only the Transformers compile wrapper with Torch's direct
FlexAttention eager backend. The same BlockMask, GQA, scale, BF16 inputs, weights,
features, anchors and stochastic state are retained; no SDPA substitution or
backward-unsafe debug flag. All five attention layers reached this path.
Its different reduction kernels are not a higher-precision oracle: its full
gradient differs from the compiled full gradient by 1.6211% in relative L2 norm.
Thus the experiment shows that compilation-specific failure is **not required**
to obtain the sum discrepancy; it does not establish identical arithmetic
across backends or rule out every shared implementation defect.

The code audit found no missing/doubled positions, alternative normalization or
teacher-row mismatch. The first pass has 441 sampled anchors,440 nonempty blocks,
3,924 supervised occurrences across441 unique positions, partitioned 1,935+1,989.
All three losses use denominator 2,374.79052734375.

| Backend | Full loss | A loss | B loss |
|---|---:|---:|---:|
| Compiled | 1.1728844643 | 0.7103841305 | 0.4625004828 |
| Eager | 1.1724951267 | 0.7098924518 | 0.4626027346 |

The compiled scalar sum differs from full by 1.4901e-7 and passes the existing
scalar-loss check. Scalar agreement does not imply coordinatewise BF16 gradient
agreement. The final-gradient-based tolerance is an admission criterion, not a
proved bound on rounding throughout all intermediate network reductions.

One reporting inconsistency remains in the inherited `AUTHOR_RUNTIME.json` recipe
summary: it shows older response/step counts. The actual saved trainer state and
ordinary-reference receipt confirm the authorized **one response and two steps**.
This is stale summary metadata, not observed execution drift or an explanation
for the gradient failure. An independent evidence audit passed **43/43 checks**.

## Supported explanation and proposed repair

**Supported:** separately rounding the A and B backward paths can produce a sum
that differs from one full-loss backward. The discrepancy is small globally but
violates the declared coordinatewise gate. Reuse/order corruption is not observed in this case;
a compilation-only explanation is contradicted by the eager failure. The exact
first rounding/reduction operation and the effects on actual filtered updates
remain unresolved. No learning-quality claim follows from this failed gate.

**One proposed repair, not implemented:** construct the ordinary AdamW proposal
from a direct full-loss gradient, retaining raw A/B gradients only for G's sign
filter and N's own hypothetical filter. Requalify the disabled-filter path against
the unchanged ordinary OSD reference. Retain gradient-sum differences as diagnostics
rather than substituting their sum for the ordinary proposal.

This **changes the numerical update construction and qualification protocol**;
it is not a tolerance relaxation or logging-only fix. It also adds a third
backward to production grouped updates, whose cost must be charged. It requires
review before implementation or execution. No precision, learning-rate or
acceptance-threshold change is proposed here.

## Cost, evidence and stopping state

- Diagnostic wall time, including model setup, state hashing/copies and CPU
  gradient statistics: **495.98s (8.27min)**. This is not serving/training throughput.
- Original warm-up/reproduction: 34.41s; repeated-full probe: 92.23s;
  reordered probe: 159.47s; eager probe: 149.10s. Probe times include gradient
  transfer/statistics; differences do not measure attention-kernel speed.
- Observed PyTorch peak allocated/reserved: **45.04 / 47.09GiB**;
  process peak host RSS: **30.14GiB**. These are allocator/process measures,
  not a complete breakdown of all driver allocations.
- **131 local CPU tests and 32 focused CPU tests under A800 Torch2.9.1 passed.**
  Diagnostic process exited 0; every variant completed and exact final restoration
  passed. No new serving checkpoint was exported. The native constructor wrote
  its small initial RNG/empty-optimizer receipt, not learned model weights.
- Raw diagnostics, parameter tables, sampled-anchor/RNG metadata and private
  runtime records remain outside Git, preserved on A800 and Fenrir.
  Verified compact evidence archive SHA256:
  `bd358dff981a4c62a9a89741511e76c73eefa5add483aed9346bbd97b6c8e834`.
- Both A800 GPUs are idle. Remaining A800 storage: 41,706,151,936bytes;
  no cleanup/deletion was needed. The 8GiB reserve was preserved.

**Stop here. Full qualification and the main campaign remain paused.**
