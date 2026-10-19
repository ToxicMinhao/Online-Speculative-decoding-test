# Implementation contract

## Material passport

Algorithm: user-supplied v0.2 PDF, SHA256 recorded in README. Additional user
clarifications: exact Armijo rule; memory reservation must leave usable source
positions; unchanged inference/output validity; independent pre-read snapshots;
per-turn exposure and forecast reporting. This is a fidelity implementation,
not an expanded mechanism, extra baseline campaign or performance-tuning pass.

## Pinned numerical choices

- Four slots; module rank32; at most two active modules. Initial modules have
  zero-width factors representing the exact zero operator.
- FP32 master factors, stored full hidden features `h` and teacher moments
  `mu = E^T p`. No fixed output basis, random correction input map, clipping or
  universal correction radius. The separate prompt descriptor remains a
  fixed seed17 Gaussian orthonormal projection to64dimensions.
- Mathematical loss **changes** and gradients use FP64. Native teacher logits
  use the qualified BF16 shared head; probabilities and moment accumulation
  use FP64 before FP32 moment storage. The unchanged OSD baseline retains its
  native softmax arithmetic; the bank records the probability and mass difference
  from that native arithmetic. Qualification measures this storage
  discrepancy against uncompressed teacher distributions. Absolute KL is not
  reported without its teacher constant.
- Chunk32 rows and4096 vocabulary entries for mathematical calculations;
  no full FP32/FP64 copy of the vocabulary matrix merely to obtain moments.
- `Q=(S+M)/(trace(S+M)/d)+1e-3 I`; fixed for all attempts of one owner in an
  event. Memory covariance is balanced by response and weighted by immutable
  squared action coefficients. Multiple certificates do not duplicate data.
- Exact reduced-space, nonpivoted economic QR and SciPy LAPACK `gesvd`, in
  FP64. Square-root metric products use the covariance's low-rank factorization.
  Source gradients remain factored. Reduced problems can still reach full
  hidden dimension; the spectral work may be substantial and is timed.
- The proposed matrix is truncated to rank32 **on every backtrack**, then
  converted to the actual FP32 factors. Its numerical quadratic descent check
  uses those persisted factors. No interpolation and no zero-factor SGD.
- Armijo coefficient `1e-4`; metric epsilon `1e-3`; rho `0.25`.
  Numerical proposal/source tolerance `1e-10 + 1e-7*max(1,|terms|)`;
  SVD residual tolerance `1e-10 + 1e-8*max(1,|operand|)`;
  owner loss tie tolerance `1e-10 + 1e-7*max(1,|losses|)`;
  nonzero step threshold `1e-12`; positive KL-gain threshold `1e-10`. These are pinned numerical tolerances,
  not claimed exact-real-arithmetic guarantees.
- At most two accepted steps, eight attempts per step. Each step starts at
  twice the most recently accepted eta, or1 without one. Thus a fresh owner's
  first step starts at1; its second starts at twice its first accepted eta.
  Halve eta on rejection and recompute. Only the winning trial persists eta.
  An exhausted second step retains the first accepted state. Only a nonzero
  committed write allocates a slot.
- Choose minimum final singleton loss; ties prefer an occupied slot, then
  lower index. Read selection does not restrict candidate owners.

## Memory, protection and progress

Native memory blocks are selected by seed17 SHA256 of stream, turn and exact
output-relative block start, without inspecting gains. Candidate positions
exclude the target seed. Remove the union of reserved output positions from
every source anchor, including overlaps. Training uses a single fixed sampled
objective for all owners/attempts. Empty original objectives are explicit no-ops;
reservation is never allowed to create one from a nonempty objective.

Store original native frozen features and canonically conditioned teacher
moments, canonical IDs, block identity, response weights, prompt descriptor,
served coefficients and precision metadata. Native feature rows are joined to
the qualified verification trace, and their head predictions must reproduce the
actual served proposals. Post-rejection target vectors are not substituted for
canonical teacher conditioning. Memory keeps all nine original native head rows
even at EOS/cap, then selects eligible rows after the native head calculation;
changing the GEMM row count must not masquerade as a correction effect.

Before owner trials, assess the served action on newly retained blocks. Create
separate KL/prefix records only for demonstrated pre-write gains. Keep references
and cumulative allowances fixed. KL requires loss at most reference plus25% of
its gain. Prefix certificates protect every position through the improved
prefix, using all-vocabulary mathematical margins and native argmax. Positive
reference margins retain75%; nonpositive mathematical margins rely on native
checks without an invented positive floor. Bounds are scoped to retained
features and historical action coefficients, not future requests or routing.

Mathematical protection tolerance is `1e-8+1e-7*max(1,|terms|)`; direct
qualification additionally reports FP32-moment storage discrepancy. Native
prefix loss is a rejection even when the mathematical bound passes. A finite
search failure does not establish mathematical infeasibility.

## Current-version routing

Allocated slots define frozen, singleton, pair-mean and pair-sum actions (at
most17). Pair fusion is factor concatenation, never rank32 recompression.
After a write, refresh only actions whose module-version tuple changed; populate
new actions and new memory entries. A stale version cannot be read.

The prompt descriptor uses the last prompt token's normalized target feature,
with stable chunk length/prefix offsets captured before native prefill mutates
them. Reads use only prior memory. Gaussian kernel weights are response-balanced;
predicted gain has the specified additional unit zero-gain prior. No-memory
fallback is frozen; turn2 can use a newly learned module. Every coefficient
remains fixed during its response.

Use observed prefix progress and a qualified cycle-cost proxy. EOS/cap records
use clipped runtime progress rather than assuming one plus prefix length.
Qualification profiles controlled zero/rank32/rank64 native head paths: five
warmups and20CUDA-event measurements per path, with no new answers. Initialize
context bins8192/16384/32768 from the observed canonical cycle and measured
head increments; unmeasured bins explicitly use that initial profile. Update
bins only from subsequent canonical executions. Timing replays never update
this model. Profile/canonical instrumentation costs are disclosed; the proxy
is not a measured deployment speedup or detailed kernel attribution.

## Retention and qualification boundary

Qualification is exactly the first three WildChat00 requests plus deterministic
fixtures, then clean restart. Algebra, masks, canonical conditioning, source
compression, rank, fusion, cumulative protection, cache versions and output
preservation are correctness gates. Fit/gain measurements are descriptive,
including negative results. Extra learning-rate sweeps, ablations and dataset
expansion are excluded.

Peak storage includes one11-model-unit rolling OSD envelope, all six compact
bank envelopes (512MiB each),2GiB qualification evidence,2GiB staging and8GiB
free reserve. Final OSD retirement occurs only after valid paired results and
worker shutdown; failed episodes keep recovery state and stop. Native capture
has its own512MiB per-request guard and is transient except qualification.
No unverified offload or undeclared tensor deletion is assumed.
