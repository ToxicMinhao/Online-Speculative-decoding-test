# Point3 cumulative-maximum assessment — implementation and storage admission

**Status: blocked before GPU qualification. No scientific comparison has run.** The full comparison cannot currently retain both new recoveries and exports while maintaining the required A800 disk reserve. Under handoff §8, implementation and CPU checks are delivered, and GPU work remains unstarted. **Keep native C+K+PAIR as the working configuration; no SUM/MAX benefit or final adoption is established.**

Implementation revision: `2e2cb39222ff54f0de5f351b2ecb530d1e0a7f42`. Base: PR62 publication `a1558c0c78c543fb257d1e21889b333be1675eec`. Native execution commit: **none**. ARS experiment-agent run/validate workflow; independent mathematical, native-integration, serving and resource reviews. Graduate and prior experiments remain unchanged.

## Scope implemented

Only the outer preservation assessment changes. Both new arms use the actual normalized native C preclip gradient, the unchanged FP32 optimizer sensitivity, beta-centered allocator and native C+K+PAIR teaching. Native BF16 parameters/head/forward KL/Adam, FP32 PAIR, learning rate3e-4, decay0.01, clipping1, two-response accumulation, gamma7, eight-block head chunks, full attention, horizon7424 and original sampling remain fixed.

The no-step C-reference path performs ordinary native forward/backward, clears reference gradients, and restores RNG, runtime and observation state without an Adam call or history/counter commit. Qualification will compare this route bitwise with the gradient captured during the planned disposable C step. SUM must reproduce PR62 F_C's coefficients and complete scientific identity. Those GPU checks have **not** passed yet.

MAX uses each block's earliest smallest native margin in every assessed prefix. Selection is detached; gathered margins remain differentiable. It excludes the seed and invalid positions, respects canonical EOS and fixed origin K, and applies no triangular multiplier to the cumulative maximum. Empty sets retain graph-connected zero. Readouts independently retain SUM and MAX from the same native margins, with native FP32 and matched-contribution FP64 reductions.

The event wrappers retain explicit `F_C_SUM` and `F_C_MAX` identities, exact warm restoration and normal anchor/RNG progression checked against retained C. Each commits128 updates and256 presentations; reference work is counted separately. Both complete recovery states and exports remain mandatory. The serving adapter genuinely executes three arms, rather than duplicating a two-arm label. Its bootstrap resamples whole requests jointly across arms; timing repeats are averaged first. Freeze decisions use actual exported tensor contents, so safetensors header/metadata differences cannot manufacture a distinct model.

## Mathematical note and limits

For fixed K, let `phi(m)=softplus(-m)/log(2)`. Replace

`R_SUM = sum_j (K-j+1) phi(m_j)`

with

`R_MAX = sum_k max_(j<=k) phi(m_j)`.

For greedy canonical matching prefix A, `D=K-min(A,K)` counts the prefixes containing an error. A wrong native argmax has canonical-minus-best-wrong margin at most zero, including a wrong-at-tie decision, so its phi is at least1. Therefore **D≤R_MAX≤R_SUM**. The upper inequality follows from nonnegative terms. On a fixed selected branch, the derivative at position j is `-c_j sigmoid(-m_j)/log(2)`, where c_j is its bottleneck multiplicity and `sum c_j=K`.

This is a pointwise, fixed-K greedy shortfall bound. It establishes neither better gradients nor stochastic preservation, transfer or serving benefit. A worst surrogate obstruction need not be the first actual error. If the first error remains worst, matching followers can increase its multiplicity while receiving no direct MAX gradient. Sparse diagnostics explicitly measure that behavior. The fact that MAX is below SUM at one endpoint is structural, not an improvement result.

## Validation completed and pending

**291 CPU tests pass** in90.339seconds; CUDA remained uninitialized. This combines the preservation package, legacy two-arm serving/analysis, Point1/Point2A readouts, native PAIR objective checks and the new Point3 tests. `git diff --check` passes. [Test receipt](CPU_TESTS.json).

Tests cover first/last/multiple/no errors, empty blocks, EOS/gaps, deterministic rival and prefix ties, fixed-branch FP64 finite differences, multiplicities, full-block/chunk normalization, unchanged direct-loss support, native-reference isolation using small CPU fixtures, warm-event identities, continuous scheduling, unknown-work reservations, failure/clock preservation, three-arm balance, invalid-record exclusion, paired bootstrap and the exact development-selection rule. CPU tests do not establish native GPU parity, production memory fit or endpoint output equality.

Independent source review corrected explicit historical coefficient authentication, resource peak/reserve reconciliation, failure-saving order, stable sparse-score diagnostics, tensor-content endpoint identity, and complete warmup/group reconciliation. Scientific algorithms and inherited GPU numerical tolerances were not changed. The historical 2.047197% versus1.6% hidden-gradient failure remains a failed diagnostic; it is not relabeled a pass.

Pending GPU gates: three independently restored C/SUM/MAX steps, exact C-reference and SUM historical parity,28 fixed serial MAX token-gradient backwards and repeated fast scoring, full restoration, both complete events, matched TRAIN readouts, and all native serving checks. No qualification-only substitute was run to bypass the full-plan storage gate.

## Live resource admission

All units below are decimal GB. The two permitted A800 storage roots resolve to the same `/dev/sdb2` filesystem. Both physical80GiB A800 cards are idle; available host RAM is103.889GB. The inherited55GB workspace plus32GB host reserve and60GB free-training-VRAM requirement pass. The source-only resource audit used no model or optimizer.

| Item | Bytes / result |
|---|---:|
| A800 free space at admission |407,182,004,224|
| Required original-filesystem reserve |400,000,000,000|
| Available headroom |7,182,004,224|
| Two endpoint pairs at retained observed sizes |16,796,734,838|
| Endpoint-only capacity deficit |9,614,730,614|
| Conservative entire remote peak:17GB endpoints +3GB source/log/cache allowance |20,000,000,000|
| Additional headroom needed for that plan |**12,817,995,776**|
| Local delivery/source allowance |600,000,000|
| Whole-study peak budget |20,600,000,000 /25,000,000,000 cap|
| Fenrir free space |74,588,794,880|
| Declared Fenrir safety allowance |20,000,000,000|

The regular publisher temporarily retains a recovery plus its export: conservatively8.5GB even for one endpoint, exceeding the available7.182GB before source/log growth. An atomic pending-recovery rename does not create an extra recovery copy; no such artificial duplicate was charged. No distinct permitted durable A800 destination was found. Unapproved overlay storage and RAM/tmpfs are not substitutes.

An existing Fenrir retrieval route was measured with a16MiB prefix of the retained parent:64.547656s end-to-end, about0.260MB/s. Proportional extrapolation for even the minimum9.615GB offload is10.28hours, exceeding the six-hour phase limit. **This small sample is a planning estimate, not a proven lower bound or bulk-transfer guarantee**; it does not establish that a future network session cannot be faster. Regardless, the normal A800 single-endpoint publication peak already fails admission. No new streaming persistence architecture was substituted for the qualified publisher.

The16MiB probe copy was verified against the unchanged source prefix and removed. No inherited checkpoint, feature, recovery, export, source, cache or evidence was deleted or relocated. Exact private inventory and retirement paths remain outside Git. [Resource table](RESOURCES.json) and [independent ten-check audit](INDEPENDENT_RESOURCE_REVIEW.json).

## Planned work, duration and actual work

| Work | Fixed plan | Actual |
|---|---:|---:|
| Qualification Adam calls |3; maximum6 across one identified correction retry|0|
| MAX serial token-gradient backwards |28; maximum60 across permitted attempts|0|
| New continuation events |2|0|
| Committed presentations / Adam updates |512 /256|0 /0|
| Final complete recoveries / exports |2 /2|0 /0|
| Endpoint TRAIN witness response forwards |64 across parent/C/SUM/MAX|0|
| Scored serving generations |1,152|0|
| Included warmup generations |36|0|
| New occupied card-seconds |maximum14,400|0|

Each new event starts independently from the exact PAIR1 step128 state, reuses the immutable PR60 train[128:256] feedback/features, and ends at256 steps/512 presentations/event2. Retained C is authenticated and reused; C is not retrained. There is no new feedback or canonical generation. All1,188 serving executions are unattempted, not invalid or unknown. The phase clock has not started; prior preservation-development compute remains5,790.93129579185card-seconds, not the entire project's historical total.

Once storage is admitted, the prospective compute schedule is about **2h55m raw or3h38m with25% contingency**, including qualification, two serial events, witness/readout/persistence costs and three-arm serving. This is based on PR60's40.66-minute candidate event, with a35% allowance for the new reference route; Point3 runtime remains unmeasured. Hard limits stay six elapsed hours and four occupied card-hours. There is no current completion time while storage is blocked. No relaxed reserve, omitted recovery, inherited deletion or full-event retry is authorized.

## Remaining execution and freeze status

Use the published CPU-only prepare and guarded launch entries described in [COMMANDS.md](COMMANDS.md). Authentication resolves private paths from the original receipts and rechecks full inputs/states before any future model action. Native source staging, full fresh tensor authentication and GPU launch are unperformed; this is a prospective contract, not an admitted native plan.

The intended report contrasts MAX−SUM first, then SUM−C and MAX−C. Eligibility requires a complete valid comparison, distinct serving parameters, unrounded summed mean latency below C and calls no worse than C. Among eligible candidates select lower latency, then fewer calls, then SUM. Identical SUM/MAX serving models use SUM evidence. A selected latency interval crossing zero remains provisional and does not establish speedup, equivalence or absence of degradation.

**Current §9 outcome: incomplete because of storage; retain C+K+PAIR as the working configuration.** No assessment candidate is adopted, Point2B remains unexecuted, and no broader mechanism search or additional run follows automatically. This readiness result says nothing about whether MAX improves drafting.
