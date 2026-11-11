# Successor-shuffle implementation and native qualification

## Material Passport

ARS-Codex 3.24.2, experiment-agent. Implementation and bounded native qualification are complete. Endpoint results will be reported separately. The executed code is commit `4d977e9ffd2cab9d58ae8806c29df11d843353da` (PR #39); this document is a later publication artifact describing that execution.

## Scientific contract

The study forks the complete original PR #32 KPLUS4 state at Adam step 512, including its weights, BF16 Adam moments, counters and RNG state. It reuses PR #37's 128 canonical feedback responses, teacher packets, two-pass order and anchor draws. Ordinary K receives no redistribution. TRUE is the authenticated native PR #37 KPLUS5 export at step 640, with required weight hash `9dcb0d637eabef6bd4e34c552da0448faa9510e6bcaf5c753a6bd1b43674b795`. K and SHUFFLED each perform 128 new steps across 256 response presentations, continuing from the complete parent state.

SHUFFLED changes only the detached successor-priority assignments, separately within each response and first-error proposal position. Recipients are ordered by canonical anchor offset. Donors are ordered by the SHA-256 hash of compact UTF-8 JSON `[20261111,response_id,position,anchor_offset]`, with offset as the tie-break. This mapping uses no training RNG. Unavailable or post-terminal successors remain low priority; canonical EOS successors are eligible. Fixed points are allowed. Single-class strata retain their bit values, although the donor map still depends on group membership. The assignment pools all sampled blocks across the fixed eight-block head chunks.

A detached student-head prepass uses the current hidden states already computed by the drafter. Immutable assignment tensors are passed explicitly into checkpoint recomputation. Native BF16 KL, the FP32 response-wide reduction, masks, denominator and redistribution formula remain unchanged. The added work consists of head projections and priority assignment; it requires no second backbone or teacher forward. Those costs are recorded separately.

## Checks before training

- All 63 new CPU tests pass locally and on A800; 50 inherited K+ regression tests pass locally. The independent analysis has separate preparation tests, which do not establish native correctness.
- Source staging verified 3,799 files and one symlink against a clean checkout. The plan binding for the actual execution paths is `d57554b8d8dd9a072b30b0e639cda1beba56f140cadca1165da17a3c97341eb7`.
- Storage admission compared a planned peak requirement of 103,198,974,562 bytes, including the reserve, with 714,132,443,136 bytes free. Both physical A800s were idle. Qualification and training use physical GPU 1. Serving uses GPU 0 after the training processes finish, with checks that both cards are quiet.
- The 256-request panel was selected by input hash from the 295 remaining PR #38-eligible rows. All 256 have recorded exposure in older projects. The plan reuses 221 canonical references and prepares the 35 missing references separately. The 64 matched starts were selected by hash before outcomes.

## Native qualification

All three prespecified disposable two-response cases completed. Each includes a separate startup procedure whose state is restored before the qualification window. The resulting qualification states are retained as evidence and are not used to initialize main training.

| Case | Outcome | Temporary Adam steps | Case wall seconds | Peak allocated bytes |
|---|---|---:|---:|---:|
| TRUE | Native reference completed with finite values | 1 | 116.865194 | 15,649,520,640 |
| IDENTITY | Exact equality of native loss, gradients, weights, Adam, RNG and next anchor draw | 1 | 143.682950 | 15,649,619,968 |
| SHUFFLED | Same-state manipulation and sampling checks passed; changed disposable update retained | 1 | 152.390037 | 15,649,619,968 |

The identity comparison covers 351 tensor leaves and 5,243,137,914 coordinates, including unclipped and clipped gradients, weights, Adam and RNG. It found no tensor or metadata differences. SHUFFLED's gradients and weights are allowed to differ from TRUE's. Its compact comparison checks sampling, RNG and counters; its saved masks are independently reconstructed on the same state.

The two real-shuffle responses contain 100 and 194 eligible errors, 13 and 14 mixed strata, and 34 and 72 flipped bits: 106 flips in total. Their uniform-permutation benchmarks are 36.661111 and 75.480302 expected flips, totaling 112.141413. Reassigned first-error weights are 3.712706 and 7.093583, totaling 10.806288 when summed before display rounding. Checks of class counts, the selected prefix, denominator, and native FP32 mass and coefficients all passed.

Total qualification work was 3 temporary Adam steps, 6 qualification response forwards and backwards, 6 additional disposable startup forwards and backwards with 0 Adam steps, and 3 sampler-only next-draw checks under RNG restoration. Actual head projections, chunks and recomputation are recorded separately. The case wall times include startup and persistence work, so those nested components must not be added again. These times describe qualification cost, not serving performance.

The first local aggregate CPU invocation allowed the inherited CUDA-RNG test helpers to initialize CUDA, causing six no-CUDA teardown assertions to fail. The CPU checks passed with `CUDA_VISIBLE_DEVICES=''`. This changed neither native model execution nor numerical tolerances. The failed CPU log is retained privately. No native qualification retry was required.

## Evidence limits

Qualification establishes that the assignment control, state preservation and work accounting satisfy their contracts on this bounded fixture. Useful learning and serving benefit remain untested here. The primary endpoint compares fresh pooled verification-call counts for TRUE and SHUFFLED; ordinary K provides a separate control for redistribution. All three arms use the same prespecified panel. Endpoint interpretation must report uncertainty and remain bounded to one continuing training seed and one permutation.
