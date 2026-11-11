# Successor-shuffle qualification evidence review

## Material Passport

- Mode: ARS 3.24.2 experiment-agent / validate.
- Verification status: **ANALYZED — compact receipts and scalar reconstruction passed**.
- Execution: `4d977e9ffd2cab9d58ae8806c29df11d843353da`, execution001; binding `d57554b8d8dd9a072b30b0e639cda1beba56f140cadca1165da17a3c97341eb7`.
- Evidence: 39 local JSON/JSONL files, 4,160,565 bytes. Input hashes and exact command are in [QUALIFICATION_REVIEW_FINAL.json](REVIEW.json); the earlier scalar review remains preserved separately.
- Review scope: same-team independent arithmetic with implementation-author overlap. No remote access, model work, or rereading `.pt` endpoints, gradients, weights or Adam tensors. Full serving analysis remains pending.

The completed qualification supports the intended identity control and confirms that this two-response permutation changes learning. It provides no held-out serving result.

## Identity comparison

All three cases restored step512/presentation1024 with recorded exact resume, no RNG reset, and the same three packet identities. The first two packets are `train_00512` and `train_00513`; the third supplies a sampler-only next-anchor check. This is a fresh native TRUE reference on the same qualification pair, not an original event-five loss from another presentation order.

IDENTITY and TRUE report identical full and compact endpoint file hashes. Their exact comparison covers **351 tensor leaves / 5,243,137,914 coordinates**: 58 normalized pre-clip gradients, 58 gradients at Adam entry, 58 weight tensors, 174 optimizer tensors, and three RNG tensors. It records zero tensor and zero metadata differences. The producer also compares response losses, anchor observations, counters, accumulation state and the next-anchor draw. Both native response losses are exactly `0.35443469882011414` and `0.19132587313652039` in the local observations. Startup restoration comparisons pass in all cases.

Tensor equality and next-anchor parity are supported by the recorded native comparisons; this reviewer did not reopen the remote tensors. The reported pre-clip norm is 0.5703125, so this window does not exercise an active norm-one clipping threshold.

## Same-state shuffle and actuation

The local IDENTITY and SHUFFLED snapshots are exactly equal for anchors, canonical labels, logged weights, selected masks, first-error masks, eligibility, true successor bits and EOS IDs. Only assigned bits differ. Independently rebuilding the prescribed SHA-256 donor order reproduces every assignment, class count and changed bit. Selected prefixes and successor eligibility were checked from the compact masks, labels and weights; the native true successor bits themselves were not regenerated from logits/proposals.

| Response | Eligible errors | Ineligible errors | Mixed strata | Changed bits | Uniform-permutation benchmark |
|---|---:|---:|---:|---:|---:|
| train_00512 | 100 | 2 | 13 | 34 | 36.661111 |
| train_00513 | 194 | 9 | 14 | 72 | 75.480302 |
| Total | **294** | **11** | **27** | **106** | **112.141413** |

All 27 mixed strata span projection chunks. One valid canonical EOS successor is eligible. This native window has one single-class stratum, no singleton stratum, and no first error on a terminal EOS token; those absent cases must not be described as covered by this native window.

The 106 changes conserve per-position class counts. Logged native class masses, coefficients and error budgets pass their prescribed FP32 tolerance. Independently summing logged FP32 weights in Python FP64 also matches within that tolerance; this is not a bitwise reconstruction of the native reduction. Native recorded reassigned weight sums to **10.806288242340088**. The uniform benchmark is descriptive, not a test over randomized permutations.

SHUFFLED's `EXACT.json` intentionally covers the compact control fields and three RNG tensors. Its separate full `NATIVE_DIFFERENCE.json` reports changed losses, all 58 pre-clip and all 58 Adam-entry gradient tensors, 39 weight tensors and 116 optimizer tensors, without nonfinite differences. Its two unpermuted same-state losses exactly equal native TRUE; assigned losses are 0.3500054180622101 and 0.19107738137245178. Lower fitted losses are actuation evidence, not efficacy.

## Work and costs

Raw started/completed work ledgers reconcile three disposable Adam steps, six ordinary response forwards/backwards, three sampler-only next-anchor draws and zero unknown work. Separate discarded startup adds six forwards/backwards and zero Adam steps: **12 response forwards and 12 backwards total**. Qualification states are recorded as excluded from main training.

IDENTITY and SHUFFLED each add 47 detached vocabulary projections, **94 total**, matching their physical head-call excess over TRUE. The SHUFFLED projection and assignment clocks are 0.296113s and 0.058982s. The three sequential case-wrapper walls sum to **432.147379s (7.202456min)**. Startup's 58.175848s, projections, assignment and worker clocks are nested; they must not be added again. Maximum recorded reserved GPU memory is 17.677734375GiB. These qualification costs are separate from the two main 128-step arms and fresh serving.

## Interpretation checks

All 11 ARS fallacy categories were considered for this bounded evidence review. Simpson's paradox: response-level changes and pooled totals are both retained. Ecological fallacy: no per-request efficacy is inferred from aggregate actuation. Berkson/selection and collider bias: the fixed qualification pair is not treated as a representative held-out panel or selected by a favorable endpoint. Base-rate neglect: eligible, ineligible and mixed-stratum denominators are explicit. Regression to the mean: this is a same-state comparison, not a selected pre/post benefit. Survivorship: all three cases and their startup/work receipts are included. Look-elsewhere and forking paths: no new permutation, seed, window or analysis was selected. Causal attribution: changed assignments and updates are demonstrated within this window, but no serving mechanism or benefit follows. Reverse causality: qualification precedes serving and is not explained using later outcomes.

No qualification blocker was found within the compact-evidence scope. Final main-training manipulation, complete serving validity, paired intervals and the prespecified decision remain separate. Stage 1 / project None and the authorized stopping rule are unchanged.
