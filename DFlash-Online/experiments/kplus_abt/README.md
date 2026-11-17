# K+ adversarial block training: fixed four-arm pilot

This is a two-view, single-ascent adaptation of [FreeLB's embedding training](https://github.com/zhuchen03/FreeLB/blob/eb2370161037f3b03c5ab1262fb455dc7da4d361/huggingface-transformers/examples/run_glue_freelb.py), under the user's explicitly scoped exception to the author-reference-reproduction prerequisite. It is **not a FreeLB reproduction**, an established DFlash improvement, or a claim that adversarial training itself is new.

Each dataset independently continues its exact PR32 KPLUS4 state (weights, Adam, counters and RNG) from step 512 to 768. The four arms are ordinary KPLUS, two genuinely clean views (CLEAN2), a clean plus independent keyed Gaussian view (RANDOM), and a clean plus teacher-loss-gradient view (ADV). KPLUS4 supplies event-five feedback; KPLUS5 supplies event-six feedback. All arms use the same packets and naturally matched presentation/anchor draws. The original GSM8K and MathQA 256-request development panels are evaluated separately, with four checkpoints and two balanced repeats.

For each augmented presentation, the clean forward fixes K+'s selected prefix-plus-first-error positions, priority groups, weights and response-wide denominator. Both views use the **original aggregate K+ numerator graph**, with the same full-block attention, teacher conditioning and BF16 probability arithmetic. Only valid artificial mask embeddings are perturbed. Seed slots, canonical inputs, frozen teacher features, positions and attention topology remain unchanged. The proposed per-block radius is 0.01 times the clean eligible embedding Frobenius norm. The perturbation is detached before the second view; serving has no perturbation.

The clean backward simultaneously supplies the parameter contribution and the FP32 perturbation-leaf gradient. It uses the actual quarter-response scaling (one half per view and one half per response in the two-response window); this positive scalar cancels in exact normalized-direction arithmetic, while its native numerical realization is explicitly pinned. There is no extra gradient-extraction backward. Sequential view graphs accumulate into the qualified BF16 parameter gradients. Native clipping and AdamW happen once per two-response window. Disabled augmentation and zero attack radius directly return the original KPLUS core. CLEAN2 keeps the real two-view path and its differences from ordinary BF16 accumulation are measured, not presumed absent.

## Work and qualification

| Scope | Response forwards | Response backwards | Adam steps | Native generations |
|---|---:|---:|---:|---:|
| Main training | 7,168 | 7,168 | 2,048 | 0 |
| Main disposable startup | 32 | 32 | 0 | 0 |
| Fixed post-update probes | 128 | 0 | 0 | 0 |
| Qualification, including startup/resource fixtures | 68 | 68 | 12 | 8 |
| Shared feedback | teacher preparation separately counted | 0 | 0 | 512 |
| Final serving comparison | 0 | 0 | 0 | 4,096 |

The first four feedback responses are collected for qualification and reused within the 512 total. Retained authenticated canonical records supply evaluation references: no extra reference generations. Native qualification requires exact disabled/zero-radius endpoint equality, exact ordinary sampling/RNG/counters for active arms, actual paired context/position/attention checks, correct frozen selections and groups, nonzero BF16 perturbation realization, one Adam step per window, preserved target outputs, and a synthetic 2,272-token/512-anchor allocation fixture. Intended FP32 radius checks use `rtol=1e-5, atol=1e-8`; realized BF16 radii are reported without changing them to force equality. Adversarial loss need not rise for every finite nonlinear step.

The four fixed diagnostic windows per event are 1, 32, 65 and 128. Each reuses its actual two fitting responses and performs two clean post-update forwards on the original anchors, selections and K+ priorities, replaying their original stochastic state. Prefix repairs, lost old-prefix occurrences and net prefix progress are source-fitting evidence. They are not held-out serving evidence.

## Code and launch

- `online_feedback/adversarial_block.py`: frozen selection and perturbation primitives.
- `online_feedback/adversarial_block_runtime.py`: sequential two-view response/core implementation.
- Three narrow optional hooks in `online_feedback/objective.py` preserve the native objective without rewriting the author training or optimizer.
- `training.py`: complete-state continuation and one logical response observation per presentation.
- `seal.py`, `preflight.py`: pin actual source, inputs, runtime, original checkpoints and storage.
- `run_qualification.py`: bounded disposable qualification, with no automatic retries.
- `native_pilot.py`: shared-feedback collection, training and four-arm evaluation.
- `diagnostics.py`, `report.py`: fixed probes and source-bound result interpretation.

Use the separate sealed run directory. `python -m experiments.kplus_abt.run_qualification RUN_ROOT` must pass before `run_campaign.sh RUN_ROOT CODE_ROOT`. The production launcher requires the admission receipt for the same source commit and binding. No incomplete run is treated as a completed comparison. Ordinary historical K+/ROLE/FSG guards remain unchanged.

Storage admission uses measured checkpoint/state sizes, eight final complete arm boundaries, one atomic overlap, one shared feature pool, compact records, compiler/workspace allowance and an 8 GiB reserve. Disposable qualification weights are retired only after their declared comparisons and serving checks; failures retain evidence. No disposable state initializes the pilot.

## Interpretation

ADV must improve clean serving beyond KPLUS and the CLEAN2/RANDOM controls to support the adversarial direction in this fixed pilot. Better perturbed loss alone is insufficient. Mixed datasets and unresolved intervals stay mixed/unresolved; absence of a demonstrated difference is not equivalence. Bootstrap requests, not timing repetitions, and report datasets separately. This pilot neither establishes training-seed robustness nor the full 58-event ranking. There is no automatic radius tuning, extra arm or later event.
