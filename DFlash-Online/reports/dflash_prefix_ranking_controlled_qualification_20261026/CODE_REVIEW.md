# Controlled initialization qualification: independent code review

Reviewed 2026-10-26 using the ARS experiment/code-runner workflow. This review covers the bounded qualification driver, its reused production qualification helpers, CPU fixtures, and launch boundary. It does not execute a remote or GPU experiment.

**Verdict: no blocking implementation finding remains for the authorized qualification.** This is approval of the check's implementation, not evidence that native qualification has passed or that U/P improves drafting.

## Reviewed controls

- One ordinary process warmup performs two presentations and one temporary optimizer step. Its weights, optimizer state and counters are discarded. Every following arm constructs a fresh released draft and fresh optimizer, then restores the same pinned complete pre-sampling RNG state. Exact comparisons validate construction rather than silently replacing mismatching weights with the pin.
- Oa, Ob, U0 and P0 compare the initial state, each completed backward, normalized pre-optimizer gradients, final weights, Adam state, RNG, counters, gradient layouts and scalar records. Comparisons use exact native tensor equality; runtime duration is excluded. Empty-container and list/tuple structure is checked too.
- Each numerical boundary is saved before its equality assertion. Failures retain detailed differences and an additional available-state/gradient snapshot, report failure, and stop subsequent trials. Nonfinite values cannot pass the equality gate unnoticed.
- Sampling observes the actual full 128-element first-pass permutation and validates it against the original event. Both executed anchor draws are checked against the retained schedule. The observer restores the ordinary `torch.randperm` function before model/compiler work.
- Active U1/P1 retain the existing forward-KL arithmetic and add the implemented ranking objective without changing scientific settings. Both require two independent prefix-mask checks and two bounded rival-gradient checks. The independent reference has released weights, no parameter aliasing or gradients, and eval mode; reference setup/forwards preserve complete RNG and current-model modes.
- Four detached comparison forwards check the two retained full blocks under matching eval/no-grad conditions. They preserve RNG and restore model modes. Temporary states are never exported or published.
- The complete budget is seven discarded optimizer steps, fourteen training presentations/backwards, four reference training forwards and four detached comparison forwards. There are no generations, new teacher-model forwards, main U/P steps or automatic campaign commands. Layer/checkpoint recomputation remains separately counted.
- Storage admission includes retained snapshots, failure evidence, atomic-write overlap and reserve. Loading and overall CUDA peak memory are recorded, including failure paths. Snapshot synchronization and all measured qualification costs are diagnostic costs.

## CPU evidence and resolved launch issue

The independent five-test suite passes with GPUs hidden. It exercises all seven windows, deliberate post-step corruption with persisted failure evidence, deliberate fresh-weight corruption, exact/finite comparison behavior and storage accounting. Existing helper/source tests are separate evidence; CPU success does not establish native Flex/BF16 equivalence or GPU memory fit.

The first independent test command left GPUs visible. Four tests passed; the full-window fixture failed before any qualification backward because its CPU pin had no CUDA RNG state while the fixture's checkpoint path initialized CUDA later. The only initial-state mismatch was `rng.cuda.0`; all sixteen shared tensors and scalar metadata matched. The process exited. This was a test isolation failure, not a model discrepancy, and the failed observation is retained here rather than being described as a qualification pass. The launcher now runs the CPU child with `CUDA_VISIBLE_DEVICES=''`; the fixture also rejects incidental CUDA initialization. Actual GPU qualification keeps its separately selected GPU.

## Interpretation limit

Passing controlled warm Oa/Ob/U0/P0 checks establishes equality for this two-presentation, one-step window under the explicit initialization procedure. It does **not** retroactively pass the earlier failed gate, identify a specific kernel as its cause, or prove equivalence to the retained ordinary O checkpoint's complete 128-step trajectory. Original scalar metrics and anchor equality do not replace that missing full-trajectory state evidence. Reusing the old O artifact in a future main comparison therefore retains this disclosed limitation; this qualification does not itself authorize that campaign.

The active checks establish implementation properties and memory feasibility, not useful transfer or serving improvement. No acceptance threshold, optimizer, precision, anchor, denominator or learning-rate change is introduced.

Reviewed final driver SHA-256: `b9df0a9f73fae84d61b83102c2c7a6b618886f6d846e718a71829b2f2ffd10fd`.

Reviewed final test SHA-256: `0031f3656f3ad49c1f008b2ab1035f173b1cb065dd16ed4ca7ee84eb6502e006`.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_controlled_qualification_20261026/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
