# Initial-state mismatch: cause and experimental relevance

The bounded native trace identifies an import-order issue, not a C objective or model-update failure. After the original seed-17 call, Rich's terminal-formatting module calls `getrandbits(24)` to initialize terminal hyperlink identifiers. This changes Python's global random state without changing any model tensor, optimizer state or scientific sampling state.

## Evidence

The actual failed campaign source remains `2897d483241e0c24b64deab31db09cc10669d9c7`. Diagnostic source is `d2cfcba7a3c0e079ae8e27b36bc7766b6ea2206f`; it constructs one trainer and performs **zero forwards, backwards, optimizer steps or generations**. The original failed candidate was not saved completely; the diagnostic reproduces its exact error class and first mismatch paths and supplies the missing complete evidence. It is not a retrospective all-component verification of the unsaved candidate.

| Component | Exact comparison with original K0 |
|---|---|
| Released weights | Pass |
| Empty Adam state and options | Pass |
| Torch CPU and CUDA RNG | Pass |
| NumPy RNG | Pass |
| Scientific configuration, counters and history | Pass |
| Python RNG | Fail: 625 state entries differ |

These 625 differences are entries of one generator's state, not 625 different model errors. An independent CPU reconstruction confirms that the current state is **exactly** fresh seed17 followed by one `getrandbits(24)` draw. The native trace records that call at `rich/style.py:22`; its MT index changes from624 to1. Seven trace entries comprise nested seeding, this one draw, and unchanged-state preservation calls; no trace entries were dropped.

The pinned Rich file hashes to `ca9e13b0d28d3d4c41cada0559fcbb9331361a1ffd4d803b7ecd942d77db9b4d`. Its identifier is used for terminal hyperlink escape codes. The actual response shuffle uses Torch CPU RNG; the author anchor sampler uses Torch on the training device. K+/C coefficient construction and AdamW do not draw from Python's generator in these inspected paths. A CPU control using the actual sampler and C allocator confirms unchanged outputs under different Python states with identical Torch/NumPy state. This is a scoped dataflow finding, not a claim that no dependency ever uses Python RNG.

## Repair and decision

The user's conditional permission allows ignoring a scientifically irrelevant mismatch. A smaller repair is available: import only `rich.style` before the **existing** seed call. This lets the terminal library initialize before the scientific RNG boundary. Keep the exact initial-state comparison, all model/optimizer checks, and all RNG components in save/reload checks. Do not reseed after model construction, load the witness's RNG to force agreement, alter C, or relax tolerances.

The repair is committed as `2601743d8f5197aa120c1b3e05d5bd02d501edbc`; nine focused CPU tests pass. A separate same-team reader passed34 causal/accounting checks. The original gate remains failed and retained. **The repaired startup passes that same original gate on A800 at execution `a32f259d87697f09653c2e2eab786bb17603de1b`: zero mismatches and all four RNG components exactly equal the original witness.** A separate same-team receipt audit passes27 checks. All54 full-C CPU tests pass. This is an original-gate pass after a narrow startup fix, not a waived or relaxed gate. **All remaining native qualification now passes:** ordinary/no-op/active-C checks, exact next-update save/reload, CPU export reconstruction, and24/24 serving checks with no invalid or unknown generation. The retry used6 disposable Adam steps,14 response forwards/backwards each and one constructor. No production training has begun at this report revision. Native/cap-zero equivalence, actual C intervention, exact own-state recovery, CPU export and serving checks remain required. Identifying this administrative cause is not a full qualification pass or an efficacy result.

## Cost and evidence limits

The native constructor diagnostic used52.902s wall time and52.894 occupied-card seconds, peaking at6.691GB allocated /9.209GB reserved VRAM. At the end of this diagnostic, cumulative model work was two trainer constructions and no F/B/Adam/generation work; subsequent qualification is accounted separately. One earlier diagnostic metadata error occurred before trainer construction or ledger mutation and is retained separately; its fix reads installed package metadata without changing import order.

Original WORK/CLOCK and failure evidence remain intact. Storage remains sufficient. The original24-hour forecast gate correctly stopped production after qualification:18,710.930s consumed plus79,540s remaining with contingency equals98,250.930s (27.292h). The user then authorized **only the elapsed cap to30hours**, retaining the original clock,40occupied-card-hours, settings, panels and storage limits. Source-authenticated production continuation `b4459820b3822c3d760bb3fd880e4d139fa2bbc2` has started after fresh CPU/source/storage/resource admission. Qualification is reused, not repeated. The serving engine is loading for the first feedback window; no production efficacy result exists. Raw RNG/stack/state comparison records remain private with authenticated compact public summaries.

## Passed qualification and disposal

The active-C fixture changed2,496 coefficient occurrences and40 parameter tensors; this verifies intervention, not benefit. Qualification training wall time was217.006s; peak allocated/reserved VRAM15.122/17.060GB. Serving qualified24/24 executions across both datasets and allthree checkpoints, with zero invalid, unknown or unattempted work. CPU reconstruction reproduced export bytes exactly. No full training event or efficacy endpoint has run.

After preservation of tensor fingerprints and compact validity evidence, six disposable qualification payloads were retired (12.584GB logical bytes;10.486GB observed free-space increase). Original K0, controls and historical evidence remain. Deleted qualification tensors are not archived.136 compact files (10.863MB) were transferred and individually authenticated on Fenrir. Prior failed attempts and their work remain separate and counted.

Independent requalification audit passes3,688 checks over all136 file identities, six training cases,24 raw serving records, cache boundaries, failure preservation, work/clock and forecast. Qualification records are disposable correctness checks even where inherited metadata uses endpoint labels; they are not C58 efficacy measurements. The production-only continuation patch `ec56400` passes16 focused CPU tests, preserves the default24h policy for other plans, and accepts only the explicitly authorized30h amendment for this run.

Production launch admission observed517.398GB shared free space and71.011GB projected peak. The complete remaining raw forecast is59,800s (16.61h), with79,540s (22.09h) including the unchanged contingency. The original first-model clock remains1794993369.5107875;30h is a maximum total elapsed allowance, not a fresh30h clock. The final comparison remains the full58-event candidate and both unchanged1,024-request panels.
