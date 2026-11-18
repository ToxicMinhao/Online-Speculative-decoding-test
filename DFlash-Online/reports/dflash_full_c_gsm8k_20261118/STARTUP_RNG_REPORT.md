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

The repair is committed as `2601743d8f5197aa120c1b3e05d5bd02d501edbc`; nine focused CPU tests pass. A separate same-team reader passed34 causal/accounting checks. The original gate remains failed and retained. The repaired startup must pass that same gate during the already authorized bounded qualification before production. Native/cap-zero equivalence, actual C intervention, exact own-state recovery, CPU export and serving checks remain required. Identifying this administrative cause is not a full qualification pass or an efficacy result.

## Cost and evidence limits

The native constructor diagnostic used52.902s wall time and52.894 occupied-card seconds, peaking at6.691GB allocated /9.209GB reserved VRAM. Cumulative model work is two trainer constructions and no F/B/Adam/generation work. One earlier diagnostic metadata error occurred before trainer construction or ledger mutation and is retained separately; its fix reads installed package metadata without changing import order.

Original WORK/CLOCK and failure evidence remain intact. Storage remains sufficient; production admission must still apply the original24-hour/40-card-hour forecast with consumed time included. Startup diagnosis does not waive that resource gate. Raw RNG/stack/state comparison records remain private with authenticated compact public summaries.
