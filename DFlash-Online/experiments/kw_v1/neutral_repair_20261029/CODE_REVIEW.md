# Separate neutral-repair code review

## Material Passport

ARS-Codex 3.24.0 experiment-agent run/validate; separate same-team review by `kw_eval`, 2026-10-30 UTC. Verification status: **ANALYZED**. Reviewed the new diagnostics and revised qualification implementation authored by other agents, its CPU regressions, repair admission/launcher and source-diagnosis limits. No implementation file was edited by this reviewer. No GPU access, remote operation, full-model pass, serving generation or production optimizer update was performed. The focused tiny CPU fixtures are separate debugging/test work, not GPU qualification work.

**Conclusion: no remaining source-level blocker in the reviewed qualification-only change after three concrete issues were corrected.** Actual full-model requalification remains pending. This review does not authorize production, establish a numerical root cause, relax native-bit equality, or change the fixed KW algorithm. Original failed execution `9fd9fe6f7db9796a49e2a6879b5a79e7f01fbd22` remains historical evidence. The new code must be published and sealed at its own execution revision before launch.

## Findings resolved during review

1. **Diagnostic serialization could lose failure evidence.** A finite FP64 fixture `[1e308]` versus `[0]` overflowed its sum-of-squares diagnostic to infinity. The strict JSON writer then raised before publishing the intended mismatch receipt. The helper now encodes nonfinite aggregate summaries explicitly, as it already did coordinate examples. The regression confirms a durable exact-failure receipt with the overflow disclosed. No numerical tolerance was introduced.
2. **A cleanup-only failure could skip the final worker receipt.** The first worker-finalization draft raised immediately from `trainer.close()` on an otherwise successful path. That bypassed the final failed receipt. The final implementation stores the cleanup error, writes the failure record, releases remaining resources and then propagates the error. A separate test also preserves the original training failure when cleanup fails afterward.
3. **Restoring optimizer state could mutate the saved comparison snapshot.** Passing the saved optimizer dictionary directly into PyTorch `load_state_dict` retains nonfused/noncapturable Adam step tensors by reference. Subsequent temporary applications could advance the captured first-window step counter. A local load-only probe confirmed the alias without running any optimizer application. Restoration now deep-copies the saved optimizer state. The regression runs temporary native steps and verifies that the complete saved snapshot remains unchanged. This repair is limited to qualification restoration; it does not alter live KW training arithmetic.

The campaign author also placed completion of the two preview-ledger entries before the new post-preview assertions. Consequently, an isolation failure there retains the two completed previews while stopping the live update.

## Exact gates and observations

The diagnostic helper calls the existing native-bit `differences` comparator. It preserves all differing paths rather than truncating to 100, and adds chunked coordinate counts, finite-pair magnitudes, explicit nonfinite counts and a bounded set of examples. Signed zero remains a bit mismatch. Structural shape/type/active-set mismatches remain failures. Reads, casts and reductions do not write into the compared tensors; no RNG draws are added. The scalar diagnostics are descriptive and never decide whether the exact gate passes.

The revised qualifier compares exact initial scientific state, gradients after all four backwards, both normalized pre-clip gradients, parameters/Adam/RNG and gradients across previews, both complete neutral-update boundaries, and complete recorded metrics/anchors. Both neutral allocations and controller window/preview/credit counts are still required. The existing nonzero optimizer-only checks reuse the first captured window, with the saved state now protected from mutation. No response forward/backward is added for these observations.

The sole production-trainer edit introduces an optional `post_previews` observation point. Without a qualification observer it performs no comparison or state write. Masks, features, controller constants, gradient arithmetic, clipping, native AdamW settings, precision, RNG restoration, training inputs and evaluation scope are unchanged.

## Process and budget controls

The default coordinator launches K and KW sequentially through two fresh Python `EXEC` children, not `fork`. Workers must begin without an initialized CUDA context or existing compiled Flex/block-mask singleton, then initialize the pinned GPU runtime. Startup identity/environment is retained, and successful production qualification requires distinct child PIDs and GPU execution. The explicit injected CPU fixture mode cannot set `full_model_memory_checked=true` and is not exposed as a normal CLI qualification mode.

Both children reopen the same durable `WORK.json`. Every call is charged before dispatch and the hard ceiling is shared across children: eight response forwards, eight backwards, sixteen disposable optimizer applications, zero serving generations and zero persistent updates. K must consume exactly four/four/two completed calls before KW starts. Final success requires exactly eight/eight/fourteen and no started-but-uncompleted record. The expected fourteen applications are two K updates, two KW neutral updates, four KW previews and six optimizer-only applications. The previous failed six/six/five work remains separate.

The coordinator reloads the shared ledger even after child failure and records unknown started attempts rather than erasing them. Tests cover a child abort without a child result receipt, reopening a consumed budget, zero-work constructor/input failures and final-receipt write failure without hiding the original error. The original evidence is protected by exclusive output creation, per-worker receipt existence checks and one-shot launcher sentinel. There is no automatic retry or production entrypoint in the repair launcher.

## State, storage and admission

K's initial state, four backward gradient boundaries and two completed states are saved as immutable comparison artifacts with hashes. The normalized pre-clip entries reuse the appropriate saved backward artifact after an exact equality check; no duplicate tensor copy is needed. Artifacts are rehashed before loading. Saved tensors remain for explicit review and verified retirement; they are not promoted to serving checkpoints.

The transient writer imposes a 30 GiB stored-tensor cap and checks the unchanged 8 GiB reserve before each write. The launcher preflight requires 40 GiB for transient/overlap writes plus the 8 GiB reserve, at least 80 GiB available host memory, idle GPUs and exact source/configuration/model/runtime/input/original-state checks. The K artifact payload is approximately 27.27 decimal GB before serialization overhead; one largest atomic write overlap remains inside the 40 GiB allowance. This is a source/arithmetic review, not a claim about actual remote availability at launch.

Original K8 is read, identity-checked before each worker and checked again after successful work; the admission check covers original O8/K8 state/export identities. Temporary optimizer checks restore only disposable in-memory training objects. Post-execution original-state/source validation remains necessary as an operational result check.

## Independent CPU validation

The diagnostics suite passed **4 tests in 0.025 s**. After the qualification fixes and formatting, the focused independent suite passed **11 tests in 3.220 s**: `QualificationLedgerTests`, the complete tiny qualification flow, and the saved-optimizer-snapshot immutability regression. The full tiny flow verifies exact eight/eight/fourteen accounting with all 30 calls completed, exact check results, controller accounting and rejection of rerunning the same output. The author subsequently reported **68 combined CPU tests passing in 39.871 s**; I checked the retained `CPU_COMBINED_TESTS.log` summary, rather than running that combined suite independently. The final cosmetic helper expansion and removal of the unused CHECKS dictionary-envelope fallback were inspected, and the final qualifier hash below was refreshed. These suites overlap and are not additive independent scientific trials. `git diff --check` passed.

Commands used `/home/minhao2/.venvs/dflash-specforge/bin/python`, `CUDA_VISIBLE_DEVICES=''` and `PYTHONPATH=.:tests`. No full-model or GPU requalification was attempted by the reviewer.

## Interpretation limit

Separate fresh processes remove the confirmed shared-process compiler-history asymmetry of the old harness and avoid retention of K objects during KW. That is a justified control repair. It is not proof that a particular compiled kernel caused the earlier mismatch. A successful rerun would establish the requested exact comparisons under the repaired startup; it would not establish arbitrary process/shape-history reproducibility or KW efficacy. The repair scope ends at bounded requalification results, with production still stopped.

## Reviewed file identities

Paths are relative to the shared `DFlash-Online` checkout. These hashes identify the reviewed working-tree bytes; publication assigns their execution commit.

| File | SHA-256 |
|---|---|
| `experiments/kw_v1/qualify.py` | `07201e49d4d57054b2cc0864a69254d671e5861b14270e3ca8e12143ea12e2f9` |
| `experiments/kw_v1/diagnostics.py` | `d8535f4a4279dd0b59858e8356ee754c73a8fd234804e932b6cdd398384cbf85` |
| `experiments/kw_v1/test_diagnostics.py` | `4f9558a4428ad55cd1f4ab782f9d7b04639977db38ce5adedd5dce74e896c2c9` |
| `online_feedback/kw_training.py` | `926a25d2426c10544af6ccc29ef9107983a89936c1b032825666d4427b26b74b` |
| `tests/test_kw_training.py` | `e2e49a3e2be41fe3e232c75f4cacbc92717dc1944aac720a68c5e33cdf28c850` |
| `experiments/kw_v1/neutral_repair_20261029/repair_preflight.py` | `59a4ffbdf0b80ffa49998af3bd71b4d47fbda5c0d2b63cf4431b6477cacc778a` |
| `experiments/kw_v1/neutral_repair_20261029/qualify_a800.sh` | `959b77f791cae7dd54a5a80d50a96651623d975a52e207ecd0166c1dd13666c5` |
