# Persistent trainer and bounded qualification implementation review

Scope: updated version-2 handoff. This is a CPU/source review, not a native GPU qualification pass. No remote command or GPU execution was performed by this reviewer.

## Implemented contracts

- `PersistentPrefixTrainer` uses the existing SpecForge event runtime and native K+ response arithmetic. Production E installs its current row controller before every original response forward and advances its own weights, optimizer, counters and RNG continuously. N/S have no role controller.
- Event boundaries are explicit: E/S start from original complete K6, resume their own E7/S7 for event 8; N forks the authenticated original N7 only. Qualification's N6 exception is explicit and cannot authorize production N7 replay.
- S changes both current optimizer recipe LR and parameter-group LR using the exact result of `float(3e-4)*0.96`, while preserving weights, moments, counters, RNG and historical receipts. Own S7 resumes without another fork. Historical validators remain unchanged.
- Event 7 naturally draws the retained response order and anchors and checks them against the original bindings. Event 8 uses natural own-arm sampling. There is no anchor substitution or reseeding.
- E boundary presentations 0/1,126/127,254/255 retain the existing row-gradient observations and immediately check finite gradients and zero withheld/excluded direct gradients. Empty current-error support remains valid; zero current gradients do not skip Adam.

## Qualification

One trainer initialization per physical card; A performs legacy N, all-enabled N, E, repeat E, S (five disposable Adam applications); B independently performs N,E,S (three). Every case restores original K6, optimizer and RNG before the same natural first pair. The qualified startup adds two forwards/backwards, zero Adam, per initialization and restores scientific state.

Compact exact identities bind every represented gradient/parameter/moment tensor bit, with None gradients explicit. No model-sized gradient archive is written. Local all-enabled/native and repeated-E checks, cross-card N/E/S checks, full source restoration and the original E row-gradient support gate are retained. Failed/unknown work is preserved before raising. The production trainer cannot invoke qualification state switching.

Serving qualification uses one fixed original-K6 backend per card and exactly two authenticated old requests twice (four calls/card, eight total). Raw results precede validation. Exact canonical tokens/termination, retained native target-commit reconstruction, within-card counts and cross-card counts must agree. Before/after loaded-weight checks add no generation. There are no extra generation warmups. Failed returned records are invalid; transport outcomes without returned evidence remain unknown.

## Integration review

Own-feedback collection and teacher preparation occur in separate processes from training, so they cannot consume the saved training RNG. Each arm collects with its own event-7 export and retains a separate prepared feature pool. Two separately normalized responses retain native accumulation and one Adam step. Main accounting is 640 Adam, 1280 original forwards/backwards plus ten declared startup forwards/backwards across five production initializations.

The campaign serializes checkpoint publication overlap and only retires a newly created E/S event-7 recovery after its complete event-8 recovery and export pass validation. Original K6/N7 and event-7 serving exports remain protected.

Two integration findings were communicated to and repaired by the campaign owner: production now applies the same donated-buffer runtime pin before trainer construction (covered by a CPU call-order test), and writes a production memory receipt before runtime cleanup. Parent is integrating sampled serving VRAM in the worker receipt. The qualifier's missing required runtime-pin directory argument was corrected, with an autospec regression check.

## CPU evidence

The 24 focused trainer/state/qualification/serving tests passed in 21.84 seconds before the final CPU-only fixture-isolation change. The real tiny-runtime integration performs complete E7, own-state E8, S migration, and both qualification case sets; its naturally tiny row support cannot guarantee both production roles, so only that support gate is mocked in that synthetic integration. Separate row-gradient tests cover the actual error/prefix calculus. Native qualification still requires the unchanged support gate on the real fixed pair.

The CPU synthetic origin fixture now explicitly captures CPU RNG only, avoiding pollution when unrelated prior tests initialized visible CUDA. This changes tests only; complete production RNG validation was not weakened. Parent's integrated suite should always run with `CUDA_VISIBLE_DEVICES=''`.

Remaining limitations: actual pinned GPU compatibility, BF16 exact parity across the two physical cards, production memory/duration admission, complete output preservation and performance remain unestablished until the bounded native checks execute. No efficacy claim follows from these CPU checks.

## Launcher, preparation and preflight follow-up

Reviewed `launch.py`, `prepare.py` and `preflight.py`. Added nine CPU launcher tests: exact carryover of eight disposable Adam applications, twenty response forwards/backwards (including startup), two trainer initializations and eight serving qualification generations; failed/unknown/wrong-budget rejection; failed CPU/native/cross-card gates preventing production; deadline termination limited to the owned child and lease release; UUID isolation; and no automatic restart. The combined launcher/entry/serving test set passes 20 tests in 0.142 seconds.

One preparation omission was found and corrected by the parent: the native trainer constructs from the released draft directory before restoring K6, so its five previously pinned files must remain included among immutable dependencies, alongside K6/N7 exports and target files. Actual old/live identities for K6 recovery, N7 recovery, original config and event7 schedule agree with the current inspected dependency receipt. The CPU preflight log is now explicitly closed through a context manager.

Own-state ancestry, original N7 parity receipt, field-by-field S migration and full peak storage calculation remain required before CUDA. The launcher starts the clock once, carries completed qualification work into the shared production ledger, and never launches main work after an unpassed gate.

Host-memory bound advice: qualification process-tree peak plus 12 GiB for trainer/publication has a source basis in simultaneous ~6.30 GB recovery deserialization plus at most ~4.20 GB materialized draft construction. Serving qualification peak plus 4 GiB covers bounded native/prepared feature replacement and metadata; the global 32 GiB free-headroom rule remains. These are measurement/source-based bounds, not measured production peaks; actual per-task memory receipts must still be retained.
