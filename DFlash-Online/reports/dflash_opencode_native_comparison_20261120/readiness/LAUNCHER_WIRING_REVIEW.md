# CODE launcher and qualification-disposal review

Implementation source: `c0a0794e8a7a42a3e2e70833a73992a6201ea6d6`. Native execution: **not started**.

Scope: independent CPU/source review of the newly added `launch.py`, `forecast.py` and `qualification_cleanup.py` against the actual training, qualification, serving-gate and native timing APIs. No GPU execution, environment installation, scientific setting or resource-cap change.

Three concrete issues were repaired:

1. A newly started child was registered only after its process receipt was written. A receipt-write failure could therefore bypass cleanup. The launcher now registers the owned process first and attempts cleanup of every owned child even if another cleanup fails, preserving the original exception.
2. Qualification completion relied too heavily on the plan's totals and generic success flags. It now independently requires the fixed 64 generations, 17 Adam steps, 40 forwards/backwards, 16 teacher preparations and three constructors; the exact five/five/seven case order; startup counts; exact restored state; all60 initial serving checks and four publication checks. This adds no work and changes no tolerance.
3. Disposable cleanup trusted paths reported by receipts without checking each expected role. It now validates the six exact disposable state paths, C's sole active export, two exact synthetic fixtures and all16 sets of native/terminal/prepared features before hashing or unlinking anything. Duplicate paths, symlinks, foreign ownership and substituted paths stop cleanup. All57 expected tensor files are retired only after complete qualification; compact records and deletion identities remain.

The forecast uses the correct native field names. It retains the whole1,024-input AR/F panel, all16 teacher receipts and all bounded training cases, including the full4,096 synthetic pair. Measured durations must be finite and nonnegative, completed arms must follow the prescribed prefix order, and a rejected resource forecast now saves its complete estimate and explicit failure status. Qualification's observer-adjusted training cost excludes measured observer hashes and whole-case endpoint/state serialization. These are planning estimates, not guaranteed duration bounds or serving-benefit claims.

The root's final-evaluation storage-phase marker remains intact. No phase-aware budget policy was changed by this review.

Validation: **nine new focused CPU tests pass**. The combined new tests plus existing entry, plan-consumer and runtime tests pass **17/17** using the existing `moziflux` environment. Native qualification remains unexecuted; a separate parent-owned resource-admission decision remains necessary before GPU work.
