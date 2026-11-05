# KW neutral qualification repair

ARS experiment-agent run/validate. User-authorized implementation repair and bounded requalification only. Algorithm, hyperparameters, exact equality gate, inputs and production scope remain unchanged; no production campaign is launched here.

The original harness ran cold ordinary K followed by KW in the same process, after four attention shapes had changed compiler specialization history. A bound optimizer method also retained K tensors during KW. These are confirmed harness asymmetries, not yet a confirmed cause of the numerical mismatch.

Run K and KW in two separate fresh interpreter processes, each with the same retained K8 state, full RNG, four sealed responses, model/runtime and environment. No warmup or additional model call. Capture exact initial state, all four backward gradient boundaries, both normalized pre-clip gradients and both final boundaries. Check live parameters/Adam/gradients/RNG across previews. Record complete mismatch paths and bounded coordinate statistics before raising. Nonzero optimizer-only checks reuse the already captured first window.

New qualification limits: eight response forwards, eight backwards, sixteen disposable optimizer applications, zero serving generations and zero persistent updates. Expected complete execution is fourteen optimizer applications. The previous failed six/six/five work remains separate. No added diagnostic model execution is planned outside this qualification.

Resource admission: allow 40 GiB additional storage for transient host comparison tensors, atomic overlap, logs and retrieval staging, plus the unchanged 8 GiB reserve; require at least 80 GiB available host RAM and an idle qualified A800. Preserve failed-run artifacts. Temporary qualification tensors are never serving checkpoints; retire only after compact comparisons and result evidence are verified.

Interpretation: a passing fresh-process comparison validates neutral equivalence under the repaired harness. It does not by itself identify a specific compiler kernel or prove arbitrary cross-process/shape-history reproducibility. If the first divergence remains, its exact stage and magnitude are reported; no tolerance change or automatic scientific retry.
