# Independent implementation review: one KC event

**No remaining blocking finding in the reviewed implementation.** The scientific correction is byte-for-byte unchanged from the successful single-step diagnostic. The added manager applies that same correction once after each ordinary K accumulation window, then lets the existing trainer continue from the corrected parameters.

## Material Passport

- Workflow: ARS experiment-agent validation and independent code review.
- Scope: local source inspection and CPU tests only; this reviewer performed no GPU work or remote execution.
- Inputs: HANDOFF.md, NUMERICS.json, the retained single-step result, and the new core manager/campaign/evaluator.
- Status: implementation reviewed; full-event native validity and results remain pending execution.

The manager constructs a fresh recorder before each window's first forward. Both presentations contribute to the same protected-weight denominator. Ordinary K gradients, clipping and AdamW execute through the existing trainer; correction gradients use separate FP32 buffers and do not enter Adam. The tested FP32 coefficient, BF16 application, zero/tie behavior and no-op branches are retained. Corrected parameters become the next window's starting point without resetting sampling, optimizer moments or counters. The exported checkpoint's algorithm contract explicitly identifies KC.

Exact device-local copies verify Adam and existing gradient preservation around auxiliary work. RNG, module modes, caches and counters are checked, with parameter identity/version checks during read-only passes. CPU integration fixtures additionally compare actual parameter values and compare the manager's two consecutive windows against manually invoking the tested correction. Snapshot/check costs are recorded separately from intrinsic correction work. Each recorder and its feature/gradient buffers are released before the next window.

The step-1 gate compares the complete retained response, prefix, margin and correction records, not just aggregate improvements. It runs within the main event before window 2. Only steps 1, 64 and 128 receive corrected-state scoring. The budget remains 256 K forwards, 256 post-K forwards and 6 corrected forwards; 128 native Adam steps; at most 256 correction backwards and 128 corrections. The existing disposable startup adds two forward/backwards and zero optimizer steps, separately counted.

The genuine K/KC evaluator uses the original 64 development requests, three balanced-order repeats and retained F tokens/termination. It validates checkpoint publication, exact native outputs, measurements and repeat-consistent counts. It stops on correctness failures while preserving partial records, and imposes no performance gate or historical K-count gate. Training closes before serving begins. Reports pool one validated count trajectory per request and use per-request mean latency across repeats; benchmark activation and adaptation costs remain separate.

Two review findings were repaired before execution: nonfinite diagnostics now preserve the original exception rather than failing JSON serialization, and actual completed native Adam/presentation counts remain visible when a post-step failure leaves the trainer's return-time counters behind. Focused fixtures cover both failure paths.

I independently ran all **30 new manager, campaign and evaluator CPU tests** successfully with CUDA disabled. The unchanged correction file also matches the tested single-step source exactly. These checks establish CPU integration and scope compliance, not a full-event performance result. Step-1 native reproduction and the fresh evaluation validity gates remain required.

Interpretation limits: later probes compare the ordinary K proposal and corrected proposal from the **same current KC state**; they are not historical K64/K128 checkpoints. The reused panel is development evidence. Historical K training duration is not a fresh matched overhead control. Valid null or harmful outcomes must complete unchanged, and this run authorizes no tuning or follow-up.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_kc_event_20261027/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
