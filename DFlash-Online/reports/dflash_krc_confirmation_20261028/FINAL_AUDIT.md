# Independent K1/KRC confirmation audit

## Material Passport

- Origin Skill: ARS academic-research-suite, experiment-agent validate.
- Verification Status: ANALYZED; retained-data audit only, no model execution.
- Provenance: separate agent review and audit in the same project team; not a blinded external review.
- Scope: fixed checkpoints and 64 inputs new to this study; older project exposure remains disclosed.

Audit status: **passed**; 0 issue(s).

Reconstructed 18,825 native verification blocks across 384 returned executions. First-K1 output/termination binding, pending KRC-first validation, fixed order, repeat-consistent counts and the included 48-execution gate were checked.

| Arm | Verification calls | Accepted draft tokens | Sum of mean request times (s) |
|---|---:|---:|---:|
| K1 | 3150 | 16769 | 55.833496780 |
| KRC | 3125 | 16795 | 55.088521150 |

Pooled call reduction: **+0.793651%**. Request gains/regressions/ties: **25/14/25**. Gross calls saved/added: 42/17. Request-time reduction: +1.334281%.

Per-repeat time reductions: +0.989022%, +1.277880%, +1.735990%.

Leave-one-request-out call reduction ranges +0.678295% to +0.869845%. This is sensitivity analysis only; the primary comparison retains every valid request.

This is a fixed-checkpoint confirmation. Repeated timings are not new request samples. No training, new feedback, model correction, generation warmup or tuned follow-up was performed. Loading and switching are separate from request latency. Development results remain separate; this panel does not establish training-seed robustness or adaptation-inclusive speedup.

Passed pre/postflight receipts cover 368 sealed files; 341 locally available files were independently rehashed. Checkpoint immutability is supported by the remote seal and loaded-tensor equality receipts. All 64 requests have disclosed older-project exposure, while all 64 are new to the current study.

The first audit incorrectly mapped 163 unchanged author-training files into the partial run worktree. INITIAL_AUDIT_001 preserves that diagnostic. The final audit hashes their canonical DFlash-Research copies read-only and verifies every one against the remote seal; no experiment source or result changed.

Statistical fallacy scan: 11/11 considered. Repeat-level results address aggregate masking; request-level counts prevent assigning the mean benefit to every request; the fixed exclusions and old exposure limit selection/collider interpretations; explicit denominators prevent base-rate confusion; fixed-checkpoint confirmation does not remove earlier screening or regression-to-the-mean concerns; all completed and failed work remains visible; there is one prespecified primary comparison, with no outcome-dependent analysis paths; a bounded checkpoint comparison is not a general causal explanation; and inputs were sealed before outputs. No p-values, population significance, cross-panel pooling or automatic further execution are claimed.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_krc_confirmation_20261028/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
