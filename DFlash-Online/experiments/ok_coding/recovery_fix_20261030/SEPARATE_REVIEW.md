## Material Passport

- Origin: ARS-Codex 3.24.0, experiment-agent / validate, October 30, 2026.
- Reviewer: separate same-team read-only reviewer `pr20_recovery_review`; not external validation.
- Verification status: **VERIFIED for the focused CPU paths below**. Native scientific execution remains unverified.
- Reviewed implementation commit: `a0ba4c23e28bef6a16d79866b251fb1b6c480358`; baseline `5ec126a189170235b8e89e390261904ddc691dcc`. the accompanying `SEPARATE_REVIEW.json` binds nine reviewed implementation, launcher, documentation and test files by SHA-256. Publication must bind the corrected source rather than reuse the old source seal.

**Review result: passed within the stated scope; no unresolved actionable finding.** No production source was edited by this reviewer. There was no remote operation, model loading, GPU work, or scientific experiment.

### Recovery and admission

`campaign.py:422` completes fresh/resume preflight before `campaign.py:442` reserves a native attempt. The parent is created exclusively; its absent `native/` child is passed to the unchanged backend. Attempt records retain source, mode, absolute path, admission references, startup failure and closure. Scientific counters are reconstructed before recording the attempt.

`attempts.py:8–30` uses the greatest reserved/existing number, includes numbering gaps, and fails exclusively on unexpected collisions without using Python, NumPy or Torch RNG. Both native attempts and the existing admission-receipt namespaces use this rule. Legacy native files and absolute references remain intact.

`recovery.py` verifies a chronological prefix of complete pools, own-arm jobs, evaluations and retirement, with exact artifact identities and latest-state consistency. It rejects unknown/partial work, missing or changed required artifacts, unbound export files and later completed actions after an unstarted phase. Retirement intent/progress without completion rejects recovery even before any unlink. Completed feedback, preparation, training and evaluation are reused by the real campaign branches.

### Storage accounting

The fresh threshold remains **366,710,105,681 bytes** and still rejects the published **150,373,031,936 bytes**. At a hypothetical fully completed seed-17 boundary, the unchanged cap/retention model requires **209,627,864,400 bytes** for remaining work; the old fresh-again gate would reject that threshold. This is deterministic fixture arithmetic, not a measured live recovery.

The remaining model retains both future capture/prepared copies, missing initial states, Adam states/exports, 5W pending-save/export overlap, newly created rolling predecessors, required endpoints, the full 2,048-token response cap, 16 GiB compiler allowance, 10% new-allocation slack, 4 GiB compact/workspace allowance and 8 GiB reserve. Existing allocations cancel against their current occupied baseline; they are neither new writes nor free capacity. The physical inventory counts `(st_dev, st_ino)` once and records allocated blocks separately from logical sizes.

**Every pre-existing allocation receives zero retirement credit.** Link or extent information cannot establish immediate release while an open descriptor, mapping or snapshot retains storage. This conservative policy follows the handoff's zero-credit fallback and removes the need for FIEMAP assumptions. Future newly allocated rolling payload still follows the established retirement model. The pure arithmetic check confirms slack before hypothetical release: `floor(1.10*100)-90 = 20`, while fixed reserves never cancel. No artifact is deleted to calculate admission.

### Findings resolved during review

An unbound extra export file initially passed boundary validation and could receive retirement credit; a tiny CPU fixture reproduced that path. The final validator rejects unbound export entries, and candidate accounting is restricted to validated bindings. The subsequent zero-credit policy independently prevents that allocation from inflating capacity.

The initial physical-extent proposal could not exclude remaining open holders. The implementation replaced it with zero existing-retirement credit and removed the now-unreachable positive-credit branch. No scientific arithmetic or settings changed.

### Independent final validation

| Check | Result |
|---|---|
| `test_ok_coding_recovery` | 4 passed, 24.717 s |
| `test_ok_coding_resume_storage` | 14 passed, 0.057 s, after final dead-branch removal |
| Launch-wrapper shell syntax | Passed |
| Original scientific contract, training, fresh storage model and upstream backend/training versus baseline | Unchanged |

The successful recovery fixture executes real campaign/preflight/source-seal/boundary/capacity/skip decisions with a strict backend fake that creates a genuinely absent directory and restarts serial numbering at zero. It preserves legacy traces/captures and earlier receipts, reuses completed events, dispatches the next sealed feedback/preparation and O job from the correct parent, and deliberately stops at the next K dispatch. Model/configuration/runtime/machine probes and scientific model/subprocess work are fixtures; directory behavior, recovery reuse and admission decisions are not replaced. Other checks reject changed/missing/partial artifacts and unfinished retirement before native allocation, and verify startup-failure records, exclusive collisions and RNG preservation.

The 14 storage checks cover exact/one-byte-below thresholds, remaining-headroom acceptance, caps and overlap, missing initial jobs, inode aliases, sparse size, other filesystems, unbound files, zero credit with open handles, and fixed reserves. The wider 58-test suite is executor evidence, recorded separately rather than claimed as this reviewer's independent rerun.

### Limits

These checks do not qualify native model startup, full-model state restoration or GPU behavior. Positive credit for existing-file retirement is deliberately unsupported, so some physically feasible recoveries may still be conservatively rejected. No current fresh capacity shortfall is solved, no campaign restart is authorized by this review, and the exact data binding, seeds, scientific settings and work budget remain unchanged. All 11 ARS statistical-fallacy categories were considered; none is applicable to the deterministic implementation claims made here.
