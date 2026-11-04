## Material Passport

| Field | Value |
|---|---|
| Task | MathQA O58/K58/K+58 raw correctness and accounting audit |
| Mode | ARS experiment validation, CPU-only evidence reconstruction |
| Status | Passed for the audited retained evidence; no new GPU reproduction |
| Execution source | `12583640baf2210f767e108e3632c0e5f9b8caaa` |
| Evidence root | `/home/minhao2/EAGLE_SPEC/research/dflash_mathqa_transfer_20261104/retrieved_002/production_002` |
| Review provenance | Separate same-team calculation by the original data-preparation author |

All **9,216 timed executions and three historical controls pass** this separate reconstruction. There are zero detected output, prompt, count, native-trace or accounting failures. The resource agent first verified every file in the retained private copy against the remote inventory; this audit then read and hashed **18,466 distinct source files**. The helpers import no production validation functions.

The exact 9,216-slot schedule was rebuilt from the handoff's block/parity/rotation rule and matches the sealed plan and chronological measurements. All six arm orders occur eight times. Each of the 1,024 sealed requests has all three arms and all three repeats. All prompt-token arrays match the sealed arrays; all nine output-token sequences and termination records agree with that request's fresh O58 repeat-zero record. Every arm's four count fields agree across repeats.

All **663,173 native verification blocks**, across all executions including controls, were checked. The original sidecar equals the capture embedded in its corresponding raw record. Proposed tokens, posterior argmaxes, first mismatch, committed tokens, accepted counts, absolute positions and response coverage reconstruct consistently. Token-domain, length, EOS/cap, finite timing and stored-scalar checks pass. The 663,173 blocks are repeated execution work, not independent statistical samples.

Historical controls reproduce these retained values:

| Arm | Verification calls | Accepted draft tokens | Committed progress | Output tokens |
|---|---:|---:|---:|---:|
| O58 | 23 | 157 | 180 | 181 |
| K58 | 23 | 158 | 180 | 181 |
| K+58 | 22 | 159 | 180 | 181 |

All six loaded-weight receipts and 142 activation receipts bind to their intended exports and record empty caches and refreshed helpers. All 147 batch-boundary receipts record sufficient storage and no competing training, transfers or foreign GPU processes. Authenticated pre/postflight export receipts agree. These are recorded runtime checks, not newly executed tensor comparisons.

The original server log records **40 CPU tests passing**. Successful campaign, postflight and report exits are retained. There are exactly 9,219 numbered native generation sidecars and the expected raw per-execution file sets, with no extra generation record. Work accounting reports zero optimizer steps, training presentations, feedback collections, teacher preparations or generation warmups; there are no generated training tensor files in this run. The original `production_001` CPU-only failure remains preserved with no campaign directory or native generation record. It is not hidden or counted as successful scientific execution.

The core audit performed 3,552,555 checks and the supplement 663,618, with zero failures, in 25.236 seconds of local audit wall time. No generation, training or GPU work was added. Detailed source hashes are retained in `RAW_RECORD_REVIEW_001.sources.json` and `RAW_RECORD_REVIEW_001.supplement.sources.json`; audit logic and exact results are retained alongside this report.

This audit establishes consistency of the preserved experiment evidence. It does not provide a new target-only decode, independent GPU replication, fresh model-tensor audit, or a guarantee about activity outside the retained run. The reviewer authored the data-preparation module; separate calculation does not make the review institutionally independent. Endpoint statistics and interpretation are handled separately, and this report makes no additional performance claim.
