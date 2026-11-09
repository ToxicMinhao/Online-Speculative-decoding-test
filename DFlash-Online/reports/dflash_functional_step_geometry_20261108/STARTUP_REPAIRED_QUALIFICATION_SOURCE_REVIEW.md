# Remaining qualification: final source and admission review

Review status: local source review and focused CPU verification; no remote or GPU execution by this reviewer. Reviewer is the same-team runtime implementer and has authorship overlap. ARS experiment-agent fidelity workflow; no scientific protocol revision proposed.

## Disposition

No blocking implementation issue found in the reviewed `resume_after_startup.py`, `seal_after_startup.py`, `run_after_startup_qualification.sh`, and the explicit-limits addition to `work.py`. Seven focused `test_fsg_resume_after_startup.py` tests passed (0.701 seconds test time).

The new runner preserves the latest failed qualification_003 ledger byte-for-byte as its prefix. Its admission checks authenticate the failed pre-Adam SCALAR gate, exact consumed work, unchanged geometry/serving/numerical recipe, and the new named startup contract before model work. Historical diagnostics are separately identified; their work is not silently folded into or removed from that ledger.

GSM performs separately discarded startup and exact restoration before recapturing the ordinary reference. It reuses the authenticated completed ordinary/neutral/FSG endpoints and passed GSM same-layout/resource checks, then executes only the missing SCALAR window. MathQA performs its existing discarded startup and four required cases, including its independent same-layout oracle. Existing native gradient/Adam/neutrality/state gates remain exact. The runner closes each trainer before its four native serving checks, for eight serving generations total, and contains no pilot launch.

The explicit remaining ceiling is five Adam steps, ten ordinary forwards/backwards, at most 33 additional forwards and 16 additional backwards, and eight serving generations. Adding the retained ledger gives eight cumulative Adam steps, 18 ordinary forwards/backwards, 67 additional forwards, 36 additional backwards, and eight serving generations. Historical default limits remain unchanged; this fresh plan explicitly admits the repair work instead of relabeling the old plan.

The sealer reads the latest failed compact qualification evidence, retains its immutable dependency identities, copies the original preparation, includes the two startup diagnostic evidence trees, and binds the fresh published clean source mirror and new qualification_004 destination. Launcher source path and sealer source path agree. The launcher runs CPU checks and preflight before the native runner, uses the qualified training GPU/environment, preserves exit/postflight receipts, and stops after qualification.

## Storage and remaining execution admission

The earlier PREFLIGHT_REVIEW.md remains applicable: budget 32 GiB additional writes plus 8 GiB reserve, with disposable checkpoint payloads retired only after decoded-state and exact export checks. This covers the calculated approximately 23.44 GiB conservative two-payload overlap plus all six fresh exports and compact evidence. The mandatory 40 GiB free-space check must run against actual current A800 capacity, along with idle GPU, host-memory, dependency hash and environment admission. Local tests do not establish native correctness, current availability or eventual activity of a nontrivial accepted correction.

No source edits, GPU calls, new feedback, training steps or serving generations were performed for this review.

## Sealing input packaging correction

The first local sealing attempt exposed that the compact retrieved evidence intentionally omitted YAML files. The sealer now copies the complete original local qualification_003 sealed directory only after verifying every non-SEAL file against the retained remote SEAL identities. Independent local verification checked all 32 such files successfully; all 16 explicitly required configuration/model/runtime/manifest paths are present, including seven YAML files. No new blocking issue found. The incomplete sealing attempt is preserved, and this repair performs zero model work or scientific changes.
