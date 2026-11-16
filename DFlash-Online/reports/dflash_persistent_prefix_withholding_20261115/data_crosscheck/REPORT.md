## Material Passport

- Origin Skill: ARS-Codex 3.24.2, experiment-agent
- Origin Mode: validate
- Origin Date: 2026-11-16 UTC
- Verification Status: ANALYZED
- Version Label: independent_data_crosscheck_v1
- Scope: independent local CPU evidence audit; no model execution or remote access.

## Decision

**STOP_BEFORE_MODEL_WORK — the required 128 unused same-split requests cannot be formed.**

The pinned official `openai/gsm8k`, configuration `main`, `test` split at revision
`740312add88f781978c0658806c59bc2815b9866` contains 1,319 unique questions.
Two completed prior project evaluations alone exclude 1,300 distinct questions.
At most 19 remain, at least 109 short of the required panel. The 19 are **not**
certified eligible: other historical use, adaptation and prompt admissibility
can only reduce that set. No exhaustive project exposure union is claimed or
needed for this failed gate.

| Evidence | Completed rows | Unique source questions |
|---|---:|---:|
| AdaFlash `serving_c1_reference_001/gpu0_ar` timed evaluation | 1,024 | 1,024 |
| Same run's warmup | 1 | 0 additional |
| October 5 objective check, `code_s0_buffered_gsm8k` | 276 | 276 |
| Overlap between these two histories | — | 0 |
| Prior-use union lower bound | — | 1,300 |
| Eligible pool upper bound | — | 19 |

## Source validation

[The executable audit](crosscheck_freshness.py) passed 6,354 assertions. Its
[compact receipt](CROSSCHECK.json) records exact source paths, SHA256 hashes,
sizes, environment, counts and uncovered source indices without raw prompts.

- Rehashed the official test parquet against both recorded SHA256 pins and its
  byte count. Its revision equals the native `ORIGINAL_MANIFEST.json` revision.
- Established a full bijection between all 1,319 historical opaque IDs and the
  pinned source questions by exact text equality. Exact and NFKC/whitespace
  normalized questions are each unique.
- Checked actual `REQUESTS.jsonl`/`EVALUATION.jsonl` records against their request
  manifests and passed terminal results. The objective-check output file also
  reproduces its terminal SHA256. The historical map bytes equal the dataset
  SHA256 declared by that execution.
- All 1,024 native evaluation IDs and prompt-token arrays exactly match the
  completed AdaFlash timed evaluation. The extra warmup repeats an existing ID.
- Recomputed every AdaFlash rendered-text/token hash. All 276 objective-check
  content hashes match their exact source questions plus the common recorded
  reasoning instruction; all 276 rendered-prompt hashes match the recorded
  common wrapper, and token-array hashes agree.

This was independently calculated from source records; no other agent's audit
or answer was read. The audit selected no new panel and inspected no candidate
model outcome to select data. It is not a GPU reproduction or an efficacy test.

## Interpretation limits and work

This is a data-readiness failure, not evidence that persistent prefix withholding
fails scientifically. No old freshness exception, panel reuse, split change or
reduction in panel size is authorized by this audit. Stage 1/project None remain
unchanged. Reopening needs a user-approved revision of the data contract or
source evidence establishing that these executed histories were misidentified;
the current authenticated records support the stop.

New model forwards, Adam applications, complete generations, GPU model work,
SSH/network calls and changes to experiment code: **zero**. Only this independent
audit script and compact evidence were written.

Statistical fallacy scan: **11/11 checked**. Simpson's paradox, ecological
fallacy, Berkson's paradox, collider bias, base-rate neglect, regression to the
mean, correlation/causation and reverse causality are inapplicable to this
deterministic set bound. Survivorship is addressed by counting actual completed
rows and stating the exclusion union as a lower bound. Look-elsewhere and
forking-path risks are addressed by preserving the fixed 128-request requirement
and stopping once two documented histories make it impossible. No statistical
significance, uncertainty interval or algorithm performance claim is made.
