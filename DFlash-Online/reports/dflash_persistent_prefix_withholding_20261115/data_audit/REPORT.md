# Fresh-panel feasibility audit

## Material Passport

- Workflow: ARS-Codex 3.24.2, experiment-agent; CPU source and identity validation.
- Status: **ANALYZED — required data gate fails before model execution**.
- Scope: the current persistent-prefix handoff's original GSM8K source split and its globally unused 128-request evaluation requirement.
- Execution: local CPU only; no model loading, GPU operations, training, generations, network access or dataset mutation.

**The specified evaluation panel cannot be formed.** Retained completed execution records prove that at least **1,311 of the 1,319** questions in the pinned GSM8K test split were already used. At most **eight** can remain, before additional exclusions. The requested panel therefore has a shortfall of **at least 120** questions.

This is a bound, not a certification that the eight remaining IDs are unused. Later studies, other aliases, adaptation overlap or admissibility checks can only reduce this remainder. No panel was selected, no old panel substituted, and the former O/K study's older-exposure exception was not applied to this new handoff.

## Reconstructed completed-use evidence

The audit reads the Parquet question column only and maps native opaque IDs to official source rows by unique NFKC/whitespace-normalized question equality. Both the 7,473-row train map and 1,319-row test map are bijective. It checks actual completed output/request IDs against their original manifests, rather than treating reserved requests as execution.

| Retained execution | Completed unique test rows | Additional rows in the union |
|---|---:|---:|
| AdaFlash serving reference, `gpu0_ar` | 1,024 | 1,024 |
| Objective check, `code_s0_buffered_gsm8k` | 276 | 276 |
| October 22 projected-target F on GSM8K B8 | 8 | 8 |
| Async passes, `final_one_pass` | 256 | 3 |
| **Union** | **1,311** | **1,311** |

The first two completed runs alone establish an upper bound of 19, already insufficient for 128. The other two tighten it to eight. The audit checks successful completion receipts, exact requested/returned ID sets, the objective-check output digest, and nonempty completed October 22 responses. It independently recomputes the union and checks overlapping source hashes against the retained October 27 inventory.

Source: `openai/gsm8k`, `main`, revision `740312add88f781978c0658806c59bc2815b9866`.

- Test Parquet: 1,319 rows, SHA-256 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`.
- Train Parquet: 7,473 rows, SHA-256 `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`.

The current native manifest uses that exact revision. All 1,024 inherited native evaluation IDs resolve to the test split, and every saved prompt-token sequence equals its corresponding completed historical serving prompt exactly. Thus this is an actual source/normalization/token-identity overlap, not an inference from similar request names.

## Event 8 input check

The retained native stream's positions **896–1,023** contain 128 distinct requests and are disjoint from the first 896 by native source ID, normalized problem text and saved prompt-token hash. They are labeled `event_008`. The event-7 feature-manifest and presentation-schedule hashes match the retained native training plan, and their request identities equal stream positions 768–895, twice each across the two passes.

This input-only event-8 check passes. It does not replace live checkpoint/optimizer/RNG ancestry checks, which belong to later preparation. These training questions have older-project exposure through the original full-stream work; the current handoff permits that and does not require globally unused training questions.

## Decision boundary and artifacts

The fresh final-panel condition is explicitly a CPU stop gate in §4–5 of the handoff. The correct status is **`FAIL_FRESH_PANEL_INSUFFICIENT` / experiment not started**, not an algorithm failure. No scientific runtime estimate or qualification pass is established by this audit. Running the prescribed experiment would require a new user decision about evaluation data; this audit does not make that decision or propose a replacement panel.

- `audit_freshness.py`: reproducible, CPU-only source audit.
- `FRESHNESS_AUDIT.json`: pinned identities, source-bound completed-use sets, current native token checks and event-8 validation.
- `EXCLUSION_LOWER_BOUND.csv`: all 1,311 proven used test source IDs, normalized-question hashes and supporting completed-run identities; no question or answer text.
- `EVENT8_INPUT_IDENTITIES.json`: 128 retained source/request IDs and hashes, without private prompt or token arrays.
- `AUDIT_RUN.log`: successful audit execution reporting the failed eligibility gate.

The audit deliberately stops at a sufficient bound. It does not claim to enumerate all later exposures or certify global semantic novelty. Existing model results, raw records and the original dataset remain unchanged.
