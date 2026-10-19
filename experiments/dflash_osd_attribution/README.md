# Bounded OSD gain attribution

**Review stage.** This change implements the approved attribution handoff. It
does not run the analysis or native diagnostic. CPU tests use synthetic inputs;
they do not establish GPU compatibility or a scientific result. Existing
campaigns remain paused. This is a separate entry point; historical experiment
code and measurements are unchanged.

The question is where the qualified OSD path improves native drafting and which
of those local repairs the original correction space cannot express. There is
no new mechanism, training, learning-rate change, checkpoint reconstruction,
generation campaign, extra case selection, or fresh timing-repeat campaign.

## Fixed evidence and scope

| Completed main stream | Requests | Diagnostic request | OSD state |
|---|---:|---:|---|
| `wildchat_00` | 20 | 20 | After response 19 |
| `wildchat_01` | 16 | 16 | After response 15 |
| `wildchat_02` | 17 | 17 | After response 16 |
| `wildchat_19` | 23 | 23 | After response 22 |
| `personamem_persona42` | 18 | 18 | After response 17 |

Only these 94 requests are admitted. Qualification and the unexecuted
PersonaMem404 stream are excluded. Checkpoints must be the exact released
draft and exact pre-request OSD states recorded in each result, verified by all
original file hashes. An unavailable checkpoint is not replaced by a later
state or retrained. Projection hashes come from the **original preparation
receipt**, not a fresh checksum used as its own provenance.

The native model pair remains Llama-3.1-8B-Instruct and its released DFlash
draft, with the campaign's source-pinned native runtime, full causal context,
greedy decoding, block size 10, BF16 precision and original 1,024-token-or-EOS
canonical outputs. Exact identities/settings are recovered from each original
`NATIVE_CONFIG.json`, rather than guessed from these descriptive model names.
No dataset assistant answers are introduced.

## Existing measurements

`analysis.py` validates all three original frozen/OSD timing rows per request.
It averages repeats within a request and pools ratios from sums. Calls and
token observations use **one validated repeat-zero trajectory per method and
request**, not three independent examples. The primary paired inputs are the
BANK arm's saved trajectory; OSD learned from its own earlier responses.

The tables retain gains, regressions, exclusions, every raw timing repeat,
conversation contributions to net saved seconds/calls, and separate WildChat
and synthetic PersonaMem results. Contributions can be negative or exceed
100% when other conversations regress. There is no inferred kernel breakdown.

Token decisions are aligned by exact canonical prefix IDs and absolute output
position. Verification-call number is never the join key. Target predictions
conditioned on rejected hypothetical proposals are not relabeled as canonical
supervision. Unmatched starts/positions and beyond-output proposals remain in
coverage tables. Native accepted-prefix lengths and first-rejection positions
are preserved separately from these position-level associations.

The declared, mutually exclusive lexical categories are whitespace/punctuation,
code-syntax lexemes, identifiers/numbers, ordinary text, and mixed/uncategorized.
`analysis.CODE_LEXEMES` fixes the short syntax list before results. They are
token-text heuristics, not semantic task labels. Exact token-ID occurrence in
earlier OSD responses with validated completed updates is distinguished from
repetition only earlier in the current answer. Neither association proves
memorization, domain specialization, or semantic novelty. If the pinned
tokenizer is unavailable, categorization is marked unavailable; supported
timing, call and exact-ID recurrence analyses continue without substitution.

Cost rows retain canonical OSD serving, adaptation, export, activation and cache
costs. The outer recurring-adaptation timer is used once in episode totals.
Teacher preparation is a component of the training timer; those nested timers
are not added again. Capture already included in request time is not charged
twice. Missing cost fields stay unavailable. Local diagnostic wall time is
reported separately from serving measurements.

## Deterministic matched-prefix diagnostic

For each final request, sort distinct native frozen block starts. Select
`k=min(n,32)` indices using `floor(i*(n-1)/(k-1))`, including endpoints; handle
zero/one start directly. This uses no OSD outcomes, margins or token categories.
No replacement anchor is selected after invalid reconstruction or missing data.

At most 160 paired anchors and **320 draft-block forwards** are admitted. Each
anchor/checkpoint is durably claimed before model work; failure does not permit
retry. A plan-level execution directory prevents accidental reruns into a new
output directory. The only native worker is a serving worker, without a trainer.

The scheduler RPC cold-prefills the full canonical context **excluding the
already-known seed token**, in the qualified 8,192-token chunks. It uses native
target conditioning and KV materialization, then calls the original single-block
DFlash draft operation with the saved seed. A private stop exits after the
unchanged greedy vocabulary head, before target verification. It does not call
the full-response generation API or append any tokens to history. Frozen and
OSD use the same canonical token prefix and valid checkpoint-specific caches.
Cold reconstruction is preferred over reusing potentially incompatible caches.

Each call records nine proposals, the normalized native BF16 features, native
canonical-token margins against the strongest competitor, and hook/cache/state
checks. All nine reconstructed frozen proposals must match the retained native
block; the reconstructed target seed must match too. Independent prefill can
differ numerically from the original incremental path. Such discrepancies are
retained and excluded from valid-pair claims, never silently corrected, replaced
or rerun. The exact saved seed is used for the diagnostic even if the fresh
target seed differs, but that pair remains invalid.

Correct-prefix length stops before the first mismatch or EOS; a matching EOS is
recorded separately. `block_position` is zero-based over nine proposals, while
`native_block_slot` is 1–9 after seed slot 0. Repairs/new errors after an earlier
mismatch are local positional observations, not realized accepted progress.
Checkpoint grouping reduces reload cost; there are no timed deployment claims
from these diagnostic calls and no newly generated histories.

## Original bank's mathematical capacity

For an OSD-repaired frozen error, let `y` be the canonical target token and `j`
the frozen native winning token. Use the fixed vocabulary rows, frozen hidden
feature and original projection artifacts:

```text
phi(h) = (h @ V_in) / max(1, ||h @ V_in||_2)
r = 1 / kappa                        # bound on combined C_eff, including fusion
d = (E_j - E_y)^T h
B = r * ||U^T(E_y - E_j)||_2 * ||phi(h)||_2
tol = 1e-8 + 1e-7 * max(1, |d|, |B|)
```

Both `d` and `B` use float64 consistently. The constants above are fixed before
outcomes. There is **no multiplication by the number of modules**. Classification
is finalized only after both checkpoints are paired and validated:

- `d-B > tol`: excluded under the specified mathematical correction model.
- `abs(d-B) <= tol`: numerical borderline, inconclusive.
- Otherwise: not excluded by this necessary pairwise test.

Native BF16 logit margins are separate measurements, never substituted for
`d`. Exact float32 representations of the captured BF16 `h`, `E_y` and `E_j`
are retained with fixed projection bindings, allowing independent float64
recalculation. Survival of rounding does not establish useful drafting.

Passing the bound does not show that one shared correction can repair all
competitors, be learned, satisfy protection constraints or accelerate native
serving. Exclusion is mathematical and does not automatically apply to every
rounded implementation. Changed OSD features do not prove full-backbone
training necessary. Sampled local repair frequency cannot be converted into a
percentage of total native time saved.

## Entry points after review

Run from the repository root in the existing pinned environment. Paths below
are illustrative; source the server's approved environment helper before any
native operation and follow its active-job/storage checks. No installation,
download, source mutation, cleanup or remote launch is automated here.

```bash
python -m experiments.dflash_osd_attribution.run inventory \
  --campaign-root /absolute/path/to/original/attempt_001 \
  --out /absolute/path/to/attribution/inventory

python -m experiments.dflash_osd_attribution.run analyze \
  --plan /absolute/path/to/attribution/inventory/PLAN.json \
  --out /absolute/path/to/attribution/offline

python -m experiments.dflash_osd_attribution.run probe \
  --plan /absolute/path/to/attribution/inventory/PLAN.json \
  --reviewed-source-digest SOURCE_DIGEST_FROM_REVIEWED_PLAN \
  --out /absolute/path/to/attribution/native

python -m experiments.dflash_osd_attribution.run report \
  --plan /absolute/path/to/attribution/inventory/PLAN.json \
  --analysis /absolute/path/to/attribution/offline/report/REPORT.json \
  --diagnostic /absolute/path/to/attribution/native/DIAGNOSTIC.json \
  --out /absolute/path/to/attribution/combined_report
```

Commands are separate; inventory and analysis never launch the diagnostic. A
source digest is a reproducibility guard, not a substitute for the user's
review approval. `--map-root ORIGINAL RETAINED` permits explicit artifact
relocation with original hashes; it does not change model/runtime identities.
Native execution still requires the original pinned environment/target paths.

Inventory does not create missing projections or states. Probe revalidates
retained artifacts before loading models and records unavailable cases.
The storage preflight budgets 1 GiB for compact evidence/configuration and
2 GiB for runtime/cache growth, plus an 8 GiB reserve. No model copies, optimizer
states or checkpoint exports are created. The inherited worker checks GPU
occupancy, process liveness and RPC/startup deadlines. Failures stop without
automatic retry; partial results and claims are retained.

## Deliverables and review checks

The report writer produces Markdown plus JSON/CSV tables covering per-turn and
conversation gains, every original timing repeat, native blocks, paired
decisions, recurrence/category denominators, missing evidence, separate costs,
local anchors, native margins and mathematical capacity classifications.
Interpretation ends at attribution and unresolved explanations; no replacement
mechanism is implemented. Private inputs, artifacts and results stay outside
this repository.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -B -m pytest -q experiments/dflash_osd_attribution
```

Tests cover scope/admission, historical artifact identity, shifted block starts,
canonical-prefix alignment, recurrence leakage, missing evidence, raw repeats,
cost overlap, combined-radius/clipping mathematics, native wrapper mutation and
chunking, exact one-block limits, discrepancy exclusion and no automatic retry.
Native GPU qualification remains pending code review.
