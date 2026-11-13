# Panel-publication review

Status: **passed local metadata checks**. This review performed no model work,
remote operations or inspection of partially completed performance outcomes.

`publish_panel_selection.py` adds two publication artifacts from the original
sealed panel manifest, without modifying live scientific code:

- `PANEL_SELECTION.json`: source revision/binding, candidate and eligibility
  totals, exposure tiers, exclusion reason counts, deterministic ordering rules,
  input-only restrictions and source/helper/output identities.
- `panel_exclusions.csv`: all 1,319 test candidates, including selected,
  eligible-but-unselected and excluded rows, with identifiers, selection/path
  hashes, exposure flags and allowlisted exclusion matches.

The independent metadata reduction checks **1,319 candidates = 1,063 eligible +
256 excluded**, with **256 selected + 807 eligible unselected** and 64 selected
fixed-start requests. All 256 selected inputs have disclosed older endpoint
exposure. The exclusions comprise 192 PR43 outcome inputs and 64 PR34/PR40
discovery inputs; these are per-candidate matches in this test population, not
the size of the original training/exclusion inventory. No exclusion was relaxed.

The selected order exactly matches the retained panel. Its original binding is
`a53e38cc65fd9fd474e994fdbf7f7467dfa00951e1ff511e9b61ff57557c6f5e`.
The helper independently recomputes tier/hash ordering and the separate path
subset. It pins the manifest bytes; the binding itself is carried from the
already sealed original panel, whose private token fields are intentionally
absent from the public-format input manifest.

Four synthetic CPU tests pass: full-population/privacy, invalid
counts/order/outcome-selection, unsafe identifiers, and shortfall preservation.
A separate check of the actual output confirms all candidate IDs, exact selected
order and counts; no prompt/response/token arrays, stop-token values or private
filesystem paths appear. Free text and unrecognized nested fields are omitted
by allowlists. Outputs are under `final_publication_additions`.

Keep the full `audit_final.py` result private because its input-identity keys and
missing-file fields retain exact private artifact paths. Publish only a redacted
audit receipt after the complete experiment has finished and passed raw checks.

This is a separate reduction by the same research team. Its author also
contributed panel preparation; it is not an independent GPU replication.
