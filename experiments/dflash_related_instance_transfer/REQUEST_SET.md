# Sealed request preparation — 2026-10-21

All **12 children** were constructed from the frozen prompt-only packet and
reviewed before any model generation. No parent was replaced, no case was
removed, and no saved response was used for construction. The author/reviewer
had prior aggregate project context; **construction was not blinded**.

| Group | Parent suffix | Child change | Prompt tokens |
|---|---|---|---:|
| GSM8K 00 | 01 | New quantities in comparison/addition/subtraction problem | 78 |
| GSM8K 00 | 02 | New total and fraction in complement-count problem | 61 |
| GSM8K 01 | 00 | New quantities/prices in revenue-difference problem | 105 |
| GSM8K 01 | 01 | New combined count in linked linear-relation problem | 73 |
| CodeAlpaca 00 | 03 | Even-filtered sum of squares instead of plain sum | 65 |
| CodeAlpaca 00 | 00 | SQL completed-order counts, including zero-order customers | 92 |
| CodeAlpaca 01 | 00 | Required/forbidden characters and case-insensitive filtering | 112 |
| CodeAlpaca 01 | 03 | Reverse temperature-conversion direction | 46 |
| ShareGPT 00 | 01 | SAS SVG output instead of EPS | 45 |
| ShareGPT 00 | 03 | Substantively different fictional platform/reliability CV | 1,457 |
| ShareGPT 01 | 01 | New eighteen-item bilingual Markdown extraction content | 568 |
| ShareGPT 01 | 02 | Multilabel support-ticket analysis instead of single-label news | 195 |

The code children change required behavior, not only examples. Review confirmed
well-posed math changes, complete task inputs, task/format/language continuity,
and no mere paraphrases of the original 48 or other children. Approximate
complexity was reviewed, not proven equal: explicit SQL/filtering/multilabel
constraints add some processing. The original character-filter parent is
underspecified; the child defines a substantive new constrained task without
claiming knowledge of the unknown parent answer. No external links in prompts
were opened or needed for construction.

Two revisions were recorded before sealing: correct a revenue-ordering statement
in construction metadata, and align the fictional CV's stated experience duration
with its dates. The private record preserves the original draft, exact revisions,
final construction and independent review.

The pinned tokenizer reproduced **all 48 original input-token sequences** before
sealing the twelve new ones. Every child fits the unchanged effective prompt
limit. No model answers or new performance measurements exist.

## Artifact bindings

| Binding | SHA-256 |
|---|---|
| Original manifest file | `1411a2b76a2b4c4b6352b1f338f3ce47125ae4b929f7f715876d8a590a787f0d` |
| Frozen parent-selection payload | `85c3c4c597a50b8586fa7b8915063bcaf3858483a161e22adc893595d4754fe8` |
| Original prompt-only catalog payload | `db5c96964065bbf89250e62f86fe997a60fc8d670432dcca8fc3aeeb28b21702` |
| Final construction file | `4c132b0cc69d100aac0cb448fdace662196e0964c8a452a1895b38d8487df327` |
| Sealed request-manifest file | `71a6388c44b79f19571542d2fee42dcdbe0c2cbe4d654c2666dde5e49b0e53b6` |
| Request-manifest payload seal | `3ae6c9adf2248be9158de410c961fcb28a02e1ce6d72affce6c1026c8179621c` |

The payload seal excludes its own seal field; file SHA hashes exact saved bytes.
The complete manifest and readable parent/child preview remain in the private
review package. They include source material and artifact paths and are not
uploaded to this code repository. Code provenance is bound separately so code
repairs do not silently reseal or change the request set.
