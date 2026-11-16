# Independent partial production audit

Audit: **passed** (4,121,758 assertions; 0 errors). Experiment: **INCOMPLETE_NO_EFFICACY_CLAIM**.

Execution commit: `db6e1221d8d1b234d9d07e246025d1afd5dc30d5`. This CPU-only review performs no model work and does not inspect efficacy.

The reference task failed before model loading because its mutable `references/CANONICALS.json` destination was absent. The supervisor then interrupted S8; final evaluation never started. This is execution failure, not evidence about E or S.

| Work | Main planned | Main completed | Pending/unresolved | Main unattempted | Cumulative completed |
|---|---:|---:|---:|---:|---:|
| adam | 640 | 535 | 0 | 105 | 545 |
| response_forward | 1290 | 1080 | 1 | 209 | 1106 |
| response_backward | 1290 | 1080 | 0 | 210 | 1106 |
| feedback_generation | 384 | 384 | 0 | 0 | 384 |
| reference_generation | 24 | 0 | 0 | 24 | 0 |
| qualification_generation | 0 | 0 | 0 | 0 | 8 |
| warmup_generation | 72 | 0 | 0 | 72 | 0 |
| scored_generation | 1152 | 0 | 0 | 1152 | 0 |
| teacher_preparation | 384 | 384 | 0 | 0 | 384 |
| training_initialization | 5 | 5 | 0 | 0 | 8 |

E7/S7/E8/N8 completed and passed retained resume/publication checks. Their exact endpoint counters are896/1,792 for event7 and1,024/2,048 for event8. All384 own-feedback responses and384 preparations completed. The four completed training trajectories and46 complete S8 presentations reconcile to535 new Adam updates and1,070 new training presentations; ten disposable startup forwards/backwards bring completed new F/B work to1,080.

S8 completed23 updates (global step919,1,838 cumulative presentations), then was interrupted during presentation47’s forward. The shared ledger has one unclosed forward reservation. Its copied `unknown=0` fields are stale qualification metadata; this audit reports the pending operation explicitly. No backward or Adam step was entered for that presentation.

S8 has **no durable partial recovery or export**. The remote inventory retains S7’s complete recovery/export. E7’s optimizer recovery was retired only after E8’s exact state/export check, with its export retained. E8 and N8 complete recovery/export remain. Repeating S8 from S7 would redo23 already charged updates; it is not an exact continuation from step919.

| Task | Status | Task wall seconds | Occupied card seconds |
|---|---|---:|---:|
| E_collect | completed | 173.399 | 173.011 |
| S8 | failed | 115.963 | 114.614 |
| references | failed | 0.516 | 0.099 |
| S_collect | completed | 172.477 | 172.085 |
| N_collect | completed | 177.014 | 176.626 |
| E7 | completed | 483.471 | 483.079 |
| S7 | completed | 437.722 | 437.336 |
| E8 | completed | 437.889 | 437.480 |
| N8 | completed | 420.190 | 419.779 |

Original clock epoch remains1794802603.911516. Prior occupation435.094867s plus main2414.109661s equals2849.204527s cumulative. Original elapsed at stop7061.426059s; production wrapper1271.667178s. Both physical cards’ task leases are nonoverlapping; all retained storage boundaries preserve the110GB allocation cap and400GB reserve. These are nested/overlapping time scopes and must not be added together.

The complete private machine-readable audit includes source-file identities, startup state-equality receipts, per-task memory, training/load/publication components, and teacher preparation/persistence costs. The public PARTIAL_AUDIT.json contains its scalar/accounting summary. It preserves the failed/pending operation and all unattempted work. No retry, resource amendment, or experiment completion is implied.
