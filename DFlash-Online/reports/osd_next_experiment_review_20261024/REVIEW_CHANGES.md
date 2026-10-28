# Review disposition: OSD checkpoint component test

## Material Passport

- Date: October 24, 2026.
- ARS workflow: experiment-agent, plan mode; design review and revision only.
- Sources: user's [review addendum](/home/minhao2/.codex/attachments/4a5419b0-8f4f-45f3-bba6-014037597fc2/OSD_CHECKPOINT_TEST_REVIEW_MODIFICATIONS_20261024.txt),
  [original proposal](PROPOSAL_BEFORE_REVIEW.md), successful-run
  [results](../dflash_online_osd_rerun_20261023/RESULTS.md), sealed configuration,
  final publication receipt and local SpecForge DFlash source.
- Output: consolidated [PROPOSAL.md](PROPOSAL.md), revision 2.
- Verification: local design/source review. Two independent internal agent reviews
  checked analysis and decision logic; they are not external peer review or GPU
  qualification. No new model performance was measured.
- Scope: no implementation, dataset sampling/seal, remote operation, training or
  benchmark launch. Stage 1 / project None and paused campaigns remain unchanged.

## Decision

Accept the review modifications. They improve interpretation without expanding
model execution. The question remains whether retained conditioning changes,
retained body changes, or their combination preserve the successful endpoint's
useful drafting. This is a standard component diagnostic, not a new algorithm.

The source report supports the motivating current Qwen/GSM8K result: 11.57% fewer
calls and 11.33% less complete request time on 1,024 held-out questions, after
58 events / 7,424 optimizer steps. The report excludes adaptation cost from those
serving gains. This review read that evidence and the publication receipt; it did
not re-execute or independently re-audit all raw measurements. Current A800
availability and the native runtime still require qualification before a run.

## Incorporated review requests

| Review item | Disposition in consolidated proposal |
|---|---|
| Competent pure-FKL OSD reference; pause CE-plus-margin design | Section 1; no objective or optimizer change. |
| Fixed F/O/C-only/B-only arms and exact partition | Sections 2–3; names avoid confusion with historical CE arm C. |
| Same 64 requests, 3 repeats, 768 generations | Section 4; first 8 requests/96 runs remain inside this budget. |
| Designated repeat 0 for deterministic call statistics | Section 6; all timing repeats and discrepancies retained. |
| Gross gains, gross harm and descriptive sign cross-tabs | Section 6.2; common cohort, no post-outcome subset selection. |
| Required conditional effects and interaction | Section 6.3; all algebraic identities and signs checked. |
| Absolute paired bootstrap effects first; ratio instability explicit | Section 6.4; no removal of adverse/undefined resamples. |
| Endpoint usefulness is distinct from independent trainability | Section 6.5; no necessity or cheap-learning claim. |
| Later restricted training requires matched exposure | Section 6.5; frozen body still lies on conditioning's backward path. |
| Expanded machine-readable/report outputs | Section 7; CPU analysis only, no extra model probes. |

## Additional clarifications from this review

1. **One common valid cohort.** All four arms and all three repeats must qualify
   for a request to enter the primary table. Every contrast, G/H statistic,
   interaction, cross-tab and bootstrap uses that same cohort. Keep individual
   failures visible. Partial-cohort results do not repair exclusion bias.
2. **Unambiguous rotating order.** Use selected-panel ordinals 0–63, not original
   manifest indices, with explicit base order and left rotation. Both identifiers
   remain in raw records. This preserves the intended exact positional balance.
3. **Ordinary generation, validation afterwards.** Saved output IDs are equality
   references, not teacher-forced inputs. Each checkpoint generates normally from
   the exact saved prompt IDs.
4. **Valid null is distinct from failed qualification.** O can reproduce its old
   per-request evidence yet have weak gain on the selected subset. This remains a
   valid outcome; continue the declared panel and do not replace it. Incorrect
   outputs, counts, publication or cache state follow the correctness gates.
5. **Explain what intervals mean.** Bootstrap intervals describe empirical
   request-composition sensitivity, conditional on this trained endpoint/cohort.
   Qualified fixed-panel totals are exact. Pointwise intervals are exploratory,
   not simultaneous guarantees, training-seed uncertainty or fresh confirmation.
6. **Keep time pooling explicit.** Use mean summed request latency across all
   three repeats and ratios of summed measurements, never favorable-repeat
   selection or averaging per-request percentage reductions.

None changes the parameter partition, request count, generation count, model,
precision, output cap, correctness tolerance, storage reserve or stopping budget.

## Outcome-to-next-step interpretation

- A useful C-only hybrid motivates a later conditioning-only training comparison
  from released weights, using the successful OSD recipe and competent matched-
  exposure full training. It does not establish that restricted training works.
- A useful B-only hybrid motivates the analogous body-only question. Both useful
  hybrids mean multiple retained configurations work, not uniquely located learning.
- Similar gross savings with less harm supports a promising trade-off. Less harm
  accompanied by much less useful gain can simply be weaker adaptation.
- If both hybrids fail but O works, keep joint OSD; compatibility/coadaptation is
  unresolved. Neither restricted training nor a different mechanism is disproved.
- If O is weak on the valid subset, report absolute differences and inconclusive
  retention. If correctness fails, do not interpret affected gains.

No result from this test alone supplies a theorem, new loss or complete online
acceleration claim. The immediate value is choosing a justified training question
and avoiding another mechanism whose assumed adaptation location is untested.

## Operational record

Fenrir had approximately 78 GiB available at the local session check. The process
snapshot showed no task-owned training process among the inspected entries. Only
small review documents and ledger updates were written; no files were retired,
no cleanup was necessary, and A800 was not accessed. No prospective manifest has
been sampled or sealed. Original artifacts and the pre-review proposal remain.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_next_experiment_review_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
