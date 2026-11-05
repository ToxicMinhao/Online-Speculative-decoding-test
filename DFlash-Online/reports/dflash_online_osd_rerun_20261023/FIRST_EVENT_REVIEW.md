# First full OSD event: independent compact-evidence review

## Material Passport

ARS experiment-agent; review of the authorized running campaign. October 24,
2026 UTC (October 23 local). Sources: copied `first_event/` evidence and the
[sealed config](sealed/config.yaml)/[manifest](sealed/MANIFEST.json). Local
read-only analysis; no remote operation, model execution or training change.
**Verdict: the reviewed first-event correctness/accounting checks pass. This is
not evidence of adaptation efficacy or completion of the 58-event campaign.**

## Findings

- [Trainer result](first_event/trainer/events/event_001/RESULT.json), controller
  receipt and committed buffer agree: exactly the first 128 sealed requests,
  128 distinct fresh responses, two passes/256 presentations and 128 optimizer
  steps from step 0 to 128. Each pass contains every response once; both orders
  differ from source order and from each other. Every second presentation steps;
  all 128 recorded step learning rates are `3e-4`; all recorded metrics are finite.
- Serving prompt IDs match the seal; prepared IDs equal each complete prompt
  plus its actual response. All preparation receipts report response-only masks,
  native label binding and zero label mismatches. Responses span 80–516 tokens;
  maximum total length is 619, with no output-cap hit. Thus this event does **not**
  exercise the new 2,272-token training boundary or 2,048-output boundary.
- Every recorded kept anchor is unique, lies at a response position with a
  following response token, and respects `min(512, response_tokens - 1)`.
  There are 65,096 anchor presentations. The cap binds twice, for one response
  across its two passes; its anchor sets differ. The other 127 responses use all
  eligible anchors each pass. Reconstructing up to 15 subsequent supervised
  positions per anchor gives **949,560 exposures**. All 128 optimizer-boundary
  exposure metrics equal the corresponding two-presentation sums. Nonstepping
  rows contain only the current presentation, while stepping rows contain both:
  summing all 256 metric rows would double-count alternate presentations.
- [Publication check](first_event/events/event_001/PUBLICATION_CHECK.json)
  reports equality for all 58 source tensors mapped into 43 native tensors,
  matching fused KV helpers, zero accumulation remainder and horizon 7,424.
  Capture-disabled frozen replay preserves the initial response and its 24 calls.
  Updated/frozen/reloaded replay of the next sealed request preserves all 172
  output IDs with capture disabled; calls are 33/32/33. Reload equality passes
  without requiring a performance win. This single qualification probe supplies
  no heldout efficacy estimate.

## Cost and retention

The 128 serving receipts sum to **139.201452 s**; the controller's complete
blocking update is **375.471793 s**. Their nonoverlapping production subtotal is
**514.673244 s**. Serving already includes native request/capture time
(101.985349 s) and buffer persistence (36.734452 s). Update components are nested:

| Component | Seconds |
|---|---:|
| Teacher preparation | 4.950080 |
| Prepared-feature persistence | 5.897937 |
| Training | 330.198854 |
| Checkpoint save | 17.198155 |
| Export | 11.581939 |
| Activation | 1.395893 |

The 103.706466 s teacher-head timer is already inside training. None of these
child timers should be added again to the subtotal. Loading, qualification,
publication checks and retirement remain separate. The copied
[STATUS](first_event/STATUS.json) was taken at 200 served requests; its
`reconstructed_episode_s` still reflects the event-1 boundary, while current
serving plus blocking-update fields sum to 589.977201 s. This is a stale interim
subtotal, not evidence of double-counted execution.

[Retirement](first_event/events/event_001/RETIREMENT.json) records exactly 384
removed tensors: 128 native captures, 128 prepared features and 128 terminal
features, totaling 4,147,545,152 bytes. Its protected current-state/export paths
match the trainer result and are absent from the removal list. Recorded free
space rises from 55,131,222,016 to 59,279,106,048 bytes. These event tensors were
deleted, not archived; their direct replay is unavailable.

All 653 compact files listed in the retirement receipt are now copied locally
and match their recorded byte counts and SHA-256 values, including the 128 native
request JSON files retrieved separately. See the
[transfer verification](FIRST_EVENT_EVIDENCE_VERIFIED.json). This verifies the
copied compact evidence, not the current remote state/export contents.
Actual retired mask tensors, optimizer moments and RNG state were not reloaded;
the mask/precision/recovery conclusions are limited to compact validation
receipts, sampled positions, configuration and publication evidence.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_rerun_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
