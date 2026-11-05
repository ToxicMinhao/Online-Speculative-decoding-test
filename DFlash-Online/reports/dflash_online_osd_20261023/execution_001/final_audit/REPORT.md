# Independent final OSD follow-up audit

**All 77,592 checks pass.** The completed campaign contains 96 fresh generations,
three independent OSD events, 48 response presentations and 24 optimizer steps.
The audit reconstructs 8,496 native verification blocks and 126,286 proposal
positions without executing a model. There are no detected output, accounting,
sampling, RNG or trace-commit discrepancies.

| Dataset | Set | Frozen calls | OSD calls | Call change | Fewer / more / tied requests |
|---|---|---:|---:|---:|---:|
| GSM8K | A: trained source | 356 | 380 | +6.74% | 1 / 6 / 1 |
| GSM8K | B: held out | 303 | 361 | +19.14% | 1 / 6 / 1 |
| CodeAlpaca | A: trained source | 456 | 460 | +0.88% | 5 / 3 / 0 |
| CodeAlpaca | B: held out | 598 | 659 | +10.20% | 0 / 8 / 0 |
| ShareGPT | A: trained source | 1,440 | 1,275 | −11.46% | 7 / 1 / 0 |
| ShareGPT | B: held out | 1,072 | 1,136 | +5.97% | 1 / 7 / 0 |
| Pooled | A: 24 source requests | 2,252 | 2,115 | **−6.08%** | 13 / 10 / 1 |
| Pooled | B: 24 held-out requests | 1,973 | 2,156 | **+9.28%** | 2 / 21 / 1 |

Positive changes mean more verification work. Ratios use summed calls. All 48
F/OSD pairs produce identical output IDs and termination, and all fresh F traces
match the retained original F traces exactly. The previous ShareGPT P mismatch
remains historical evidence; it did not occur in this F/OSD comparison.

The pooled source benefit comes from ShareGPT: GSM8K and CodeAlpaca together
increase source calls from 812 to 840 (+3.45%). Every dataset's held-out total is
worse than frozen. Thus this run demonstrates some source-fitting benefit under
the fixed recipe, but does not demonstrate useful held-out adaptation.

## Training and validity

Each group's 16 chronological presentations and eight steps match the approved
schedule. Exact sampled anchors, weight/exposure calculations, learning-rate
sequence and initial/final RNG states match historical C. Supervised-position
exposures are 15,014 / 14,914 / 14,948 for GSM8K / CodeAlpaca / ShareGPT,
respectively. Prompt/response boundaries, native matched prefixes, first rejection
offsets, EOS/cap truncation and target correction/bonus commits reconstruct the
saved answers. Per-request CSV values and all grouped/pooled sums match raw records.

Qualification receipts report released initialization, fresh optimizer,
publication equality and frozen-target guards. The independent audit checks
those receipts and the compact evidence; it does not rerun a whole-weight scan.
Changing-anchor training losses do not establish fixed-example KL improvement.

## Historical C/P context — counts only

Only GSM8K and CodeAlpaca were fully qualified for all F/C/P arms previously.
Their exact retained outputs and matched training exposure support this contextual
count comparison. No historical timings are included.

| Same first two datasets | Frozen F | Hard-CE C | Projected P | New OSD |
|---|---:|---:|---:|---:|
| A: 16 source requests | 812 | 775 | 890 | 840 |
| B: 16 held-out requests | 901 | 1,023 | 987 | 1,020 |

On B, OSD removes three of C's 1,023 calls (−0.29%) but requires 33 more calls
than P (+3.34%) and 119 more than F (+13.21%). This is not a substantive
demonstration of improved transfer from replacing hard CE with this OSD loss.
It is also not evidence that forward KL is generally unsuitable: the result is
limited to the fixed model, feedback, sampling and optimization recipe.

## Costs and timing limits

The actual campaign driver wall time is **672.8745 s (11.21 min)**. New training
totals 98.9683 s, including 4.9692 s of teacher-head work. Checkpoint persistence
is 114.0264 s and export 55.0708 s. These are inside the 268.6069 s summed event
wall boundary. Activation totals 1.0953 s; trainer loading is 51.5338 s. Historical
teacher preparation is 78.4055 s and is reused, not newly spent campaign time.
These timer boundaries are not an exhaustive additive decomposition.

Fresh single-request time sums are A: 87.7758 s F versus 82.9357 s OSD (−5.51%),
and B: 75.2496 s F versus 81.8395 s OSD (+8.76%). Measurements include prefill,
drafting, verification and compact tracing but exclude training and activation.
There is one fixed-order instrumented observation per request, with no added
generation warmups or timing repeats. The timings are descriptive; they do not
establish robust serving speedup or adaptation-inclusive acceleration.

The evidence supports stopping at the completed comparison and discussing the
optimization/source-fit/transfer distinction. It does not identify a specific
failure cause, validate the proposed replacement mechanism, or authorize retuning.

Raw audit: `AUDIT.json`. Reusable local-only audit entry:
`../first_group_audit/audit.py`. This audit made no remote call, generated answer,
optimizer step or scientific code change.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
