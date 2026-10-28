# Independent first-group audit

The GSM8K group passes **14,388 independent checks** across 32 generated records,
1,400 verification blocks, exact source-anchor exposure and initial/final RNG state.
No model execution, training or remote operation was performed by this audit.

| Workload | Frozen calls | OSD calls | Call change | Frozen / OSD request seconds |
|---|---:|---:|---:|---:|
| A: eight trained source requests | 356 | 380 | +6.74% | 15.4457 / 15.3988 |
| B: eight held-out requests | 303 | 361 | +19.14% | 11.8009 / 13.8760 |

Positive call change means additional verification work. All paired output IDs
and termination agree. Frozen traces reproduce the retained original trajectories.
The fresh timing measurements are single, instrumented observations in fixed method
order; the tiny source-time reduction does not establish a speedup despite the
increased call count.

The OSD event has 16 presentations, eight optimizer steps and 15,014 supervised
position exposures. Every sampled anchor, exposure, learning rate and initial/final
RNG component matches the historical C arm. Training loss values from changing
anchor layouts do not establish fixed-example loss improvement.

Training is 33.2021 s, containing 1.6560 s of teacher-head work. Checkpoint save is
38.7087 s and export 18.5287 s; these are inside the 90.5941 s event wall boundary.
Activation is 0.3425 s. Historical teacher preparation, 13.8745 s plus 0.3782 s
persistence, is reused evidence and not newly spent campaign time. The reporter
keeps these boundaries separate and does not add nested teacher-head cost twice.

The raw trace audit reconstructs matched draft prefixes, first rejection offsets,
EOS/cap truncation, target correction/bonus commits, accepted-token counts and
progress. Per-request CSV fields and aggregate sums agree with the raw records.
Execution receipts report frozen-target and publication checks; this audit does not
repeat a full model-weight comparison.

Machine-readable evidence: `AUDIT.json`. The local-only `audit.py` accepts another
retrieved campaign directory and `--groups` to audit the final complete campaign.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
