# Matched OSD follow-up: results

## Material Passport

Bounded code experiment, completed on A800 GPU 0, 2026-10-23 UTC. Execution
exit 0; all three groups qualify. This is the authorized OSD follow-up to the
projected-target screen, not an AdaFlash paper reproduction or a persistent
per-turn conversation campaign. No settings were retuned after observing results.

## Main finding

**The new OSD objective did not improve held-out drafting under the previous
experiment's training recipe.** On 24 held-out requests, verification calls
increased from **1,973 to 2,156 (+9.28%)**, with identical output token IDs.
Two requests improved, 21 worsened and one tied. All three dataset totals worsened.

There is one useful positive observation: on the eight ShareGPT source requests
used for training, calls fell from 1,440 to 1,275 (11.46% fewer), while its eight
held-out requests worsened. This demonstrates better native drafting on those
fitted examples without a corresponding held-out gain. It does not establish a
particular cause, fixed-example KL improvement, or that forward KL generally fails.

## What stayed matched

- Execution uses A800 `/data/sources/minhao/DFlash-Online`: its `online_feedback`
  objective and trainer, current SpecForge, and official DFlash inference. It does
  not execute an AdaFlash runtime. Fenrir DFlash-Research remains unchanged.
- Qwen3-8B target and released Qwen3-8B-DFlash-b16; frozen target, fresh released
  draft and optimizer at seed 17 independently for GSM8K, CodeAlpaca and ShareGPT.
- The original sealed 8 A + 8 B requests per group and exact saved A responses,
  masks and target features. No newly selected feedback or dataset answers.
  B never trains in this screen; it is previously evaluated project data, not a
  newly sealed final test set.
- OSD uses teacher forward KL at temperature 1, no hard-label mixture. Target
  final features reconstruct teacher distributions in bounded vocabulary chunks.
  Earlier prompt/context, anchors and padding do not receive response loss.
- Two passes, effective response batch size 2, 64 sampled anchors per response
  presentation, identical position weights, response averaging, clipping and
  FP32-master AdamW. All anchors, exposure and initial/final RNG match historical C.
- Eight optimizer steps per group, retaining the prior nominal 6e-4 cosine/.04
  warmup schedule. The exact eight learning rates are in `PLAN.json` and each
  `TRAINING.json`; the first is approximately 6.2374e-4. This is the preserved
  scheduler, not eight constant-rate steps and not the historical 3e-4 recipe.
- BF16 models, FP32 loss arithmetic, greedy block size 16, 512 output tokens or
  EOS, prompt cap 1,024, context limit 2,048, concurrency 1 and fresh request caches.
  Training ends before timed evaluation. No extra warmups or timing repeats.

The newer sampler, optimizer arithmetic and synchronous fixed-pool organization
are disclosed adaptations; see the [coding survey](../CODING_SURVEY.md).

## Paired native results

Changes below are `100 × (sum(OSD) / sum(Frozen) − 1)`: **positive means worse**.
Every row contains eight valid identical-output request pairs, except the pooled
rows, which contain 24. Calls are sums; timing percentages are also ratios of sums.

| Dataset | Set | Frozen calls | OSD calls | Call change | Request-time change | Better / worse / tied calls |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | Source A | 356 | 380 | +6.74% | −0.30% | 1 / 6 / 1 |
| GSM8K | Held-out B | 303 | 361 | +19.14% | +17.58% | 1 / 6 / 1 |
| CodeAlpaca | Source A | 456 | 460 | +0.88% | +4.73% | 5 / 3 / 0 |
| CodeAlpaca | Held-out B | 598 | 659 | +10.20% | +10.14% | 0 / 8 / 0 |
| ShareGPT | Source A | 1,440 | 1,275 | −11.46% | −10.34% | 7 / 1 / 0 |
| ShareGPT | Held-out B | 1,072 | 1,136 | +5.97% | +5.42% | 1 / 7 / 0 |
| **Pooled** | **Source A** | **2,252** | **2,115** | **−6.08%** | **−5.51%** | **13 / 10 / 1** |
| **Pooled** | **Held-out B** | **1,973** | **2,156** | **+9.28%** | **+8.76%** | **2 / 21 / 1** |

Held-out committed progress per verification fell from 4.058 to 3.713; calls per
1,000 output tokens rose from 245.704 to 268.493. Output length is matched at 8,030
tokens. Source pooled improvement is driven by ShareGPT; neither GSM8K nor
CodeAlpaca improves in aggregate calls on A.

Each timing is one instrumented, fixed-order observation including prefill,
drafting, verification and compact trace collection. Large feedback capture,
training, publication and record persistence are outside that timer. Held-out
request time was 75.250 s frozen versus 81.839 s OSD. These descriptive measurements
do not establish a repeated serving-speedup estimate. The small GSM8K A time gain
despite more calls is not evidence of better drafting agreement.

## Relation to the previous C/P screen

For the 16 qualified GSM8K/CodeAlpaca held-out requests, calls are frozen 901,
historical hard CE C 1,023, historical projected-target P 987, and new OSD 1,020.
OSD needs three fewer calls than C but remains 119 calls worse than frozen and
33 worse than P. This is no meaningful rescue of the prior held-out result.
Only exact-token/count evidence is compared across runs; historical timings are
not combined with these measurements. Historical ShareGPT P remains invalid and
incomplete after its preserved output mismatch; this new OSD run does not repair it.

## Correctness and budget

Completed **96/96 native generations, 48 response presentations and 24 optimizer
steps**, with 44,876 supervised-position exposures (including repeated and
overlapping sampled positions; not distinct tokens). All fresh frozen outputs
and verification counts reproduce the original reference. Every OSD output and
termination agrees with its fresh frozen pair. No records are removed.

The first complete GSM8K group passed before automatic continuation; negative
efficacy did not block CodeAlpaca or ShareGPT. All groups record released-weight
initialization, fresh optimizer, target exclusion, unchanged target tensor
identities/versions, publication equality, and exact anchors/exposure/RNG checks.
These are execution checks, not a proof of a globally optimal learning recipe.
All 162 CPU tests passed on both hosts before launch. The independent audit passed
77,592 checks over all 96 records and 8,496 native verification blocks. See the
[raw-record audit](final_audit/REPORT.md) for post-run reconciliation.

## Cost and resource accounting

| Measured boundary | Total across three groups |
|---|---:|
| Training | 98.968 s |
| Full recovery-state checkpoint persistence | 114.026 s |
| Serving export | 55.071 s |
| Training-event wall before commit | 268.607 s |
| Publication / activation | 1.095 s |
| Training-model loading | 51.534 s |
| Native serving initialization | 8.801 s |
| Recorded diagnostic validation | 2.144 s |
| Driver campaign wall | 672.874 s |
| Controller process wall | 675.295 s |

Training includes 4.969 s of teacher-head work; do not add it again. Event wall
contains training, checkpoint persistence and export; it is another boundary,
not an additional additive cost. Native activation and activation describe the
same publication. Teacher preparation is reused, so new preparation time is zero;
the retained original preparation took 78.405 s and remains a real cost for a
from-scratch workflow. Costs are retained separately rather than disguised as
serving time. This fixed-pool campaign includes evaluations of both controls and
source examples, so its wall time is not a deployment episode time.

Maximum recorded reserved CUDA memory was 35.35 GiB. Both A800 GPUs are idle after
completion. Three final OSD model exports remain available; full optimizer states
were retired only after valid native evaluation and verified compact metadata,
as recorded in each group's `trainer/RETIREMENT.json`. Exact optimizer continuation
from those retired tensors is no longer available.

Before launch, 20.973 GB of ten completed October 16 first-update checkpoint
exports were retired after rechecking 568 retained compact evidence files;
wildchat_00 D1/L1 remain as representative exports. These ten weights are not
archived and cannot be directly replayed now. All prior projected-target source,
features, checkpoints and recovery state remain untouched. The aborted slow
offload transferred no complete replacement and deleted no remote source.
[Retirement receipt](RETIREMENT.json), [final resource check](FINAL_RESOURCES.txt).
A800 has 28.128 GB free after the run, above the 8 GiB reserve.

## What this supports, and where to stop

The implementation is natively qualified on this bounded setup. Changing the
objective to teacher forward KL alone did not make the previous training recipe
transfer better than the released draft. ShareGPT supplies a concrete source-fit
versus transfer contrast. Learning-rate effects, finite feedback coverage and
objective-versus-native-metric alignment remain possible explanations, not causes
identified by this run. Changing-anchor training logs cannot establish final loss
improvement on a fixed set of examples. No new learning-rate sweep, alternative
objective or paused campaign has been launched.

## Evidence

- [Per-request raw measurements](final/campaign/PER_REQUEST.csv)
- [Pooled measurements and progress metrics](final/campaign/POOLED.csv)
- [Qualification and comparison summary](final/campaign/SUMMARY.json)
- [Detailed costs](final/campaign/COSTS.json)
- [Pinned plan](final/campaign/PLAN.json), [preflight](PREFLIGHT.json)
- [Exact retrieval check](RETRIEVAL_CHECK.json): 162 compact files, 8,948,638 bytes
- [Remote execution log](final/controller.log), [exit status](final/controller.exit)

The retained `final/campaign/<group>/` trees contain all token IDs, compact native
traces, sampled anchors, loss/exposure logs, initial/final RNG state, configs and
qualification receipts. Final model exports remain on A800 under the corresponding
`campaign/<group>/O/trainer/events/update/export/` paths.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_osd_20261023/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
