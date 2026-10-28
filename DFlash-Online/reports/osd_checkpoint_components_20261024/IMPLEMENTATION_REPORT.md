# Four-checkpoint OSD comparison: implementation and readiness

October 24, 2026. ARS experiment-agent implementation and reproducibility review.
**Implemented and installed on A800; CPU preparation passed. No GPU qualification
or scientific generation has been run.** The scientific stage remains Stage 1 /
project None. No new objective, online update, or paused campaign was started.

## What was implemented

Production location: `/data/sources/minhao/DFlash-Online/experiments/osd_components/`.
Use [the reading guide](worktree/experiments/osd_components/README.md) and the
[local source snapshot](worktree/experiments/osd_components/run.py) for review.

The new package uses the existing DFlash-Online native backend, checkpoint
publication, cache handling, generation recorder and scoring. The successful run's
26 pinned source files, original manifest and training config all match their
execution seal on A800. `online_feedback`, upstream training/inference, model
weights and Fenrir `/home/minhao2/DFlash-Research` were not modified.

| Arm | Target-feature conditioning C | Draft decoder body B |
|---|---|---|
| F | Released | Released |
| O | Final successful OSD | Final successful OSD |
| C-only | Final successful OSD | Released |
| B-only | Released | Final successful OSD |

C is exactly `fc.weight` and `hidden_norm.weight`: 83,890,176 parameters. B is the
remaining 56 tensors (`layers.*` and `norm.weight`): 964,736,256 parameters. C is
about 8% of these draft parameters; two tensors do not imply negligible training
cost. All substitutions copy BF16 bits exactly, without interpolation or updates.

The two hybrid exports have passed CPU checks of all 58 tensor keys, shapes,
dtypes, finite values, exact source membership, complete/disjoint partition,
endpoint reconstruction, configuration equivalence and unchanged source files.
Native publication/58-source-to-43-native equality remains a GPU check.

## Settings and budget

| Item | Prepared value |
|---|---|
| Models | Pinned Qwen3-8B target and released Qwen3-8B-DFlash-b16 |
| Trained checkpoint | Original event 58 export, after 7,424 optimizer steps |
| Evaluation | 64 uniformly selected original held-out GSM8K requests; seed 20261024; original order |
| Repeats / order | Three capture-disabled repeats; rotating four-arm order by selected-panel ordinal |
| Total | 768 generations, including first-eight qualification's 96; zero training steps |
| Inference | Greedy BF16, SGLang/FA3, block 16, concurrency 1 |
| Length | 2,048 output tokens or EOS; canonical context 4,096 plus existing internal 16-token allowance |
| Prompts | Exact saved token IDs; selected prompt length at most 152 tokens |
| Feedback / warmup | No capture, no training, no extra generation warmups |
| Cache policy | Existing cold-cache maintenance before requests and publications |

The original OSD learning recipe is preserved as provenance: pure forward KL,
constant 3e-4, BF16 AdamW, weight decay .01, 128 responses/event, two shuffled
passes, effective response batch two and at most 512 anchors per response
presentation; Flex attention and BF16 probability arithmetic. **These settings
are not run again** in this checkpoint-only experiment. Training sequence length
2,272 is distinct from the 2,048 generated-token cap and 4,096 serving limit.

The selected IDs and canonical references are sealed in [PANEL.json](a800_prepared/PANEL.json).
Selection precedes hybrid construction and does not inspect improvements. F/O must
reproduce saved output IDs and verification counts. Every arm generates naturally.
A common cohort requires all four arms and all three repeats to be valid; invalid
or incomplete work remains visible. Structural errors stop. Later isolated output
mismatches exclude the entire request from gain claims. No performance win is
required to pass qualification or continue the remaining 56 requests.

## Review and validation completed

The actual A800 coordinator environment passed **322 CPU tests**, including
**46 new focused tests**. [Full log](ALL_CPU_TESTS.log).
CPU preparation finished in 49.66 seconds. A separate post-preparation recheck of
input seals and actual model/checkpoint/source files passed in 23.26 seconds.
These are implementation checks, not GPU qualification.

Two independent internal reviewers checked checkpoint provenance, native APIs,
execution/failure accounting, statistical pairing and timing. Their actionable
findings were fixed:

- Seal construction receipts and revalidate actual checkpoint, target and code
  files before loading. Check added files as well as changed files.
- Explicitly publish initial F; require exact checkpoint identity in native
  equality replies and valid cache/helper state on every publication.
- Preserve a rejected measurement, then stop on structural/runtime/capture/count
  errors. Only coherent canonical output/termination discrepancies may enter the
  later partial-cohort path.
- Keep the 80% descriptive threshold unavailable when O's gain denominator is
  unstable; independently reject impossible accepted-token accounting in analysis.

Regression tests cover these boundaries. Neither reviewer ran a GPU experiment.
No repeated model-file hashing or whole-model equality scan was added inside
request timing. Full native tensor equality runs once per arm; normal publication
checks still run every switch.

Historical target/released artifacts match the original transfer manifest,
including tokenizer/configuration/source files. No historical file digest for the
final O export was found: its retained event-58 receipt identifies the checkpoint
and step count; its current digest is sealed. Historical output/count reproduction
is therefore still required. This is not claimed to be a historical O hash match.

## Measurement and output

Request latency includes prefill, drafting, verification and the existing
per-request cache maintenance. Cache time is nested, so it is neither subtracted
nor added twice. Loading, checkpoint activation and full equality checks remain
outside request latency. CPU construction is already inside preparation time.
The campaign wall time includes all benchmark overhead. No adaptation-inclusive
speedup can be inferred from this no-training experiment.

Reports retain all raw repeats and per-request results. Verification accounting
uses designated repeat 0; request time uses all three repeats. Pool ratios from
sums, report gross saved/extra calls, residual gap to O, required marginal effects
and interaction, and descriptive 3x3 cross-tabs. The 10,000 paired request
bootstrap resamples the same request indices across methods. It describes panel
composition sensitivity, not training-seed robustness or uncertainty in the exact
fixed-panel totals. Do not filter zero/negative denominator resamples or clip
benefit-retention ratios.

Expected files: prepared inventory/config/qualification receipts; raw native
records and `MEASUREMENTS.jsonl`; `ACTIVATIONS.jsonl`, per-arm equality receipts,
status and costs; per-request/pooled/interaction/cross-tab CSVs and a JSON/Markdown
report. A failed or timeout run still reports attempted, unknown and unattempted
work. No automatic retry or extra scientific case is authorized by the runner.

## Storage and estimated duration

A800 free space before preparation: 52,772,810,752 bytes (52.77 GB).
After preparation: 48,560,406,528 bytes (48.56 GB). The two hybrids account for
about 4.19 GB; admission required at least 18 GB, including atomic overlap,
workspace and an 8 GiB reserve. No files were deleted. Fenrir had about 83.48 GB
free; only compact evidence/source was retrieved, not the hybrid weights. The
user's live Fenrir F5 debug process was left untouched.

**Expected GPU benchmark duration: 35–60 minutes. First eight-request
qualification: approximately 4–8 minutes, included in that total.**

The actual selected historical F/O requests take 304.21 s for 384 generations.
Using their mean to estimate the two hybrids gives 608.41 s for 768 generations.
Historical activation averaged 1.757 s; 768 publications contribute about
1,349.41 s. Together with loading, the subtotal is approximately 33 minutes.
The range allows current disk/GPU variation, four first-use equality checks,
artifact preflight and report generation. Hybrid generation speed is unknown
until measured. [Estimate and sources](RUNTIME_ESTIMATE.json).

The wrapper enforces a two-hour limit, followed by up to 120 seconds for orderly
cleanup. It requires idle GPU serving and does not launch background training.
No GPU run has begun as part of this implementation task.

## What the eventual result can justify

- If C-only preserves much of O's benefit, consider a separately reviewed,
  matched-exposure test of training only conditioning weights.
- If B-only preserves the benefit, investigate the body under the corresponding
  matched control.
- If neither hybrid preserves O's gain, examine their interaction/co-adaptation;
  this does not prove full-weight training is necessary.
- If both work, the retained changes may provide overlapping routes to benefit.
- If O has little gain on this valid panel, report weak evidence and unstable
  ratios rather than selecting a mechanism from the hybrids.

This experiment identifies useful retained checkpoint changes. It does not prove
that a restricted update can learn them cheaply or establish a new mechanism.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/osd_checkpoint_components_20261024/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
