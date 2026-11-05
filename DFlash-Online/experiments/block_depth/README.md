# K+58 block-depth: first native milestone

Study `vfd_depth35_gsm8k_v1` tests a frozen five-layer K+58 draft. This entry
point implements only native cache/parity qualification and real-skip pricing.
It stops before shallow-readout fitting, gate calibration or FINAL evaluation.

The original FIVE runner remains unwrapped. THREE executes layers 1–3 and the
rank-32 sidecar. Forced adaptive modes execute the real gate and synchronization
before taking the requested branch. ALTERNATE switches 3→5→3→5 across rounds.
All five committed-prefix KV caches are still populated by the native worker.
Only one vocabulary projection supplies the live block, and target verification
and speculative cache cleanup remain the existing implementation.

The sidecar norm starts from the original final norm, A uses seed-17 uniform
initialization in ±1/sqrt(hidden_width), and B starts at zero. Gate coefficients
start at zero with the actual 97-feature shape. These initial values are used
only to price the specified implementation, never to judge learned draft quality.

## Inputs and controls

`prepare.py` authenticates the pinned GSM8K parquet files and Qwen tokenizer,
decodes questions only, and seals READOUT/GATE/FINAL partitions of 384/128/384.
It preserves PR25's raw-question training prompt and suffixed evaluation prompt.
The test partition is disjoint from fitting, not globally unused by the project.
Private rendered inputs and token records stay outside Git.

The first 16 READOUT requests run unmodified FIVE with capture disabled. Their
complete native timings and canonical records are retained. Before the next
cache flush, idle replay reconstructs up to eight evenly spaced original round
starts. Every original-FIVE and forced-five replay must match all 15 recorded
proposals before its measurements are eligible. No reconstructed output changes
history, supplies learning feedback or becomes an additional request sample.

One additional generation per request exercises forced FIVE (first four),
THREE (next four), or 3→5→3 (last eight). Forced FIVE requires identical draft
proposals, verification counts, tokens and termination. Shallow/mixed routes
require identical final tokens/termination and matched-start deep-path parity;
their proposals and number of rounds may differ.

Route pricing measures native graph input/attention metadata preparation through
proposal-ready, including dispatch, device synchronization, readout and one
vocabulary projection. Common committed-prefix KV materialization/allocator
work preceding this boundary is unchanged and cancels in route differences;
it is not mislabeled as separately measured kernel time. The denominator is the
unaltered complete-request clock, including the established cache policy.
Offline cache cloning/restoration is diagnostic cost, not live serving cost.
All graph builds, warmups, shadows and scratch work are counted separately.

The four pricing paths are original FIVE, static THREE, adaptive forced-three
and adaptive forced-five. Up to 16×8×4×4 = 2,048 synchronized measurements use
balanced orders. Continuing cost is adaptive-five minus adaptive-three, never
original-five minus static-three. The optimistic same-progress screen weights
each prompt's samples by its native number of rounds, allows zero saving, and
bootstraps 16 paired prompts 10,000 times with PCG64 seed 20261105. An upper
95% bound below 2% stops on price. A crossing/straddling bound permits considering
fitting; this executable nevertheless stops at the requested first milestone.

## Run after publication and storage admission

```sh
python -m experiments.block_depth.qualify --plan /approved/PLAN.json --output /approved/qualification_001
```

Use the qualified SpecForge interpreter for the parent and the pinned SGLang
interpreter named by the plan for the native worker. `preflight.py` authenticates
the exact K+58 export, target/runtime/source, sealed inputs, idle hardware and
6 GiB additional-write allowance plus an 8 GiB disk reserve. No old optimizer
state or retired feedback packet is needed. Do not run other GPU work or bulk
transfers while collecting route prices. Preserve any failed attempt.

This milestone has 16 READOUT completions, at most 16 extra qualification
completions, zero optimizer steps and zero FINAL completions. It neither tests
the trained gate's endpoint benefit nor establishes online recalibration value.
