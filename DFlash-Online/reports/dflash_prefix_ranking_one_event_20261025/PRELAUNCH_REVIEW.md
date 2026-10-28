# U/P one-event prelaunch review

## Material Passport
ARS experiment-agent, run mode. User-approved correctness qualification and bounded one-event comparison. Internal delegated review; native qualification pending. No external reviewer or scientific outcome is implied.

## Preserved setup
The production algorithm implementation is unchanged from the CPU-reviewed U/P version. Only a thin experiment runner, qualification observer and four-arm evaluator were added. The historical one-event O configuration, 128 prepared responses, two shuffled passes, all 256 presentation/anchor entries and 64-request development panel are reused exactly. Each U/P arm is fresh seed17, 128 steps, lambda1; O is retained. No feedback collection, teacher reconstruction, tuning or second event.

The training code remains DFlash-Online online_feedback over the pinned SpecForge components. Qualified BF16 AdamW, BF16 forward KL, constant3e-4, weightdecay.01, batch1/accum2, up-to512 anchors,8-block objective chunks, gamma7 and Flex attention remain unchanged. Serving remains Qwen3-8B/DFlash-b16, greedy,2048orEOS,4096context,coldcache,C1,SGLangFA3.

## Correctness and accounting
Five disposable qualification trials O/U0/P0/U1/P1 use the first two original presentations after the full128 shuffle. Budget10 current forwards +4 reference +4 detached comparisons,5 temporary steps,0 generations; internal checkpoint recomputation/head work recorded separately. Exact compatibility checks include weights, Adam, RNG, anchors, counters and losses. Prefix masks use an independent sequential oracle; rival-gradient checks use real projected values; reference weights/modes/RNG/activation release and memory are checked. All temporary state is removed before main training on success, retained on failure.

All new training stops before timing. Evaluate768 generations in three rotating orders; exact canonical output/termination, trace validity, fresh F/O retained counts and repeat counts are required. Every correctness failure stops with evidence. First96 generations qualify native switching within the budget. Performance never gates continuation. Counts use one trajectory/request; time uses all repeats. PrimaryP/O, thenP/U,P/F; O/F also retained. Do not pool historical timings or multiply the sample size by repeats.

## Review disposition
The independent reviewer identified model-file membership verification as an incomplete preflight gate. Fixed: exact target/draft file sets and exact retained O-export membership now required, alongside content bindings. No scientific or tolerance changes. All original O artifacts are read-only.

Local full suite:393 tests passed before the final three main-accounting tests were added. All27 focused tests including those three pass. Final installed A800 full suite and source/model/runtime/storage preflight run before GPU launch.

## Storage and timing
Read-only admission found403.31GB free and both80GB A800s idle. No cleanup. Conservative additional allowance includes both model+Adam states and exports, qualification state, serialization slack,8GiB working/staging and8GiB reserve. Preflight records exact required capacity. TrainingGPU1; servingGPU0 after all trainers close. Preserve inherited scheduler allocations. No transfers during timed evaluation. Expected45–70minutes; hard execution bound3hours, cleanup120seconds. No automatic retry or restart after failure.

Final installed A800 suite: **396 tests passed**, zero failures. CPU source/model/runtime/artifact preflight passed;132 retained O artifacts bound. Actual additional peak budget42,556,704,342bytes fits403,312,914,432bytes free. GPU qualification launch is recorded separately.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_prefix_ranking_one_event_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
