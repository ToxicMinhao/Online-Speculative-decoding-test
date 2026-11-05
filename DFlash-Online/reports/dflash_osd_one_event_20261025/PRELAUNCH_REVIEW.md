# One-event OSD review

## Material Passport

ARS experiment-agent, run mode. User-approved baseline-only execution, October 25. Stage 1/project None unchanged.

## Implementation

Only `experiments/osd_one_event` and focused tests are added to A800 DFlash-Online. Existing inference, controller, feedback conditioning, sampling, objective, optimizer and SpecForge trainer remain unchanged. An independent local agent inspected storage, data/config equality and failure reporting. Its repeat-count reporting finding was fixed before sealing; all337 A800 CPU tests passed. No new ranking objective is implemented.

## Protocol

Qwen3-8B target / released Qwen3-8B-DFlash-b16; BF16, SGLang FA3, greedy, block16, concurrency1, cold caches, 2048 output tokens or EOS, full exact prompt IDs. Training: frozen-target forward KL, temperature1/no hard-label mixture, response_mean, BF16 AdamW constant3e-4, weight decay.01, clipping1, two shuffled passes, effective response batch2, up to512 eligible anchors per response presentation, eight-block objective chunks, gamma7, Flex attention,2272 training tokens,seed17.

A is the original first128 chronological GSM8K train requests, served fresh by released F. B is the exact previous64-request development panel, not a new or blind test set. Prompt-ID overlap is zero. One128-step update,256 response presentations. The original7424-step constant-LR horizon remains in the configuration; the runner invokes exactly one event and then closes training.

Evaluation has384 fresh capture-disabled generations, with balanced rotating F/O order. First8requests/48generations are the embedded correctness qualification; no extra generation warmup or U/P check. Historical F counts, exact canonical outputs, checkpoint/helper equality, traces, measurement bounds and repeat counts are checked. The old7424-step O counts are not a validity reference for new128-step O. Invalid outcomes stay visible and do not enter gain claims.

## Operations

Both A800 GPUs were idle;48.40GBfree. Fenrir83.56GBfree, userF5debugger untouched. No deletion. Collection storage admission budgets the full capped inputs; a second gate uses completed response lengths before teacher preparation/training. Both retain8GiBreserve,2GiBwork and8GiBcompact allowances. One new state and one export coexist with native/prepared features. All captures are retained unless a later explicit audited retention action is recorded.

No background training or artifact transfer during timed replay. Retrieve evidence only after run completion. Detached supervisor records start/end/exit;7200-second hard limit plus120-second shutdown grace. Estimated30–45minutes after launch. No automatic retry or additional event.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_osd_one_event_20261025/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
