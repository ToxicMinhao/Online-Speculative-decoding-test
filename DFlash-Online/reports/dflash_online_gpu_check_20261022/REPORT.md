# Online-feedback real-GPU qualification

## Material Passport

Verification: VERIFIED for this bounded fixture. A800 GPU0; unchanged Qwen3-8B / released DFlash-b16 foundation, seed17, BF16, block16, cap128 tokens or EOS, context2048. DFlash pin `07ebd93db9f472af339b644bb70221ad8428328a`; SpecForge pin `1839aae30003e3573d00979eb1edd57347b3578c`. Current weighted hard-token CE, response-mean accumulation; not the historical AdaFlash forward-KL recipe or projected-target mechanism. Stage1/projectNone unchanged. Fenrir DFlash-Research unchanged.

## Outcome

**Passed**, exit0. Qualification body 201.26s; wrapper204.343s (3m24s), excluding storage preparation. Peak PyTorch GPU allocation 34.19GiB; reservation 34.54GiB. Both GPUs idle afterward.

- Four actual serving answers:128,128,81,128 tokens; three cap hits and one EOS.
- Two updates, each using its own two responses for two passes: four optimizer steps/eight response presentations overall. Each event steps after presentations2and4. Update2 does not retrain update1 records.
- Frozen-target replay agrees with all saved response labels; full input tokens retained, prompt positions masked, only each current response supervised.
- All58 published tensors exactly match trainer weights after each update. Real weights change: U1 changes50tensors/972,820,120coordinates versus released; U2 changes46tensors/555,325,010coordinates versus U1.
- Live-to-saved and saved-to-restored draft, optimizer/master/scheduler, RNG and counters compare exactly after update1.
- Three held-out generations have identical input/output token IDs. Updated and reloaded updated have identical verification counts. Evaluation leaves learning counters and pending feedback unchanged.

## Held-out replay (correctness fixture only)

| Draft | Output tokens | Verification calls | Accepted draft tokens | Instrumented request seconds |
|---|---:|---:|---:|---:|
| Frozen released |128|19|109|0.9586|
| Updated after U2 |128|23|105|0.8922|
| Reloaded U2 |128|23|105|0.8952|

The update needs **more** verification calls on this one held-out request. It therefore does not demonstrate improved drafting here. Single instrumented timings, with fixed evaluation order and no repetitions, establish no speedup despite the lower recorded updated time. Passing means generation, feedback, actual learning, checkpoint switching and evaluation work together on this fixture; it does not establish general correctness across all workloads or learning benefit. No target-only held-out generation was added.

## Actual procedure

1. Create the frozen target, a serving draft initialized from released weights, and a persistent SpecForge training copy with optimizer/scheduler.
2. Caller supplies exact prompt/history token IDs to `OnlineController.serve()`. Native DFlash drafts and target-verifies the answer. Save verified output IDs and trace; queue that response once. Dataset answers and rejected proposals do not become response labels.
3. When the caller requests an update, `update()` reserves the oldest requested number of pending responses. Here it reserves two.
4. `NativeDFlashBackend.prepare_feedback()` replays each saved prompt+answer through the same frozen target. It extracts training features and verifies next-token label alignment. Earlier history supplies context; only that answer's positions receive loss. Feature preparation is currently a separate forward replay.
5. `SpecForgeEventTrainer.train_event()` calls upstream `TrainerCore.train_step()` in pass/response order. With two responses/two passes/accumulation2: response1→response2→optimizer step, then response1→response2→optimizer step. Model, optimizer, scheduler and RNG continue across events.
6. Save restart state, export the draft, validate/load it into inference, then mark the reserved responses consumed. The next `serve()` uses the published checkpoint. Every request creates new target/draft caches.
7. `evaluate()` uses fixed saved inputs, performs no update and supplies no feedback. It checks exact output token IDs and unchanged learning state.

A conversational caller must append each generated verified answer and the next user message before the next `serve()`. This fixture uses independent prompts; it does not itself test growing-history assembly. With full histories supplied, old answers remain conditioning and are masked when a newer answer is trained. Consumed feedback is not automatically replayed in later update events.

## Costs and current boundaries

Across both events: training 18.63s, checkpoint persistence 76.72s, export 38.60s, publication 0.742s. These are component timers; their sum is not the complete wall time, which also includes setup, generation, restart and qualification comparisons. Full restart state is about14.68GB and each model export about2.10GB. This test intentionally exercises persistence; low-latency online deployment is not qualified.

- Current objective: upstream weighted **hard-token cross entropy**, not full-teacher forward KL.
- Teacher features: reconstructed after generation, not reused directly from verification-time hidden states.
- Scheduling/publication: synchronous, with export/loading; no asynchronous worker or cache-reuse claim.
- Current four-step cosine horizon is a qualification configuration, not an indefinite serving schedule.
- No new algorithm, projected target, baseline campaign or Fenrir port was introduced.

## Storage and operations

Initial A800 free14.03GB. Retired33.556GB across exactly three superseded/completed restart payloads, after keeping model weights and verified compact metadata. Newly preserving the uncommitted old draft costs2.10GB. Freebeforelaunch45.49GB exceeded44.299GB admission, including8GiBreserve. Freeaftertest26.58GB (24.76GiB). Old exact Adam/master continuation is lost; weights/results/config/RNG/counters and failure evidence remain. See STORAGE_RETIREMENT.json for exact paths. Compact metadata is also retained on Fenrir.

First wrapper attempt failed before Python/GPU because `/usr/bin/time` was absent. Its logs are preserved remotely in `launch_wrapper_failure_001`; replacing only that OS wrapper with Bash's timer enabled the sole actual GPU execution. No experiment retry, setting change or learning rerun occurred.

## Evidence and entry points

- [Raw result](evidence/RESULT.json), [independent tensor/mask/count audit](AUDIT.json), [run log](QUALIFICATION.log).
- [Original plan](evidence/PLAN.json), [configuration](evidence/config.yaml), [storage admission](evidence/STORAGE_ADMISSION.json), [retirement](STORAGE_RETIREMENT.json).
- `RUN_SOURCE.tar.gz` preserves executed Online source/config; `COMPACT_RESULT.tar.gz` preserves compact run evidence. Full current restart state, exported weights and teacher features remain on A800.
- A800 code: `/data/sources/minhao/DFlash-Online/online_feedback/`. Start at `controller.py` (`serve`, `update`, `evaluate`), then `native.py` (`prepare_feedback`, `publish`) and `training.py` (`_build_runtime`, `_batch`, `_train_passes`).
- A800 run: `/data/sources/minhao/runs/dflash_online_foundation_20261022/refactor_qualification_001`.

Bounded task complete. Stop after reporting; no scientific campaign follows automatically.


<!-- dflash-online-publication-provenance -->

## Published experimental code

Archived source commit: [`52e5a8a870f4691a44caf1428e21110b539a984a`](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/commit/52e5a8a870f4691a44caf1428e21110b539a984a).

Exact retained source variants and their original hash contracts: [run source manifest](https://github.com/ToxicMinhao/Online-Speculative-decoding-test/blob/52e5a8a870f4691a44caf1428e21110b539a984a/DFlash-Online/provenance/runs/dflash_online_gpu_check_20261022/sources.json).

This is a post-execution publication of retained source bytes, not a claim that the experiment originally launched from this Git commit. The manifest distinguishes historical revisions and records unavailable evidence.
