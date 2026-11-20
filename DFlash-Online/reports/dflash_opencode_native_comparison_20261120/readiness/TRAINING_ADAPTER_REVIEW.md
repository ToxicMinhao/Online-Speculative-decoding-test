# CODE training adapter and resource review

Implementation source: `c0a0794e8a7a42a3e2e70833a73992a6201ea6d6`. Native execution: **not started**.

Review scope: inspection of the adopted November 20 handoff and retained DFlash-Online training, publication, storage and cost evidence, followed by the parent-authorized dedicated CODE adapter, bounded qualification and retention implementation. ARS experiment-agent execution review. Only CPU tests, including a tiny native training runtime, have executed. No GPU model work, remote operation, production training or commit was performed by this reviewer. Research remains Stage 1 / project None.

## Conclusion

The three numerical algorithms already exist and can be reused without a new training implementation. A separate CODE orchestration/validation adapter is required: old runners encode GSM8K, context 2,272, 7,473 served requests including a 49-request tail, and retirement of the immediate predecessor. Those assumptions conflict with this handoff. Source inspection identifies the exact hooks below.

Resource admission is **not established yet**. An actual sealed-window prompt sum can give a substantially tighter prospective feature bound than 128×4,096, while preserving the 2,048-token cap for every response. However, the 80 GB ceiling includes all three arms' retained event-57/58 recoveries, publication overlap and compact evidence. A one-arm historical storage plan cannot be reused. The current 493.8136 GB free-space observation leaves 93.8136 GB above the shared 400 GB reserve, so the experiment's 80 GB cap is the tighter limit before other users' writes.

## Narrow implementation boundaries

| Concern | Existing source | Required CODE behavior |
|---|---|---|
| Training engine | `online_feedback/training.py:173,225,482,543,580` | Keep SpecForgeEventTrainer, native shuffle/anchor sampling, response averaging, accumulation, clipping, BF16 AdamW and export. Context admission changes to 4,096 in the new adapter. |
| O | `online_feedback/training.py:235,302` | `objective='osd'`, no point-loss extension, no first-error redistribution or C controller. |
| AREPO | `online_feedback/adaflash_repo.py:67,113,144,199`; `experiments/adaflash_arepo/training.py:101` | `objective='osd'` with `install_adaflash_repo`. Retain the literal source probability operations, global response denominator, normalized-position clipping and late mixture. No head. |
| C | `experiments/full_c_gsm8k/training.py:147,160,195`; `online_feedback/kplus.py:138`; `online_feedback/prefix_allocation.py:170` | `objective='osd_first_error'`, install K+ then unchanged C relative cap .25/smoothing .5. Preserve full-valid scoring, prefix-only allocation, neutral additive operations and observations. |
| Recipe checks | `experiments/ok_coding/training.py:35`; `experiments/full_k_gsm8k/training.py:35,55,101` | New strict CODE validator accepts 4,096 and the named arm/dataset/binding only. Do not widen or mutate historical validators. A read-only compatibility view is acceptable only after actual CODE fields are checked. |
| Event sizing | `experiments/full_k_gsm8k/training.py:378`; `experiments/osd_persistent/run.py:203,272` | Exactly 58 windows of 128; no tail loop, no partial event or extra intermediate efficacy evaluation. Each event has 256 presentations/128 steps; cumulative horizon 7,424. |
| Boundary publication | `online_feedback/training.py:663–686`; `experiments/full_k_gsm8k/campaign.py:107` | Pending recovery is saved, export produced, state atomically committed, exact state/export checked and serving publication verified before next feedback request. Retained boundary link shares the committed inode, not a second copy. |
| Short responses | `online_feedback/sglang_backend.py:160–178` | Existing preparation returns `ineligible_short_response` for <2 tokens. New controller must stop the dependent window while preserving it; do not omit/replace the request or count the skipped preparation as a forward. |
| Retention | `experiments/kplus_full_gsm8k/storage.py:253–302` | Old code deletes event n−1 state/export. New CODE policy must keep n and n−1 complete states, retire n−2 only after full successor publication, and retain 57/58 plus export58 for all three arms. |

Use fresh isolated processes for each O/AREPO/C trajectory. Installers are model-specific, but samplers/observers and training backends have lifetime-bound state; arm separation must not depend on cleanup of another arm's installed controller. One process contains one continuous arm. No trainer reset/reseed between windows.

## Initialization and two-card sequence

PR54's confirmed startup repair imports `rich.style` before the existing native seed-17 call (`full_c_gsm8k/training.py:160–166`). Keep that order in original and adapted qualification branches. Preserve the native fixed compiler/startup procedure, ordinary train mode, fresh empty Adam, all RNG components and exact released weights. No post-construction RNG overwrite is a substitute for equal initialization.

A new initial witness may normalize only declared differences: output/run identity, experiment/arm/dataset/binding metadata, intended loss-controller identity, and the common context change from 2,272 to 4,096. It must independently check both values before normalization. It may not normalize model/Adam tensors, seed, LR, counters, accumulation, sampling settings, optimizer options or unexplained RNG components. `full_k_gsm8k/training.py:224` provides the old exact initial comparison; `full_c_gsm8k/training.py:36` retains mismatch evidence before raising. Old witnesses are immutable.

Card A serves target/current drafter and captures the arm's own 128 responses. Card B trains only that arm from those prepared rows. This uses both GPUs but remains causally synchronous: finish both passes, exact recovery/export checks and publication before collecting the next window. No independent arm concurrently consumes a future window. Final timing needs both cards and competing project CPU/I/O work quiet except the chosen serving process. Count both allocation intervals as occupied card-time, even while a card is waiting.

The inherited four correctness generations per arm remain: one capture-disabled reproduction of first feedback response; after event 1, updated/frozen/reloaded executions of the designated next request. Their tokens/termination must agree, checkpoint publication must be exact, and these diagnostic calls supply no feedback or training RNG change. Three arms therefore use 12 correctness generations, separately from 22,272 feedback generations.

## Exact work reconciliation

| Work | Production total |
|---|---:|
| Feedback generations | 22,272 |
| Events | 174 |
| Adam steps | 22,272 |
| Response presentations / core forwards / core backwards | 44,544 each |
| Terminal teacher preparations | 22,272 |
| Inherited correctness generations | 12 |
| AR/F full-panel precheck | 2,048 |
| Final scored generations | 15,360 |
| Final activation warmups | 240 |
| Fixed generations | 39,932 |
| Extra bounded qualification generations | at most 64 |
| Total generation ceiling | 39,996 |

Qualification additionally permits at most 48 disposable Adam steps, 128 response forwards and backwards each, 32 teacher preparations and eight trainer initializations. Allocate exact cases first, rather than spending unused allowance. A full-4,096 synthetic qualification fixture tests GPU memory and shape compatibility; it does not imply every production feedback record must permanently occupy 4,096 rows. Actual startup/graph initialization work stays separate. A real save/reload test must compare the next update with uninterrupted execution, not just confirm that a file loads.

## Storage derivation

Measured retained weight file: **W = 2,097,259,104 B** (`configs/runs/dflash_osd_one_event_20261025/sealed/MODEL_PIN.json:80`). PR54 event-58 recovery: **S_hist = 6,532,631,103 B**, including **240,853,791 B** above 3W. Source: private retained PR54 `final_result_check_20261119/evidence/campaign_001/arms/KPLUS_C/STATUS.json:2635`. S_hist is a measured example, **not an upper bound for CODE metadata**. New history/serialization size needs a declared conservative envelope or CPU serialization bound before launch.

Prepared tensors have `hidden_states=[1,L,20480]`, `target_last_hidden_states=[1,L,4096]`, BF16; `input_ids=[1,L]` int64; `loss_mask=[1,L]` FP32. Thus the main payload is **49,164 B/row**. Native captures omit the final row; terminal preparation supplies one row, then the full prepared packet is atomically written. `sglang_backend.py:199–221`, `kplus_full_gsm8k/storage.py:213–239` check exact equivalence. A per-record header/metadata and allocation margin remains necessary.

For sealed window w, define L_i = rendered_prompt_tokens_i + 2,048. Every L_i ≤4,096 by input eligibility. Define T_w = sum_i L_i and M_w = max_i L_i. A safe feature-payload bound for incremental preparation/retirement is:

`F_w = 49,164 × (T_w + M_w + 1) + serialized-header/metadata allowance`.

This includes the complete active window (captures or prepared packets), overlap while replacing the largest capture by its prepared packet, and terminal row. It assumes the checked original capture and terminal tensors are retired immediately after exact prepared-packet equivalence is durably recorded. If that order is changed to retain all raw captures through preparation, the whole extra window must be added. Do not use observed generated lengths to reduce prospective admission before collection.

With two completed arms retained and the third writing its next recovery/export, a conservative envelope is:

`P_train = 7 × S_max + 4 × W + I + max_w(F_w) + C_compact + C_compiler + C_source + C_stage + C_archive + margin`.

The seven recovery inodes cover four states of the two completed arms plus predecessor/current/new atomic state of the active arm. Four exports cover the two finished endpoints and active old/new publication overlap. Hardlinks count once by device/inode. I is the actual initialization witness footprint: three inherited full initial snapshots add about 3W; a genuinely implemented compact witness bound to protected released weights can be much smaller. Do not silently assume the old `save_initial` snapshots are compact.

At the historical state size, seven states and four exports are **54.117454 GB**, before initialization, features and miscellaneous costs. Final required six recoveries and three exports are **45.487564 GB**. A universal 128×4,096 window plus capture replacement costs **25.977520 GB** before headers: together these already reach **80.094974 GB** with zero initialization or compact allowance. Therefore that universal storage envelope fails the 80 GB cap. The correct admission must use the fixed sealed prompts and capped outputs, not erase required states or assume future offload.

Illustrative sensitivity only (not a storage admission): at S_hist with one full initial witness and an 8 GB combined miscellaneous allowance, the available feature budget is 15.785287 GB, corresponding roughly to mean prompt length ≤441 under an equal-length approximation. Three full initial snapshots leave only 11.590769 GB, less than even 128 capped 2,048-token responses with no prompt. These figures show why actual initialization policy and bounded compact evidence matter. They are not permission to lower cap, omit evidence or assume a smaller future recovery.

Use a phase-specific maximum for qualification, AR/F precheck, third-arm publication and final packaging. Disposable qualification tensors can retire only after their checks pass and compact diagnostics are retained; their temporary peak still counts. Packaging overlap, compiler growth and any dataset/source stage must be charged. No historical dependency deletion or hoped-for Fenrir offload may be credited. At every actual admission require both `charged_new_peak ≤80e9` and `current_free − remaining_forecast ≥400e9`; avoid subtracting already allocated study bytes twice.

## Duration evidence and gate

Retained PR31 report: AREPO trajectory worker **31,668.327 s (8.797 h)**, training 19,164.221 s, update calls 22,170.554 s. Retained PR54 `final/COSTS.json`: C trajectory worker **35,935.149 s (9.982 h)**, training 21,092.391 s, update calls 24,539.599 s. Nested timers must not be added. Both are GSM8K, not the new CODE length distribution or 4,096 context. Three historical trajectories alone imply approximately **26.4–29.9 h elapsed**, with both GPUs held for much of each trajectory (roughly 52.8–59.9 occupied-card hours), before qualification, 2,048 AR/F preparation requests, genuine AR switches, 15,360 scores, 240 warmups, audit and packaging.

The parent's initial 35–45 h end-to-end planning range is a conditional forecast, not established by these reports. A raw 45 h forecast cannot satisfy `consumed +1.25×remaining +1 h ≤48 h` at startup. With zero consumed, remaining must be ≤37.6 h. Measured qualification and full-panel precheck must supply CODE length/serving costs, while reserved long/synthetic fixtures constrain training/resource extrapolation. Independently forecast occupied-card time ≤90 h. Recheck after each complete arm and safe boundaries; preserve rather than auto-shrink or auto-extend on a concrete failure.

## Required CPU coverage before staging

- 7,424 requests yield exactly 58 windows and no request 7,425/tail execution.
- Separate O/AREPO/C installer membership and formula-neutrality checks; active C coefficient observations, literal AREPO cap operations.
- Actual 4,096 context accepted; all other recipe alterations rejected by a narrow allowlist.
- Independent fresh initialization witness; exact model/Adam/RNG/counter differences cannot be normalized away.
- Causal save/export/validate/publish ordering; no next collection after failed publication.
- Ineligible short response preserves record and stops its window without false preparation-forward accounting.
- Completed-event retention keeps current/previous, deletes only n−2 through allowlist; final57/58 kept for every arm.
- Peak forecast includes all completed-arm states, atomic/export overlap, largest replacement packet and archive copies; shared inodes counted once.
- Resume uses exact own state/history, natural sampling and original ledger; unknown work is never converted to zero or retried automatically.

## Implemented adapter and CPU evidence

The new `opencode_native_comparison/training.py` calls the unchanged SpecForge numerical training engine and existing O/AREPO/K+/C installers. `configuration.py` admits the declared 4,096 context and run/output metadata only. Fresh arms import `rich.style` before native seed initialization and retain empty Adam/RNG/weight tensor identities in compact initialization witnesses. The historical released-K0 comparator independently validates its full state before allowing only declared CODE metadata, objective and context changes. Failed comparison evidence is saved before raising.

Production preserves natural sampling; there is no forced anchor schedule. Full own-state restoration retains model, Adam, counters, RNG and complete event history. `authenticate_resume` is a read-only complete-boundary check; it does not authorize a retry, construct a trainer or restore RNG. The controller must separately reconcile attempted work, publication, failure status and resource/constructor budgets before any later continuation.

The bounded qualification component uses three fresh isolated constructors and **17 disposable Adam steps, 40 response forwards and 40 backwards**, including six qualified startup forwards/backwards with zero Adam. O and AREPO compare independent original objective paths against the adapter, then next-update uninterrupted versus serialized/reloaded execution. C additionally checks bare/native/cap-zero neutrality and active allocation invariants. Each arm exercises a disclosed synthetic 4,096-token pair without another teacher forward or scientific feedback. Native serving checks remain separate. It records physical chunks/recomputation and per-response native training timers, with measured preclip/block-observer costs removed for a forecast estimate; whole-case hashing, serialization and resource costs remain separately visible. No positive efficacy or mandatory nonzero allocation is required to qualify.

CPU evidence: 20 focused contract, qualification, exact read-only resume, envelope and rolling-retention tests pass. The three-arm tiny native-runtime integration and bounded C qualification also pass: the former covers continuous two-event native training and exact own-state continuation in isolated processes; the latter exercises all seven C cases, including the 4,096 fixture, with exact 7-Adam/16-forward/16-backward accounting. These are CPU correctness checks, not native A800 qualification or production resource admission.

The actual sealed largest window has 293,039 capped rows, largest single sequence 3,397 rows. The replacement-overlap feature payload is therefore **14,574,028,668 B**. A counting serializer built from authenticated historical schema, expanding all 14,848 presentations to 512 anchors and all legal per-position C summaries, yields **363,263,390 B** of metadata plus declared reserve in the private sizing witness. The implementation reserves **500,000,000 B** and enforces a complete-state bound **6,791,777,312 B** before/after serialization. Seven such recoveries plus four exports plus prospective feature payload are **70,505,506,268 B** before the declared margin and compact/compiler/source allowances. This remains a component bound; the parent's complete capacity plan is the admission authority.

Every published boundary now retires the predecessor's redundant serving export only after exact tensor comparison with its retained complete recovery. Its fingerprint and retirement receipt remain durable. The n−2 recovery is later retired using those unchanged authenticated files. Thus two complete states remain, but only the current serving export occupies each active boundary; old-state serving requires rematerializing the verified export. Final57/58 states and export58 are unchanged. This is what makes the four-export publication-overlap bound valid. No inherited source retention policy is weakened.

Native validity, measured GPU memory, native timings and final capacity admission remain unresolved. No mechanism change is proposed.
