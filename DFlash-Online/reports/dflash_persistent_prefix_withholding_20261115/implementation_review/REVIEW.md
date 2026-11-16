# Persistent prefix withholding: implementation and duration review

ARS-Codex 3.24.2 experiment-agent; source review only. No production code modification, remote operation, GPU forward, optimizer application or generation was performed. This separate same-model-family review is not native qualification. The handoff's fresh-data gate remains a prior requirement.

Parent coordination subsequently reported the independent inventory result: 1,311 of the pinned 1,319 GSM8K test requests have already been used, leaving at most eight eligible before any additional exclusions. This cannot supply the required128-request globally fresh panel. This reviewer has not independently repeated that inventory; its source review is consistent with stopping before production implementation or execution pending the parent’s inventory confirmation and an explicit user decision about the panel. No CPU training tests were run here.

## Assessment

The intended E operation is already implemented and qualified for disposable PR47 steps. It can support persistent learning without changing objective arithmetic, but **the disposable trainer cannot safely be reused unchanged**. The new adapter, endpoint migration/validation and scheduler are substantive work. Existing three-arm evaluation code is useful infrastructure, not a drop-in implementation of this handoff.

PR47's local result, PR48's unresolved two-size-control test and PR32's different ROLE/MIX objectives were read. The new question is a continuing E trajectory with its own changing roles, Adam history and subsequent feedback. The earlier experiments do not answer that question. The new test does not retrospectively pass PR48's gate.

## Narrow implementation boundary

1. Reuse `online_feedback/native_prefix_roles.py:26` and the hook at `online_feedback/objective.py:206`. The hook is before both the selected-loss terms and the first-error redistribution terms. `torch.where(errors, token_loss, token_loss.detach())` preserves native forward values and lets context gradients flow normally. No new loss or error-only denominator is needed.
2. Use a new persistent event adapter directly around `SpecForgeEventTrainer` and the qualified full-K observations. `experiments/role_loss_pilot/training.py` illustrates metadata-scoped contract projection, own-state restore and persistent observations, but its event0–4/released-initialization/ROLE contracts must not be silently weakened for this run.
3. Keep a serialized immutable arm/algorithm contract and explicit event7/8 identity. Validate historical receipts under their original contracts; validate new event receipts under the new arm identity. Dynamic expected boundaries are768/1536 →896/1792 →1024/2048 with inherited horizon7424. Reinstall and validate E before every training forward and after each restore.
4. `NativePrefixTrainer.run_pair` defaults to role N and restores the previous controller; `ThreatReplayTrainer._train_passes` calls the default and hardcodes768/1536. Its owner/window controller restores N intentionally. These are concrete reasons to avoid that production path, not hypothetical style concerns.
5. N7 reuse needs complete scientific ancestry and its recovery/export, not just a final model. Bind PR48's exact final-native parity and original identities. The two new E/S event7 trajectories start from authenticated K6; all event8 trajectories resume their own complete states.
6. S migration must change current config/contract and **both** optimizer recipe LR and saved parameter-group LR. Preserve weights, moments, counters, RNG and historical receipts. In Python the mandated product is `0.00028799999999999995`, hex `0x1.2dfd694ccab3ep-12`; it is not equal to the literal `2.88e-4`. Store the computed value once. Historical event1–6 LR remains3e-4: the existing FullK validator's single-arm LR applied to every historical event cannot simply be pointed at S.
7. Each worker is a one-GPU process, visible by physical UUID as logical cuda:0. Scientific CUDA RNG has one state; physical metadata is compared separately. Use separate ports, arm state and directories. Qualification must prove cross-card equality before that parallel route is admitted.
8. Adapt the native scorer/activation backend, but supply the new exact128-request/three-repeat/four-block order and72 explicit warmups. Existing `role_loss_pilot/evaluation.py` hardcodes256 requests, old arm names and old event2/4 timing. Preserve native versus stop-clipped accepted-count definitions and full-request latency boundaries. A global quiet barrier must cover both cards and heavy host/transfer work.

## Qualification and resource risks to settle

- CPU tests need two consecutive E steps plus a save/load boundary, role changes after repair, no hidden restoration to N, zero-error gradients with native Adam semantics, and exact LR migration. Boundary observations must deduplicate checkpoint recomputation and must not retain vocabulary arrays.
- The inherited discarded startup at `experiments/functional_step_geometry/native_pilot.py:184` uses two forward/backwards, no Adam, then restores scientific RNG/runtime and verifies exact state. Charge at most two per trainer initialization; clear observers as well as gradients.
- Native `SGLangDFlashBackend` sets `skip_server_warmup=True` but enables CUDA graphs (`cuda_graph_max_bs=8`). This flag does not mean initialization has no model work. The complete installed graph-runner source and initialization callbacks need finite work bounds before launch; the checked-in external-runtime overlay is only partial.
- No independent target-only canonical generator with a matching prior qualification receipt was identified in the bounded online/role-loss/current report search. Existing role-loss panels reuse native-verifier references. This is an **unresolved provenance lookup**, not proof that such a path does not exist elsewhere. The handoff's target-only path must be located/authenticated before canonical generation; a draft-generated canonical must not be substituted.
- Two persistent trainers may fit more cheaply than PR48's old model-sized snapshot/calibration design, but that is not yet measured. Preserve32GiB host headroom and measure peak working sets during the eight-case qualification. Do not import the old80GiB-per-trainer rule twice or assume two available GPUs imply enough host RAM.
- Code-wide elapsed/card-time accounting must include model residency, startup, exports and idle waits. Parallelism saves elapsed time, not summed card time.

## Source-backed duration components

These are historical planning references, not a launch admission or guaranteed completion time. `REVIEW.json` records exact calculations and source paths.

| Component | Historical measurement / projection |
|---|---:|
| PR48 actual native event training sum |485.753s/event |
| PR32 GSM8K KPLUS training mean |341.922s/event |
| Five events at PR48 actual native training rate |2,428.766s =40.48 card-min |
| PR48 qualification pair rate |9.456s/pair |
| Five events priced at that pair rate |6,051.744s =100.86 card-min |
| Final1,152 scored +72 warmups at PR32 mean |1,052.119s =17.54 card-min |
|384 feedback generations at PR32 mean |304.126s =5.07 card-min |
|384 terminal preparations at PR32 mean |15.259s |

The9.456s PR48 qualification pair includes5.185s of model-sized gradient observation. Its full5,522.87s trajectory clock includes scalar calibration; it is **not** the cost of an ordinary persistent event. Conversely, the faster rates above cannot be promised for the new workers, different output lengths or parallel contention.

At PR48's actual training sum, the listed training/feedback/speculative-serving/preparation components total about63.3 card-min; at its conservative qualification pair rate they total about123.7 card-min. **Both totals omit** target-only autoregressive canonicals, eight-case qualification/state comparisons, finite startup/graph captures, loads,36 activations, checkpoint hashing/validation/publication and resident waits. No justified full-campaign estimate is available until those are priced and the eligible panel is sealed.

Admission must use the measured worst arm/card and metadata-based autoregressive cost, then satisfy `spent + 1.5 × remaining + 1800 <= 14400` card-seconds and the four-hour elapsed schedule. With zero spent, the unbuffered remaining estimate must be at most8,400s (140 card-min); qualification consumption reduces that ceiling. The conservative123.7-minute partial projection leaves little room for omitted costs. That is a reason to check admission carefully, not permission to change counts, precision or caps.

No scientific implementation or execution should start if the prior128-fresh-request gate fails. A conditional wall-time estimate should be labeled pending that gate and measured resource qualification; report a concrete blocker rather than an invented finish time.
