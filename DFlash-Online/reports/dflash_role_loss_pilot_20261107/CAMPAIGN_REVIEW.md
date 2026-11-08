# Campaign implementation review

## Material Passport

- Mode: ARS experiment-agent / code_runner, authorized implementation and local validation.
- Status: local implementation checks passed; native A800 qualification remains pending.
- Research stage: Stage 1 / project None unchanged.
- Provenance: same implementation subagent, shared parent context. This is not an independent external audit.
- Publication: no commit created by this agent. Reviewed working-tree bytes are pinned below; base is `abb28033e6676acac5b92c1086399c4886e093e8`, branch `codex/kplus-role-loss-pilot`.

## Implemented scope

Six separate fresh seed-17 trajectories use released weights and authenticated empty native BF16 Adam/RNG. The inherited 7,424-step constant-LR configuration is unchanged; the isolated pilot trainer enforces four events/512 updates per trajectory and records arm plus dataset in state and event contracts. Metadata-only projections reuse the old exact state validators without modifying their guards. Original KPLUS arithmetic uses its unchanged installer; MIX and ROLE use the dedicated reducer supplied by the core worker.

Every response loss and pre-Adam gradient is checked for finiteness. Training observations retain raw role weights, selected/full denominators, priority counts, actual losses, pre-clipping gradient norms and native optimizer windows. KPLUS denominator diagnostics derive from existing chunk-reduced forward metrics, so their FP32 reduction association can differ slightly from candidate full-flat D; this observation does not change KPLUS loss arithmetic. KPLUS has no CE/RKL cap observations because it does not evaluate those branches.

The qualification accepts the authenticated six-feature inventory and presents only its first four responses as two fixed pairs per arm. Six separate fresh native trainer constructions each take one disposable update. It performs exactly six updates and twelve response forward/backwards, with zero startup response passes. The updated state is saved and decoded; the live trainer is first reset to the released state, then decoded weights, moments, RNG, counters and core accumulation count are restored and compared exactly. It finally restores the released state again. Target-derived parameters remain frozen and outside the optimizer. A failure preserves any surviving disposable tensor and reports its path; successful retirement records exact identity, free space and lost capability. Production cannot reuse qualification state.

The six production workers execute serially, preserving state within a trajectory and resetting between trajectories. Shared fixed canonical feature pools support the initial and event-2/4 diagnostics. Event-2 and event-4 exports are evaluated only after all training; their bytes are checked against immutable training-boundary identities before and after each stage. Exact publication checks remain at every event. The additional eight inherited capture/reload checks are charged separately, four per dataset on KPLUS.

Storage uses the maximum of two real phase peaks, not a sum of mutually exclusive maxima. With measured draft size 2,097,259,104 bytes, the conservative training requirement is 145,736,919,695 bytes and evaluation requirement is 149,072,926,082 bytes. The latter includes all 8,204 bounded generations, all six initial containers, all six final recovery states, twelve selected exports, diagnostic features, compiler/workspace allowances and an 8 GiB reserve. No future deletion or historical retirement receives credit. Within-run exact successor validation permits event-1/3 export retirement and obsolete recovery/feature retirement; event-2/4 exports remain.

The launcher performs CPU tests, qualification admission, bounded qualification, full production admission, campaign, report and postflight. SIGTERM/SIGINT raise into normal failure/cleanup paths; the parent allows 90 seconds for detached trainer/backend cleanup inside the 120-second hard-timeout grace. There is no automatic retry or efficacy-based event-2 decision.

## Validation and review corrections

All 21 owned CPU tests passed: five state/training tests, twelve campaign/storage tests and four qualification tests. The real tiny native path runs a complete 128-update event for every arm and verifies exact resume. The bounded native CPU fixture performs exactly six updates/twelve forwards/backwards; forced nonfinite gradients stop before Adam. Contract tests reject crossed datasets/arms, a fifth event, changed moments/RNG, bad diagnostics, changed source inputs, changed selected checkpoints and failed publication. Retention tests preserve event-2 exports and prevent deletion on failed publication. Shell syntax validation passed.

Peer integration comments were incorporated: evaluator boundary schema normalized; selected exports pinned to training-boundary hashes; launcher CPU module names corrected and checked; peak-memory reset moved after runtime initialization; early CPU qualification config identity; actual reset-before-decoded-restoration; correct retained-failure tensor reporting; current runtime-pin scope labels; and timeout cleanup.

The sealer interface was reviewed against campaign/preflight expectations. The parent separately binds original initial-state and retained qualification inventory identities in the seal after the review identified that gap. GPU execution, runtime precision, native compilation and real qualification behavior remain unverified until the bounded A800 gate succeeds. This agent performed no remote or GPU operation and created no commit.

## Reviewed file fingerprints

| File | Bytes | SHA-256 |
|---|---:|---|
| `experiments/role_loss_pilot/__init__.py` | 71 | `aebe8df5540a7ce96d122d49ca38c7df5d3cc94e118768aa8ba4181a2a4d18c2` |
| `experiments/role_loss_pilot/contract.py` | 4766 | `cbb39d4e188a863aeaa170bfbc65c26251531f5a19e67c0e054c69859fc066d7` |
| `experiments/role_loss_pilot/training.py` | 14466 | `ed9a41014e317e972fa275ccc548ddb55190379dee21ac3c9eb5b68716a1d8da` |
| `experiments/role_loss_pilot/campaign.py` | 18613 | `00fe0d962651eafda3d497348b1107623612fa278bf8e9353aa2cf3a8a958ae9` |
| `experiments/role_loss_pilot/qualify.py` | 14652 | `90d14233672dc2945ddc5d8d94dfe9e23de86637976cf585caad6294e438c66f` |
| `experiments/role_loss_pilot/preflight.py` | 4819 | `6aac5184a9fb6ae2bb45ac619c6a68d5378fa00b7898007cba382b34e4141a62` |
| `experiments/role_loss_pilot/storage.py` | 8764 | `22f26542a0a0d405e1d557e1d2e680cc0280b9e11a3356e1496a157efdd0ae2e` |
| `experiments/role_loss_pilot/run_campaign.sh` | 3061 | `b33b5e07ef53bd902b3b958cab73389914b33d50631e23e17ea09041e7db6608` |
| `experiments/role_loss_pilot/README.md` | 2990 | `a0adcd2cdcce054e772e96c98b4ae15c590222585eac6013545cd505a7d94e35` |
| `tests/test_role_loss_campaign.py` | 11759 | `c409993038841a0290398397e02e461fde105a1793edb20ff3a75ed4fc7c4b56` |
| `tests/test_role_loss_campaign_training.py` | 5664 | `e7a4d7d3e6260ab2b8d6b060e90ef1643937a3d24050548b0887cdb60d5957dc` |
| `tests/test_role_loss_campaign_qualification.py` | 4971 | `d382e046543dab39639da2ef1e46352ade5a91de0b40a3dd37ef4a6b43846f02` |
