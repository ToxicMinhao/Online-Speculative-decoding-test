# Track2 cost, resource and retention accounting

Prepared by the author of the additive trainer and seal. This is source-informed compact-receipt accounting, not an independent trainer review, full-state redecoding, GPU reproduction or endpoint/bootstrap audit.

Producer status: **completed**, committed events: 8. Receipt coverage: required_receipts_present_inventory_not_verified. This extraction is ANALYZED, not an independent GPU verification.

## Timing

Only sum the same timer across disjoint events/workers. Do not sum different scopes into a total; campaign wall is the outer measured total.

| Recorded scope | Seconds |
|---|---:|
| campaign_wall_s | 17,525.487 |
| adaptation_ready_wall_s | 15,327.051 |
| evaluation_wall_s | 2,191.939 |
| initialization_s (component; nested) | 81.290 |
| backend_loading_s (component; nested) | 20.507 |
| training_worker_wall_s (component; nested) | 10,016.630 |
| activation_s (component; nested) | 91.322 |
| teacher_preparation_s (component; nested) | 84.036 |
| feature_persistence_s (component; nested) | 48.736 |
| descriptor_s (component; nested) | 0.453 |
| selection_s (component; nested) | 0.683 |
| memory_maintenance_s (component; nested) | 23.960 |
| export_s (component; nested) | 434.004 |
| evaluation_wall_s (component; nested) | 2,191.939 |
| resource_check_s (component; nested) | 1,384.528 |

## Arm totals

Presentation-weighted token/anchor totals, not unique tokens or FLOPs; replay repeats are intentionally counted.

| Arm | Event receipts | Presentations | Replay | Prompt tokens | Response tokens | Active anchors | Adam steps | Native training s | Worker s | Export s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K | 8/8 | 2048 | 0 | 114374 | 588760 | 542308 | 1024 | 2,806.719 | 3,355.996 | 148.328 |
| U | 8/8 | 2048 | 224 | 116786 | 585134 | 543435 | 1024 | 2,764.578 | 3,323.331 | 143.833 |
| R | 8/8 | 2048 | 224 | 115765 | 587487 | 545614 | 1024 | 2,775.998 | 3,337.304 | 141.843 |

Each sum has observation coverage in JSON; incomplete coverage is not filled with zero.

## Resources

Maximum of per-worker CUDA allocator peaks reset after construction/restore. Window includes train_event loops and checkpoint persistence/commit; excludes later verify_boundary, state hashing and cleanup. This is not whole-job GPU, device, or host-memory peak.

Discrete filesystem observations, not continuous peaks. Compiler/compact allocated blocks are inode-deduplicated by producer. pending_write_bytes is the sampled gate argument (campaign uses default zero), not observed pending allocation; separate require_write_space checks are not logged here. No host or whole-device peak was measured here.

| Discrete observation | Minimum bytes | Maximum bytes | Samples |
|---|---:|---:|---:|
| free_bytes | 43447312384 | 74626428928 | 3150 |
| compiler_growth_bytes | 0 | 100327424 | 3150 |
| compact_allocated_bytes | 6103040 | 1953501184 | 3150 |
| pending_write_bytes | 0 | 0 | 3150 |

| Arm | Maximum allocator allocated bytes | Maximum allocator reserved bytes |
|---|---:|---:|
| K | 20723064832 | 27577548800 |
| U | 20788358144 | 27573354496 |
| R | 20788358144 | 27573354496 |

Recorded CPU scoring/greedy timings nest inside selection total. Array working bytes describe explicitly counted NumPy arrays, not process RSS or measured host peak; retain each producer working_bytes_scope.

| Selection measurement | Value |
|---|---:|
| total_wall_seconds | 0.683 |
| scoring_seconds | 0.548 |
| greedy_seconds | 0.082 |
| Maximum persistent array bytes | 8520704 |

## Retention

Completed retirement receipts: 2986; distinct retired paths: 2986.

| Category | Paths | Nominal logical bytes | Lost capability |
|---|---:|---:|---|
| native_feature_tensor | 1024 | 17247630656 | Original native capture tensor removed; compact conditioning evidence survives. Prepared supervision remains only while retained in reservoir. |
| preceding_export | 21 | 44042441184 | This intermediate serving-weight file is unavailable; compact serving checks/configuration remain. |
| predecessor_full_state | 21 | 132205995767 | Direct exact resume from this intermediate full model/Adam/RNG boundary is lost; compact validation survives. Immediate successors can also be retired; later retained boundaries are listed separately. |
| prepared_packet | 896 | 15019087580 | Exact evicted supervision tensor is unavailable for direct replay. Descriptor, source identity and compact membership/validation receipts survive. |
| terminal_feature_tensor | 1024 | 53998656 | Separate terminal teacher tensor removed; compact evidence survives. Prepared supervision remains only while retained in reservoir. |

- Retired tensors are deleted, not archived.
- Receipt sizes count removed directory entries. Allocated-byte sums are nominal and are not measured reclaimed space: hardlinks and open references may retain blocks.
- No inode-only deduplication: receipts lack filesystem device IDs; inode reuse can also occur.
- Retained evidence means production validated these artifacts and no matching retirement is recorded. Compact-only retrieval does not rehash their bulk bytes or prove current remote presence.
- trainer/training_state.pt is a producer-maintained latest hardlink alias; it is not an additional independently retained payload.

Exact paths, reasons, retained evidence, state/export identities and final reservoir membership appear in the JSON and CSV companions.

## Scope notes

- **campaign_wall_s**: Outer campaign wall, including cleanup; excludes separate preflight/postflight and report built after this timer.
- **adaptation_ready_wall_s**: Nested in campaign wall: initialization, serving backend load, eight adaptation events and their checks.
- **event_wall_s**: Disjoint sequential event scopes; each includes pre-update serving, preparation, selection, training workers, maintenance, exports and checks.
- **worker_wall_s**: Inside event wall: construction/restore, feature admission, native loops, checkpoint commit, validation, state hashing and cleanup; excludes Python process startup.
- **process_wall_s**: Outer subprocess wall including startup and worker wall; never add to worker wall.
- **construction_and_restore_s**: Sum scope of construction_s and restore_s; do not add all three.
- **feature_admission_io_and_validation_s**: Worker subcomponent before native pass loops.
- **total_loop_s**: Includes training_s and input_load_and_loop_bookkeeping_s; do not add loop total to its children.
- **training_s**: Native per-response timer sums, including forward/backward/optimizer and observations; excludes input file loading.
- **checkpoint_save_s**: Worker subcomponent; export_s in trainer RESULT is deferred zero, not the later materialized-export cost.
- **export_and_verify_s**: Post-eviction native export, load and exact-weight verification, outside training workers but inside event/campaign wall.
- **initialization_s**: Sum of three initialization workers; separate process-wall scopes include startup.
- **backend_loading_s**: One serving-backend load within campaign/adaptation wall.
- **activation_s**: Pre-update checkpoint publication and validation inside serving/event wall; excludes final-evaluation activations.
- **teacher_preparation_s**: Shared K feedback preparation includes native/terminal loading, binding checks, concatenation and terminal teacher forward; not isolated GPU compute or separately attributable to each arm.
- **feature_persistence_s**: Prepared-packet save inside event wall, not all native capture I/O.
- **descriptor_s**: Shared CPU descriptor construction within event wall.
- **selection_s**: Shared CPU pass selection within event wall.
- **memory_maintenance_s**: Reservoir commit/eviction and associated retirement within event wall.
- **resource_check_s**: Disk/cache scan sum inside event/evaluation/campaign scopes; RESOURCE_CHECKS.jsonl is canonical, STATUS copies are not additional samples.
- **evaluation_wall_s**: Final capture-OFF evaluation scope, including its publication, checks and barriers; inside campaign wall.
- **compiler_timers**: Per-name sums over sequential workers; names can nest, and are already inside native training time. Never sum across timer names.
