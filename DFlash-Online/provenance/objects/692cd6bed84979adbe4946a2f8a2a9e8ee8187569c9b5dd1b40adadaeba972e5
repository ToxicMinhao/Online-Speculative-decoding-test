"""Summarize the completed bounded continuation without additional computation on models."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
raw = root/'raw'
a = json.loads((root/'analysis/ANALYSIS.json').read_text())
c = a['campaign_costs']
audit = json.loads((root/'FINAL_AUDIT.json').read_text())
assert a['status']=='complete' and audit['status']=='passed'
rows=[]
for event in (2,3,4):
    f=c[f'event_{event:03d}_feedback']
    records=[json.loads(p.read_text()) for p in (raw/f'campaign_001/event_{event:03d}/responses').glob('*.json')]
    rows.append(dict(event=event,arm='shared_feedback',responses=len(records),
        output_tokens=sum(len(r['response_ids']) for r in records),cap_hits=sum(r['finish_reason']=='length' for r in records),
        collection_wall_s=f['collection_wall_s'],request_latency_s=f['request_latency_s'],
        capture_host_s_nested=f['capture_nested']['capture_host_s'],
        capture_persistence_s_nested=f['capture_nested']['feature_persistence_s'],
        cache_maintenance_s_nested=f['capture_nested']['cache_maintenance_s'],
        preparation_wall_s=f['preparation_wall_s'],teacher_preparation_s_nested=f['teacher_preparation_s'],
        prepared_feature_persistence_s_nested=f['feature_persistence_s'],
        activation_s=f['publication']['activation']['activation_s'],
        native_activation_s_nested=f['publication']['activation']['native_activation_s'],
        weight_validation_s=f['publication']['weight_validation_s']))
    for arm in ('O','K'):
        own=c[f'event_{event:03d}_{arm}']
        rows.append(dict(event=event,arm=arm,**{k:own[k] for k in ('loading_and_resume_validation_s','training_s','checkpoint_save_s','export_s','event_wall_s','peak_allocated_bytes','peak_reserved_bytes')},
                         selection_s_nested=own['objective_work'].get('selection_s',0),
                         teacher_head_s_nested=own['objective_work']['teacher_head_s']))
    if event in (2,4):
        s=json.loads((raw/f'campaign_001/evaluation_{event:03d}/STATUS.json').read_text())
        rows.append(dict(event=event,arm='paired_benchmark',evaluation_wall_s=s['evaluation_wall_s'],
                         activation_s=s['activation_s'],weight_validation_s=s['weight_validation_s']))
fields=list(dict.fromkeys(key for row in rows for key in row))
with (root/'analysis/COSTS.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def total(arm,key):return sum(r.get(key,0) for r in rows if r['arm']==arm)
def link(label,path):return f'[{label}](/home/minhao2/EAGLE_SPEC/research/dflash_ok_continuation_20261027/{path})'
text=['## Material Passport','',
'ARS academic-research-suite, experiment-agent run/validate. The declared run completed; numerical analysis and independent compact-evidence audit passed. No independent scientific rerun is claimed. Stage 1/project None is unchanged.','',
'**K’s advantage persists through four events, but it does not grow steadily.** Its verification-call advantage over ordinary OSD is 1.56% after event 1, 2.81% after event 2, and 1.74% after event 4. Both arms improve relative to retained frozen DFlash. Between events 2 and 4, O saves another 65 calls and K another 31 calls on the same evaluation inputs, narrowing the gap from 86 to 52 calls. This is evidence for a modest sustained benefit on this trajectory, not a growing advantage or a general robustness result.','',
'**What was run.** Continue each arm’s own complete event-1 weights, BF16 Adam moments/counters and Python/NumPy/Torch/CUDA RNG. Events 2–4 each collect 128 responses with O’s preceding checkpoint, prepare one immutable shared pool, and train O and K separately for two passes, 256 presentations and 128 steps. K recomputes first-error selection from its own current logits. Both updates finish before the next collection. This controls feedback while comparing objectives; it is not two independently measured online deployments.','',
'Qualified Qwen3-8B target / DFlash-b16 pair; constant LR 3e-4, BF16 AdamW and KL arithmetic, weight decay 0.01, clipping 1.0, shuffled responses each pass, up to 512 anchors per response, two-response accumulation, Flex training attention, response averaging and full-block attention are unchanged. O supervises every valid proposal; K includes the current correct prefix and first error, with selected-weight normalization. Training length is 2,272. The inherited 7,424-step horizon is preserved while orchestration stops each arm at 512 cumulative steps. No seed reset or event-1 repeat occurred.','',
'Feedback inputs are the next sealed GSM8K train indices 128–511, excluding the original 128 feedback inputs and both current evaluation panels, with exact pinned preprocessing and prompt deduplication. All 384 have disclosed older-project exposure, which the user permitted; fresh responses were generated for this continuation. Dataset answers are unused. Evaluation reuses the original 64-request development panel; the later confirmation panel is not pooled here.','',
'Native evaluation retains greedy decoding, block size 16, 2,048 tokens or EOS, context 4,096, concurrency 1, capture disabled, and the pinned SGLang/FA3 runtime with cache invalidation. Three fresh alternating-order repeats were measured after events 2 and 4, with training closed. Event-1 measurements are retained.','',
'**Learning curve.** Positive percentages mean reductions. Frozen uses 3,140 retained calls on the identical 64 inputs. Calls use one validated trajectory per request, not all three repeats as independent samples.','',
'| Event | Cumulative steps/arm | O calls | K calls | K/O call reduction | O/F reduction | K/F reduction | K/O request-time reduction |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for event in (1,2,4):
    e=a['events'][str(event)]; d=e['comparisons']
    text.append(f"| {event} | {event*128} | {e['O']['verification_calls']:,} | {e['K']['verification_calls']:,} | {d['K_vs_O']['call_reduction_pct']:.2f}% | {d['O_vs_F']['call_reduction_pct']:.2f}% | {d['K_vs_F']['call_reduction_pct']:.2f}% | {d['K_vs_O']['request_time_reduction_pct']:.2f}% |")
text += ['',
'Each time ratio compares O and K measured together at that event. Historical-versus-fresh raw latency is not interpreted as learning progress, and no fresh F latency claim is made. These are available-checkpoint serving measurements; training, loading, export and activation are excluded. All three timing repeats favor K at each event.','',
'| Event | Requests with fewer K calls | More K calls | Ties | Net calls saved |',
'|---|---:|---:|---:|---:|']
for event in (1,2,4):
    q=a['events'][str(event)]['comparisons']['K_vs_O'];d=q['call_request_directions']
    text.append(f"| {event} | {d['improved']} | {d['regressed']} | {d['tied']} | {q['call_net_saved']} |")
text += ['',
'At event 4, K improves over retained F on 50 requests, regresses on 8 and ties on 6. All evaluated checkpoints produce exactly the same 19,547 output tokens across the panel, with EOS termination on every request and no cap hits. O/K event-4 calls per 1,000 output tokens are 153.169/150.509; accepted draft tokens per verification are 5.526/5.643 and committed progress per verification is 6.507/6.622. Request-level regressions remain in the tables.','',
'Two feedback responses reached the 2,048-token cap: train_00187 in event 2 and train_00399 in event 4. Their total training lengths were 2,133 and 2,231, within the unchanged 2,272 limit. They were retained without truncation or substitution.','',
'**Exposure.** The arms use identical feedback, normal shuffled presentation order and anchor draws within every event. K’s selection changes which occurrences and weights receive loss; its mask and selected-weight renormalization are jointly tested. These are repeated position occurrences across blocks and passes, not unique tokens.','',
'| Event | Sampled anchors per arm, both passes | Available occurrences | K selected occurrences | Selected fraction | K selected weight fraction |',
'|---|---:|---:|---:|---:|---:|']
for t in a['training']:
    if t['arm']=='K':
        text.append(f"| {t['event']} | {t['kept_anchors']:,} | {t['available_occurrences']:,} | {int(t['selected_occurrences']):,} | {100*t['selected_occurrence_fraction']:.2f}% | {100*t['selected_weight_fraction']:.2f}% |")
text += ['',
'O uses all available occurrences. Per-event weighted numerators/denominators, no-error blocks and all 15 first-error-position counts are retained in TRAINING.csv and the raw event receipts. Boundary accumulation rows are counted once. Raw O/K losses supervise different selected distributions and are not direct efficacy comparisons.','',
'**New continuation costs, events 2–4 only (seconds).**','',
'| Arm | Training | Checkpoint persistence | Export | Complete update events | Loading + resume validation |',
'|---|---:|---:|---:|---:|---:|']
for arm in ('O','K'):
    text.append('| '+arm+' | '+' | '.join(f'{total(arm,k):.3f}' for k in ('training_s','checkpoint_save_s','export_s','event_wall_s','loading_and_resume_validation_s'))+' |')
text += ['',
f"K’s measured training total is {100*(total('K','training_s')/total('O','training_s')-1):.2f}% higher than O’s. Its recorded selection computation is {total('K','selection_s_nested'):.3f} s, already included in training. Teacher-head work is also nested in training, not added again. Complete update-event time includes training, persistence, export and event bookkeeping; loading/resume validation is separate.",'',
'| Shared feedback or benchmark component | Seconds | Accounting |',
'|---|---:|---|',
f"| Shared O feedback collection wall | {total('shared_feedback','collection_wall_s'):.3f} | Collected once across the system |",
f"| Feedback request latency | {total('shared_feedback','request_latency_s'):.3f} | Included in collection wall |",
f"| Capture host work | {total('shared_feedback','capture_host_s_nested'):.3f} | Nested in captured requests; not separately added |",
f"| Capture persistence | {total('shared_feedback','capture_persistence_s_nested'):.3f} | Nested in captured requests |",
f"| Feedback preparation wall | {total('shared_feedback','preparation_wall_s'):.3f} | After collection |",
f"| Teacher preparation | {total('shared_feedback','teacher_preparation_s_nested'):.3f} | Nested in preparation; 384 terminal feature forwards |",
f"| Prepared-feature persistence | {total('shared_feedback','prepared_feature_persistence_s_nested'):.3f} | Nested in preparation |",
f"| Activation for O feedback collection | {total('shared_feedback','activation_s'):.3f} | Outside captured request latency |",
f"| Paired benchmark activation/switching | {total('paired_benchmark','activation_s'):.3f} | Benchmark overhead; outside serving latency |",
f"| Paired benchmark wall | {total('paired_benchmark','evaluation_wall_s'):.3f} | Includes repeats, activation and validation |",
f"| Initial native model loading | {c['backend_loading_s']:.3f} | One-time campaign setup |",'',
f"Campaign driver wall time was {c['campaign_wall_s']:.3f} s. The supervisor, including preflight, CPU checks and postflight, ran from 02:05:43 to 03:15:03 UTC on October 27 (69 min 20 s). Timers above overlap as labeled and must not be summed indiscriminately. Benchmark repeats are measurement cost, not a deployed serving episode. No independent adaptation-inclusive O/K deployment speedup is established by this shared-feedback design. Peak training-process allocation/reservation was {max(r.get('peak_allocated_bytes',0) for r in rows)/2**30:.2f}/{max(r.get('peak_reserved_bytes',0) for r in rows)/2**30:.2f} GiB; these are process peaks, not an isolated algorithm-memory attribution.",'',
'**Validity and accounting.** Exactly 384 feedback plus 768 evaluation generations = 1,152 new complete-response generations; 1,536 presentations and 768 optimizer steps. No extra optimizer qualification steps, generation warmups, retries, invalid comparisons, unknown or unattempted work. The initial 48 event-2 evaluations formed the native correctness gate within the budget. Both arms finish at 512 steps / 1,024 cumulative presentations.','',
'All six saved/live restore checks passed for own-arm weights, Adam, counters, config, training mode and complete RNG, with zero mismatch. Final CPU reload independently checks both completed states against event-4 exports and 512-step Adam counters; final O/K RNG payloads match. All 185 sealed-artifact checks and the seven external runtime source checks passed before and after execution. Independent compact auditing reconstructs 52,270 native blocks and verifies all 1,152 response records, 384 terminal feature forwards, shared feedback and matched exposure. All 4,001 retrieved compact files match their manifest. Original event-1 artifacts remain unchanged; final full states, all six new exports and feature pools remain on A800.','',
'Only the experiment orchestration, state validation, two-arm evaluator and local analysis were added. The 117 inherited scientific source files remained unchanged. The 36 focused execution CPU tests, five selection tests and separate reviewed analyzer/auditor tests passed. Pre-launch test-fixture and dictionary-representation corrections are preserved; they did not change learning settings or introduce GPU work. Exact state restoration does not prove an uninterrupted GPU execution would be bitwise identical; no extra warmup or tolerance change was introduced.','',
'**What this establishes.** With this seed, shared O-generated feedback sequence and fixed development panel, K remains modestly better than ordinary OSD after more feedback. Both continue improving versus frozen; O catches up partly between events 2 and 4. It does not establish monotonic growth, independent training-seed/feedback-window robustness, that masking alone caused the benefit separately from renormalization, or net serving-plus-adaptation acceleration. The original development panel is reused. The retained O/K checkpoints remain the agreed starting points; no full-trajectory bitwise-equivalence claim is made. No algorithm change or additional run was made.','',
'**Evidence files.**','',
'ALL_REPEATS.csv contains 384 retained event-1 O/K measurements plus the 768 new event-2/4 measurements. PER_REQUEST.csv contains 64 reused requests at each of three checkpoints, not 192 independent questions. The 384 new feedback generations are retained separately in the raw window records.','',
'- '+link('Event comparison','analysis/EVENTS.csv'),
'- '+link('Per-request results, including regressions','analysis/PER_REQUEST.csv'),
'- '+link('Every O/K timing measurement, events 1/2/4','analysis/ALL_REPEATS.csv')+' and '+link('repeat totals','analysis/REPEATS.csv'),
'- '+link('Training exposure and source costs','analysis/TRAINING.csv')+' and '+link('separate cost accounting','analysis/COSTS.csv'),
'- '+link('Independent execution audit','FINAL_AUDIT.md')+' and '+link('final full-state CPU audit','raw/FINAL_STATE_AUDIT.json'),
'- '+link('Pinned execution plan','raw/sealed/PLAN.json')+' and '+link('configuration','raw/sealed/config.yaml'),
'- '+link('Selection and older-exposure provenance','data_preparation/DATA_PREPARATION_REPORT.md'),
'- '+link('Raw compact artifacts','raw')+' and '+link('retrieval verification','RETRIEVAL_VERIFIED.json'),
'- '+link('Implementation','IMPLEMENTATION.md')+'. Authoritative A800 module: /data/sources/minhao/DFlash-Online/experiments/ok_continuation; artifacts: /data/sources/minhao/runs/dflash_ok_continuation_20261027.','']
(root/'REPORT.md').write_text('\n'.join(text))
print(json.dumps({'report':str(root/'REPORT.md'),'cost_rows':len(rows),'K_extra_training_pct':100*(total('K','training_s')/total('O','training_s')-1)}))
