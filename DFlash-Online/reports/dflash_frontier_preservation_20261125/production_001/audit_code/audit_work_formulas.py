"""Independent stdlib-only reconciliation; reads saved receipts, never imports models."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw'
manifest = json.loads((ROOT / 'COMPACT_MANIFEST.json').read_text())
identities = {row['path']: row for row in manifest['files']}
checked = {}
checks = []


def require(condition, label):
    checks.append({'check': label, 'passed': bool(condition)})
    if not condition:
        raise AssertionError(label)


def read_raw(relative, lines=False):
    data = (RAW / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = identities[relative]
    require(len(data) == expected['bytes'] and digest == expected['sha256'],
            'raw identity: ' + relative)
    checked[relative] = {'bytes': len(data), 'sha256': digest}
    return [json.loads(line) for line in data.splitlines()] if lines else json.loads(data)


def chunks(row):
    calls = row['sampled_anchors']
    require(len(calls) == 1, 'one sampler receipt per response')
    positions, keep = calls[0]['positions'], calls[0]['keep_mask']
    require(len(positions) == len(keep) == 1, 'native response batch size one')
    require(len(positions[0]) == len(keep[0]), 'sampler width equals keep-mask width')
    return (len(positions[0]) + 7) // 8


def sum_counts(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    return dict(total)


retrieval = json.loads((ROOT / 'RETRIEVAL.json').read_text())
require(retrieval['status'] == 'verified', 'retrieval gate verified')
require(manifest['execution_commit'] == '225802657d630c0985abd68d6e7453f3e47c5b74',
        'production execution identity')
export_path = ROOT / 'export_001/COSTS_SUMMARY.json'
export_bytes = export_path.read_bytes()
export = json.loads(export_bytes)
observations = {}
events = {}
committed = {}
chunk_totals = {}
startup_counts = []
startup_backward = 0
startup_chunks = 0
for arm in ('CONTROL', 'FRONTIER_P'):
    event = f'arms/{arm}/trainer/events/event_002/'
    rows = read_raw(event + 'RESPONSE_OBSERVATIONS.jsonl', lines=True)
    require(len(rows) == 256, arm + ' 256 response observations')
    require([row['presentation_ordinal'] for row in rows] == list(range(256)),
            arm + ' presentation order complete')
    observations[arm] = rows
    chunk_totals[arm] = sum(chunks(row) for row in rows)
    events[arm] = read_raw(event + 'RESULT.json')
    require(events[arm]['response_presentations'] == 256
            and events[arm]['optimizer_steps'] == 128, arm + ' native event counts')
    work = read_raw(f'arms/{arm}/WORK.json')
    require(work['attempted'] == work['completed'] and not work['pending'],
            arm + ' committed work complete')
    require(work['completed'] == {'response_forward': 256, 'response_backward': 256, 'adam': 128},
            arm + ' committed ledger counts')
    committed[arm] = work['completed']
    startup = read_raw(f'arms/{arm}/startup/RESULT.json')
    start_rows = read_raw(f'arms/{arm}/startup/response_observations/RESPONSE_OBSERVATIONS.jsonl', lines=True)
    require(len(start_rows) == 2, arm + ' startup two responses')
    c_start = sum(chunks(row) for row in start_rows)
    startup_chunks += c_start
    physical = startup['physical_work']['counts']
    require(physical['draft_forwards'] == 2 and physical['vocabulary_head_calls'] == 4 * c_start
            and physical['objective_chunks'] == 2 * c_start, arm + ' startup native chunk formula')
    require(startup['work']['attempts'] == startup['work']['completed']
            and all(value == 0 for value in startup['work']['unknown'].values())
            and startup['work']['completed']['response_backwards'] == 2
            and startup['optimizer_steps'] == 0, arm + ' startup backward complete without Adam')
    startup_counts.append(physical)
    startup_backward += startup['work']['completed']['response_backwards']

C = chunk_totals['CONTROL']
require(chunk_totals['FRONTIER_P'] == C, 'both arms same total native head chunks')
require(all(a['sampled_anchors'] == b['sampled_anchors'] for a, b in
            zip(observations['CONTROL'], observations['FRONTIER_P'])),
        'all paired anchor receipts identical')
scoring = Counter()
score_s = 0.0
window_s = 0.0
adam = 0
for index in range(1, 129):
    stem = f'arms/FRONTIER_P/trainer/events/event_002/allocation_{index:03d}/'
    work = read_raw(stem + 'WORK.json')
    result = read_raw(stem + 'RESULT.json')
    pair = observations['FRONTIER_P'][2 * index - 2:2 * index]
    c = sum(chunks(row) for row in pair)
    expected = {
        'allocation_native_backbone_forward': 8,
        'allocation_freeze_native_head_chunk': c,
        'allocation_native_teaching_head_chunk': 2 * c,
        'allocation_native_reference_backward': 2,
        'allocation_fp32_adam_shadow': 1,
        'allocation_preview_native_head_chunk': 2 * c,
        'allocation_preview_native_backward': 2,
        'allocation_native_token_head_chunk': c,
        'native_committed_response_forward_backward': 2,
    }
    require(work['attempted'] == work['completed'] == expected and not work['pending'],
            f'window {index} complete and matches its own sampled chunk counts')
    require(work['native_adam_steps'] == 1 and result['window'] == index,
            f'window {index} one committed Adam')
    require(result['native']['sampling_replays'] == 0
            and result['native']['cached_anchors'] is True
            and result['native']['native_adam_steps'] == 1,
            f'window {index} cached replay without resampling')
    scoring.update(work['completed'])
    adam += work['native_adam_steps']
    score_s += result['score_s']
    window_s += result['total_window_s']
require(adam == 128, 'candidate scoring window Adam count')
require(dict(scoring) == export['allocation_observed']['completed_operation_counts']
        == export['allocation_observed']['attempted_operation_counts'],
        'independent scoring totals equal exported totals')
require(score_s == export['allocation_observed']['score_s_sum']
        and window_s == export['allocation_observed']['total_window_s_sum'],
        'independent score/window timer sums equal export')

physical = {}
for arm, multiplier, draft, objective_multiplier in (
        ('CONTROL', 4, 256, 2), ('FRONTIER_P', 13, 1280, 4)):
    recorded = events[arm]['objective_work']['work']
    require(recorded['draft_forwards'] == draft
            and recorded['layer_forwards'] == 5 * draft
            and recorded['vocabulary_head_calls'] == multiplier * C
            and recorded['objective_chunks'] == objective_multiplier * C
            and recorded['response_forwards'] == 256,
            arm + ' measured physical counters match independent formula')
    physical[arm] = recorded
training_physical = sum_counts(physical.values())
require(training_physical['vocabulary_head_calls'] == 17 * C
        and training_physical['objective_chunks'] == 6 * C,
        'combined training equals 17C head calls and 6C objective chunks')

witness_counts = []
for endpoint in ('parent', 'CONTROL', 'FRONTIER_P'):
    row = read_raw(f'witness/{endpoint}.json')
    require(row['draft_forwards_attempted'] == row['draft_forwards_completed'] == 16
            and row['head_calls_attempted'] == row['head_calls_completed'] == 16,
            endpoint + ' witness all 16 forwards and heads complete')
    require(row['backwards'] == row['optimizer_steps'] == row['teacher_forwards'] == 0,
            endpoint + ' witness zero backward/Adam/teacher work')
    witness_counts.append(row['physical_work'])
startup_total = sum_counts(startup_counts)
witness_total = sum_counts(witness_counts)
combined = sum_counts([training_physical, startup_total, witness_total])
combined['first_order_backwards'] = 512 + scoring['allocation_native_reference_backward'] + scoring['allocation_preview_native_backward'] + startup_backward
combined['native_adam_steps'] = 256
require(combined['draft_forwards'] == 1588 and combined['first_order_backwards'] == 1028
        and combined['vocabulary_head_calls'] == 17 * C + 4 * startup_chunks + 48
        and combined['objective_chunks'] == 6 * C + 2 * startup_chunks,
        'training plus startup plus witnesses total formula')

require(export['selected_update_displacement_norms']['status'] == 'unavailable',
        'missing production step norms disclosed')
source_hashes = {}
source_root = ROOT.parent / 'implementation_checkout/DFlash-Online/experiments/frontier_preservation_pilot'
for name in ('allocation_scoring.py', 'allocation_training.py'):
    source_hashes[name] = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
result = {
    'schema': 'independent_production_work_formula_audit_v1',
    'status': 'passed',
    'execution_commit': manifest['execution_commit'],
    'scope': 'Local saved-receipt arithmetic and source-bound operation classification only; no Torch, model, GPU or remote operations.',
    'checked_raw_files': len(checked),
    'all_checked_raw_identities_match_compact_manifest': True,
    'source_hashes': source_hashes,
    'retrieval_sha256': hashlib.sha256((ROOT / 'RETRIEVAL.json').read_bytes()).hexdigest(),
    'compact_manifest_sha256': hashlib.sha256((ROOT / 'COMPACT_MANIFEST.json').read_bytes()).hexdigest(),
    'costs_summary_sha256': hashlib.sha256(export_bytes).hexdigest(),
    'checked_raw_bindings_sha256': hashlib.sha256(json.dumps(checked, sort_keys=True).encode()).hexdigest(),
    'checks_passed': len(checks),
    'failed_checks': [],
    'C_from_sampled_anchor_widths': chunk_totals,
    'C_startup_from_sampled_anchor_widths': startup_chunks,
    'scoring_completed_operation_counts': dict(scoring),
    'scoring_attempted_equals_completed_every_window': True,
    'scoring_pending_operations': 0,
    'training_physical_module_counts': physical,
    'combined_training_physical_module_counts': training_physical,
    'combined_training_first_order_backwards': 1024,
    'combined_training_native_adam_steps': 256,
    'startup_physical_module_counts': startup_total,
    'startup_first_order_backwards': startup_backward,
    'startup_native_adam_steps': 0,
    'witness_physical_module_counts': witness_total,
    'witness_first_order_backwards': 0,
    'witness_native_adam_steps': 0,
    'training_startup_witness_totals': combined,
    'source_derived_counts_not_separately_instrumented': {
        'directional_backbone_forwards': 256,
        'optimizer_map_vjps': 128,
        'basis': 'One fast directional pass per response; one optimizer VJP inside each completed allocation_fp32_adam_shadow scope. Directional forwards are already included in total backbone counts.',
    },
    'timer_checks': {
        'scoring_s': score_s,
        'candidate_window_s': window_s,
        'scopes_overlap_do_not_add': True,
        'training_s': {arm: event['costs']['training_s'] for arm, event in events.items()},
    },
    'limitations': [
        'Counts are instrumented module calls and first-order backward invocations, not kernel launches or FLOPs.',
        'Optimizer VJP and directional subcounts are source-derived from completed scopes, not independent runtime counters.',
        'Actual/shadow displacement norms at production updates 1, 64, 128 were not retained; no reconstruction or model rerun.',
        'Per-window attention tile counts and scratch maxima were not retained.',
        'Feedback, teacher preparation, serving and initialization work are outside this scoped audit and must remain separately reported.',
    ],
}
(ROOT / 'WORK_FORMULA_AUDIT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': result['status'], 'checked_raw_files': len(checked), 'checks_passed': len(checks),
                  'C': C, 'training': training_physical, 'all_three_scopes': combined}, indent=2))
